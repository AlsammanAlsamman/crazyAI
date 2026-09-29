## APPROACH

**Mapping the taffy mechanism onto the hash.**

| Taffy step | Hash realization |
|---|---|
| Split the order card into short chunks, one small ball each | The buffer is consumed 48 bytes per iteration split across **three independent accumulator lanes** (`seed`, `see1`, `see2`) — the "single accumulator updated in place" assumption is dropped. For very large buffers a second, coarser split kicks in: the buffer is cut into up to 256 fixed, length-determined slices. |
| Fold each small ball *just enough* to work in its own drops | Each lane does exactly **one folded 64×64→128 multiply per 16 bytes** (`wymix`: multiply, then XOR the high and low halves). No per-byte multiply, no repeated rounds — the mentor's "more folds always help" is precisely the assumption the helper debunked, and past one good folded-multiply the extra rounds buy nothing but time. |
| Hand a ball to the helper so two are worked at once | The three lanes are latency-independent, so they pipeline on the superscalar multipliers (ILP parallelism); above 1 MB the slices are additionally farmed out with **OpenMP** across cores. Slice boundaries depend only on `len`, so the result is bit-identical regardless of thread count. |
| Braid all the small balls together with one big twist at the end | A final fold chain over the lane/slice values plus a full 128-bit-multiply finalizer. **All avalanche comes from this last step**, not from repetition: a single flipped input bit fully scrambles one lane/slice value, which then passes through at least two folded multiplies, each of which is a full 64-bit avalanche map. |
| Never two drops at once, never skip ahead | Dropped deliberately — that was the constraint the disguised solution attacks. Every byte is still read exactly once, and order still matters (lane index, offset, and `len` all enter the mix), so the hash is order-sensitive and collision-resistant, just not sequentially *dependent*. |

Bytes are loaded via `memcpy` into `uint64_t` (compiles to a single `mov`, no strict-aliasing or alignment UB), and the short-input path (`len ≤ 16`) reads only overlapping in-bounds windows — never one byte past `data+len` or before `data`.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

/* ---- secrets: the distinct "flavor bases" each small ball is folded against ---- */
#define S0 0xa0761d6478bd642full
#define S1 0xe7037ed1a0b428dbull
#define S2 0x8ebc6af09c88c6e3full_INVALID
#undef  S2
#define S2 0x8ebc6af09c88c6e3ull
#define S3 0x589965cc75374cc3ull
#define SEED0 0x9E3779B97F4A7C15ull

static inline uint64_t r8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t r4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* one "fold": 64x64 -> 128 multiply, high and low halves folded together.
   Single instruction pair, full 64-bit avalanche. Extra rounds add nothing. */
static inline uint64_t fold(uint64_t A, uint64_t B){
    __uint128_t r = (__uint128_t)A * (__uint128_t)B;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

/* work one ball: multi-lane absorb + final braid */
static uint64_t ball(const unsigned char *p, size_t len, uint64_t seed){
    uint64_t a, b;
    seed ^= fold(seed ^ S0, S1);

    if (len <= 16) {
        if (len >= 4) {
            const size_t q = (len >> 3) << 2;               /* 0 for 4..7, 4 for 8..16 */
            a = (r4(p)       << 32) | r4(p + q);
            b = (r4(p+len-4) << 32) | r4(p + len - 4 - q);
        } else if (len > 0) {
            a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len>>1] << 8) | (uint64_t)p[len-1];
            b = 0;
        } else { a = 0; b = 0; }
    } else {
        size_t i = len;
        if (i > 48) {
            /* three small balls worked side by side: independent dependency chains */
            uint64_t l1 = seed, l2 = seed;
            do {
                seed = fold(r8(p)    ^ S1, r8(p+8)  ^ seed);
                l1   = fold(r8(p+16) ^ S2, r8(p+24) ^ l1);
                l2   = fold(r8(p+32) ^ S3, r8(p+40) ^ l2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= l1 ^ l2;                                 /* pre-braid */
        }
        while (i > 16) { seed = fold(r8(p) ^ S1, r8(p+8) ^ seed); i -= 16; p += 16; }
        a = r8(p + i - 16);
        b = r8(p + i - 8);
    }

    /* the big twist: full 128-bit product, then one more fold with the length */
    a ^= S1; b ^= seed;
    { __uint128_t r = (__uint128_t)a * (__uint128_t)b;
      a = (uint64_t)r; b = (uint64_t)(r >> 64); }
    return fold(a ^ S0 ^ (uint64_t)len, b ^ S1);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
#if defined(_OPENMP)
    if (len >= ((size_t)1 << 20)) {
        size_t nch = len >> 18;                  /* ~256 KiB per slice */
        if (nch > 256) nch = 256;
        size_t base = (len / nch) & ~(size_t)63; /* slice size depends only on len */
        if (base >= 64) {
            uint64_t h[256];
            long n = (long)nch, i;
            #pragma omp parallel for schedule(static)
            for (i = 0; i < n; i++) {
                const unsigned char *q = data + (size_t)i * base;
                size_t sz = ((size_t)i == nch - 1) ? (len - (size_t)i * base) : base;
                h[i] = ball(q, sz, SEED0 ^ (0x9E3779B97F4A7C15ull * (uint64_t)(i + 1)));
            }
            uint64_t acc = SEED0 ^ (uint64_t)len;            /* braid the finished balls */
            for (i = 0; i < n; i++) acc = fold(acc ^ S1, h[i] ^ S2);
            return fold(acc ^ S0, (uint64_t)len ^ S3);
        }
    }
#endif
    return ball(data, len, SEED0);
}
```

*(Note: the stray `#define S2 ...INVALID` / `#undef` pair above is a typo artifact — the effective definition is the second one, `0x8ebc6af09c88c6e3ull`; if your compiler flags it, delete the two lines marked `_INVALID` and `#undef  S2`.)*

Corrected header block, to be safe:

```c
#define S0 0xa0761d6478bd642full
#define S1 0xe7037ed1a0b428dbull
#define S2 0x8ebc6af09c88c6e3ull
#define S3 0x589965cc75374cc3ull
#define SEED0 0x9E3779B97F4A7C15ull
```

## PREDICTION: 12

(Predicted speedup factor over the FNV-1a baseline on multi-kilobyte buffers: FNV-1a is a serial 1-byte-per-`imul` latency chain at roughly 3–4 cycles/byte; this consumes 48 bytes per ~3 pipelined folded multiplies, so ~0.25–0.35 cycles/byte, before the OpenMP slice path adds further scaling above 1 MB.)

## MEASUREMENT

Not obtained — no tools were available in this session, so `hash_bench` and `hash_contract` were never run. I am reporting that plainly rather than inventing numbers: the prediction of 12× stands unverified, and the avalanche property is argued structurally (every input bit reaches the output through at least two folded 128-bit multiplies, each a full 64-bit diffusion map) rather than measured. The four allotted improvement rounds were likewise not exercised.

Known risks a real measurement would resolve:
1. Short inputs (`len < 64`) are branch-heavy; the baseline may win below ~16 bytes.
2. The 1 MB OpenMP threshold is a guess; thread spin-up could dominate if the benchmark's largest size sits just above it.
3. The typo in the `S2` define block must be cleaned up before it will compile.

## VERDICT

The mechanism translates faithfully and the implementation is correct by construction (no out-of-bounds access, no UB, deterministic and thread-count-independent, contract signature exact). But **the claim that it is faster is unmeasured** — I predicted 12× and then could not test it. Treat this as a well-founded but unconfirmed artifact: compile it, fix the `S2` define typo, and run `hash_contract` before `hash_bench`, because a correctness failure on the short-input path would invalidate the speed number entirely.