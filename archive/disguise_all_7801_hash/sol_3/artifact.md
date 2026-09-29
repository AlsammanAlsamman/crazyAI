## APPROACH

**Mapping the disguised solution onto the real problem.** The strict-Granny baseline (FNV‑1a) is a *lockstep* loop: read one byte, fold it into the one accumulator, multiply (the "twist"), and only then read the next byte. Every byte sits on a serial `mul` dependency chain, so throughput is pinned at one 3–5‑cycle multiply *per byte* — roughly 0.2–0.3 B/cycle, and adding rounds ("just twist it a few more times") makes it strictly slower without making it meaningfully better.

Mabel's fix, translated literally:

| Disguised step | Real implementation |
|---|---|
| "Walk the whole sack out into labeled saucers first, no wax touched" | Bulk **load** phase per block: six independent unaligned 8‑byte loads (`memcpy` → single `mov`), issued with zero dependency on the hash state. Loads run ahead in the out‑of‑order window; nothing waits on a multiply. |
| "Only then start folding, in any order you like" | The absorbed words are combined **order‑free** within a block — three lanes touch disjoint 16‑byte slices, so their schedule is up to the machine, not fixed by the byte order. |
| "Several charms shoved in per twist" | **16 bytes per multiply** instead of 1. Each `mum` is a 64×64→128 widening multiply (`mulx`) that swallows two whole words, keeping *both* halves of the product so no entropy is thrown away. |
| "One blob of wax on the stick" | The contract is still a single 64‑bit value. Three lanes are saucers, not blobs: they collapse (`seed ^= s1 ^ s2`) into one accumulator, then one final `mum` + strong finalizer. |
| "Reading and twisting are two separate errands, back to back" | Per 48‑byte block the critical path is *one* multiply per lane; the three lanes are independent, so the serial chain is ~1 mul per 48 bytes rather than 48 muls. |

**Why this doesn't cost avalanche.** Each byte still passes through a full 64×64→128 multiply, and both product halves are retained (`a^b`), so a single input bit flip propagates across the whole lane word. The lanes merge by XOR and the result goes through a final `mum` plus a `mix` with the length folded in — this is the wyhash‑final3 structure, which clears SMHasher's avalanche and bit‑independence tests. I deliberately did *not* take the tempting extra speedup of XOR‑combining two input words into one multiply operand (32 B/mul): that creates exact differential collisions (flip the same bit in both words), which is the "bad blob" Granny warned about.

**Invariant held:** no read ever goes outside `[data, data+len)`. Tails re‑read the last 16 bytes with overlap, and the 1–3 byte case reads only `p[0]`, `p[len>>1]`, `p[len-1]`.

**Bottleneck analysis (why I stopped where I did).** At 16 B per multiply and one multiplier port at 1 `mulx`/cycle, the ceiling is ~16 B/cycle ≈ 45–50 GB/s at 3 GHz — already at or past DRAM bandwidth for large buffers and near L1 limits for small ones. Adding a 6‑lane/96‑byte loop would not move the mul‑port bound, so extra lanes are dead weight; an AVX2 `vpmuludq` (xxh3‑style) absorb is the only route past it, and I judged its avalanche/collision risk unjustified here.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Mixing secrets: high-quality 64-bit constants (wyhash default secret),
   each with balanced popcount and no small factors. */
#define K0 0xa0761d6478bd642fULL
#define K1 0xe7037ed1a0b428dbULL
#define K2 0x8ebc6af09c88c6e3ULL
#define K3 0x589965cc75374cc3ULL

/* ---- "labeled saucers": pure loads, no state dependency ---- */
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; __builtin_memcpy(&v, p, 8); return v;
}
static inline uint64_t rd4(const unsigned char *p) {
    uint32_t v; __builtin_memcpy(&v, p, 4); return (uint64_t)v;
}
static inline uint64_t rd3(const unsigned char *p, size_t k) {
    /* touches only p[0], p[k>>1], p[k-1] : never out of bounds for 1<=k<=3 */
    return (((uint64_t)p[0]) << 16) | (((uint64_t)p[k >> 1]) << 8) | (uint64_t)p[k - 1];
}

/* ---- "one hard twist": 64x64 -> 128, both halves kept ---- */
static inline void mum(uint64_t *a, uint64_t *b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)(*a) * (__uint128_t)(*b);
    *a = (uint64_t)r;
    *b = (uint64_t)(r >> 64);
#else
    uint64_t ha = *a >> 32, la = (uint32_t)*a, hb = *b >> 32, lb = (uint32_t)*b;
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32), c = t < rl;
    uint64_t lo = t + (rm1 << 32); c += lo < t;
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    *a = lo; *b = hi;
#endif
}
static inline uint64_t mix(uint64_t a, uint64_t b) { mum(&a, &b); return a ^ b; }

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    uint64_t seed = mix(K0, K1);          /* constant-folded at compile time */
    uint64_t a, b;

    if (len <= 16) {
        if (len >= 4) {
            size_t d = (len >> 3) << 2;
            a = (rd4(p) << 32)           | rd4(p + d);
            b = (rd4(p + len - 4) << 32) | rd4(p + len - 4 - d);
        } else if (len > 0) {
            a = rd3(p, len); b = 0;
        } else {
            a = 0; b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            /* three independent lanes: 48 bytes absorbed per 3 multiplies,
               critical path = ONE multiply per lane per block */
            uint64_t s1 = seed, s2 = seed;
            do {
                uint64_t w0 = rd8(p),      w1 = rd8(p + 8);
                uint64_t w2 = rd8(p + 16), w3 = rd8(p + 24);
                uint64_t w4 = rd8(p + 32), w5 = rd8(p + 40);
                seed = mix(w0 ^ K1, w1 ^ seed);
                s1   = mix(w2 ^ K2, w3 ^ s1);
                s2   = mix(w4 ^ K3, w5 ^ s2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= s1 ^ s2;              /* saucers collapse into the one blob */
        }
        while (i > 16) {
            seed = mix(rd8(p) ^ K1, rd8(p + 8) ^ seed);
            i -= 16; p += 16;
        }
        a = rd8(p + i - 16);              /* overlapping tail, always in bounds */
        b = rd8(p + i - 8);
    }

    a ^= K1; b ^= seed;
    mum(&a, &b);
    return mix(a ^ K0 ^ len, b ^ K1);    /* finalizer: length folded in */
}
```

## PREDICTION

PREDICTION: 12

(Predicted speedup factor over the FNV‑1a reference on a mixed-length workload. Reasoning stated before any measurement: FNV‑1a is latency-bound at ~1 byte per ~4–5 cycles; this kernel is throughput-bound at ~16 bytes per multiply with a ~1-multiply-per-48-byte serial chain. Pure long-buffer ratio should be 20–60×, but short inputs and per-call overhead in a mixed sweep drag the harness average down, so 12 is my honest expected aggregate.)

## MEASUREMENT

**Not performed.** No tools were available in this session — `hash_bench` and `hash_contract` were listed but are not callable here, so I could not compile, run the avalanche check, or time anything. I am reporting that plainly rather than inventing numbers: every figure above is an a‑priori estimate from instruction-level reasoning (mul-port throughput, dependency-chain length), not a measurement, and the four allotted improvement rounds were not exercised.

What I would expect the harness to show, and what would falsify me:
- Long buffers (≥64 KiB): 8–16 B/cycle, i.e. 25–45 GB/s. If it lands under ~5 GB/s, my mul-port model is wrong and the loop is load- or bandwidth-limited instead.
- Short inputs (≤16 B): a handful of cycles plus call overhead; speedup here will be small (~2–4×) and will dominate a mixed-size average.
- Avalanche: ~50% ± 1–2% output bit flips per input bit flip, uniform across all bit positions. A localized dip (e.g. only the top bits of the output moving for late-buffer bytes) would indicate the final `mum`/`mix` is not scrambling enough and would call for one extra xor-shift in the finalizer.

Had measurement been available, my ordered improvement list was: (1) verify avalanche before touching speed; (2) unroll the 48-byte loop 2× to cut loop overhead on large inputs; (3) add a software prefetch at +256 B if large-buffer throughput sits below DRAM bandwidth; (4) only if still mul-port-bound *and* avalanche has headroom, an AVX2 `vpmuludq` absorb for `len ≥ 512` with the scalar path retained for everything else.

## VERDICT

The artifact is complete, obeys the contract exactly (`uint64_t kernel(const unsigned char *, size_t)`), reads no byte outside the buffer, handles `len == 0`, and faithfully implements the disguised mechanism: a dependency-free bulk-load phase followed by an order-free multi-byte-per-twist absorb into a single accumulator, replacing the byte-at-a-time lockstep chain. Structurally it is the wyhash-final3 core, which is known to pass SMHasher's avalanche suite, so I have high confidence in correctness and quality.

**But the honest verdict is incomplete, not successful.** The task asked me to measure and then improve up to four times; I did neither, because the tools were absent. My speed claim is a prediction resting on an instruction-throughput model, and the avalanche claim rests on the published provenance of the construction rather than on a run in this harness. Treat `PREDICTION: 12` as an unvalidated hypothesis, and please re-run with `hash_bench`/`hash_contract` available before relying on it.