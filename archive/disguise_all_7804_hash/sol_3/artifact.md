## APPROACH

**Mapping the bakery story onto the hash.**

| Disguised element | Real implementation |
|---|---|
| The shelf of jars | The byte buffer `data[0..len)` |
| One bowl, one long march jar-by-jar | FNV‑1a: one accumulator, one byte per multiply, strictly in order |
| Two people, two bowls | Two independent 64‑bit accumulators `A` (forward) and `B` (backward) |
| Person 1 starts at jar 1 walking forward | Pointer `p` ascending from `data` |
| Person 2 starts at the last jar walking backward | Pointer `q` descending from `data + len` |
| "A scoop" | A 16‑ or 32‑byte block load, not one byte — the story never said the scoop must be one grain; it only said the *order* of visits is what's being freed |
| "Twist‑stir" | A 64×64→128 multiply folded by XOR of the halves (`fold`), the strongest single-instruction diffuser available, plus a rotate-and-add into the bowl so the stir is order-dependent within a lane |
| They meet in the middle | Loop runs while `q - p >= 32`; the leftover middle is absorbed by one final overlapping scoop per walker |
| Pour the two bowls together, one last firm twist‑stir | `fold(A^K2, B^K3) ^ (A+B)` then a MurmurHash3‑class 2‑round finalizer — this is what guarantees the avalanche property |
| Every jar folded in exactly once | Every byte is read at least once; near the meeting point a few bytes may be re-read by the overlapping tail scoop, which is harmless |

**Why this is fast, mechanistically.** FNV‑1a is *latency*-bound: the next byte cannot be mixed until the previous multiply retires, so throughput is ~1 byte per 4–5 cycles (≈0.2 B/cycle). The two-walker structure breaks that dependency chain in two ways at once: (1) two bowls means two independent chains running in parallel, and (2) the `rotl(A,29) + fold(...)` update keeps the *multiply off the critical path* — the fold of the freshly loaded block does not depend on the accumulator, so the per-block critical path is rotate+add ≈ 2–3 cycles while several multiplies are in flight. The bulk loop therefore becomes load-port/uop bound rather than multiply-latency bound: ~64 bytes per ~6–7 cycles (≈9–10 B/cycle).

**Why avalanche survives.** Each block enters through `fold(x ^ K, y ^ K')` (constants XOR'd on *both* operands so an all-zero block still produces a non-degenerate product — dropping either constant would make a one-bit flip in an all-zero buffer vanish entirely). A single flipped input bit perturbs one 128-bit product by `operand << k`, which changes many bits of the folded 64-bit result, propagates through rotate/add carries into the bowl, and finally passes through the two multiply–xorshift rounds of the finalizer, which is the standard construction measured at ~50% bit-flip probability.

**Rejected alternatives** (would have been a different mechanism, not this one): OpenMP shards, AVX2 `vpmuludq` lanes, or 4–8 accumulators. All faster in principle, all abandon "two people from opposite ends."

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Odd, high-entropy mixing constants (wyhash-family). */
#define KP0 0xa0761d6478bd642fULL
#define KP1 0xe7037ed1a0b428dbULL
#define KP2 0x8ebc6af09c88c6e3ULL
#define KP3 0x589965cc75374cc3ULL
#define KP4 0x1d8e4e27c47d124fULL
#define KP5 0xeb44accab455d165ULL

static inline uint64_t hk_r8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t hk_r4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t hk_rotl(uint64_t x,int r){ return (x<<r)|(x>>(64-r)); }

/* "twist-stir": 64x64 -> 128 multiply, folded to 64 bits by XOR of halves. */
static inline uint64_t hk_fold(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al=a&0xffffffffULL, ah=a>>32, bl=b&0xffffffffULL, bh=b>>32;
    uint64_t ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t cross=(ll>>32)+(lh&0xffffffffULL)+hl;
    uint64_t hi=(lh>>32)+(cross>>32)+hh;
    uint64_t lo=(cross<<32)|(ll&0xffffffffULL);
    return lo ^ hi;
#endif
}

/* final firm twist-stir: MurmurHash3-class finalizer (full avalanche). */
static inline uint64_t hk_final(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t A = KP0 ^ (uint64_t)len;   /* bowl of the walker going forward  */
    uint64_t B = KP1 + (uint64_t)len;   /* bowl of the walker going backward */

    if (len < 16) {                      /* short shelf: the two meet immediately */
        if (len >= 8) {
            A ^= hk_r8(data);
            B ^= hk_r8(data + len - 8);
        } else if (len >= 4) {
            A ^= (uint64_t)hk_r4(data);
            B ^= (uint64_t)hk_r4(data + len - 4);
        } else if (len) {
            A ^= ((uint64_t)data[0] << 16) |
                 ((uint64_t)data[len >> 1] << 8) |
                 ((uint64_t)data[len - 1]);
            B ^= (uint64_t)data[len - 1] * KP5;
        }
        return hk_final(hk_fold(A ^ KP2, B ^ KP3) ^ (A + B));
    }

    const unsigned char *p = data;        /* forward walker  */
    const unsigned char *q = data + len;  /* backward walker (exclusive) */
    size_t rem = len;                     /* bytes still between them: q - p */

    /* Both walk toward the middle, 32 bytes each per step. */
    while (rem >= 64) {
        uint64_t a0 = hk_r8(p),      a1 = hk_r8(p + 8);
        uint64_t a2 = hk_r8(p + 16), a3 = hk_r8(p + 24);
        q -= 32;
        uint64_t b0 = hk_r8(q),      b1 = hk_r8(q + 8);
        uint64_t b2 = hk_r8(q + 16), b3 = hk_r8(q + 24);
        A = hk_rotl(A, 29) + hk_fold(a0 ^ KP1, a1 ^ KP2)
                           + hk_fold(a2 ^ KP3, a3 ^ KP4);
        B = hk_rotl(B, 31) + hk_fold(b0 ^ KP4, b1 ^ KP3)
                           + hk_fold(b2 ^ KP2, b3 ^ KP1);
        p += 32;
        rem -= 64;
    }
    /* 16 bytes each per step as they close in. */
    while (rem >= 32) {
        uint64_t a0 = hk_r8(p), a1 = hk_r8(p + 8);
        q -= 16;
        uint64_t b0 = hk_r8(q), b1 = hk_r8(q + 8);
        A = hk_rotl(A, 29) + hk_fold(a0 ^ KP1, a1 ^ KP2);
        B = hk_rotl(B, 31) + hk_fold(b0 ^ KP4, b1 ^ KP3);
        p += 16;
        rem -= 32;
    }

    /* The meeting: one last overlapping scoop each, covering every leftover jar.
       In-bounds for all len >= 16 because p and q are symmetric about the middle. */
    A = hk_rotl(A, 29) + hk_fold(hk_r8(p)      ^ KP1, hk_r8(p + 8)  ^ KP2);
    B = hk_rotl(B, 31) + hk_fold(hk_r8(q - 16) ^ KP4, hk_r8(q - 8)  ^ KP3);

    /* Pour the two bowls together, one final firm twist-stir. */
    return hk_final(hk_fold(A ^ KP2, B ^ KP3) ^ (A + B));
}
```

## PREDICTION

Speedup over the FNV‑1a baseline (harmonic-ish average over the benchmark's buffer sizes; L1/L2-resident sizes should show ~25–35×, main-memory sizes ~10× because they become bandwidth-bound):

PREDICTION: 18

## MEASUREMENT

**Not performed — and I am not going to dress that up.** The session states plainly that no tools are available (`hash_bench`, `hash_contract`, and the rest are listed but not callable here), so there is no measured number to report and the four-round improvement loop never ran. Everything below is *estimate*, not evidence, and should be treated as such until the pipeline compiles and runs it:

- Expected bulk throughput: ~8–10 bytes/cycle (load-port and uop bound; critical path per 64‑byte iteration is rotate+add+add ≈ 3 cycles per lane, with 4 `mulx` in flight off the chain).
- Expected FNV‑1a baseline: ~0.2–0.25 bytes/cycle (multiply-latency bound, 4–5 cycles per byte).
- Implied ratio: ~35× while data is cache-resident, collapsing toward ~10× once the buffer exceeds L2 and DRAM bandwidth dominates — hence the single blended figure of 18.
- Avalanche: expected ~50% output-bit flip per input-bit flip, inherited from the MurmurHash3 finalizer; I did *not* run a bit-independence test, so this is an argument from construction, not a measurement.

Risks I would look for first if the numbers come back wrong: (a) the benchmark weighted heavily toward very short inputs, where fixed finalizer cost caps the win near 2–4× and 18 would be a large overprediction; (b) the dual opposite-direction streams interacting badly with the hardware prefetcher on very large buffers; (c) a contract checker objecting to the overlapping tail reads near the meeting point (they are in-bounds, but they do read a few bytes twice).

## VERDICT

The mechanism translates cleanly and honestly: two accumulators, two pointers converging from opposite ends, wide multiply-fold stirring, a single joint finalizer at the meeting point. It should be substantially faster than byte-at-a-time FNV‑1a for a structural reason, not an incidental one — FNV‑1a's cost is a serialized multiply chain, and two converging lanes with the multiply lifted off the accumulator's critical path attack exactly that. The avalanche requirement is carried by the final two-round multiply–xorshift stir, which is the part of the design I would defend most confidently.

But the claim is unverified. **I predicted 18× and then could not measure it**, so this answer is a design plus a falsifiable number, not a demonstrated result. If the harness reports anything under ~8×, the prediction was wrong and the most likely cause is a short-input-dominated benchmark — the fix would be to cheapen the short path (single-round finalizer below 16 bytes), not to abandon the two-walker mechanism.