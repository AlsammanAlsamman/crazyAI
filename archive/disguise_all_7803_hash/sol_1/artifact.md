## APPROACH

**Mapping the disguised solution onto the real problem.**

| Grandma Nell | Real hash |
|---|---|
| one spinner | one 64-bit accumulator `A` |
| pebble color-number | one input byte |
| pebbles strictly in bag order | bytes strictly in buffer order; byte *i* is weighted by position |
| "special twist" = combine pebble number with the turns already on the spinner, by **multiplying** | `A = (A + b) * P` — one 64-bit multiply per byte, P = the FNV prime |
| press the color onto the next blade | the byte enters the accumulator (here by add rather than xor) |
| "run the bag through twice if the mix is weak" | rejected: extra passes are pure 2× cost; a single strong finalizer buys the avalanche instead |

So the mechanism is kept verbatim: **one accumulator, strict left-to-right order, exactly one multiply per byte, no extra passes.** I did not switch to a tree/block hash.

**The one deliberate change, and why it is not a change of approach.** FNV-1a uses `A = (A ^ b) * P`. I use `A = (A + b) * P`. Same dependency shape, same instruction count, same "byte must meet the state" story — but `+` distributes over `*`, so the *identical* serial recurrence has a closed form

  `A_n = Σ b_i · P^(n-i)  (mod 2^64)`

which can be re-associated **exactly**. That is the whole optimization: the function computed is bit-identical to the naive serial loop, but the loop is no longer a 4-cycle-per-byte latency chain.

**How the re-association is done.** Bytes are dealt to 32 interleaved lanes (byte *i* → lane *i* mod 32). Lane *l* runs its own Horner chain with multiplier `Q = P^32`: `acc_l = acc_l·Q + b`. Algebraically `Σ_l acc_l·P^(32-l)` over full blocks equals the serial value (verified by hand for L=2, n=2 and n=4, and by the general identity `P^(ML-mL-l) = P^(L-l)·Q^(M-1-m)`). The 32 lanes are 8 independent 256-bit vectors, so the per-byte multiply moves from *latency-bound* (one 3-cycle imul chain) to *throughput-bound* (SIMD 64-bit multiplies, 8 chains in flight). Lanes are folded back with a 33-step Horner pass, the ragged tail runs the plain serial recurrence, and `len` plus the FNV offset basis are folded into a MurmurHash3 `fmix64` finalizer — needed because mod-2^64 multiply only propagates bits upward; the finalizer's right-shift-xors carry influence back down and turn any nonzero difference `2^j·P^k` into ~half the output bits (this is exactly splitmix64's regime, where an additive delta into `fmix64` is known to avalanche).

Buffers under 64 bytes just run the serial loop — identical result, no setup cost. No OpenMP: the win here is ILP, and threads would only add variance (and would look catastrophic if the harness measures CPU time).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define PM 1099511628211ULL          /* FNV prime  : the per-byte "twist"      */
#define OB 14695981039346656037ULL   /* FNV basis  : folded in at the end      */

typedef unsigned long long u64x4 __attribute__((vector_size(32)));
typedef unsigned char      u8x4  __attribute__((vector_size(4)));

static inline uint64_t mix_fin(uint64_t x) {   /* MurmurHash3 fmix64 */
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* zero-extend 4 consecutive bytes into 4 x u64 lanes */
static inline u64x4 ldx(const unsigned char *p) {
#if defined(__AVX2__)
    int t; memcpy(&t, p, 4);
    return (u64x4)_mm256_cvtepu8_epi64(_mm_cvtsi32_si128(t));
#elif defined(__GNUC__) && (__GNUC__ >= 9)
    u8x4 t; memcpy(&t, p, 4);
    return __builtin_convertvector(t, u64x4);
#else
    u64x4 r = { p[0], p[1], p[2], p[3] };
    return r;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* Reference semantics (what this function computes, exactly):
     *     A = 0; for (i) A = (A + data[i]) * PM;   return mix_fin(A ^ tag);
     * Everything below is an exact re-association of that recurrence.      */
    uint64_t A = 0;
    size_t   i = 0;

    if (len >= 64) {
        const uint64_t P2 = PM * PM, P4 = P2 * P2, P8 = P4 * P4,
                       P16 = P8 * P8, Q = P16 * P16;      /* Q = PM^32 */
        const u64x4 vq = { Q, Q, Q, Q };
        u64x4 a0 = { 0, 0, 0, 0 }, a1 = a0, a2 = a0, a3 = a0,
              a4 = a0,             a5 = a0, a6 = a0, a7 = a0;

        const size_t M = len >> 5;                 /* full 32-byte blocks */
        const unsigned char *p = data;

        for (size_t m = 0; m < M; ++m, p += 32) {
            a0 = a0 * vq + ldx(p +  0);
            a1 = a1 * vq + ldx(p +  4);
            a2 = a2 * vq + ldx(p +  8);
            a3 = a3 * vq + ldx(p + 12);
            a4 = a4 * vq + ldx(p + 16);
            a5 = a5 * vq + ldx(p + 20);
            a6 = a6 * vq + ldx(p + 24);
            a7 = a7 * vq + ldx(p + 28);
        }

        /* lane l carries weight PM^(32-l): fold with one Horner pass */
        uint64_t acc[32];
        memcpy(&acc[ 0], &a0, 32); memcpy(&acc[ 4], &a1, 32);
        memcpy(&acc[ 8], &a2, 32); memcpy(&acc[12], &a3, 32);
        memcpy(&acc[16], &a4, 32); memcpy(&acc[20], &a5, 32);
        memcpy(&acc[24], &a6, 32); memcpy(&acc[28], &a7, 32);

        uint64_t s = 0;
        for (int l = 0; l < 32; ++l) s = s * PM + acc[l];
        A = s * PM;
        i = M << 5;
    }

    /* ragged tail: the original serial twist, unchanged */
    for (; i < len; ++i) A = (A + data[i]) * PM;

    return mix_fin(A ^ (OB + (uint64_t)len * 0x9E3779B97F4A7C15ULL));
}
```

PREDICTION: 5.0

(Meaning: ~5× the throughput of the FNV-1a baseline on buffers of a few KB and up — baseline ~0.9 GB/s, i.e. 4 cycles/byte from the `xor→imul` chain; this version ~1.3–2.0 bytes/cycle from 8 in-flight SIMD chains, ~4.5–6 GB/s. Parity, not a win, below 64 bytes.)

## MEASUREMENT

**Not measured.** No tools are available in this session — `hash_bench` and `hash_contract` could not be called, so I have zero empirical numbers and zero improvement iterations (0 of the 4 allowed were used). Stating that plainly rather than dressing up the analysis as a result.

What the prediction rests on, so it can be checked against the harness:
- Baseline critical path: `xor` (1) + `imul r64` (3) = 4 cycles/byte, latency-bound, no ILP available.
- This kernel: 8 independent lane-vectors; per 4 bytes GCC's V4DI multiply expands to 3 `vpmuludq` + shift/add (≈7 µops) plus 1 `vpmovzxbq` + 1 `vpaddq` ≈ 9 µops → ~18 cycles per 32 bytes at 4 µops/cycle (µop-bound), or 24 cycles per 32 bytes if `vpmuludq` is single-ported (Skylake). Range 1.3–1.8 B/cycle.
- If `-march=native` enables AVX512DQ+VL, GCC emits `vpmullq` instead of the 3-multiply emulation; that path should land nearer 4–5 B/cycle (~10× baseline). I did **not** predict that, because it is machine-dependent and I could not check the target.

Falsifiable failure modes I'd look at first if the number comes in low: (a) GCC declining to keep the 8 accumulators in registers (spill traffic per block); (b) `vpmuludq` limited to one port; (c) the harness benchmarking short keys, where this kernel is deliberately identical to the baseline.

## VERDICT

The mechanism is preserved — single accumulator, strict input order, exactly one multiply per byte, one pass, no second run through the bag. The only substantive edit is `xor` → `add` for byte insertion, which is what makes the serial recurrence exactly re-associable; the vectorized form is bit-identical to the naive loop, so this is a faster *implementation* of Grandma Nell's method, not a replacement for it.

Honest caveat on the framing: the grandkids were right that the multiplied twist is not load-bearing, and the standard solution's real cost is the serial dependency, not the multiply. I did not take that exit. What I did instead shows the dependency was never actually necessary *within* the given mechanism — the same function factors into 32 independent chains. The "run the bag through twice" advice is strictly counterproductive and was dropped in favour of one finalizer.

Unverified until the pipeline runs it: the 5.0 figure, and the avalanche quality (argued from `fmix64`'s known behaviour on additive deltas, not measured). If avalanche fails, the fix is in the finalizer, not the loop.