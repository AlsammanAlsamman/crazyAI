## APPROACH

**Mapping the disguised solution onto the real problem**

| Granny's field | Hash kernel |
|---|---|
| The row of flowers, west→east, visited once, no doubling back | One forward sequential pass over `data[0..len)`, each byte read exactly once (except the deliberate tail overlap) |
| The single jar on the belt | The standard solution's single accumulator `h` — the assumption being broken |
| Four jars A, B, C, D | Four independent 64-bit accumulators `A,B,C,D`, each with a *different* nonzero initial "starter" so identical pours don't produce identical jars |
| Pouring flower *i* into jar *i* mod 4 | Round-robin: consecutive chunks of the buffer go to lanes A,B,C,D,A,B,C,D… so the four dependency chains are disjoint |
| One firm twist-stir per pour, the other three jars untouched | Exactly one mixing step per chunk, applied only to the receiving lane. The three idle lanes are what the CPU actually exploits: four independent multiply chains issue in parallel instead of serializing. This is where the speed comes from — not from stirring less, but from the stirs no longer queueing behind each other. |
| The younger beekeepers' three-or-four stirs per flower | The "more rounds = better mixing" assumption, explicitly rejected: one stir per pour, but a *strong* stir |
| Pour all four into one big jar, one final hard twist at the gate | Final combine: fold the lanes pairwise through the mixer, xor in `len`, then a proven xor-shift-multiply avalanche finisher so a one-bit change in any lane reaches all 64 output bits |
| "Roughly half of color, thickness, scent must change if one flower changes" | Avalanche requirement: one input bit flip ⇒ ~32 of 64 output bits flip |

**What the "hard twist-stir" is.** The single stir per pour is a folded widening multiply: `fold(x,y) = lo64(x*y) ^ hi64(x*y)`, one `mulx` instruction on x86-64 that yields both halves. It takes two operands, so one stir absorbs **16 bytes** — I define a "spoonful" as 16 bytes rather than 1 byte, and I say that plainly: that is the one place I loosened granularity, because absorbing 8 bytes per multiply operand is what makes the mechanism beat FNV-1a by more than the 4× that four lanes alone would buy. Structure (round-robin into four separately-stirred jars, one stir per pour, single combine at the gate) is preserved exactly. One operand of each stir carries the lane's own secret, the other carries the lane's running state, so state and data both enter the nonlinear step.

**Why this is faster than the known way.** FNV-1a is latency-bound: every byte sits behind a 3-cycle multiply plus xor in one chain — ~4–5 cycles *per byte*. Here the critical path is one `mulx`+`xor` per lane per 64 bytes, four chains deep in parallel, 4 multiplies and 8 loads per 64 bytes. The bottleneck moves off the multiplier entirely and onto load/memory bandwidth (~0.06 cycles/byte in cache). No threads and no SIMD: the four jars are ILP lanes, not four beekeepers — one walker, one pass, as specified (and 64-bit×64-bit→128-bit has no AVX2 equivalent worth emulating).

**Deliberate safety choices:** lane starters are all nonzero, so an all-zero buffer never feeds a zero operand into a multiply (which would collapse a lane to 0 and lose history); `len` is folded in so length changes are visible; the tail reads the last 8/4 bytes overlapping to stay branch-light while never reading out of bounds; `len == 0` dereferences nothing.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Four jars on the belt: distinct nonzero starters / stir secrets. */
#define S0 0xa0761d6478bd642fULL
#define S1 0xe7037ed1a0b428dbULL
#define S2 0x8ebc6af09c88c6e3ULL
#define S3 0x589965cc75374cc3ULL
#define S4 0x1d8e4e27c47d124fULL

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;          /* one mov at -O3 */
}
static inline uint64_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* ONE hard figure-eight twist-stir: widening multiply, halves folded together.
   Absorbs two 64-bit operands for a single multiply instruction. */
static inline uint64_t stir(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    uint64_t hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (a * b) ^ hi;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;

    /* Four jars, each starting from its own nonzero base. */
    uint64_t A = S0, B = S1, C = S2, D = S3;

    /* The walk: one pass, round-robin A,B,C,D; each pour gets its own single
       stir while the other three jars sit untouched (4 independent chains). */
    while (n >= 64) {
        A = stir(ld64(p +  0) ^ S1, ld64(p +  8) ^ A);
        B = stir(ld64(p + 16) ^ S2, ld64(p + 24) ^ B);
        C = stir(ld64(p + 32) ^ S3, ld64(p + 40) ^ C);
        D = stir(ld64(p + 48) ^ S4, ld64(p + 56) ^ D);
        p += 64; n -= 64;
    }
    /* Keep going round the jars for whole 16-byte spoonfuls that remain. */
    if (n >= 16) { A = stir(ld64(p) ^ S1, ld64(p + 8) ^ A); p += 16; n -= 16; }
    if (n >= 16) { B = stir(ld64(p) ^ S2, ld64(p + 8) ^ B); p += 16; n -= 16; }
    if (n >= 16) { C = stir(ld64(p) ^ S3, ld64(p + 8) ^ C); p += 16; n -= 16; }

    /* Last partial spoonful (0..15 bytes) goes into D. Overlapping reads stay
       strictly inside the buffer; nothing is read when len == 0. */
    uint64_t x, y;
    if (n >= 8)      { x = ld64(p);            y = ld64(p + n - 8); }
    else if (n >= 4) { x = ld32(p);            y = ld32(p + n - 4); }
    else if (n != 0) { x = ((uint64_t)p[0] << 16) |
                           ((uint64_t)p[n >> 1] << 8) |
                            (uint64_t)p[n - 1];       y = 0; }
    else             { x = 0;                  y = 0; }
    D = stir(x ^ S4, y ^ D);

    /* At the east gate: pour all four jars into the big jar, one final hard
       twist (fold + full 64-bit avalanche finisher). */
    uint64_t h = stir(A ^ S1, B ^ S2) ^ stir(C ^ S3, D ^ S4) ^ (uint64_t)len;
    h ^= h >> 33; h *= 0xc2b2ae3d27d4eb4fULL;
    h ^= h >> 29; h *= 0x165667b19e3779f9ULL;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

PREDICTION: 20

(Meaning: ~20× the throughput of the FNV-1a reference on a cache-resident buffer of a few KB. Reasoning stated before any measurement: FNV-1a is a serial 1-byte-per-multiply dependency chain at ~4–5 cycles/byte; this kernel needs 4 multiplies and 8 loads per 64 bytes across 4 independent chains, so it should land at ~0.1 cycles/byte in cache — a ~40× ceiling, which I discount to 20 for loop/tail overhead, small-size cases in the mix, and memory bandwidth on larger buffers. Avalanche: expect ~50% bit-flip rate, passing the one-bit-flip test.)

## MEASUREMENT

**Not measured — and I will not dress that up.** This session was invoked with no tools available: `hash_bench` and `hash_contract` could not be called, so there is no measured number to report and no measurement-driven improvement iterations took place (0 of the allowed 4 were used). Everything above is static reasoning about instruction-level cost, not observed timing. The prediction of 20× stands as an unverified estimate for the pipeline to confirm or refute.

What I *can* assert without a benchmark, by inspection:
- **Contract**: signature is exactly `uint64_t kernel(const unsigned char *data, size_t len)`; no globals, no allocation, no seed, deterministic, thread-safe, pure.
- **Memory safety**: every read path is bounded — the 64-byte loop requires `n >= 64`, each 16-byte step requires `n >= 16`, and the tail reads at most `p[0..n-1]` for `n <= 15` with `n == 0` reading nothing (safe for `data == NULL, len == 0`).
- **Portability**: `__int128` fast path with a correct scalar 64×64→128 fallback; unaligned access only via `memcpy`.

Where it would most plausibly fall short of 20×: if the harness benches very small inputs (≤32 bytes), fixed cost and the tail branches dominate and the realistic factor is more like 3–6×; if it benches DRAM-resident buffers (≥ tens of MB), both kernels' ceiling is bandwidth and the factor lands nearer 10–15×.

## VERDICT

The mechanism translates cleanly and, I believe, wins substantially — four independent stirred accumulators fed round-robin, combined once at the end, converts FNV-1a's latency-bound serial chain into a load-bound parallel one while keeping (per the finisher) full 64-bit avalanche. But the claim is **unverified in this session**: the benchmarking tools were absent, so I have a prediction and an argument, not a result. Treat 20× as a stated bet, not a measurement; run `hash_bench` and `hash_contract` against the artifact above to settle it.