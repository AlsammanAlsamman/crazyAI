## APPROACH

**How Rosa's method maps onto the real problem.**

| Bakery | Hash |
|---|---|
| jar *i* on the shelf | input unit at offset *i* |
| one bowl | one accumulator `h` |
| scoop a spoonful in | combine the unit into `h` (add) |
| hard twist-stir | multiply by an odd 64-bit constant |
| strict shelf order, never skip or double back | the fold is order-sensitive: unit *i* carries weight `M^(N-i)` |
| "different all over if one jar changes" | avalanche |
| walking the whole bakery | the serial multiply-latency chain, 4 cycles per byte |

The candidate solution is exactly FNV-1a: one bowl, one stir per jar, strict order. I keep that mechanism and attack only the *walking*.

The key observation is that Rosa's walking order and the *recipe's* order are two different things. The stir is multiplication by a constant, and multiplication distributes over the scoop (addition), so the fold

```
h ← (h + w_i) · M       for i = 0 … N-1
```

is exactly Horner's rule, i.e. `h = Σ w_i · M^(N-i)`. Each jar's *position on the shelf* is recorded in its exponent. So the shelf order is preserved perfectly in the answer even though the jars can be visited in any physical order: **the arithmetic remembers which room each jar was in.** That is "improve how you implement this mechanism", not "replace the mechanism" — same single accumulator, same one-stir-per-unit, same strict order, same one pass.

Concretely I split the stream into 8 interleaved lanes (jars `i ≡ j mod 8`). Lane *j* runs its own Horner fold with base `Q = M^8`, then the eight lanes are recombined with the position weights `M^7 … M^0`. This is algebraically the same order-sensitive fold, but the 8 lanes are independent, so the 4-cycle multiply latency is hidden and we become multiply-*throughput* bound (1 `imul`/cycle on p1) instead of latency bound.

Two more implementation-level (not mechanism-level) choices:
- **The spoon is 8 bytes wide.** One 64-bit word per stir instead of one byte, so the multiply/byte ratio is 8× better. Order is still strictly respected.
- **Add, not xor, to scoop.** This is the FNV-1 vs FNV-1a difference and is what makes the reassociation exact — xor does not distribute over the stir. Low-bit diffusion is then weak by construction (the last unit only lands in the low bits), so the bowl gets one final hard twist-stir: a `fmix64` finalizer, plus `len` folded in. Every single input bit flip changes the pre-final state by `d · 2^(8s) · M^t` with `M^t` odd and dense, which `fmix64` avalanches to ≈50%.

Rejected on purpose: multiple independent hashes combined at the end (that's a different approach — more than one bowl), and OpenMP (large buffers are memory-bandwidth bound anyway, and it would only add small-input overhead and determinism risk).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* one odd 64-bit "twist" constant; powers folded at compile time */
#define M1 0x9E3779B97F4A7C15ULL
#define M2 (M1*M1)
#define M3 (M2*M1)
#define M4 (M3*M1)
#define M5 (M4*M1)
#define M6 (M5*M1)
#define M7 (M6*M1)
#define M8 (M7*M1)

static inline uint64_t ld8(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t ld4(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

/* the final hard twist-stir */
static inline uint64_t stir(uint64_t x){
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t h = 0;
    size_t i = 0;

    if (len >= 64) {
        /* 8 interleaved lanes of the SAME ordered Horner fold, base Q = M^8 */
        uint64_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        const unsigned char *p = data;
        size_t nb = len >> 6, b;
        for (b = 0; b < nb; ++b) {
            a0 = (a0 + ld8(p     )) * M8;
            a1 = (a1 + ld8(p +  8)) * M8;
            a2 = (a2 + ld8(p + 16)) * M8;
            a3 = (a3 + ld8(p + 24)) * M8;
            a4 = (a4 + ld8(p + 32)) * M8;
            a5 = (a5 + ld8(p + 40)) * M8;
            a6 = (a6 + ld8(p + 48)) * M8;
            a7 = (a7 + ld8(p + 56)) * M8;
            p += 64;
        }
        /* recombine with the shelf-position weights M^7 .. M^0 */
        h = a0*M7 + a1*M6 + a2*M5 + a3*M4 + a4*M3 + a5*M2 + a6*M1 + a7;
        i = nb << 6;
    }

    for (; i + 8 <= len; i += 8)
        h = (h + ld8(data + i)) * M1;

    {
        size_t r = len - i;                 /* 0..7 */
        if (r) {
            const unsigned char *q = data + i;
            uint64_t t;
            if (r >= 4) t = ld4(q) | (ld4(q + r - 4) << 32);
            else        t = (uint64_t)q[0]
                          | ((uint64_t)q[r >> 1] << 16)
                          | ((uint64_t)q[r - 1] << 40);
            h = (h + t) * M1;
        }
    }

    h ^= (uint64_t)len ^ 0xCBF29CE484222325ULL;
    return stir(h);
}
```

Contract check: signature and argument order are exactly `uint64_t kernel(const unsigned char *data, size_t len)`. No reads outside `[data, data+len)` — the `r>=4` path reads `q[0..3]` and `q[r-4..r-1]`, the `r<4` path reads `q[0]`, `q[r>>1]`, `q[r-1]`. `len == 0` returns `stir(0xCBF29CE484222325)`, a nonzero constant. No allocation, no globals, pure and deterministic.

## PREDICTION: 20

(20× the throughput of the given FNV-1a reference on a buffer that fits in cache. Reasoning: FNV-1a is latency bound at ~4 cycles/byte = 0.25 B/cycle; this is `imul`-throughput bound at 8 muls per 64 bytes = 8 B/cycle, i.e. 32× in the steady state, discounted to ~20× for prologue/tail/finalizer and for memory bandwidth clipping the large sizes.)

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were not available in this session — no tools were exposed at all — so I did not run a single timing or avalanche test. The prediction above is a static cycle-count argument, not a result, and I am not going to dress it up as one.

What I expect the pipeline to find, stated so it can be falsified cleanly:

- **Throughput vs. the FNV-1a reference:** ~20× on multi-KB in-cache buffers; falling toward ~8–12× on multi-MB buffers where DRAM bandwidth, not the multiplier port, is the limit; and possibly *below* 5× on very short inputs (< 32 B), where neither version is in its steady state and fixed overhead (the finalizer's two dependent multiplies, ~15 cycles) dominates. If the harness benchmarks only short strings, the win will be much smaller than 20 and the prediction should be scored as wrong.
- **Avalanche:** ~0.50 flip probability per output bit, with the worst single-bit case being the top bit of the final byte (it reaches the pre-final state as a lone bit-63 flip and relies entirely on `fmix64`, which is the specific case that finalizer was searched for). If `hash_contract` reports a bias outlier, that is the bit to look at first.

Known soft spots I did not measure and would check first if given the tools: (1) `M8 ≡ 1 (mod 32)` — unavoidable, since `x^8 ≡ 1 (mod 32)` for every odd `x` — so within a lane the low 5 bits accumulate additively; the recombination weights and the finalizer should cover it, but a low-bit-specific chi-square test is the honest check. (2) whether gcc spills a lane accumulator at 8 lanes; if so, dropping to 6 lanes costs nothing in throughput (6 muls still saturate p1 against a 4-cycle chain) and would be the first thing I'd try.

## VERDICT

Faithful to the mechanism and, on paper, a large win — but unverified. The solution I was handed is Rosa refusing to reorganize her morning, and the fix is not to give her a second bowl or let her skip jars; it is to notice that the exponent on `M` already records which room each jar was in, so the *result* is byte-for-byte an in-order single-accumulator multiply fold while the *work* runs in eight lanes. That converts a 4-cycle-per-byte latency chain into an 8-bytes-per-cycle throughput chain. The one real concession is scooping with `+` instead of `^`, which is what makes the stir distribute; the finalizer pays that debt back.

Confidence in correctness (compiles, obeys the contract, in-bounds, deterministic): high. Confidence in the number 20: moderate, and entirely dependent on which buffer sizes the harness times. Treat the MEASUREMENT section as a set of predictions to shoot at, not as evidence.