## MAPPING

**SEED 1 — "each mark poured into the sickened cup, the changed color seeds the next pour"**

| world object | problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| one mark | one input byte |
| the cup | the running state accumulator |
| "never pure / sickened wine" | state pre-loaded with impure (non-zero, odd-prime) basis |
| "cut wine remembers every drop" | state retains dependence on all prior bytes |
| a pull on the cup | one absorb step |
| tasting the color, it becomes the next seed | serial carried dependency: `h` of step *i* feeds step *i+1* |
| single unbroken pour | one pass, no byte judged twice |

*Breaks: nothing.* This seed **affirms** "each byte must be mixed before the next is read" and "state is a single accumulator." It is a description of FNV-1a itself.

**SEED 2 — "the final color carried through seven organs, each bending and discarding half, passing on what refuses to sit still"**

| world object | problem object |
|---|---|
| the cup's final color | the accumulator value after the buffer is consumed |
| "carry it past the cup, into the body" | a **finalizer**: mixing that happens *outside* the per-byte loop |
| an organ | one mixing stage |
| seven organs | a fixed, bounded stage count (7), not proportional to `len` |
| "bending" | a multiply by an odd constant (a bijection on 2⁶⁴) |
| "throws away half of what came before" | `h >> 32`-ish: discard half the word width |
| "keeps only what refuses to sit still" | `h ^= h >> s` — only bits that *differ* under the shift survive |
| the garden door / nightingale's note | the avalanche criterion: output half-flip |
| "trade beauty for beauty exactly" | ~0.5 flip ratio, measured, not assumed |
| "bend again" if it fails | add a stage **only** while the criterion fails |
| "knot into a jacket's collar" | 64-bit output |
| "a body compensates a sickness" | nonlinearity is injected once, at the end, not per byte |

*Breaks three:* (a) **"more mixing rounds always means better mixing"** — the organ count is terminated by a measured criterion at the garden door, and each organ *destroys* half its input rather than adding information; (b) "each byte must be mixed into the running state before the next byte is read" — the expensive mixing is deferred past the last byte; (c) "mixing one byte requires a multiplication" — multiplication is an organ's job, not a mark's.

**SEED 3 — "change one mark anywhere, re-pour from the first cup, any resemblance condemns the method"**

| world object | problem object |
|---|---|
| changing one mark, even the quietest | flip one input bit, including in the first byte |
| re-pour from the first cup | recompute the full hash |
| "resembles the old shape" | output Hamming distance ≪ 32 |
| "throw the whole method away" | reject the design |
| "a door someone forgot to turn" | an identity/near-identity map on part of the state |

*Breaks:* only the *epistemic* half of "more rounds is better" — it is a **test protocol**, not a mechanism. It cannot be compiled into a kernel by itself.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that is both a mechanism and a departure: it physically relocates mixing out of the per-byte loop and bounds it by a measured criterion. SEED 1 *is* the known way; SEED 3 is a test harness.

## ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** The native's organs each *throw away half* of what they receive — mixing is subtractive, and past the garden door's criterion extra organs buy nothing. Two consequences, both load-bearing:

1. Because a bounded end-of-pile finalizer can avalanche *any* state, the per-byte pour no longer has to carry the avalanche burden. That frees me from the companion assumptions: the cup becomes **"never pure" in the plural — four cups side by side**, so the serial multiply chain that caps FNV at one byte per multiply-latency is replaced by four independent chains limited only by multiplier *throughput*.
2. Conversely, where the pile is light, seven organs is *waste*, so the native weighs the pile at the door and uses three.

**Arriving at a validated technique rather than inventing (step 4).** This mechanism lands exactly on two published, heavily-validated constructions and I take them rather than invent: the four side-by-side cups with a prime-multiply-rotate absorb and a merge-round blend is **xxHash64's striped accumulator** (its primes and rotations, verbatim); the organ chain is the **xorshift–multiply finalizer** family (murmur3 `fmix64` constants in organs 1–2, `splitmix64` constants in 3–5). The metaphor's contribution is not a new primitive — it is the *decision* to split absorb from finalize and to make the finalizer length-dependent.

**Regime recognition (step 5).** The known_way describes one regime (whole buffer, in order, one accumulator) but the contract spans two: piles too light to wet four cups, and piles heavy enough. The native weighs the pile at the door: `len >= 32` → four cups; otherwise → one cup, and `len < 16` → three organs instead of seven, because a small sickness needs less compensating. The light path never touches the four-lane prologue, so it carries none of its overhead.

**Thread parallelism: declined.** The metaphor's unit of work is one pull on a cup — ~8 bytes. OpenMP fork/join is ~1–5 µs ≈ tens of kilobytes of hashing; at plausible benchmark sizes it would lose, and splitting across threads would make the hash value depend on thread count. Vectorization hints only: `restrict`, `memcpy` word loads (no strict-aliasing UB, compiles to a plain `mov`), four independent chains for ILP. I also considered an AVX2 `vpmuludq` 32×32→64 lane mix to beat the single-port `imul` throughput limit — dropped, because it is a novel untested mixer and step 4's rule says the validated one wins.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The cup is never pure: xxHash64's odd primes. */
#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t rd64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rd32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* one pull on a cup: the pulled word bends the cup, the cup's new color stays */
static inline uint64_t pull(uint64_t acc, uint64_t w) {
    return rotl64(acc + w * P2, 31) * P1;
}
/* blending the four cups into one */
static inline uint64_t blend(uint64_t h, uint64_t v) {
    v = pull(0, v);
    h ^= v;
    return h * P1 + P4;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = p + len;
    uint64_t h;

    /* --- weigh the pile at the door: which regime am I in? --- */
    if (len >= 32) {
        /* heavy pile: four cups side by side, four unbroken pours at once */
        uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = (uint64_t)0 - P1;
        const unsigned char *const limit = end - 32;
        do {
            v1 = pull(v1, rd64(p +  0));
            v2 = pull(v2, rd64(p +  8));
            v3 = pull(v3, rd64(p + 16));
            v4 = pull(v4, rd64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = blend(h, v1);
        h = blend(h, v2);
        h = blend(h, v3);
        h = blend(h, v4);
    } else {
        /* light pile: one cup is enough, and it is already sickened */
        h = P5;
    }

    h += (uint64_t)len;   /* the size of the pile is itself a mark */

    /* the dregs: whatever did not fill a full round of cups */
    while (p + 8 <= end) { h ^= pull(0, rd64(p)); h = rotl64(h, 27) * P1 + P4; p += 8; }
    if    (p + 4 <= end) { h ^= rd32(p) * P1;     h = rotl64(h, 23) * P2 + P3; p += 4; }
    while (p < end)      { h ^= (uint64_t)(*p) * P5; h = rotl64(h, 11) * P1;   p += 1; }

    /* --- the organs: each bends, discards half, keeps what refuses to sit still --- */
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;   /* organ 1 */
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;   /* organ 2 */
    h ^= h >> 32; h *= 0x9E3779B97F4A7C15ULL;   /* organ 3 */
    if (len >= 16) {                            /* a small sickness needs less compensating */
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;   /* organ 4 */
        h ^= h >> 27; h *= 0x94D049BB133111EBULL;   /* organ 5 */
        h ^= h >> 31; h *= 0x2545F4914F6CDD1DULL;   /* organ 6 */
        h ^= h >> 32; h *= 0xD6E8FEB86659FD93ULL;   /* organ 7 */
    }
    h ^= h >> 32;                               /* the garden door: the last discard */
    return h;
}
```

## PREDICTION

*(fixed before any measurement; no measurement had been taken, and none could be — see MEASUREMENT)*

**PREDICTION: speedup_vs_dp = 9.0**

Reasoning, so the number is falsifiable: the FNV baseline is a single `xor; imul` serial chain, one byte per iteration, so it is **latency**-bound at ≈3 cycles/byte (x86-64 `imul` latency 3). My kernel consumes 32 bytes per iteration using 8 multiplies spread over four *independent* chains, so it is **throughput**-bound on the single 64-bit multiply port at ≈8 cycles/32 B = **0.25 cycles/byte**. Ideal ratio 12×; I discount to 9.0 for loop overhead, the finalizer tax on medium buffers, and possible memory-bandwidth capping at large sizes. Avalanche prediction: ≈0.500 flip ratio, score ≥0.99, essentially identical on both regimes, because every path terminates in ≥3 bijective xorshift-multiply organs. Secondary prediction: **organs 4–7 will not measurably improve avalanche over organs 1–3** — that is the assumption-break stated as a testable claim, and if organs 4–7 *do* help, my break is wrong.

## MEASUREMENT

**I could not measure. Reported plainly as a failure of this session, not a success.** The session was invoked with no tool access — `hash_bench`, `hash_contract`, and the `symbolic_*`/`unconventional_*` tools were all unavailable, so the four permitted improvement iterations never happened and the prediction above is **unverified**. What I can state honestly:

- **Measured:** nothing. Zero runs. Throughput 0.25 cycles/byte and avalanche 0.500 are *derived*, not observed.
- **Verified by reasoning only:** all multiply constants are odd, so every organ is a bijection on 2⁶⁴ and no organ can collapse state (this is the direct answer to SEED 3's "door someone forgot to turn" — a non-odd constant would be exactly that door). `len` is folded in, so pour-length is not confusable. `len == 0` returns a deterministic finalized `P5`. The `do/while (p <= limit)` with `limit = end - 32` is exact at `len == 32` (one iteration, `p` ends at `limit + 32 > limit`). The `memcpy` loads cannot invoke strict-aliasing UB and lower to single `mov`s under `-O3`.
- **Unverified and therefore the honest risk:** the real cycles/byte, whether the `len < 16` three-organ path really is faster than the FNV baseline at `len == 8`, and the secondary prediction about organs 4–7. The pipeline's numbers govern; if `speedup_vs_dp < 3`, treat the latency-vs-throughput argument as refuted and look first at whether the benchmark is memory-bound.

## VERDICT

The native's seven organs are not mysticism — they are a **finalizer**, and the whole value of the seed is what the finalizer *permits upstream*. Once you accept that a bounded, half-discarding end-of-pile cascade can avalanche any state, the per-byte loop is released from carrying avalanche, and the single serial cup can become four cups in parallel. That one move converts a latency-bound loop into a throughput-bound one, which is where the predicted ~9× lives. The metaphor did not invent a primitive; it made a structural decision that lands on xxHash64 + `fmix64`/`splitmix64` — which is the right outcome under step 4, and I did not pretend otherwise by inventing a novel mixer I could not validate.

Stated risks and how each is addressed, not deferred:

- *"Seven organs is overhead on a tiny pile."* Guarded: `len < 16` runs three organs (a superset of validated `fmix64`), cutting ~4 multiply latencies off the short path, where they would otherwise dominate.
- *"Four cups cost a prologue a light pile cannot amortize."* Guarded: `len >= 32` gates the four-lane path entirely; below it the kernel is a plain single-cup pour.
- *"Threads could help."* Dropped, not guarded, because the metaphor's unit of work (one 8-byte pull) is orders of magnitude smaller than OpenMP fork/join, and threading would make the hash value depend on thread count. Nothing risky was shipped on a promise.

Where the native is simply right, and where I am exposed: the organ-count claim (3 suffices, 7 is the body being thorough) is my sharpest falsifiable statement and it is **untested**. Lacking the garden door, I shipped the native's literal seven on the heavy path — where their cost is amortized to nothing over ≥32 bytes — rather than trimming on an argument I could not check. That is deliberate conservatism, and it is also the one place where I did not actually obey the native's own instruction, which was to bend *until* the nightingale answers and then stop. With no nightingale available, I could not know when to stop, so I did not claim to.