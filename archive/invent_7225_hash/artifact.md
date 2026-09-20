# MAPPING

**SEED 1 — the cup**

| World object | Problem object |
|---|---|
| a "mark" | one input byte |
| "the pile of marks" | the byte buffer `data[0..len)` |
| "the cup that is never pure — sickened wine" | the running 64-bit accumulator `h`, seeded non-zero (an FNV-style offset basis — "pure wine holds no memory", so we never start at 0) |
| "pour a mark into the cup" | `h ^= data[i]` |
| "taste the cup's color, let that color become the seed for the next pull" | `h *= prime` — the just-updated state is what the next iteration reads |
| "the whole pile passes through as a single unbroken pour, no mark judged alone or twice" | one sequential pass, each byte folded exactly once, in order |

Assumption broken: **none.** This is a literal restatement of assumptions A1 (byte mixed before next is read), A2 (single in-place accumulator) and A4 (one ordered pass). It *is* the standard construction, not a departure from it.

**SEED 2 — the seven organs**

| World object | Problem object |
|---|---|
| "the pile ends... I carry its last color through my organs" | after the byte loop, feed the final `h` into a finalizer stage |
| "seven organs" | a bounded, fixed number of post-processing rounds |
| "bending" | one nonlinear scramble (shift/xor + multiply) |
| "each bending throws away half of what came before, keeping only what refuses to sit still" | each round is explicitly **lossy** — it doesn't add mixing for free, it destroys some of the previous round's information and keeps only the part that changes most under perturbation (the highest-entropy bits) |

Assumption broken: **A5 — "more mixing rounds always means better mixing."** The metaphor states plainly that every extra bending *costs* information rather than banking it for free. That is the opposite of "more rounds = strictly better"; it says rounds have a point of diminishing (and eventually negative) return.

**SEED 3 — the test at the garden door**

| World object | Problem object |
|---|---|
| "change one mark, even the quietest one" | flip one bit anywhere in the input, including the least-significant one |
| "pour the whole thing through again from the first cup" | recompute `kernel()` on the mutated buffer from scratch |
| "if the two don't trade beauty for beauty exactly... if resemblance, throw the whole method away" | the avalanche criterion itself: reject the design if Hamming distance between old/new hash isn't ~half the output width |

Assumption broken: **none of the five.** This seed describes the *evaluation* (exactly what `hash_bench`'s avalanche score already measures), not a construction mechanism — it's meta-level, not a mixing-strategy choice.

# CHOSEN SEED

**Seed 2 (the seven organs).** It is the only one of the three that breaks a listed assumption — and specifically the one the task prioritizes: "more mixing rounds always means better mixing."

# ASSUMPTION BROKEN

A5: more rounds ≠ strictly better. The metaphor frames each round as lossy, so there is a right-sized, small number of rounds after which adding more just costs time for no further avalanche gain.

# ARTIFACT

Per the instructions' step 4: a well-known, validated real-world technique already embodies exactly this "small fixed number of lossy bendings, not an ever-growing pile of rounds" idea — MurmurHash3's `fmix64` finalizer (and its relative, SplitMix64's mixer). It was arrived at empirically by searching for the *minimum* number of shift/xor/multiply rounds that saturates avalanche; going past it buys nothing. So rather than literally bolting on seven hand-invented rounds (untested, more expensive, no reason to think 7 beats the validated 3), the mechanism is let to converge to that known, validated finalizer — the "organs" become the three already-proven bendings, not a make-believe seven.

The ingest loop (Seed 1) stays a literal single-accumulator sequential fold, per contract, unrolled ×8 purely as an engineering addition (restrict + fewer loop-control instructions) — this does not reorder or skip any byte ("no mark judged alone or twice"), it only batches the bookkeeping.

```c
#include <stdint.h>
#include <stddef.h>

/* fmix64 (MurmurHash3 finalizer): the "seven organs" collapse, empirically,
   to three validated bending steps. Going further is exactly the case the
   metaphor warns about — each bending is lossy, so seven hand-picked organs
   would cost more than three validated ones without measurably improving
   avalanche. We defer to the known, tested technique rather than invent an
   untested seven-round variant. */
static inline uint64_t fmix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

uint64_t kernel(const unsigned char *restrict data, size_t len) {
    uint64_t h = 1469598103934665603ULL; /* the cup: sickened wine, never pure */
    size_t i = 0;

    /* the unbroken pour: one mark at a time, single accumulator, strict
       order. Unrolled only to shrink loop overhead -- the dependency
       chain itself stays fully serial, exactly as demanded ("no mark
       judged alone or twice"). */
    for (; i + 8 <= len; i += 8) {
        h ^= data[i+0]; h *= 1099511628211ULL;
        h ^= data[i+1]; h *= 1099511628211ULL;
        h ^= data[i+2]; h *= 1099511628211ULL;
        h ^= data[i+3]; h *= 1099511628211ULL;
        h ^= data[i+4]; h *= 1099511628211ULL;
        h ^= data[i+5]; h *= 1099511628211ULL;
        h ^= data[i+6]; h *= 1099511628211ULL;
        h ^= data[i+7]; h *= 1099511628211ULL;
    }
    for (; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }

    return fmix64(h); /* carry the cup's last color through the organs */
}
```

No OpenMP thread parallelism was added: the fold is a single serial accumulator by construction (Seed 1's "unbroken pour"), and splitting it into independent per-thread accumulators to combine later would be a different mechanism (multiple cups) not implied by the chosen seed, which only concerns the finalizer. Per the vectorization-before-threads default, the only additions are `restrict` and loop unrolling.

PREDICTION: speedup_vs_dp = 1.1

# MEASUREMENT

No tools were available in this session (explicitly stated by the environment), so `hash_bench` could not be run here. I did not execute anything and am not fabricating a number — the pipeline compiling and running this artifact will produce the actual throughput and avalanche score. My prediction above (≈1.1×, i.e. roughly parity with a slight edge) is reasoning from first principles, not a measurement:
- the ingest loop's hot path (`h ^= byte; h *= prime`) is bit-for-bit identical to the baseline's serial dependency chain, so no algorithmic speedup is possible there — only unrolling's reduced loop-control overhead can help, typically single-digit-to-low-double-digit percent at `-O3 -march=native`.
- the `fmix64` finalizer adds a small, fixed ~9-instruction cost per call, invisible for any buffer larger than a few dozen bytes but a real (if tiny in absolute terms) relative overhead on very small inputs.
- I expect avalanche score to improve substantially over the raw FNV-1a baseline (which has known avalanche weaknesses, especially in the low-order-byte-heavy tail), since `fmix64` is specifically validated for that purpose.

# VERDICT

The mechanism converges to a known, validated technique (MurmurHash3's `fmix64`) rather than inventing an untested 7-round finalizer, per instruction 4 — this is the right call: the seed's own claim ("each bending is lossy") is exactly the empirical justification for why real hash designers stopped at 2–3 rounds instead of piling on more.

Stated risk: the finalizer's fixed ~9-op cost is pure overhead relative to `dp` on very small buffers (e.g. `len` near 0), where it could make this kernel *relatively* slower than the baseline even though the absolute time difference is a few nanoseconds. I did not add a size-gated bypass for this, because the only way to avoid the cost would be skipping the finalizer for small inputs — which reintroduces the baseline's exact avalanche weakness precisely where short keys are most avalanche-sensitive, defeating the seed's purpose. Since the added cost is a true O(1) constant (not a discrete extra pass or allocation), and `hash_bench` throughput is measured over realistic buffer sizes rather than `len≈0`, I judge this risk as already bounded/negligible rather than something requiring a fallback branch — but I am flagging it explicitly rather than burying it, as required.

If measurement shows avalanche does **not** improve meaningfully over plain FNV-1a, or throughput drops materially, the honest fix is to try `fmix64` with only 2 rounds (drop the third `xorshift+multiply` pair) before reaching for anything fancier — never to quietly restore the naive "just multiply harder" textbook loop.