## MAPPING

The native's workshop, object by object. Three tables, one per SEED.

### SEED 1 — "the die-stone never resets between marks; every fold carries the callus of all folds before it"

| World object | Problem object |
|---|---|
| pile of marks, given order | `data[0..len-1]`, read in index order |
| desk's numbered groove | the buffer's own index axis |
| cycle-light, same starlight color at any hour | branch-free, data-independent inner loop (no table lookups, no conditionals on byte values) |
| the die-stone | the running hash state |
| "never lifting it fully off the desk" | state is never re-initialised mid-buffer |
| "callus of the one before it" | step *i* reads the state left by step *i−1* |
| folding until one turn per mark | exactly one fold per input word, O(n), no length-dependent round count |

**Assumption broken: none.** This seed *is* FNV-1a. A single never-reset accumulator chained over the buffer is precisely the known way. Honest reading: SEED 1 describes the baseline, not a departure.

### SEED 2 — "each mark's weight is pressed into the stone's already-turned position, not onto a clean face"

| World object | Problem object |
|---|---|
| "turn it a quarter through the chalk-bank groove" | `rotl64(state, 16)` — a quarter of a 64-bit word is 16 bits |
| "drop the next mark's weight into the same seated place" | XOR the new weight into a *fixed* bit position of the now-rotated state |
| already-turned position, not a clean face | the injection point is fixed but the state under it has moved, so each mark lands against a different part of the accumulated callus |
| "let the stone's memory of the last press bend how deep this one goes" | multiply the xor-result by an odd constant: the product's depth is a function of the entire prior state |
| "no two folding-paths that started differently walk the same last step" | the per-mark step must be a **bijection** on the state (rotate = permutation, xor-by-constant = permutation, multiply by odd = permutation mod 2⁶⁴) |

**Assumption broken: "mixing one byte requires a multiplication"** — partially. The rotate-then-inject is what does the positional mixing; the multiply becomes the coupling, not the diffusion. It does **not** break the single-accumulator assumption.

### SEED 3 — "only the final seated number against the wooden keeps ever leaves the desk; every intermediate turn and residue is swept away"

| World object | Problem object |
|---|---|
| a **die**-stone — a cube with several faces, turned a *quarter* at a time (four quarters to a revolution) | a **four-lane** state: `f0,f1,f2,f3`, 256 bits of working state |
| "the stone has turned once for every mark" | lane index advances with the mark index — mark *i* goes to face *i* mod 4 |
| "the seated face" | the one lane that touches the desk, i.e. the only thing readable |
| groove-dust, intermediate turns, chalk residue — swept off | the other 192 bits of state are never output; they exist only during folding |
| "lift the stone" | the folding chain ends and a separate, off-chain operation begins |
| "wooden numbered keeps mounted at the desk's edge" | a fixed finalizer with fixed constants, applied once at the edge of the process |
| "only the last seated number leaves" | a 256-bit → 64-bit compression, output strictly narrower than state |
| "two different piles essentially never seat the same way" | collision resistance claimed for the *compressed* output, not the wide state |

**Assumption broken: "the state is a single accumulator updated in place, one value."** The desk during folding holds strictly more than what ever leaves it. That is a wide internal state with a narrow, discarded-residue output — the sponge/truncation discipline, and in hashing specifically the striped multi-accumulator discipline.

---

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks *"the state is a single accumulator updated in place, one value"*, which the instructions tell me to prefer. It is also the most different from the known way: FNV-1a's state *is* its output, so FNV has no residue to sweep away at all. SEED 1 is literally the baseline and SEED 2 is a re-arrangement inside one accumulator.

SEED 3 does not discard SEED 2 — SEED 2 supplies the per-fold step (`rotl 16` → inject → multiply) that runs on each of SEED 3's four faces. SEED 1 supplies the no-reset chaining *within* each face. The three nest.

### Step 4 check: does a validated technique already satisfy this?

Yes, and I let the metaphor arrive at it rather than inventing:

- "a die turned a *quarter* at a time, one turn per mark" → **four independent accumulator lanes striped across the buffer**. That is exactly XXH64's main loop. I use XXH64's lane seeding (`K1+K2`, `K2`, `0`, `−K1`) and its validated merge rotations **(1, 7, 12, 18)**.
- "lift the stone and read it against the wooden numbered keeps" → an off-chain finalizer with fixed constants. I use **MurmurHash3's `fmix64`** verbatim, unmodified. This is the known, measured fix for the one real weakness of a multiply-xor chain: multiplication only diffuses bits *upward*, so the low bits of a bare accumulator are weakly mixed. `fmix64`'s `x ^= x >> 33` steps are precisely the downward diffusion the chain lacks.

So the mechanism is novel in *derivation* and validated in *components*. I did not invent a finalizer.

### Step 5: two regimes, recognised in-world

The known-way section describes both "the whole buffer read once, start to end" and the per-byte regime. Four faces are only worth setting up if the pile is long enough to seat all four and pay for lifting the stone and merging the faces. The native's own regime test is the glance down the numbered groove before starting: **a pile shorter than one full revolution of the stone (32 bytes = 4 faces × one 8-mark weight) never gets the four-faced layout at all** — it goes down the single-seated-face path, which is SEED 1 / the simple chain, and still ends at the same wooden keeps. One `if (len >= 32)`, one fallback, no overhead for short piles. This also discharges the risk my own verdict names (striping costs merge work that short inputs can't amortise) — per step 4, I guard it rather than ship it unaddressed.

I did **not** add OpenMP. The metaphor's unit of work is one 32-byte revolution — far too small to amortise thread spawn, and at benchmark sizes the loop is already at ~80% of the core's 64-bit-multiply port limit and will be memory-bandwidth-bound before it is thread-bound. Vectorization hints only (`restrict`, `memcpy`-based word loads that `-O3` folds to single `mov`s, four-way ILP that saturates the multiplier). Adding threads would also break the "nothing washes clean between marks" chain the native insists on.

## ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."**

Here the desk carries **256 bits** of state (four faces of the die) through the whole fold, and **64 bits** leave. The buffer is striped across the four faces, so four independent multiply chains run concurrently and the dependent-multiply latency that throttles FNV-1a is hidden four ways. Secondarily this also breaks *"each byte must be mixed into the running state before the next byte is read"* (eight marks are read as one 64-bit weight — "a flat thing invested into three") and *"more mixing rounds always means better mixing"* (one fold per weight, ever; all the remaining strength comes from one off-chain reading against the keeps, not from more rounds).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four wooden numbered keeps mounted at the desk's edge. */
#define KEEP1 0x9E3779B185EBCA87ULL
#define KEEP2 0xC2B2AE3D27D4EB4FULL
#define KEEP3 0x165667B19E3779F9ULL
#define KEEP4 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* turn the stone a quarter through the chalk-bank groove: 64/4 = 16 bits */
static inline uint64_t turn_quarter(uint64_t x) { return rotl64(x, 16); }

/* read eight flat marks as one weight -- the flat thing invested into three */
static inline uint64_t weight8(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;
}

/* one fold: turn the stone a quarter, drop the weight into the same seated
   place (so it lands on already-turned callus, never a clean face), and let
   the stone's memory of every prior press bend how deep this one goes.
   Rotate, xor-constant-free injection, and odd multiply are each bijections,
   so no two folding-paths that started differently walk the same last step. */
static inline uint64_t press(uint64_t stone, uint64_t w, uint64_t keep) {
    return (turn_quarter(stone) ^ w) * keep;
}

/* lift the stone off the desk and read its seated number against the keeps.
   Off-chain, fixed constants, once.  This is MurmurHash3's fmix64, unchanged:
   the fold's multiplies diffuse only upward, and the >>33 steps supply the
   downward diffusion the fold cannot. */
static inline uint64_t read_keeps(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t seated;

    /* Regime test: one glance down the numbered groove.  A pile shorter than
       one full revolution of the stone (4 faces x 8 marks = 32) never gets
       the four-faced layout -- it goes straight down the single-seated-face
       path below, so short piles pay nothing for the wide desk. */
    if (len >= 32) {
        /* the four faces of the die-stone: 256 bits on the desk, of which
           only 64 will ever leave it.  The rest is groove-dust. */
        uint64_t f0 = KEEP1 + KEEP2;
        uint64_t f1 = KEEP2;
        uint64_t f2 = 0ULL;
        uint64_t f3 = 0ULL - KEEP1;

        const unsigned char *const stop = end - 32;
        do {
            /* the stone turns once per mark-weight; mark i seats on face i%4.
               Four independent callus-chains, no reset, nothing washed clean. */
            f0 = press(f0, weight8(p +  0), KEEP1);
            f1 = press(f1, weight8(p +  8), KEEP2);
            f2 = press(f2, weight8(p + 16), KEEP3);
            f3 = press(f3, weight8(p + 24), KEEP4);
            p += 32;
        } while (p <= stop);

        /* lift the stone: collapse the four faces onto the one seated face.
           Rotations 1/7/12/18 and this merge shape are XXH64's, validated. */
        seated = rotl64(f0, 1) + rotl64(f1, 7) + rotl64(f2, 12) + rotl64(f3, 18);
        seated = press(seated, f0, KEEP1);
        seated = press(seated, f1, KEEP2);
        seated = press(seated, f2, KEEP3);
        seated = press(seated, f3, KEEP4);
    } else {
        seated = KEEP1 + KEEP2;   /* short pile: the seated face alone */
    }

    /* whatever marks remain, keep pressing them into the seated face */
    while (end - p >= 8) { seated = press(seated, weight8(p), KEEP1); p += 8; }
    if (end - p >= 4) {
        uint32_t v; memcpy(&v, p, 4);
        seated = press(seated, (uint64_t)v, KEEP2);
        p += 4;
    }
    while (p < end) { seated = press(seated, (uint64_t)(*p), KEEP3); p++; }

    /* the stone turned once for every mark: the count of folds is itself
       read off against the keeps, so piles of different length cannot seat
       alike even when their marks agree. */
    seated = press(seated, (uint64_t)len, KEEP4);

    /* only the last seated number ever leaves the desk. */
    return read_keeps(seated);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 9.0**

Stated before any measurement, and it has to stand as stated — no tools were available in this session, so nothing below revises it.

Reasoning behind the number: FNV-1a's cost is one dependent `imul` per byte, so ~4 cycles/byte (3-cycle multiply latency + xor on the critical path). The four-faced fold consumes 32 bytes per round of four *independent* multiplies, chain latency ≈ 5 cycles (mul 3 + rotl 1 + xor 1), giving 4 multiplies per 5 cycles ≈ 80% of the single multiply port's throughput ceiling, i.e. ~0.16 cycles/byte — a raw ~25×. I discount that hard to 9.0 because (a) large buffers will hit memory bandwidth long before the multiplier saturates, and (b) any short-input cases in the benchmark mix run the single-face fallback, where the merge-free path still pays for the finalizer and the speedup is closer to 2–4×.

Secondary prediction: **avalanche ≈ 0.50**, and I expect it to be near-ideal rather than merely adequate, because every path into `read_keeps` is a composition of bijections (so one flipped input bit always reaches the finalizer as a nonzero difference, amplified by at least one multiply) and `fmix64` is tuned for exactly the single-bit-difference case.

## MEASUREMENT

**Not performed. No tools were available in this session** — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all unreachable, so the artifact is unmeasured and the prediction above is untested. I will not dress that up: the honest status is *predicted, not measured*, and the four permitted improvement rounds were not spent because there was no feedback to spend them on.

What a run should report, as falsifiable claims:

| Claim | Falsified if |
|---|---|
| speedup_vs_dp ≈ 9.0 on large buffers | below 4× — would mean the four chains are not actually overlapping (check that `memcpy` loads folded to single `mov`s and the loop wasn't vectorized badly) |
| avalanche ≈ 0.50 | below ~0.45 — would point at `fmix64` being fed a difference confined to the top bits by the `rotl 16`; the fix is a fifth keep (one extra `press` before `read_keeps`), not more rounds |
| short-pile path ≥ 1.0× baseline | below 1.0× at len < 32 — would mean the finalizer's 2 extra multiplies aren't amortised; the guard threshold would move up, not the mechanism |

Since I could not iterate, I pre-empted the two failure modes this family of design is actually known to have, rather than shipping and hoping: weak low-bit diffusion in a multiply chain (→ validated `fmix64` finalizer) and striping overhead on short inputs (→ the `len >= 32` regime guard with the single-face fallback).

## VERDICT

The native's die-stone is a **wide-state, narrow-output striped hash**, and taken literally it lands on a real, validated construction rather than a curiosity: a quarter-turn per mark means four quarters to a revolution, four faces, four lanes — XXH64's main loop, derived from the metaphor instead of copied from it. "Only the last seated number leaves the desk" is the 256→64 truncation; "lift the stone and read it against the wooden keeps" is the off-chain finalizer, which is also exactly where the one genuine weakness of the fold gets repaired. That the metaphor insists on sweeping the residue away is the whole content of the idea: 192 bits exist purely to let four multiply chains run at once, and are then thrown out.

The mechanism should beat FNV-1a substantially on throughput and roughly match a good modern hash on avalanche. Where it could be **worse** than the known way: on inputs under 32 bytes, where four faces cannot be amortised. I have not left that as a caveat — it is guarded by the `len >= 32` test with a single-accumulator fallback that is itself the simpler path, so the risk my own verdict names is closed in the artifact rather than noted in prose. Likewise I declined thread parallelism outright instead of adding it with a caveat: a 32-byte revolution is not a thread-sized unit of work, and splitting the buffer across threads would break the one thing the native is emphatic about — that nothing washes clean between marks.

The weakness I cannot discharge here is empirical, not structural: **this is a prediction with no measurement behind it.** Treat the speedup figure as a claim to be shot at, not a result.