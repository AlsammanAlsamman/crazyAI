No tools are available in this session, so everything below is done by hand: the mapping, the prediction (stated before any measurement, as required), and the complete artifact for the pipeline to compile and measure.

---

# MAPPING

### SEED 1 — "The die-stone never resets between marks, so every fold carries the callus of all folds before it."

| World object | Problem object |
|---|---|
| die-stone | the running accumulator (a 64-bit register) |
| never resets | no re-initialization inside the loop; loop-carried dependency |
| callus of all folds before it | state at step *i* is a function of bytes 0..*i* |
| fold | one mix step |

**Assumption broken: none.** This seed *affirms* the textbook design — it is a literal description of FNV-1a's carried accumulator. It rules out per-block re-seeding (tree hashing), i.e. it argues *for* assumption (a) and (b), not against them.

### SEED 2 — "Each mark's weight is pressed into the stone's already-turned position rather than onto a clean face, so early marks bend how later marks land."

| World object | Problem object |
|---|---|
| mark's weight | the byte/word as a numeric value, not a symbol |
| already-turned position | the current rotated accumulator |
| pressed into, not onto a clean face | `acc += in*P; acc = rotl(acc); acc *= P` — input enters a state-dependent position |
| early marks bend how later land | non-commutative, order-sensitive mixing |

**Assumption broken: (c) partially** — "mixing one byte requires a multiplication" is not broken; it is *required* by this seed ("the next fold multiplies it, and the one after that multiplies it again"). Again this is FNV/xxHash's round, not a departure.

### SEED 3 — "Only the stone's final seated number against the wooden keeps ever leaves the desk; every intermediate turn and residue is swept away and discarded."

| World object | Problem object |
|---|---|
| desk's numbered groove | the byte buffer, indexed, laid out contiguously |
| a mark | one byte; but read "as weight, not as word" and "flat invested into three" → 8 flat bytes lifted into one wide 64-bit weight |
| cycle-light, same color no matter the hour | fixed constants, branch-free inner loop, no data-dependent timing |
| the stone | one lane accumulator |
| **turn it a quarter** before the next mark seats | the stone visits the **four** numbered keeps in round-robin: 4 accumulators, one per 8-byte word of a 32-byte stripe |
| the wooden **numbered keeps** (plural) at the desk's edge | the lane array + the fixed merge constants read against at the end |
| intermediate turns and residue swept away | **intermediate states are never observed** → their serialization order is a free design choice, not a requirement |
| only the last seated number leaves | one 64-bit return value after a final avalanche |
| two piles essentially never seat the same | collision resistance / avalanche |

**Assumption broken: (a) "each byte must be mixed into the running state before the next byte is read."** If no intermediate ever leaves the desk, nothing forces byte *i* to be folded before byte *i+1* is *read*. And by direct consequence it **licenses breaking (b)**: with intermediates unobservable, how many stones are seated mid-sweep is unconstrained — the "quarter turn through four numbered keeps" is then a legal reading, not a violation.

---

# CHOSEN SEED

**SEED 3.**

Plain statement as required: **none of the three seeds directly asserts a multi-value state.** Seeds 1 and 2 actively *affirm* "a single accumulator updated in place." SEED 3 is the only one that *licenses* breaking it — by making intermediate state unobservable and therefore structurally free — and it is simultaneously the most literal (it names the plural "numbered keeps" and the "quarter turn") and the most different from FNV-1a, which publishes its single accumulator as the whole mechanism.

---

# ASSUMPTION BROKEN

Primary: **"each byte must be mixed into the running state before the next byte is read."**
Consequent: **"the state is a single accumulator updated in place, one value."**

The native's desk reads 32 bytes (four weights) off the groove *before* any of them has finished folding, because the quarter-turn sends each to a different keep. Four independent dependency chains run at once; only their final merge leaves the desk.

**Landing on a validated technique rather than inventing one (step 4).** This mechanism — 4 accumulators, 32-byte stripe, round-robin quarter-turn, merge-once-at-the-end, then a strong finalizer — *is* **xxHash64**. It is SMHasher-validated. So I do not invent a novel 8-lane or hand-rolled SIMD variant; I let the metaphor arrive at xxHash64 and implement it faithfully (its exact round, rotations, primes, tail, and avalanche). The metaphor's "chalk-bank groove" quarter rotation is xxHash64's `rotl(acc,31)`; the "numbered keeps" merge is `rotl(v1,1)+rotl(v2,7)+rotl(v3,12)+rotl(v4,18)` plus four merge rounds.

**Regime recognition (step 5).** The known-way section names two regimes (a pile that fills the keeps vs. a handful; "overhead if small"). The native encodes the check himself: *does the pile fill all four keeps?* — `len >= 32` uses the four-keep desk; a shorter pile never sets up four stones at all and folds on a single keep seeded from `P5`. The tail folds at 8-byte, then 4-byte, then 1-byte granularity, so "centuries of marks fold down just the same as a handful."

**Risk I name, and how I address it.** The four-keep setup costs ~4 register inits and 4 merge rounds; for piles under ~32 bytes that overhead would make it *slower* than FNV-1a. That is guarded by the `len >= 32` branch with the single-accumulator fallback — the risky part is not shipped unguarded. Thread parallelism (OpenMP) is **deliberately dropped**: the metaphor's unit of work is one desk, one sweep; hashing is ~4 bytes/cycle and becomes memory-bandwidth-bound long before a second core helps, and a tree-split would change the function. Vectorization is left to `-O3 -march=native` on the 4 independent chains — which is the correct ordering (SIMD/ILP before threads).

---

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four wooden numbered keeps: fixed constants, "same starlight color no
   matter the hour" -- no data-dependent branches, no per-position tables. */
#define KP1 0x9E3779B185EBCA87ULL
#define KP2 0xC2B2AE3D27D4EB4FULL
#define KP3 0x165667B19E3779F9ULL
#define KP4 0x85EBCA77C2B2AE63ULL
#define KP5 0x27D4EB2F165667C5ULL

static inline uint64_t stone_turn(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* "a flat thing about to be invested into three": eight flat marks lifted into
   one wide weight. memcpy compiles to a single unaligned load at -O3. */
static inline uint64_t weight64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t weight32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* One fold: the mark's weight is dropped into the seated place, the stone is
   turned through the chalk-bank groove, and the turned position is pressed --
   never onto a clean face. */
static inline uint64_t fold(uint64_t acc, uint64_t in) {
    acc += in * KP2;
    acc  = stone_turn(acc, 31);
    acc *= KP1;
    return acc;
}

/* Reading one keep off against the desk's edge at the end. */
static inline uint64_t read_keep(uint64_t acc, uint64_t val) {
    val = fold(0, val);
    acc ^= val;
    acc  = acc * KP1 + KP4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME CHECK: does the pile fill all four keeps? */
    if (len >= 32) {
        /* Four-keep desk: the stone turns a quarter between marks, so four
           folding-paths advance at once and no intermediate is ever read. */
        const unsigned char *const limit = end - 32;
        uint64_t k1 = KP1 + KP2;
        uint64_t k2 = KP2;
        uint64_t k3 = 0;
        uint64_t k4 = 0ULL - KP1;
        do {
            k1 = fold(k1, weight64(p));      p += 8;
            k2 = fold(k2, weight64(p));      p += 8;
            k3 = fold(k3, weight64(p));      p += 8;
            k4 = fold(k4, weight64(p));      p += 8;
        } while (p <= limit);

        /* Lift the stone: the four keeps are read off once, together. */
        h = stone_turn(k1, 1) + stone_turn(k2, 7)
          + stone_turn(k3, 12) + stone_turn(k4, 18);
        h = read_keep(h, k1);
        h = read_keep(h, k2);
        h = read_keep(h, k3);
        h = read_keep(h, k4);
    } else {
        /* FALLBACK REGIME: a handful of marks never sets up four stones. */
        h = KP5;
    }

    h += (uint64_t)len;

    /* The pile's ragged end: 8 marks at a time, then 4, then one by one. */
    while (p + 8 <= end) {
        h ^= fold(0, weight64(p));
        h  = stone_turn(h, 27) * KP1 + KP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)weight32(p) * KP1;
        h  = stone_turn(h, 23) * KP2 + KP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * KP5;
        h  = stone_turn(h, 11) * KP1;
        p++;
    }

    /* Only the last seated number leaves the desk: the groove-dust is swept
       off by a final avalanche so one hair's width of chalk moves half the
       reading. */
    h ^= h >> 33;
    h *= KP2;
    h ^= h >> 29;
    h *= KP3;
    h ^= h >> 32;
    return h;
}
```

---

# PREDICTION

PREDICTION: speedup_vs_dp = 9.0

Stated before any measurement, with the reasoning so it can be scored honestly:

- **Baseline (FNV-1a)** is one serial chain: `xor` (1 cyc) + 64-bit `imul` (3 cyc latency) per **byte** ⇒ ~4 cycles/byte ⇒ ~0.25 B/cyc ≈ **0.9 GB/s** at 3.6 GHz.
- **This kernel**, large-pile regime: 8 `imul`s per 32-byte stripe. 64-bit `imul` is ~1/cycle throughput on one port, so the loop is multiplier-port-bound at ~8 cycles per 32 bytes ⇒ **~4 B/cyc ≈ 14 GB/s** — provided the buffer is in L1/L2. Out of cache it clamps to DRAM bandwidth (~10–15 GB/s single core), so the ceiling is similar either way. That is **~14–16×** over FNV-1a.
- **Small-pile regime (≤32 B)**: the single-keep path + tail + finalizer costs ~35–45 cycles versus FNV's ~4·len; crossover is around 10–12 bytes, so at 16 B expect **~1.7×**, at 8 B roughly **~1.0×** (break-even, never a loss thanks to the guard).
- A benchmark that averages several sizes lands between those. **9.0** is my blended point estimate; I would call the prediction broadly correct anywhere in **6–16×** and would call it a miss below 3× or if avalanche scores below ~0.49.
- **Avalanche**: I expect essentially ideal, ~0.4999–0.5001 flip rate, because the final `xorshift-multiply` chain is the standard strong 64-bit finalizer and every input word reaches it through at least two multiplies.

I also predict, honestly, the thing that could make this look worse than it is: if the harness hashes **only very short keys** (≤8 bytes), the measured speedup will be near 1.0 and the four-keep mechanism will have contributed nothing.

---

# MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` are unavailable in this session — no tools were exposed. I ran zero measurements and made zero post-hoc edits; the artifact above is the first and only version, written against the prediction rather than tuned to a number. The four permitted improvement rounds were not used.

What I would have measured, in order, and the falsifiable expectation for each:

1. Throughput at 8 / 32 / 256 / 4096 / 1 MiB bytes — expect the four-keep path to overtake FNV-1a somewhere between 16 and 32 bytes and to saturate near 4 B/cyc by 256 bytes.
2. Avalanche at each size — expect ≥0.499 at every size including 1-byte inputs (the 1-byte tail path alone is `h ^= b*KP5; h = rotl(h,11)*KP1` plus the full finalizer, which is sufficient).
3. If (1) had come in below ~6× on large buffers, the first improvement would have been to check whether `memcpy` loads were being folded into single `mov`s, not to add lanes — because the mul-port analysis says a 5th–8th keep buys nothing.

---

# VERDICT

The translation is literal and it lands on a real, validated hash rather than an invention: **SEED 3's "only the final seated number leaves the desk" is exactly the permission that turns one accumulator into four**, and the native's own "quarter turn through the keeps" specifies *four* of them over a 32-byte stripe — which is xxHash64's stripe, down to the structure. I implemented xxHash64 faithfully (its primes, `rotl 31` round, `1/7/12/18` merge rotations, tail cascade, and finalizer) instead of a novel wide-lane or hand-rolled AVX2 variant, per the rule that a validated technique beats an untested one.

Where the metaphor genuinely paid: it produced the multi-lane insight *and* the regime check *and* the discard-everything-but-the-last-reading finalizer, all three of which are the actual reasons xxHash64 beats FNV-1a. Where I overrode it: the native says "a quarter turn," which read literally as a rotation would be `rotl(...,16)`; I used the validated `31`, and read "quarter" instead as the round-robin across four keeps — the better literal fit anyway, since the keeps are explicitly four and explicitly plural.

Named risks and their disposition, as required:

- **"Overhead if the pile is small."** Addressed, not waved at: guarded by `len >= 32`, with the single-keep `KP5` path as the fallback. The risky mechanism cannot execute in the regime where it loses.
- **Thread parallelism.** Dropped rather than guarded. At the sizes any hash benchmark uses, one desk saturates the memory port; OpenMP would add fork/join cost exceeding the entire hash and would require changing the function itself. This is the "drop the risky part" branch of the rule.
- **Wider lanes (8 keeps / 64-byte stripes).** Not shipped. 64-bit `imul` throughput is ~1/cycle on a single port, so 8 lanes are mul-port-bound at the same ~4 B/cyc as 4 lanes — more state, identical speed, and unvalidated. This is also the concrete sense in which the native is right that "more mixing rounds always means better mixing" is false: past four keeps, extra folds cost cycles and buy no mixing.
- **Residual honest weakness.** If the harness hashes only sub-8-byte keys, this kernel is roughly break-even with FNV-1a and the whole mechanism is inert. I predicted 9.0 on the assumption of mixed or large sizes; if the measurement comes back near 1.0 on tiny keys, the prediction is wrong and the mechanism simply did not apply — not a result I would reinterpret after the fact.