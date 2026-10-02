## MAPPING

### SEED 1 — "the die-stone never resets between marks; every fold carries the callus of all folds before it"

| World object | Problem object |
|---|---|
| bone die-stone | the running accumulator (machine word holding state) |
| never resets between marks | state is not re-initialized per byte or per block |
| callus of all prior folds | step *i*'s state is a function of the whole prefix `data[0..i]` |
| one fold | one mix step |

**Assumption broken: none.** This seed *asserts* the serial single-accumulator chain. It is a restatement of FNV-1a. Honest verdict: it breaks nothing on the list.

### SEED 2 — "each mark's weight is pressed into the already-turned position, not onto a clean face"

| World object | Problem object |
|---|---|
| mark's *weight* (not a word) | the byte as an integer operand, not a character |
| already-turned position | the rotated current state, used as the mix input |
| "bend how deep this one goes" | state-dependent (nonlinear) mixing: multiply, not just XOR |
| early marks bend later landings | prefix-dependence of the transfer function |

**Assumption broken: weakly, "more mixing rounds always means better mixing."** One press per mark suffices *because* the press is state-dependent. But structurally it is still one accumulator, in order — still the known way.

### SEED 3 — "only the stone's final seated number against the wooden numbered **keeps** ever leaves the desk; every intermediate turn and residue is swept away and discarded"

| World object | Problem object |
|---|---|
| the desk's numbered groove | the byte buffer, index-addressable in given order |
| a mark | one input byte |
| **wooden numbered keeps** (plural, a rack at the desk's edge) | a **bank of parallel 64-bit accumulator lanes** — the state is physically wide |
| the stone | the whole lane bank as one object (one state, many seats) |
| "turn it a quarter through the groove" per mark | the seat advances one keep per press → marks are **interleaved across lanes** |
| "drop the next mark's weight into the same *seated place*" | the seat is fixed in **desk coordinates** while the stone turns → byte *i* lands in lane `i mod L`: striping |
| "never lifting it fully off the desk" | lanes are never re-initialized mid-buffer |
| intermediate turns / groove-dust / chalk residue **swept off and thrown away** | intermediate states are **unobservable** → no ordering or single-valuedness constraint on them; only the final collapse is defined |
| "read off its final seated position against the keeps" — *that, and only that, is the token* | a **strong finalizer**: collapse all lanes + length into one 64-bit value with avalanche mixing |
| cycle-light "same starlight color no matter the hour" | **branch-free, data-independent** inner loop |
| "the stone has turned once for every mark" | exactly O(n), one fold per byte — no extra rounds |
| "centuries of marks fold down just the same as a handful" | two regimes must exist under one ritual: long pile vs. pile too short to fill the groove |

**Assumptions broken:**
- ✅ *"the state is a single accumulator updated in place, one value"* — the state is a **rack of keeps**; only the *reading* is one value.
- ✅ *"each byte must be mixed into the running state before the next byte is read"* — intermediates are explicitly discardable, so presses into different keeps are independent and may proceed concurrently.
- ✅ *"the whole buffer must be read once, start to end, in order"* — in the desk frame the seat is fixed and the stone turns, so the pile is consumed **interleaved**, L bytes at a time.
- ✅ *"more mixing rounds always means better mixing"* — one fold per mark, strength bought at the single final read instead.

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks the preferred assumption ("the state is a single accumulator updated in place, one value"), and its mapping is fully literal: *keeps* is plural, the seat is in desk coordinates rather than stone coordinates, and the residue is explicitly garbage. Seeds 1 and 2 map literally too, but they map onto FNV-1a itself — they are the known way, and I say that plainly rather than dressing them up.

## ASSUMPTION BROKEN

Primary: **the state is a single accumulator updated in place, one value.**
Consequences taken literally: the serial multiply-latency chain that caps FNV-1a (one `imul`, 3-cycle latency, per byte) is replaced by 8 independent chains that saturate the multiplier *throughput* port instead — because nothing but the last seated number ever has to exist.

Per instruction 4, I let this mechanism land on the **validated** technique it is describing rather than inventing one: multi-lane striped accumulation with a per-lane multiply-rotate press, a lane-merge, and the xxHash64 `fmix` finalizer. That is xxHash64's architecture (SMHasher-validated), widened from 4 keeps to 8, with xxHash64's own primes, tail ladder and finalizer kept verbatim. Short inputs take the *unchanged* xxHash64-seed-0 scalar path.

Regime recognition (instruction 5), encoded through the metaphor: *does the pile reach far enough along the groove for the stone to complete a full turn?*
- `len >= 64` → **rack regime**: 8 keeps, 64 bytes per turn.
- `len < 64` → **single-seat regime**: pure scalar, no rack setup, no merge cost.
- within the tail, the ladder 8 → 4 → 1 bytes is the native sweeping the last few marks in by hand.

This also discharges my own stated risk ("8 keeps only pay for themselves on a long pile") with an explicit size guard and a fallback to the simpler path, rather than shipping it unguarded.

**No thread parallelism.** The metaphor gives one desk and one stone with many keeps — lanes, not desks. Lane-level ILP/SIMD is the metaphor's own unit of work; spawning OpenMP teams would be me adding desks the native never had, and at unknown (likely sub-megabyte) benchmark sizes the parallel-region overhead would dominate. Stated as a choice, not an oversight. (AVX2 has no 64-bit `mullo`; the 8 scalar keeps already saturate the multiply port, so explicit intrinsics would buy nothing without AVX-512DQ, which I will not assume.)

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the five numbered keeps' constants (xxHash64 primes - validated) */
#define K1 0x9E3779B185EBCA87ULL
#define K2 0xC2B2AE3D27D4EB4FULL
#define K3 0x165667B19E3779F9ULL
#define K4 0x85EBCA77C2B2AE63ULL
#define K5 0x27D4EB2F165667C5ULL

/* a quarter turn through the chalk-bank groove */
static inline uint64_t turn(uint64_t x, int q) {
    return (x << q) | (x >> (64 - q));
}
static inline uint64_t load8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t load4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold: the mark's weight dropped into a keep's ALREADY-TURNED seat,
   the keep's memory of the last press bending how deep this one goes.
   Never lifted, never washed clean. */
static inline uint64_t press(uint64_t seat, uint64_t w) {
    seat += w * K2;
    seat  = turn(seat, 31);
    seat *= K1;
    return seat;
}

/* lifting one keep's seated position into the single reading */
static inline uint64_t lift(uint64_t h, uint64_t seat) {
    seat *= K2;
    seat  = turn(seat, 31);
    seat *= K1;
    h ^= seat;
    h = h * K1 + K4;
    return h;
}

/* reading the stone off against the wooden numbered keeps: the only thing
   that ever leaves the desk. Everything else is swept away. */
static inline uint64_t read_keeps(uint64_t h) {
    h ^= h >> 33; h *= K2;
    h ^= h >> 29; h *= K3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME CHECK: does the pile reach far enough along the groove for the
       stone to complete a full turn across all eight keeps? */
    if (len >= 64) {
        /* --- rack regime: "centuries of marks" --- */
        uint64_t s0 = K1 + K2,            s1 = K2;
        uint64_t s2 = 0,                  s3 = (uint64_t)0 - K1;
        uint64_t s4 = K3 + K4,            s5 = K4;
        uint64_t s6 = K5,                 s7 = (uint64_t)0 - K3;
        const unsigned char *const limit = end - 64;

        /* the seat is fixed in DESK coordinates while the stone turns, so
           mark i lands in keep (i mod 8): eight independent fold-paths,
           branch-free, the cycle-light the same colour at every hour. */
        do {
            s0 = press(s0, load8(p +  0));
            s1 = press(s1, load8(p +  8));
            s2 = press(s2, load8(p + 16));
            s3 = press(s3, load8(p + 24));
            s4 = press(s4, load8(p + 32));
            s5 = press(s5, load8(p + 40));
            s6 = press(s6, load8(p + 48));
            s7 = press(s7, load8(p + 56));
            p += 64;
        } while (p <= limit);

        /* lift the stone: collapse the rack to one reading */
        h = turn(s0,  1) + turn(s1,  7) + turn(s2, 12) + turn(s3, 18)
          + turn(s4, 23) + turn(s5, 27) + turn(s6, 31) + turn(s7, 37);
        h = lift(h, s0); h = lift(h, s1); h = lift(h, s2); h = lift(h, s3);
        h = lift(h, s4); h = lift(h, s5); h = lift(h, s6); h = lift(h, s7);
    } else {
        /* --- single-seat regime: "a handful" --- no rack, no merge cost.
           Identical to the validated scalar short path. */
        h = K5;
    }

    /* the stone turned once per mark, so the turn count is part of where it
       seats */
    h += (uint64_t)len;

    /* sweeping the last few marks in by hand: 8 -> 4 -> 1 */
    while (p + 8 <= end) {
        uint64_t k = load8(p);
        k *= K2; k = turn(k, 31); k *= K1;
        h ^= k;
        h = turn(h, 27) * K1 + K4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)load4(p) * K1;
        h = turn(h, 23) * K2 + K3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * K5;
        h = turn(h, 11) * K1;
        p += 1;
    }

    return read_keeps(h);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 9.0**

Stated before any measurement, with the reasoning exposed so it can be falsified:

- Baseline FNV-1a is **latency**-bound: one `imul` (3-cycle latency) in a strict serial dependence per byte → ≈0.33 B/cycle ≈ 1.0–1.2 GB/s at 3–3.5 GHz.
- The rack path is **throughput**-bound: 16 multiplies per 64 bytes across 8 independent chains, one multiply retired per cycle → ≈4 B/cycle ≈ 12–14 GB/s. Loads (8 per turn, 2–3/cycle available) are not the limit.
- Raw asymptotic ratio ≈ **12×**. I shade to **9.0** because any benchmark mix containing small/medium buffers pays merge + finalizer cost that does not amortize, and the sub-64-byte regime is deliberately *no faster* than the baseline.

Secondary prediction: **avalanche ≈ 0.50** (single-bit flip probability per output bit), i.e. a near-maximal avalanche score, because every keep passes through a multiply-rotate `lift` chain and then `read_keeps`' three-stage xor-shift-multiply finalizer before anything leaves the desk.

Failure modes I expect if I am wrong: (a) benchmark dominated by ≤32-byte buffers → speedup collapses toward **1.0×**, not below it; (b) a benchmark machine with 2-cycle multiply throughput or a narrower issue width → nearer **6×**.

## MEASUREMENT

**Not performed in this session.** `hash_bench` and `hash_contract` were not available to me — the session had no tool access — so I am reporting this gap rather than inventing numbers. The artifact above is unmeasured and unmodified: zero of my four permitted improvement rounds were spent, because I had no measurement to improve against.

What the pipeline should fill in:

| Quantity | Predicted | Measured |
|---|---|---|
| `speedup_vs_dp` | 9.0 | — |
| avalanche score | ≈0.50 bit-flip rate / near-maximal | — |
| throughput, large buffer | ≈12–14 GB/s | — |
| throughput, len < 64 | ≈ parity with baseline | — |

## VERDICT

The native was not describing FNV-1a, even though two of the three seeds read exactly like it. The tell is grammatical: **"keeps" is plural**, and the seat is fixed to the *desk* while the stone turns. Taken literally, that forces the state to be a rack of eight parallel keeps consuming the pile interleaved, with the single-valued "token" existing only at the moment the stone is lifted. The third seed's insistence that all intermediates are swept away and thrown out is the licence for this: if nothing but the final reading is ever observable, the serial dependence chain that caps FNV-1a was never a requirement of the problem — only of one implementation of it.

Where that mechanism lands is **xxHash64's architecture**, widened from four keeps to eight. I treat this as the right outcome, not a disappointment: instruction 4 asks that a validated technique win over a novel one, and the native's ritual maps onto a design that has already survived SMHasher. The only thing I added was the eighth-to-fifth keep and the extended merge rotations; the press, the tail ladder, and the finalizer are the validated ones verbatim.

Honest limits:

1. **The win is conditional on size, and I guarded it rather than hiding it.** Eight keeps cost a merge of eight `lift` rounds, which is dead weight on short inputs. The `len >= 64` check routes handfuls to the plain single-seat path, so the floor is parity with the baseline, not a regression. I did not ship the risky part unguarded.
2. **No threads.** The metaphor has one desk. Adding OpenMP would be my invention, not the native's, and at plausible benchmark sizes the parallel-region overhead would likely erase the gain. If the benchmark turns out to use multi-megabyte buffers, that is real throughput I left on the table, and the honest fix would be a second regime check at ~1 MB with a fixed (thread-count-independent) segment decomposition — a change I decline to make blind, without a measurement to justify it.
3. **No explicit SIMD.** AVX2 lacks a 64-bit low multiply; the eight scalar keeps already saturate the multiply port, so intrinsics would only help under AVX-512DQ, which I will not assume from `-march=native`.
4. **The headline number is unverified.** My 9.0 is an arithmetic argument from multiply latency versus throughput, not an observation. If the measured value comes back near 1.0, the most likely explanation is a small-buffer-dominated benchmark hitting the guarded scalar path — which would mean the guard worked and the mechanism simply never engaged, a different failure from the mechanism being wrong.