## MAPPING

### SEED 1 — "one crease per mark, angle set by the mark **and** the crease before it"

| World object | Problem object |
|---|---|
| pile of marks, read in order, never looking ahead | `data[0..len)`, strictly forward, single pass |
| one fold per mark | one state update per input word |
| fold's angle | the rotate/add pattern applied to the state |
| "angle decided by the mark *and* the previous crease" | update is a function of both `m_i` and the current state |
| paper "carries forward everything it has been told" | state is the whole history |

**Assumption broken: none.** This seed *affirms* every assumption the known way makes — it is FNV-1a/xxHash's accumulator round almost word for word. (It weakly touches "mixing requires a multiplication," since an *angle* is a rotation, not a product.)

### SEED 2 — "four wire birds as gauges, all four must agree, refold tighter, markings scrambled on purpose"

| World object | Problem object |
|---|---|
| the four wire birds | four 64-bit state words `b0..b3` — four *gauges of the same crease*, not four separate tallies |
| "press each fold's corner against a different bird's beak" | the crease absorbs four marks at once, one per bird, simultaneously |
| "only when all four beaks agree does the crease count as set" | a coupling pass after which **every** bird is a function of **all four** birds; the crease is not final until that holds |
| "if they disagree I refold tighter" | repeat the coupling pass; the refold count is the *minimum* at which full four-way agreement holds |
| spiral / snail markings "line up wrong on purpose, scrambled" | deliberate anti-alignment: no fixed points, length folded in, second crosswise press so no mark can cancel itself |
| "because a clean line-up would mean two different piles could look the same" | collision avoidance is the stated reason for the scrambling |

**Assumption broken: "the state is a single accumulator updated in place, one value."** Four gauges, and — unlike four independent lanes — they are *cross-checked against each other on every crease*, so no lane ever holds an independent partial answer. Secondarily it breaks **"mixing one byte requires a multiplication"**: the entire absorption is add/rotate/xor, zero multiplies per byte.

### SEED 3 — "thin the wad at the water's edge to one dense coin, throw away every scrap"

| World object | Problem object |
|---|---|
| the huge folded wad | wide internal state (256 bits) |
| the water's edge, "small, small, small" | finalization: three closing passes |
| the coin, no bigger than a coin | the 64-bit return value |
| scraps, failed gauge readings, mud closing over the board | all intermediates discarded; nothing carried between calls |

**Assumption broken:** "the state is a single accumulator updated in place, one value" — but only at the *end* (wide→narrow), so it is a weaker version of seed 2's break. It is also a *component* of seed 2's story, not a rival to it.

---

## CHOSEN SEED

**SEED 2.** First, plainly: **none of the three seeds breaks "each byte must be mixed into the running state before the next byte is read."** The native forbids it explicitly — *"I read the pile of marks in order, one at a time, never skipping, never looking ahead."* So per the instruction I fall back to the most literal seed that is most unlike the known way, and that is seed 2: the known way has exactly one accumulator and never cross-checks anything, while seed 2's whole content is a four-way simultaneous gauge with an agreement condition. Seed 1 *is* the known way; seed 3 is absorbed into seed 2 as its closing move ("small, small, small" → three closing passes).

My previous attempt failed precisely here: it used four accumulators that never spoke to each other. Four independent lanes are *not* four gauges of one crease. Four gauges of one crease means every bird's reading depends on every other bird's reading, every time.

### Deriving the refold count instead of guessing it

"Refold until all four beaks agree" is a condition, so I checked when it is met. Take the four-word coupling pass and trace which inputs each output depends on:

```
a  = b0+b1                  -> {0,1}      b1' = rotl(b1,13)^a -> {0,1}      b0' = rotl(a,32) -> {0,1}
c  = b2+b3                  -> {2,3}      b3' = rotl(b3,16)^c -> {2,3}      b2' = c          -> {2,3}
b0''= b0'+b3'               -> {0,1,2,3}  b3''= rotl(b3',21)^b0'' -> {0,1,2,3}
b2''= b2'+b1'               -> {0,1,2,3}  b1''= rotl(b1',17)^b2'' -> {0,1,2,3}   b2'''=rotl(b2'',32)
```

After **one** pass all four beaks already agree at word level; bit-level agreement needs **two**. So the wide crease refolds twice — "refold tighter" is one extra pass, derived, not decorated. And "small, small, small" gives finalization exactly **three** passes.

That coupling pass, written out, *is* the SipRound of Aumasson & Bernstein's SipHash — and three closing passes with one compression pass is SipHash-1-3, the default hasher shipped in Rust's `HashMap` and used in the Linux kernel and CPython. Per instruction 4 I let the mechanism land on that validated primitive rather than inventing a fresh four-way coupler; what is mine is the *rate* (how many marks one crease swallows) and the regime switch.

---

## ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."** The state is four cross-checked gauges; a crease is not set until a coupling pass has made each gauge a function of all four. Also broken: **"mixing one byte requires a multiplication"** — the per-byte path contains no multiply at all (two multiplies occur once, at the thinning). Also broken: **"more mixing rounds always means better mixing"** — the refold count is set by the agreement condition (2 for a wide crease, 1 for a narrow one, 3 to close), not by piling on rounds.

Kept intact, because the native insists: strict in-order, no-lookahead, single-pass reading. Therefore **no threads and no independent SIMD chains** — the metaphor's unit of work is one ordered chain of creases on one sheet, and four cross-coupled birds cannot be split into independent lanes. Only vectorization-friendly hints are used (`restrict`, `memcpy` word loads, no aliasing, branch-free tail).

### Regime recognition, in-metaphor

The native's own regime test is *"one of the huge sheets — pink or orange, doesn't matter, only that it's large enough"*: the sheet is chosen for the pile. Two regimes, both ending at the same water's edge:

* **Big sheet (`len >= 32`)** — four corners of one crease, one per beak, 32 bytes per crease, refold twice, then a crosswise second press so every mark is checked against a second beak.
* **Small sheet / ragged remainder (`len < 32`)** — one mark per crease, pressed on both sides of the fold, one refold (the validated SipHash-1-3 absorption).

The chosen sheet is recorded on the token: `len` is folded into the starting state, so the two regimes are domain-separated and two piles of different size can never fold to the same corner. Every input bit — including the very last ragged mark on the tiny path — passes through at least **4** coupling passes plus the final thinning, which is Rust-shipped SipHash-1-3's diffusion margin; bits in the last wide crease see **6**, matching SipHash-2-4's.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------------
   THE FOUR WIRE BIRDS.  b0..b3 are not four accumulators - they are four
   gauges of the SAME crease.  One pass of BEAKS leaves every bird a function
   of all four birds (word-level agreement); two passes agree at bit level.
   No lane ever holds an independent partial answer.
   This coupling pass is the SipRound of Aumasson & Bernstein (SipHash) -
   the validated four-word cross-check, arrived at rather than invented.
   --------------------------------------------------------------------------- */
#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define BEAKS(b0, b1, b2, b3) do {                                    \
    (b0) += (b1); (b1) = ROTL64((b1), 13); (b1) ^= (b0);              \
    (b0) = ROTL64((b0), 32);                                          \
    (b2) += (b3); (b3) = ROTL64((b3), 16); (b3) ^= (b2);              \
    (b0) += (b3); (b3) = ROTL64((b3), 21); (b3) ^= (b0);              \
    (b2) += (b1); (b1) = ROTL64((b1), 17); (b1) ^= (b2);              \
    (b2) = ROTL64((b2), 32);                                          \
} while (0)

/* read one mark: unaligned, little-endian, compiles to a single movq */
static inline uint64_t mark8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof(v)); return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;

    /* The sheet is chosen for the pile, and which sheet was chosen is
       recorded on the token: differently sized piles start scrambled
       differently and can never fold down to the same corner. */
    uint64_t b0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t b1 = 0x646f72616e646f6dULL;
    uint64_t b2 = 0x6c7967656e657261ULL ^ ROTL64((uint64_t)len ^ 0x9e3779b97f4a7c15ULL, 32);
    uint64_t b3 = 0x7465646279746573ULL;

    /* ===== BIG SHEET: one crease, four corners, four beaks, all at once ===== */
    if (n >= 32) {
        do {
            uint64_t m0 = mark8(p);
            uint64_t m1 = mark8(p +  8);
            uint64_t m2 = mark8(p + 16);
            uint64_t m3 = mark8(p + 24);

            b0 ^= m0; b1 ^= m1; b2 ^= m2; b3 ^= m3;  /* one corner per beak  */
            BEAKS(b0, b1, b2, b3);                   /* fold                 */
            BEAKS(b0, b1, b2, b3);                   /* refold tighter: now
                                                        all four agree       */
            b0 ^= m2; b1 ^= m3; b2 ^= m0; b3 ^= m1;  /* each mark must read
                                                        the same against a
                                                        second beak too      */
            p += 32; n -= 32;
        } while (n >= 32);
    }

    /* ===== SMALL SHEET: one mark per crease, pressed on both sides ===== */
    while (n >= 8) {
        uint64_t m = mark8(p);
        b3 ^= m;
        BEAKS(b0, b1, b2, b3);
        b0 ^= m;
        p += 8; n -= 8;
    }

    /* ===== the last, ragged mark (carries the pile's length) ===== */
    {
        uint64_t t = (uint64_t)len << 56;
        switch (n) {
            case 7: t |= (uint64_t)p[6] << 48;  /* fall through */
            case 6: t |= (uint64_t)p[5] << 40;  /* fall through */
            case 5: t |= (uint64_t)p[4] << 32;  /* fall through */
            case 4: t |= (uint64_t)p[3] << 24;  /* fall through */
            case 3: t |= (uint64_t)p[2] << 16;  /* fall through */
            case 2: t |= (uint64_t)p[1] <<  8;  /* fall through */
            case 1: t |= (uint64_t)p[0];        /* fall through */
            default: break;
        }
        b3 ^= t;
        BEAKS(b0, b1, b2, b3);
        b0 ^= t;
    }

    /* ===== THE WATER'S EDGE: hold it down small, small, small ===== */
    b2 ^= 0xffULL;
    BEAKS(b0, b1, b2, b3);
    BEAKS(b0, b1, b2, b3);
    BEAKS(b0, b1, b2, b3);

    /* 256 bits of sheet thinned to one dense corner; nothing else survives */
    uint64_t coin = b0 ^ b1 ^ b2 ^ b3;
    coin ^= coin >> 33;
    coin *= 0xff51afd7ed558ccdULL;
    coin ^= coin >> 33;
    coin *= 0xc4ceb9fe1a85ec53ULL;
    coin ^= coin >> 33;
    return coin;
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 7.0**

Reasoning, stated before any measurement. The baseline (`dp`) is byte-at-a-time FNV-1a: a serial `xor` + `imul` chain, latency ≈ 1 + 3 = **4 cycles/byte**. My big-sheet crease costs two coupling passes per 32 bytes; a pass has a critical path of ~5–6 cycles and ~14 ops, so ≈ 12 cycles and ≈ 40 ops per 32 bytes → **≈ 0.38 cycles/byte** latency-bound, ≈ 0.31 throughput-bound. That is ~10× on long buffers. I predict **7.0** rather than 10 because (a) very large buffers may hit memory bandwidth, which caps me but never caps 0.25 B/cycle FNV, and (b) short inputs pay a fixed 4 coupling passes + thinning ≈ 30–35 cycles, where FNV is cheap, so any size-averaged score is dragged down. Crossover is at roughly 8–16 bytes.

Avalanche: I predict **near-ideal (~0.50 bit-flip probability, score at or near the harness maximum)** at every size, because every input bit passes ≥ 4 SipRounds plus `fmix64`, and the last wide crease's bits get 6 — SipHash-2-4's own margin.

---

## MEASUREMENT

**Not run.** No tools are available in this session — `hash_bench` and `hash_contract` could not be called, so I have no measured throughput or avalanche number and I will not manufacture one. The prediction above stands as written, un-tuned against any measurement, for the pipeline to confirm or refute. What I expect it to report:

* throughput speedup vs the FNV-1a reference: **7.0** (I will accept 5–11 as the prediction holding; < 3 refutes it and would point at the coupling pass's critical path, not at the rate)
* avalanche: at or within noise of the maximum score at all sizes
* the failure mode I would look for first: sizes below ~16 bytes coming out **slower** than FNV-1a

---

## VERDICT

The core of this kernel is mechanism 2, literally: four wire birds used as four gauges of one crease, cross-checked against each other on every single crease so that none of them can ever be an independent accumulator, with the refold count *derived* from the agreement condition (one pass gives word-level agreement, two give bit-level — so the wide crease refolds twice, the narrow one once) and the closing "small, small, small" giving three. Mechanism 1 is present in the same code — the crease angles act on a state that already carries every earlier crease, strictly in order, with no lookahead — and mechanism 3 is the thinning: 256 bits of sheet collapsed to one 64-bit corner with every scrap discarded. What is *absent* is the previous attempt's mistake: there are no four parallel independent lanes here, and no per-byte multiply.

I did not invent a new four-way coupler. Taken literally, the native's cross-check *is* SipRound, and one compression pass with three closing passes *is* SipHash-1-3 — Rust's default hasher, Linux kernel, CPython. Per instruction 4 I let the mechanism arrive there. My own contribution is confined to the rate: on a big sheet one crease swallows four marks at once (32 bytes, two refolds, plus a crosswise second press), which is a parameterization of a validated primitive, not a replacement of it.

**Where this could be worse than the known way, and what I did about it.** Two honest risks, both named and both addressed in the shipped code rather than in prose:

1. *Fixed cost on tiny inputs.* Four coupling passes plus the thinning is ~30–35 cycles regardless of length, so below roughly 8–16 bytes FNV-1a can win. I guarded this with the runtime regime check the metaphor itself supplies — the sheet is chosen for the pile — so tiny piles skip the 32-byte machinery entirely and take the cheap validated 8-byte-per-crease path with zero setup. I deliberately did **not** guard it by falling back to FNV-1a for short inputs: that would trade the measured metric (avalanche, which FNV is poor at on short inputs) for a few cycles, and it would ship a quality cliff. The residual is that at len < 8 I expect parity-to-2×-slower, and I state that rather than hide it.
2. *The wide rate has a lower rounds-per-byte budget* (1/16) than canonical SipHash-1-3 (1/8). This is a cryptographic-margin reduction, not a diffusion one: the bits most at risk are the last crease's, and they still receive 2 + 1 + 3 = 6 coupling passes plus `fmix64`, equal to SipHash-2-4's margin. I bounded it by a size guard — the wide rate only engages at `len >= 32`, with the validated rate as the fallback below — and I say plainly that this kernel is a fast non-cryptographic hash with good avalanche, **not** a keyed PRF, and should not be used where HashDoS resistance is needed (for that, drop the wide branch and keep only the 8-byte path, which is SipHash-1-3).

No OpenMP and no SIMD lanes, on purpose: the native reads the marks one at a time and never looks ahead, and four mutually cross-checked birds cannot be decomposed into independent vector lanes. Splitting them would recreate exactly the independent-lane error the reviewer caught. The speed here comes from making each crease swallow more marks, not from abandoning the chain.