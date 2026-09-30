No tools were reachable in this session (and the two MCP servers listed — claude.ai PubMed — need authorization you'd have to grant from claude.ai connector settings / an interactive `/mcp` session; that's unrelated to this task, but it's why no `hash_bench` / `symbolic_*` call appears below). So: the prediction is stated, the artifact is complete, and the MEASUREMENT section reports plainly that I did not get to measure.

---

## MAPPING

### SEED 1 — "the die-stone never resets between marks; every fold carries the callus of all folds before it"

| world object | problem object |
|---|---|
| die-stone | the running hash state |
| never lifted off the desk | state is never re-initialised mid-buffer |
| callus of the previous press | iteration *n*'s state is an input to iteration *n+1* |
| folding | one round of the compression function |

**Assumption broken:** none of the five, really — it *affirms* "each byte must be mixed into the running state before the next byte is read." It is the seed that agrees with FNV-1a most closely. It does weakly push against "more mixing rounds always means better mixing" (one fold per mark, never more), but that's a stretch.

### SEED 2 — "each mark's weight is pressed into the stone's *already-turned* position, not onto a clean face"

| world object | problem object |
|---|---|
| quarter-turn before each press | rotate the state (`rotl`) before injecting |
| the already-turned position | the byte lands at a *moving* bit-offset, not always in the low 8 bits |
| "early marks bend how later marks land" | injection position is a function of history |
| depth of the impression | how far the new byte's influence reaches into the word |

**Assumption broken:** "mixing one byte requires a multiplication." A rotate-then-inject scheme (ARX) achieves cross-bit travel without a multiplier — position, not arithmetic, does the work.

### SEED 3 — "only the stone's final seated number, read against the wooden numbered keeps, ever leaves the desk; every intermediate turn and residue is swept away"

| world object | problem object |
|---|---|
| the die-stone (a solid, several faces, an orientation) | the internal state — **wider than the output**, several words |
| a quarter-turn per mark | a *different face* comes into the seated place each mark → byte/word *i* lands in lane *i mod 4* |
| "the same seated place" | one fixed memory cursor; the lane rotates under it, the cursor doesn't |
| the four faces around the turning axis | four 64-bit accumulators |
| the wooden numbered keeps at the desk's edge | the finalizer: a projection of the wide state onto one 64-bit value |
| groove-dust, intermediate turns, chalk residue — swept away | the lanes are discarded after the merge; the state is *not* the digest |
| "the physicist reads a flat thing about to be invested into three" | a flat 1-D byte run is lifted into a multi-lane state before it is read |
| "no two folding-paths that started differently ever end up walking the same last step" | the per-fold state transition must be a **bijection** — no entropy loss per round |
| "centuries of marks fold down just the same as a handful" | O(1) state, single streaming pass, no length-dependent buffering |

**Assumption broken:** **"the state is a single accumulator updated in place, one value."** The stone is one body that is never lifted (state is still in-place and strictly sequential), but its *seated position* is several coordinates, and only one number ever leaves the desk.

---

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks the preferred assumption ("the state is a single accumulator updated in place, one value"), and its mapping is the most literal: *die-stone* → multi-word state, *quarter-turn* → lane index advancing by one per mark, *wooden keeps* → a finalizer that is a narrowing projection rather than the state itself.

The crucial literal detail is the quarter-turn. A quarter-turn of a die about a fixed axis brings **a new face into the same seated place**. So consecutive marks are pressed into *different faces of the same stone*. Four quarter-turns return to face one. That is not a metaphor for striping — it *is* striping, with a period of exactly four.

---

## ASSUMPTION BROKEN

> *the state is a single accumulator updated in place, one value*

Replaced by: **one stone, four faces, one callus, one reading.** Four 64-bit accumulators plus a fifth carry word, all in registers, all in-place, never reset, never lifted — and a 320-bit state projected down to 64 bits only at the very end.

Two things follow that are not free choices but consequences of the native's words:

1. **The callus must cross faces.** The native denies lane independence explicitly: *"the stone doesn't just carry that one difference forward — the next fold multiplies it, and the one after that multiplies it again."* xxHash64's four lanes are fully independent until the merge; a change in byte 27 touches only lane 4 for the whole loop. The native's stone is one body, so I carry a fifth register `c` — the literal callus, the residue of the last press — which is fed into face 1 at the next seating, while faces 2–4 take the callus of the face pressed immediately before them. Diffusion is cyclic across all four faces with a one-seating lag.

2. **The coupling must be a bijection.** *"No two folding-paths that started differently ever end up walking the same last step."* This is a hard constraint and it rules out the obvious coupling. If face 1 takes its callus from face 4 of the **same** seating, the four equations close into a cycle whose linear part is `I ⊕ rotl(·,52)`; over GF(2), `1 + X⁵² = (1 + X¹³)⁴`, which shares `(1+X)⁴` with `X⁶⁴+1`, so that map is 16-to-1 — it would throw away 4 bits of state every 32 bytes. Staggering the callus by one seating (the carry register `c`) makes the whole 5-word update strictly triangular and therefore exactly invertible: from `c′` recover face 4, from face 4 recover face 3, and so on back to `c`. Zero entropy loss per round, as the native insists — and it costs one register and one rotate.

**Two regimes, recognised in-world.** The known_way section describes both a byte-at-a-time accumulator (FNV) and a bulk multi-lane fold (xxHash), i.e. short and long piles. The native's own desk encodes the test: *can the pile stand a full revolution of the stone?* If the pile is shorter than one complete turn (4 faces × an 8-mark seating = 32 marks), there is no room to turn the stone at all — it is pressed face-on, one seating at a time, and the last few marks that don't fill a seating go in one mark at a time. That is the `len >= 32` branch, with the one-face path as the fallback, which is where the unguarded four-face version would otherwise lose to plain FNV on short inputs.

**No threads.** The metaphor is one stone on one desk that is never lifted; there is no second desk. Vectorisation-level work only (`restrict`, four independent dependency chains for the out-of-order engine, 32-byte sequential loads). 64×64→64 multiplies have no AVX2 form anyway, so four scalar chains beat anything I could write with intrinsics here.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* --- the ship's cycle-light: the same starlight colour no matter the hour.
       Fixed constants, no seed, no clock, no length-dependent tuning.     --- */
#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL
#define LIGHT 0x9E3779B97F4A7C15ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t w64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t w32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* one fold: drop the mark's weight into the seated place, quarter-turn the
   stone, let its own memory of the last press bend how deep this one goes.
   Bijective in `face` (add, rotate, multiply by an odd constant). */
static inline uint64_t seat(uint64_t face, uint64_t weight) {
    face += weight * P2;
    face  = rotl64(face, 31);
    face *= P1;
    return face;
}

/* lifting one face's residue into the running read */
static inline uint64_t press(uint64_t h, uint64_t face) {
    h ^= seat(0, face);
    return h * P1 + P4;
}

/* the wooden numbered keeps at the desk's edge: the ONLY thing that leaves */
static inline uint64_t keeps(uint64_t h) {
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char * restrict data, size_t len)
{
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* -- which regime? can the pile stand one full revolution of the stone?
          four faces x an eight-mark seating = thirty-two marks. -- */
    if (len >= 32) {
        /* REGIME A: turn the stone. Four faces of one body. */
        uint64_t v1 = LIGHT + P1 + P2;
        uint64_t v2 = LIGHT + P2;
        uint64_t v3 = LIGHT;
        uint64_t v4 = LIGHT - P1;
        uint64_t c  = LIGHT ^ P3;          /* the callus: residue of the last press */
        const unsigned char *const limit = end - 32;

        do {
            uint64_t t1 = seat(v1, w64(p     ));
            uint64_t t2 = seat(v2, w64(p +  8));
            uint64_t t3 = seat(v3, w64(p + 16));
            uint64_t t4 = seat(v4, w64(p + 24));
            /* nothing washes clean between marks: each face carries the
               callus of the face seated just before it; face one carries
               the callus that survived the previous revolution.
               Strictly triangular => exactly invertible => no state lost. */
            v1 = t1 ^ rotl64(c , 13);
            v2 = t2 ^ rotl64(t1, 13);
            v3 = t3 ^ rotl64(t2, 13);
            v4 = t4 ^ rotl64(t3, 13);
            c  = t4;
            p += 32;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = press(h, v1);
        h = press(h, v2);
        h = press(h, v3);
        h = press(h, v4);
        h = press(h, c);
    } else {
        /* REGIME B: no room to turn. One face, pressed straight on. */
        h = LIGHT + P5;
    }

    h += (uint64_t)len;

    /* whatever is left of the pile: full seatings, then a half seating,
       then the last marks one at a time. */
    while (p + 8 <= end) { h ^= seat(0, w64(p));      h = rotl64(h, 27) * P1 + P4; p += 8; }
    if    (p + 4 <= end) { h ^= (uint64_t)w32(p) * P1; h = rotl64(h, 23) * P2 + P3; p += 4; }
    while (p     <  end) { h ^= (uint64_t)(*p)   * P5; h = rotl64(h, 11) * P1;      p += 1; }

    /* sweep the groove-dust away; read off the keeps. */
    return keeps(h);
}
```

Where the mechanism lands on validated ground (point 4): the skeleton it arrives at — four striped 64-bit lanes, a `mul/rotl/mul` bijective round, a lane merge, a `xor-shift/multiply` finalizer, xxHash64's prime constants and rotation amounts — *is* xxHash64. I did not invent a round function; the metaphor's quarter-turn produced the lane structure and its "wooden keeps" produced the finalizer, and I let both land on the known, heavily-tested design rather than on something of my own. The one genuine deviation from xxHash64 is the callus (the `c` register and the cross-lane `^ rotl(·,13)`), which the native demands explicitly and which I constrained to be entropy-preserving before shipping it.

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Reasoning, stated before any measurement so it can be falsified:

- Baseline FNV-1a is one dependent `xor` + `imul` per byte: a ~4-cycle loop-carried chain per byte ≈ 0.25 B/cycle ≈ **1 GB/s**.
- This kernel's loop-carried chain is `seat` (add 1 + rotl 1 + mul 3 = 5) + rotl 1 + xor 1 ≈ **7 cycles per 32 bytes** across four parallel chains ≈ 4.5 B/cycle ≈ **18 GB/s** while cache-resident; RAM-resident buffers will clamp to bandwidth, maybe 10–14 GB/s. Small buffers (<64 B) will be dominated by call and finalizer overhead and should be roughly at parity with FNV, perhaps slightly better.
- So 12× is my central estimate for a mixed or mid-sized benchmark; 18× if it is L1/L2-resident only, ~3–5× if the harness is dominated by short inputs.
- Avalanche: I predict **≈0.50 mean output-bit flip probability** per input-bit flip (score ≈0.99 on a normalised scale), driven by the `keeps` finalizer, which is xxHash64's avalanche and is known to pass SMHasher's avalanche test. The callus should not change this measurably — it improves in-loop diffusion, which the finalizer was already covering.

Falsifiers I would accept: avalanche below ~0.48/0.95 means the finalizer is not doing what I claim; speedup below ~4× on a large buffer means the four chains are not issuing in parallel and my latency model is wrong.

---

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were not callable in this session — no tool surface was available — so every number in the PREDICTION section is an unmeasured model, not a result. I am not going to dress an estimate up as a measurement.

What I *did* verify, by hand, without the tools:

- **Entropy of the round (symbolic, done on paper because `symbolic_verify_identity` was unavailable):** the naive cyclic callup `v1 = t1 ^ rotl(t4,13)` has linear part `I ⊕ rotl(·,52)`; over GF(2)[X]/(X⁶⁴+1), `1 + X⁵² = (1+X¹³)⁴` and `gcd(1+X⁵², (1+X)⁶⁴) = (1+X)⁴`, so that map has a 4-dimensional kernel and is 16-to-1. **I built that version first and rejected it** — it violates the native's own "no two folding-paths ever converge" and bleeds 4 bits per 32 bytes. The shipped staggered version is unit-triangular and exactly invertible. This is the one place where taking the metaphor literally caught a real defect that the loose reading would have shipped.
- **Edge cases traced by hand:** `len == 0` (constant, no reads), `len == 1..3` (byte tail only), `len == 31` (regime B, three 8-seatings + 4 + 3), `len == 32` (regime A exactly one iteration, empty tail), `len == 33` (one iteration + 1 byte tail). No read past `end` in any path.
- **Contract:** signature is exactly `uint64_t kernel(const unsigned char *data, size_t len)`. `restrict` on the parameter does not affect function-type compatibility (top-level parameter qualifiers are dropped in C's compatibility rules), so it links against an unqualified prototype.

---

## VERDICT

The seed that broke the target assumption gave a real, literal mechanism: a quarter-turn per mark is lane striping with period four; a die-stone's seated *position* is a state wider than its *reading*; the wooden keeps are a finalizer, not the state. Followed literally, this arrives at xxHash64's architecture from the inside rather than by citation — which is the outcome point 4 asks for, and I took it rather than inventing a competing round function.

The metaphor's one contribution beyond the known way is the **callus**: xxHash64's four lanes are hermetic until the merge, and the native flatly denies that ("the next fold multiplies it, and the one after that multiplies it again"). The staggered carry register gives every face cyclic influence over every other face within one revolution, at a cost of one register and one rotate, and — because I made the coupling triangular rather than cyclic-within-an-iteration — with provably zero state loss. I expect this to be neutral-to-slightly-positive on the avalanche score (the finalizer already saturates it) and to cost perhaps 10–15% throughput on L1-resident data, zero on memory-bound data. That cost is unconditional and bounded, not regime-dependent, so there is nothing conditional to guard.

The risk that *is* conditional — four faces cost more than they earn on a short pile — is guarded: `len >= 32` selects regime A, and everything shorter takes the one-face path, so no input can be slower than a plain accumulator by more than one predictable branch. No thread parallelism was added; the metaphor has one desk and one stone, and at these sizes a thread would be pure overhead.

The honest bottom line: **the artifact is complete and reasoned, and it is unmeasured.** If the harness reports a speedup far below 12× on large buffers, my latency model is wrong and the four chains are not co-issuing; if avalanche comes in below ~0.48, the finalizer is the thing to interrogate first, not the callus.