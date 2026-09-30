# MAPPING

## SEED 1 — "The die-stone never resets between marks, so every fold carries the callus of all folds before it."

| World object | Problem object |
|---|---|
| pile of marks, given order | the byte buffer `data[0..len-1]`, in index order |
| die-stone that is never lifted | a hash state that is initialised once per call, never re-seeded mid-buffer |
| callus of the previous press | the state value carried into step *i* is a function of bytes `0..i-1` |
| "centuries of marks fold down just the same as a handful" | O(1) state, O(len) work, streaming |

**Assumption broken: none.** This seed *is* the textbook assumption ("each byte must be mixed into the running state before the next byte is read"). It asserts sequential dependence rather than breaking anything. Honest note: it does rule out the obvious fast trick (four independent accumulators, xxHash-style), so it constrains me *toward* something harder, not easier.

## SEED 2 — "Each mark's weight is pressed into the stone's already-turned position rather than onto a clean face, so early marks bend how later marks land."

| World object | Problem object |
|---|---|
| **die-stone** (a die: a solid with *several numbered faces/corners*) | a **wide state: eight 64-bit lanes** `s0..s7` — the die's eight corners |
| **the seated place** (one fixed slot in the groove) | the fixed absorption slot: bytes are always xored in at the same lane offsets |
| **"already-turned position" vs. "a clean face"** | *which* corner of the stone is presenting at the seat has changed since last time — the state is **not one value in place**, it is a set of lanes that take turns at the seat |
| **turn it a quarter** | a **quarter turn**: one add–rotate–xor round on four of the eight corners (BLAKE2b's `G`), literally a "quarter round" |
| **mark's weight** | a 64-bit little-endian word of input (`weight`, not `word` — read numerically, never as a symbol) |
| **"a flat thing about to be invested into three"** | the flat 1-D byte string is lifted into a multi-dimensional (multi-lane) state — the net folded into a die |
| **"the stone's own memory bends how deep this one goes"** | the `+=` inside the quarter turn: carry propagation from the old lane value determines the new one |
| **"the next fold multiplies it, and the one after multiplies it again"** | positive differential expansion per round — the avalanche requirement, achieved by *rounds*, not by an integer `*` |
| **chalk-bank groove width** | the regime selector: how far the pile reaches down the numbered groove picks wide vs. narrow groove |

**Assumption broken: "the state is a single accumulator updated in place, one value."** A die-stone has faces that take turns at the seat; that is only meaningful if the state has several distinguishable components. Also broken as a bonus: **"mixing one byte requires a multiplication"** — the native's fold is press (xor), turn (rotate), bend (add). No multiply appears anywhere in the description; the "multiplying" is of *differences*, i.e. diffusion, not of integers.

## SEED 3 — "Only the stone's final seated number against the wooden keeps ever leaves the desk; everything else is swept away."

| World object | Problem object |
|---|---|
| groove-dust, intermediate turns, chalk residue | the 512-bit internal state, discarded |
| wooden **numbered keeps** (plural) | the final reduction constants/rotations that fold 8 lanes → 1 |
| only the last seated number leaves | return type is a single `uint64_t` |

**Assumption broken: none of the five directly** — but it *licenses* the wide internal state Seed 2 needs (a 512-bit stone is legal precisely because only 64 bits ever leave the desk), and it licenses a separate expensive final "lift" that does no absorption.

# CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption (single accumulator, one value), and its mapping is the most literal: *die-stone → multi-corner state*, *quarter turn → quarter round*, *seated place → fixed absorption slot*, *bend how deep it goes → carry propagation*, and it is maximally different from FNV-1a, which is one `uint64_t` and one `imul`. SEED 1 explicitly does *not* break it (it restates the textbook's sequential accumulator) and SEED 3 does not either (it is about output width); I am using both as supporting constraints on SEED 2's implementation, not as the chosen mechanism.

# ASSUMPTION BROKEN

Primary: **"the state is a single accumulator updated in place, one value."** The stone has eight corners; a byte lands at a fixed seat but on a corner that has already been turned, and the turn is what carries the callus sideways.

Secondary: **"mixing one byte requires a multiplication."** The whole kernel contains zero multiplies. This is the actual speed source: FNV-1a is *latency*-bound on a 3-cycle `imul` dependency chain, ~4–5 cycles/byte. The stone's quarter turn is a 12-cycle chain that absorbs **64 bytes**, and two quarter turns per layer are mutually independent, so two axes turn for the price of one.

Also broken: **"more mixing rounds always means better mixing"** — the native folds *once* per seating and does the expensive work only at the lift (SEED 3), not per byte.

A note I owe you: I did **not** vectorise with AVX2 and I did **not** use OpenMP, and this is forced by the metaphor rather than by laziness. SEED 1 says there is **one** stone that never resets. SIMD lanes and threads both require *independent* accumulators — several stones. So the parallelism I am allowed to exploit is only *instruction-level* parallelism among the one stone's own corners (two independent quarter turns per layer). If that had not been enough to beat FNV-1a I would have had to improve *that*, not add a second stone.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===========================================================================
   THE DESK.  The ship's cycle-light is the same starlight colour no matter the
   hour: one fixed constant schedule, never varying with position or round.
   These are also the wooden numbered keeps the stone is finally read against.
   =========================================================================== */
#define KEEP0 0x243F6A8885A308D3ULL
#define KEEP1 0x13198A2E03707344ULL
#define KEEP2 0xA4093822299F31D0ULL
#define KEEP3 0x082EFA98EC4E6C89ULL
#define KEEP4 0x452821E638D01377ULL
#define KEEP5 0xBE5466CF34E90C6CULL
#define KEEP6 0xC0AC29B7C97C50DDULL
#define KEEP7 0x9216D5D98979FB1BULL

static inline uint64_t turn(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

/* THE QUARTER TURN.  One 90-degree turn of the stone about one axis: four
   corners move, the other four stay.  Add-rotate-xor only -- the mark's weight
   is pressed into the corner's ALREADY-TURNED value, and the corner's own
   memory (the carry out of +=) bends how deep the press goes.  No multiply. */
#define QTURN(a,b,c,d) do {                        \
        (a) += (b); (d) = turn((d) ^ (a), 32);     \
        (c) += (d); (b) = turn((b) ^ (c), 24);     \
        (a) += (b); (d) = turn((d) ^ (a), 16);     \
        (c) += (d); (b) = turn((b) ^ (c), 63);     \
    } while (0)

/* reading a mark as a WEIGHT, never as a word */
static inline uint64_t weight64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t weight32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;

    /* --- SEED 1: the stone is seated ONCE and never reset between marks. --- */
    uint64_t s0 = KEEP0, s1 = KEEP1, s2 = KEEP2, s3 = KEEP3;
    uint64_t s4 = KEEP4, s5 = KEEP5, s6 = KEEP6, s7 = KEEP7;

    /* how far down the desk's numbered groove the pile reaches is pressed in
       first -- and it is also what the native glances at to choose a groove. */
    s0 ^= (uint64_t)len;
    s7 ^= turn((uint64_t)len | 1ULL, 32);

    /* ======================= REGIME 1: THE WIDE GROOVE =======================
       A long pile.  The groove is a full stone-width, so eight marks-worth lie
       side by side in one seating.  Two axes of the stone turn at once; they
       touch disjoint corners, so the second turn is free in latency.  The
       pairing of corners ALTERNATES between seatings -- the stone rolling one
       place along the groove -- which is what carries a difference sideways
       into corners it did not first land on.
       GUARDED: entered only for long piles; short piles fall straight through
       to the narrow groove, so the wide path can never cost a short pile. */
    if (n >= 128) {
        do {
            s0 ^= weight64(p +   0); s1 ^= weight64(p +   8);
            s2 ^= weight64(p +  16); s3 ^= weight64(p +  24);
            s4 ^= weight64(p +  32); s5 ^= weight64(p +  40);
            s6 ^= weight64(p +  48); s7 ^= weight64(p +  56);
            QTURN(s0, s2, s4, s6);
            QTURN(s1, s3, s5, s7);

            s0 ^= weight64(p +  64); s1 ^= weight64(p +  72);
            s2 ^= weight64(p +  80); s3 ^= weight64(p +  88);
            s4 ^= weight64(p +  96); s5 ^= weight64(p + 104);
            s6 ^= weight64(p + 112); s7 ^= weight64(p + 120);
            QTURN(s0, s3, s4, s7);      /* rolled one place: new pairing */
            QTURN(s1, s2, s5, s6);

            p += 128; n -= 128;
        } while (n >= 128);
    }

    /* ====================== REGIME 2: THE NARROW GROOVE ======================
       A handful of marks -- or the leftover of a long pile.  Four marks-worth
       per seating, one quarter turn.  The chalk residue creeps into the far
       corners off the critical path so that a medium pile still uses all eight
       corners of the stone. */
    while (n >= 32) {
        s0 ^= weight64(p +  0); s1 ^= weight64(p +  8);
        s2 ^= weight64(p + 16); s3 ^= weight64(p + 24);
        QTURN(s0, s1, s2, s3);
        s4 += s1;  s5 ^= turn(s2, 19);  s6 += s3;  s7 ^= turn(s0, 43);
        p += 32; n -= 32;
    }

    /* --- the last, short seating: overlapping presses, so every mark of the
       remainder is read without a byte-at-a-time loop.  Overlap is harmless
       because the groove number (len) is already pressed in. --- */
    {
        uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        if      (n >= 24) { t0 = weight64(p); t1 = weight64(p+8); t2 = weight64(p+16); t3 = weight64(p+n-8); }
        else if (n >= 16) { t0 = weight64(p); t1 = weight64(p+8); t2 = weight64(p+n-8); }
        else if (n >=  8) { t0 = weight64(p); t1 = weight64(p+n-8); }
        else if (n >=  4) { t0 = weight32(p); t1 = weight32(p+n-4); }
        else if (n)       { t0 = (uint64_t)p[0]
                               | ((uint64_t)p[n >> 1] << 16)
                               | ((uint64_t)p[n - 1]  << 32); }
        s0 ^= t0; s1 ^= t1; s2 ^= t2; s3 ^= t3;
        QTURN(s0, s1, s2, s3);
    }

    /* =================== THE LIFT (SEED 3) ===================
       Nothing new is dropped in.  The stone turns twice more with both axes
       moving and the pairing alternating, so a difference from the last
       seating reaches all eight corners; then the eight corners are read down
       against the wooden numbered keeps to ONE number.  Every intermediate
       turn, all the groove-dust and chalk residue, is swept off and thrown
       away -- only this last seated number ever leaves the desk. */
    QTURN(s0, s2, s4, s6);  QTURN(s1, s3, s5, s7);
    QTURN(s0, s3, s4, s7);  QTURN(s1, s2, s5, s6);

    {
        uint64_t a = s0 + turn(s4, 11);
        uint64_t b = s1 ^ turn(s5, 29);
        uint64_t c = s2 + turn(s6, 47);
        uint64_t d = s3 ^ turn(s7,  7);
        QTURN(a, b, c, d);                          /* the final seating */
        return (a + turn(b, 13)) ^ (c + turn(d, 41)); /* read off the keeps */
    }
}
```

**Which code is which part of the native's mechanism**

| Native's mechanism | Code |
|---|---|
| the bone-colored die-stone, eight corners | `s0..s7` (512-bit state, one stone) |
| never lifted, never reset (SEED 1) | `s0..s7` declared once; no re-seed anywhere in the loops |
| starlight colour, same at every hour | `KEEP0..KEEP7`, one fixed constant set, no round schedule |
| "turn it a quarter" | `QTURN` macro (add–rotate–xor quarter round, four corners) |
| pressing a weight into the *already-turned* corner (SEED 2) | `s_i ^= weight64(...)` **before** the `QTURN`, into a corner already turned by the previous seating |
| "the stone's memory bends how deep this one goes" | the `+=` inside `QTURN` — carry propagation from the old corner value |
| "the next fold multiplies it, and the one after again" | the alternating corner pairings `(0,2,4,6)/(1,3,5,7)` vs `(0,3,4,7)/(1,2,5,6)` — sideways differential expansion |
| reading marks as weights, flat invested into three | `weight64` / `weight32`, 1-D bytes lifted into 8 lanes |
| the desk's numbered groove, how far the pile reaches | `s0 ^= len`, `s7 ^= turn(len\|1,32)`, and the `n >= 128` regime test |
| wide groove (long pile) vs. narrow groove (handful) | the two loops, **regime chosen at runtime by pile length** |
| overlapping presses for a short last seating | the `n >= 24 / 16 / 8 / 4` ladder |
| "lift the stone" | the two extra `QTURN` layers with no absorption |
| "read off against the wooden numbered keeps" | the final `a,b,c,d` combine + `QTURN` + `(a+turn(b,13))^(c+turn(d,41))` |
| "sweep off and throw away the groove-dust" (SEED 3) | `s0..s7` never escape; a single `uint64_t` returns |

**Regime handling (step 5).** The known-way section describes two regimes (`the whole buffer must be read once, start to end` + FNV for small vs. xxHash for large). The native recognises the regime *through the metaphor*: he glances at how far down the numbered groove the pile reaches (`len`), and lays it in the **wide groove** (full stone-width, 64 marks per seating, `len >= 128`) or the **narrow groove** (`len < 128`, and the wide groove's own remainder). Both use the same stone and the same quarter turn — one mechanism, two groove widths, with the narrow groove as the unconditional fallback.

**Risk guard (step 4).** The only conditional claim I make is "the wide groove helps only when the pile is long." It is guarded by `if (n >= 128)` with the narrow groove as fallback, so the wide path can never be paid for by a short pile. No thread parallelism is used at all — the metaphor's single non-resetting stone has no independent units of work to hand to threads, so adding OpenMP would have been a second stone, which is exactly the thing SEED 1 forbids.

# PREDICTION

**PREDICTION: speedup_vs_dp = 14.0**

Stated before any measurement, with reasoning so you can grade the reasoning and not just the number:

- FNV-1a reference: dependency chain of `xor` (1 cyc) + `imul` (3 cyc latency) per byte ⇒ ~4.5 cycles/byte ⇒ ~0.9 GB/s at 4 GHz. It cannot be vectorised by `-O3` because the chain is serial.
- Wide groove: 128 bytes per loop iteration, two `QTURN` layers, each layer being two *independent* quarter turns ⇒ critical path ≈ 24 cycles per 128 bytes ≈ **0.19 cycles/byte**; uop throughput ≈ 80 uops/128 B ≈ 0.16 c/B. Both land near 0.17–0.19 c/B ⇒ ~21 GB/s at 4 GHz.
- Ideal ratio ≈ 4.5 / 0.19 ≈ 24×. I discount to ~14× for: loop/tail/lift overhead, likely DRAM-bandwidth ceiling on large buffers, and the chance the harness averages in small sizes where my fixed lift costs ~50 cycles.
- Secondary prediction: **avalanche ≈ 0.499** (four-plus quarter-round depth for even the last input byte). I expect the FNV reference to score noticeably worse than 0.5 — its final byte gets exactly one multiply, so low output bits barely move.
- Honest downside prediction I am committing to in advance: for **len < 16** I expect to be *slower* than FNV-1a, roughly 1.5–2×, because the lift is a fixed ~50-cycle price that FNV simply does not pay (at the cost of FNV's avalanche). This is a property of the mechanism — one stone, one lift — not of an optional optimisation I could have guarded away.

# MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were listed as available but no tools are actually callable in this session, so I have run zero measurements and zero compiles. I will not dress up an estimate as a result: every number in the PREDICTION section is an *a priori* cycle-count argument, not an observation. The pipeline will compile and measure this; the prediction above is the falsifiable claim, and the honest expected failure mode is the sub-16-byte regime.

Specifically, these are the outcomes that would falsify me:

1. `speedup_vs_dp < 8` on a large buffer ⇒ my cycle model of the quarter-turn critical path is wrong (most likely the compiler spilled some of `s0..s7` and the loop became load/store bound). Fix *within the mechanism*: shrink the stone from eight corners to four so all lanes stay in registers, accepting 2× the cycles/byte.
2. `avalanche < 0.48` ⇒ two lift layers plus the final quarter turn are not enough depth. Fix within the mechanism: add one more lift layer (~12 cycles, negligible on large buffers).
3. `avalanche` fine but `speedup` barely above 1 ⇒ the harness is benchmarking tiny buffers, and the lift dominates. That is the risk I already named and it is structural.

# VERDICT

The literal translation is: **the die-stone is a 512-bit eight-cornered state, the quarter turn is an add–rotate–xor quarter round, the seated place is a fixed absorption offset, and the callus is that nothing is ever re-seeded.** The mechanism is a *sequential sponge* — and that is the interesting part, because it is the opposite of how fast hashes are normally made fast. xxHash goes fast by splitting into four independent accumulators; SEED 1 explicitly forbids that ("the stone never resets, one stone"). So this kernel has to buy its speed from somewhere else, and the only place left is: **widen the rate rather than parallelise the state, and delete the multiply.** 64 bytes absorbed per 12-cycle quarter-turn layer, versus FNV's 1 byte per 4.5-cycle multiply chain.

I think that is a genuine idea and not a dressed-up FNV. It arrives at roughly the BLAKE2/ChaCha design point from a completely different direction, which is a decent sign that the native's world is describing something real rather than something I retrofitted.

What I am *not* confident about, plainly:

- **I have measured nothing.** The 14× is arithmetic on instruction latencies, not a benchmark.
- **Small inputs are the mechanism's weak regime.** One stone plus one lift means a fixed ~50-cycle floor, so below ~16 bytes FNV-1a should win on throughput while losing badly on avalanche. I chose not to paper over this with an FNV fallback, because that would be quietly replacing the native's idea with the textbook method in exactly the case where the textbook method looks better. The condition I *did* name as conditional — "the wide groove only pays off on long piles" — is guarded by `if (n >= 128)` with the narrow groove as fallback, so nothing risky ships unguarded.
- **No SIMD, no threads,** and I want to be explicit that this is the metaphor binding my hands rather than an oversight: both need several independent stones, and there is exactly one stone.

If the measurement comes back below 8×, the fix is to shrink the stone to four corners so it stays in registers — not to add a second stone.