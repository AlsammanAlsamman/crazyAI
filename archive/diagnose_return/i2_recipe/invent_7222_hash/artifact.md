# DICTIONARY

| Recipe object | Computational object | Why this and nothing else |
|---|---|---|
| the **numbered groove** | the 64-bit number line, positions `0 … 2^64−1`, wrapping | a groove is a closed track with a fixed numbering; unsigned 64-bit arithmetic is exactly that |
| the **bone-colored die-stone** | one `uint64_t` register, `stone` — the whole state | one stone, one register. "Never resets between marks" = never re-initialised inside the loop |
| the **desk / bare wood** | the function's own frame; no `static`, no globals, no `malloc` | "clear the desk down to bare wood" (step 1) |
| the **wooden numbered keeps** | the `uint64_t` return value | the keeps are only a readout at the edge of the desk |
| the **ship's cycle-light, one color at every hour** | determinism: no time, no `rand`, no pointer values, no thread count, no `char` signedness, no `-march`-dependent result | "the token must not depend on when you stood there" |
| a **mark** | one byte, `data[i]` | the pile is the buffer; "a pile of five marks and a pile of a hundred thousand" = `len` |
| a mark's **weight / depth in chalk-dust** | its value `0…255` as an integer on the groove's numbering, read through `unsigned char` | step 3: "as a weight, not as a word … never its meaning" |
| **the order of the pile** | forward index walk `i = 0 … len−1`, no sorting, no skipping zeros, no compaction | step 2 forbids all three; also forbids commutative accumulation (plain sum/XOR of bytes) |
| **the starting number cut into the wood** | `SEAT_ZERO = 0xcbf29ce484222325` — one fixed constant, identical for every call | step 4: "the same for every pile anyone ever folds … do not choose it yourself" |
| **pressing a weight into the seated face** | `stone ^= byte` | XOR *is* the recipe's press: step 7 says "a shallow seat takes this weight deep, a deep seat takes it shallow" — that inversion-by-what's-already-there is XOR, not addition |
| **not lifting the stone** | no re-seeding, no per-block restart, no second accumulator | step 5: "a stone set down fresh has forgotten, and a stone that forgets folds two different piles into one token" — i.e. collisions |
| **turn a quarter** | rotate the 64-position die by `64/4 = 16` | a die shows four faces per full turn of the groove; a quarter of 64 positions is 16 |
| **advances along the numbering by however far the press drove it** | multiply by `ADVANCE`: `s → s·M = s + s·(M−1)`, a displacement proportional to the pressed value | the only groove-displacement whose size is a function of how far the press drove it |
| `ADVANCE` **odd** | `0x9e3779b97f4a7c15` | "its new seat is where it stops" — every seat must stay a distinct seat; an even factor is not a bijection of the groove and would grind every long pile toward 0 |
| **the callused face** | the state carrying all previous presses, now rotated so its best-mixed high bits sit under the next press | this is the "bending … the whole work" (step 7) |
| **turned once for every mark** | exactly `len` iterations: no final round, **no finalizer** | step 9 forbids both stopping early and "an extra turn for luck"; step 10 says read it *straight off* |
| **sweep the desk** | return the scalar; no scratch buffer, no side effect, nothing retained | step 12 |

**Ambiguities, resolved to the most literal reading**

1. *Step 6, turn vs. advance ordering.* Step 6 is one motion with two aspects (a quarter turn of the face, a press-proportional travel). I order it **advance, then settle**, because the sentence ends "its new seat is where it stops" — the resting face is the last thing to happen: `stone = rotl(stone * ADVANCE, 16)`. The alternative (`rotl` first, then multiply) is equally grammatical but measurably worse: it pushes the freshly pressed byte up 16 positions *before* the multiply, so the last mark can never touch output bits 0–16.
2. *"A quarter" = 16 bits*, not "some rotation I like". I keep 16 even though 23 or 29 would break byte-lane alignment more aggressively — literalness over cleverness.
3. *Step 1's "sweep the groove clean of old chalk"* is the **between-calls** hygiene (purity/reentrancy); step 5's "never lift" is the **within-call** rule. No contradiction.
4. *No step is wrong*, so I make no correction. Note carefully what this costs: **step 10 forbids a finalizer**, so the final mark gets only one press+turn of mixing. I did **not** add a `fmix64` tail. That would be the textbook move and it would be a different recipe.
5. *No OpenMP, no SIMD.* Four parallel lanes would be four stones; block-wise folding would be lifting the stone and setting it down fresh — step 5 names that as "the one ruinous mistake". So the fold is strictly serial and I eat the throughput loss rather than smuggle in the standard trick.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* ============================ the world =================================
   groove           : the 64-bit number line, 0..2^64-1, wrapping
   die-stone        : one uint64_t register, `stone`
   starting number  : SEAT_ZERO, cut into the wood, same for every pile
   press            : XOR of a mark's weight into the seated face
   quarter turn     : rotate the 64-position die by 64/4 = 16
   advance          : multiply by ADVANCE (odd => every seat stays a seat)
   keeps            : the uint64_t handed out
   ======================================================================= */

#define SEAT_ZERO 0xcbf29ce484222325ULL /* step 4: the number cut into the wood */
#define ADVANCE   0x9e3779b97f4a7c15ULL /* press-proportional travel; odd      */
#define QUARTER   16                    /* 64 groove-positions / 4 die faces   */

#define TURN(s)  (((s) << QUARTER) | ((s) >> (64 - QUARTER)))
#define PRESS_AND_TURN(s, b) ((s) = TURN(((s) ^ (uint64_t)(b)) * ADVANCE))

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: clear the desk down to bare wood -- no globals, no statics, no
       allocation, nothing carried over from any earlier fold; and work only
       under the cycle-light: the result depends on (data,len) and on nothing
       else -- no clock, no rand, no pointer value, no thread count, and the
       marks are read through `unsigned char` so char-signedness cannot tint
       a weight.  No -march-dependent path exists below. */
    const unsigned char *restrict mark = data;
    const size_t n = len;
    size_t i;
    uint64_t stone, token;

    /* step 2: lay the pile along the groove in the order it was given -- first
       mark nearest the hand (i = 0), last mark nearest the keeps (i = n-1).
       Nothing is sorted, no blank is skipped, no gap is closed: the walk below
       is strictly forward, one index at a time, zero bytes included.  Hence two
       piles of the same marks in a different order fold to different tokens. */
    i = 0;

    /* step 3: each mark is read as a weight -- its depth on the groove's own
       numbering, i.e. its integer value 0..255 -- never as a word or glyph.
       No weight is written down: `mark[i]` is taken once, used once, dropped;
       there is no scratch array anywhere in this kernel. */

    /* step 4: seat the stone at the starting number cut into the wood.  Not
       chosen here, not varied by pile or by mood: one constant, always. */
    stone = SEAT_ZERO;

    /* step 5: take the first mark and press its weight into the stone's seated
       face -- the one clean face it will ever show.  Press (a full XOR into the
       face), not a tap; and the stone is not lifted: `stone` is never
       re-seeded past this point, for any pile of any length. */
    if (i < n) stone ^= (uint64_t)mark[i];

    /* step 6: turn the stone a quarter through the groove while it stays
       seated, so it advances along the numbering by however far the press drove
       it (travel = stone*(ADVANCE-1), proportional to the pressed value), and
       its new seat is where it stops (the quarter turn lands last).  Nothing is
       noted down; the stone holds it. */
    if (i < n) { stone = TURN(stone * ADVANCE); i = 1; }

    /* step 7: take the next mark and press its weight into the face the stone
       is NOW showing -- the callused face, not a clean one.  Because the press
       is an XOR against what is already there, a shallow seat takes this weight
       deep and a deep seat takes it shallow; the quarter turn has just brought
       the best-bent positions of the last press down under this one.  This
       bending is the whole work.  Then another quarter turn, as in step 6. */
    if (i < n) { PRESS_AND_TURN(stone, mark[i]); i = 2; }

    /* step 8: repeat step 7 for every remaining mark, in order, without pause
       and without ever lifting the stone.  Eight marks are taken per pass only
       so the hand keeps reaching ahead for weights while the stone turns; the
       press-and-turn is byte-for-byte identical to step 7 and the chain is
       unbroken -- no second stone, no fresh seating, no rest.  Five marks and a
       hundred thousand marks are folded exactly the same way. */
    for (; i + 8 <= n; i += 8) {
        uint64_t w0 = mark[i + 0], w1 = mark[i + 1];
        uint64_t w2 = mark[i + 2], w3 = mark[i + 3];
        uint64_t w4 = mark[i + 4], w5 = mark[i + 5];
        uint64_t w6 = mark[i + 6], w7 = mark[i + 7];
        PRESS_AND_TURN(stone, w0);
        PRESS_AND_TURN(stone, w1);
        PRESS_AND_TURN(stone, w2);
        PRESS_AND_TURN(stone, w3);
        PRESS_AND_TURN(stone, w4);
        PRESS_AND_TURN(stone, w5);
        PRESS_AND_TURN(stone, w6);
        PRESS_AND_TURN(stone, w7);
    }
    for (; i < n; ++i) PRESS_AND_TURN(stone, mark[i]);

    /* step 9: we are finished when, and only when, the stone has turned once
       for every mark and no mark is left between hand and keeps -- the loops
       above exit exactly at i == n.  No early stop because a seat looked
       handsome, and no extra turn for luck: no finalizing round is applied. */

    /* step 10: lift the stone now, for the first and last time, and read its
       final seated number straight off against the keeps -- the number the
       keeps show, unaltered, untruncated, not the number we expected. */
    token = stone;

    /* step 11: that reading, alone, is the token.  Give it out. */
    /* step 12: sweep everything else off the desk -- the groove-dust, the
       intermediate seats, the weights.  They lived only in registers and in
       the caller's buffer, which we never wrote; nothing but `token` leaves. */
    return token;
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 0.9**

Stated before any measurement, with the reasoning that produced it:

- **Throughput.** The recipe pins one dependent chain per byte: `xor` (1c) → `imul` (3c) → `rol` (1c) = **5 cycles/byte latency**, and nothing else can hide it because there is exactly one stone. At ~4 GHz that is ≈ **0.8 GB/s**, flat in `len`. Loads, loop overhead and the 8× unroll are free — they are not the critical path. If the `dp` reference is the canonical byte-at-a-time multiply-XOR (4c/byte), I am one rotate slower per byte → 0.8. If the reference does any extra per-byte work, I land slightly above 1.0. Midpoint, honestly: **0.9**. I expect to *lose* on throughput, and I expect that loss to be the price of steps 5 and 8.
- **Avalanche.** Predicted **≈ 0.49** (mean output-bit flip fraction). Every mark except the last two passes through ≥2 multiply-and-rotate stages, which is full 64-bit diffusion. The last mark gets one stage: its bit *k* enters at position *k* and a multiply propagates only upward, so its coverage is ≈ (64−k)/64 → per-bit expectation ≈ 0.47, not 0.50. Known, disclosed, and not patched: patching it means a finalizer, and step 10 forbids one.
- **Predicted structural weakness.** Output bit 16 is a *deterministic* (XOR-linear) function of the last mark's bit 0, because `ADVANCE` is odd and the quarter turn moves product bit 0 to position 16. On a mean-avalanche score this costs almost nothing (1 cell of 64 reads 1.0 instead of 0.5); on a worst-cell-bias score it would show up immediately.

# MEASUREMENT

**Not performed — I could not measure this.** No tools are exposed in this session: `hash_bench`, `hash_contract` and the symbolic/unconventional tools are all unavailable, and there is no shell to compile with. So there is no throughput number, no avalanche score, and zero of my four permitted improvement rounds were spent. The prediction above stands untested, and I am not going to dress up a hand-derived cycle count as a measurement.

For the record, the harness should report roughly: throughput ~0.8 GB/s independent of buffer size; avalanche ~0.49; `kernel("",0) = 0xcbf29ce484222325`.

Three checks I would have run first, in order: (1) avalanche on a **short** buffer (8 bytes) versus a long one (4 KiB) — the gap isolates exactly the missing-finalizer cost; (2) throughput at 64 B and 1 MiB — these should be nearly identical, which is the signature of a latency-bound serial chain and would confirm the bottleneck is the recipe, not the codegen; (3) the avalanche cell for (last byte bit 0 → output bit 16), to confirm the predicted deterministic coupling rather than assume it.

# VERDICT

The recipe is a real hash and a surprisingly good one. Its per-byte core, taken literally, is `stone = rotl((stone ^ byte) * M, 16)` — a strictly serial multiply-and-rotate fold, i.e. FNV-1a with the crucial addition of the quarter turn. That rotation is not decoration: it drags the previous multiply's *high* bits — a multiply's best-mixed bits — down under the next press, so information flows both up and down the word instead of only upward. That single difference is what separates a mediocre hash from a good one, and the native's "press into the callused face, not a clean one" is precisely the design note a hash engineer would write. Three of the recipe's stranger-sounding commandments are load-bearing and correct: *never lift the stone* is collision resistance, *do not skip a blank / do not sort* is order- and length-sensitivity, and *one color at every hour* is reproducibility across runs and machines.

Two places where following it faithfully costs something real, both reported rather than engineered away:

- **Step 8 forbids parallelism.** One stone means one dependency chain means ~5 cycles per byte and ~0.8 GB/s. Four independent lanes merged at the end would be 4–6× faster and is what every modern hash does — and it is exactly the "stone set down fresh" that step 5 calls the one ruinous mistake. I did not add lanes, and the `-fopenmp` and `immintrin.h` I was handed go unused. This is where my predicted `speedup_vs_dp < 1` comes from.
- **Step 10 forbids a finalizer.** The final mark is under-mixed (≈0.47 instead of 0.50 per bit, plus one deterministic cell). One `fmix64` on the way out would fix it for ~3 cycles total, amortised to nothing. "Read the number the keeps show, not the number you expected" rules it out.

So: literal, complete, twelve blocks in order, no step substituted and no step added — but untested, and I will not claim otherwise. If the harness comes back with good avalanche and sub-1.0 speedup, the recipe is vindicated as *cryptographically-minded and deliberately serial*. If avalanche comes back poor, the culprit is identified in advance and it is step 10, and the smallest honest fix is a single output mix that the native explicitly forbade.

*(Aside, per environment notice: the claude.ai PubMed MCP server is unauthorized and cannot be authorized from this non-interactive session — authorize it in your claude.ai connector settings if you need it. It was not needed here.)*