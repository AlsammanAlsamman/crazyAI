# MAPPING

The world is a single coiled stone trail with one sphere on it. Every noun in the description has exactly one computational referent; I fix them once and reuse them in all three tables.

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| world object | problem object |
|---|---|
| the pile of marks, in their given order | the byte buffer `data[0..len-1]`, index order |
| the sphere | one `uint64_t h` — the entire state, in one register |
| "my running weight, my only memory" | no lane vector, no side array, no table, no second accumulator |
| the trail's high mouth | the initial basis of `h` |
| carrying it to the end of the pile | one serial dependency chain over the whole buffer |

*Assumption broken: none.* This seed **affirms** "the state is a single accumulator updated in place, one value" and "the whole buffer must be read once, start to end, in order." It is the seed closest to FNV-1a. Stated plainly so it isn't quietly credited with a break it doesn't make.

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| world object | problem object |
|---|---|
| a mark | one input byte = 8 bits of the sphere's face |
| the sphere's **face** | the 64-bit width of `h`; one circumference of the sphere = **8 marks** |
| "press its shape into the sphere's face" | `h + w` — the mark's weight pressed in, carries propagating (nonlinear), one 8-byte face at a time |
| a **tumble / turn** | a bitwise **rotation** of `h` — the only shaping tool |
| "a fixed count of turns, no more, no fewer" | an exact compile-time number of rotate stages per ingest; *not* "as many rounds as fit" |
| **mason** trail (stone, struck and turned) | mixing by strike/turn only: rotate, shift, xor, add. A mason shapes stone by striking it; he does not consult the priest's tables → **no multiply** |
| the trail is **coiled**; its slope turns as you go round | one turn of the coil = 32 marks = 4 faces, each quarter with its own rotation pitch |
| a **stalk** | a scheduled xorshift point, one per coil turn |
| "sobs once, cracking a hairline into itself" | `h ^= h >> s` — the sphere xored with a copy of itself |
| "the old face I let shrink and go" | the **right**-shifted (shrunken) copy is consumed and discarded — `>>`, not `<<` |
| "I read the angle of that crack as the next weight" | the crack's result *is* the new `h` |
| "a footprint at a stalk … I never look back at it" | no intermediate state is stored or re-read: no history buffer, no per-position table, no memo array |
| "one wrong mark and every stalk downstream sobs differently" | every absorb step is a **bijection in `h`**, so a single flipped input bit can never be absorbed |

*Assumptions broken:* **"mixing one byte requires a multiplication"** (there is no multiplication anywhere in this world — only turns, strikes and pressed weight) and **"more mixing rounds always means better mixing"** ("a fixed count … no more, no fewer"). It also weakens "each byte must be mixed before the next byte is read" to *each face* (8 bytes), because the sphere's face is one circumference wide.

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token, while all intermediate cracks and dust are swept away."**

| world object | problem object |
|---|---|
| the intermediate cracks | the running values of `h` — they need not individually be well mixed, only invertibly absorbed |
| "the last, smallest crack in the final stalk" | a single finalization function applied **once**, after the last byte |
| the token handed over | the 64-bit return value |
| "I don't keep the sphere itself" | the raw accumulator is never returned |
| stone dust, discarded footprints, eggshells the trail sheds | scratch, the partial tail block, everything not in the token |
| "swept off into the jungle's open plumbing, forgotten on purpose" | discarded with zero bookkeeping; no scratch buffer, no cleanup |

*Assumption broken:* **"more mixing rounds always means better mixing."** The avalanche budget is spent **once**, at the end — at O(1) cost — instead of `len` times.

# CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks *"mixing one byte requires a multiplication"* — the preference named in the instructions — and it is the mechanism seed, so it is also the most literal: tumble = rotate, strike/crack = shift-xor, press = add. It is maximally far from the known way, which is multiply-based in both variants (FNV prime, xxHash's five multiply constants).

SEED 3 is from the same world and I use it where it belongs: it licenses the *cheap body, one expensive final crack* split. SEED 1 constrains me hard: **one sphere** means one accumulator, which forbids xxHash-style lanes, SIMD lanes and threads. I honor that and report the cost rather than quietly adding lanes.

# ASSUMPTION BROKEN

1. **"mixing one byte requires a multiplication" — broken outright.** Zero multiplications in the kernel, including in the constants. The mixing algebra is add / rotate / shift / xor. This is why it is fast: FNV's per-byte critical path is `xor`(1) + `imul`(3) ≈ 4 cycles/byte; a turn-and-press path is `add`(1) + `rol`(1) = 2 cycles per **8** bytes.
2. **"more mixing rounds always means better mixing" — broken.** The trail has exactly one stalk per coil turn and exactly one turn per face. Extra body rounds cost O(len) and buy nothing measurable, because (SEED 3) the token is the final crack. Every absorb step is a bijection, so the avalanche question reduces *entirely* to the nonlinearity of one fixed 64→64-bit bijection evaluated once per hash — independent of `len`.
3. **"each byte must be mixed into the running state before the next byte is read" — weakened.** The sphere's face is one circumference = 8 marks, so 8 bytes are pressed in before the sphere is released.
4. Untouched and stated as such: the state *is* a single in-place accumulator (SEED 1 insists), and the buffer *is* read once in order.

**Regime recognition, in-world (required because the known way spans small and large inputs).** The native's own test is *does the pile reach around the coil?* Three paths, dispatched at runtime, each falling back to the next simpler one, all ending at the same final crack:

- pile reaches round the coil (`len ≥ 32`): the full coiled path, 4 faces + 1 stalk per turn;
- shorter than a turn but at least one face (`8 ≤ n < 32`): face at a time off the coil;
- fewer marks than a face (`n < 8`): crumbs gathered into one face.
- The tail (`0 < n < 8` with `len ≥ 8`) is handled by **backing the sphere up** so the last face is full — a sphere cannot press half a face — which removes the variable-length byte loop entirely.

# ARTIFACT

```c
/* The mason trail: a hash built only from turns, strikes and pressed weight.
   No multiplication anywhere, including in the constants.                     */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the mason's only two tools */
#define TURN64(x, p) (((uint64_t)(x) << (p)) | ((uint64_t)(x) >> (64 - (p))))
#define TURN32(x, p) ((uint32_t)(((uint32_t)(x) << (p)) | ((uint32_t)(x) >> (32 - (p)))))

/* one turn of the coil holds 32 marks = four faces of the sphere; the coil's
   slope turns as you go round it, so each quarter has its own pitch, and one
   stalk stands at the end of every turn.                                     */
#define PITCH_A 23
#define PITCH_B 41
#define PITCH_C 13
#define PITCH_D 46          /* 23+41+13+46 = 123 = 59 (mod 64): odd net drift */
#define PITCH_1 29          /* a lone face, off the coil                      */
#define PITCH_0 17          /* a handful of crumbs                            */
#define CRACK   31          /* the angle of the hairline a stalk opens        */

/* press a face of marks into the sphere, then let it take one turn.
   invertible in h for fixed w, and the carries of + are the crack that
   propagates through the stone (the only nonlinearity on the trail).        */
#define ROLL(h, w, p) ((h) = TURN64((h) + (uint64_t)(w), (p)))

/* a stalk: the sphere sobs once -- the old face shrinks (>>) and goes.       */
#define SOB(h) ((h) ^= (h) >> CRACK)

/* one circumference of the sphere = 8 marks. memcpy is a single mov at -O3
   and is safe for unaligned data; the kernel never writes memory, so there
   is no aliasing hazard to annotate away.                                   */
static inline uint64_t face(const unsigned char *p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

/* ------------------------------------------------------------------------ *
 * THE FINAL STALK.  The sphere itself is not kept.  The last stalk cracks
 * it clean in two; the halves grind against each other a fixed count of
 * times, and the hairline left behind -- small as a token -- is what is
 * handed over.  Every stage below is a bijection of 64 bits, so no output
 * entropy is lost:
 *   h ^= TURN(h,a) ^ TURN(h,b)  is invertible because 1 + x^a + x^b has odd
 *   weight, hence is not divisible by (x+1), hence is coprime to
 *   x^64 - 1 = (x+1)^64 over GF(2);
 *   h + K, TURN, h ^= h>>s and the ARX grind (add/rotate/xor on the two
 *   halves) are each invertible.
 * Grinding on 32-bit halves with add-carries is where all the nonlinearity
 * comes from -- this is the multiply's replacement, and it is paid once per
 * hash, not once per byte.                                                  *
 * ------------------------------------------------------------------------ */
static inline uint64_t final_crack(uint64_t h) {
    uint32_t x, y;

    /* the crack opens across the whole sphere, not a sliver of it */
    h ^= TURN64(h, 49) ^ TURN64(h, 24);
    h  = TURN64(h + 0x9E3779B97F4A7C15ULL, 32);

    x = (uint32_t)(h >> 32); y = (uint32_t)h;
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 1 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 2 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 3 */
    h = ((uint64_t)x << 32) | (uint64_t)y;

    h ^= TURN64(h, 17) ^ TURN64(h, 47);               /* re-cross the halves */

    x = (uint32_t)(h >> 32); y = (uint32_t)h;
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 4 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 5 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 6 */
    h = ((uint64_t)x << 32) | (uint64_t)y;

    /* the hairline itself */
    h ^= TURN64(h, 43) ^ TURN64(h, 11);
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;
    /* the sphere at the trail's high mouth, weighed against the pile */
    uint64_t h = 0x243F6A8885A308D3ULL + (uint64_t)len;

    /* REGIME 1 -- the pile reaches round the coil: four faces, one stalk. */
    while (n >= 32) {
        uint64_t w0 = face(p), w1 = face(p + 8);
        uint64_t w2 = face(p + 16), w3 = face(p + 24);
        ROLL(h, w0, PITCH_A);
        ROLL(h, w1, PITCH_B);
        ROLL(h, w2, PITCH_C);
        ROLL(h, w3, PITCH_D);
        SOB(h);
        p += 32; n -= 32;
    }

    /* REGIME 2 -- shorter than a turn of the coil, but at least one face. */
    while (n >= 8) {
        ROLL(h, face(p), PITCH_1);
        SOB(h);
        p += 8; n -= 8;
    }

    /* REGIME 3 -- the eggshells the trail sheds.  A sphere cannot press half
       a face, so it backs up onto ground it has already rolled over and
       presses one full face; only a pile smaller than a single face has to
       be swept up mark by mark.                                            */
    if (n) {
        uint64_t w;
        if (len >= 8) {
            w = face(p + n - 8);          /* always in bounds: p+n == data+len */
        } else {
            size_t i;
            w = 0;
            for (i = 0; i < n; i++) w |= (uint64_t)p[i] << (8 * i);
        }
        ROLL(h, w, PITCH_0);
        SOB(h);
    }

    return final_crack(h);   /* the token: only the last, smallest crack */
}
```

Deliberate omissions, with reasons: **no OpenMP** — the metaphor has exactly one sphere and one native, so there is no second unit of work to give a thread; per the instructions I stopped at vectorization-level hints (single-`mov` unaligned loads, a fixed 32-byte unrolled block, a write-free kernel so no aliasing barrier exists, loads independent of the `h` chain so the scheduler can run them far ahead). **No SIMD** — a single bijective accumulator chain has no data parallelism to vectorize; faking one would require several spheres, which SEED 1 explicitly forbids.

# PREDICTION

Hand-derived critical paths (no tool was available to check them; `rol r64, imm` is 1 uop / 1 cycle latency, `imul r64` is 3):

- reference FNV-1a: `xor`(1) + `imul`(3) = **4.0 cycles/byte**
- this kernel, regime 1: 4×(`add`+`rol`) + (`shr`+`xor`) = 10 cycles per 32 bytes = **0.3125 cycles/byte** → compute-bound ceiling **12.8×**
- fixed finalizer cost ≈ 30 cycles, so the crossover with FNV is at `len ≈ 9` bytes

Shaving that to a prediction: large buffers will lose some of the 12.8× to DRAM/L3 bandwidth (10 GB/s of loads is needed at 3+ GHz) and small buffers pay the finalizer.

PREDICTION: speedup_vs_dp = 9.0

Also predicted, so it can be falsified: avalanche ≈ **0.50**, inside [0.47, 0.53]. The argument, which is the real claim being tested: every absorb step is a bijection in `h`, so a one-bit input flip *cannot* be absorbed and always reaches `final_crack` as a nonzero difference (worst case — flipping the MSB of the last face — arrives as a difference of exactly one bit, since carries cannot propagate past bit 63). Avalanche therefore depends on one fixed 64→64 bijection, not on `len` at all. If avalanche is good at `len = 8` it is good at every length.

# MEASUREMENT

**Not performed. No tools were available in this session** — `hash_bench`, `hash_contract` and the `symbolic_*`/`unconventional_*` helpers were all listed but none were callable, so I used **0 of my 4 permitted improvement rounds** and every number above is a hand-derived latency count, not an observation. I am reporting that as plainly as I would report a bad measurement; the prediction stands unverified and the pipeline's numbers govern.

What I checked by hand instead: bijectivity of each finalizer stage (the `1 + x^a + x^b` odd-weight / coprime-to-`(x+1)^64` argument over GF(2)[x]/(x⁶⁴−1), done for {49,24}, {17,47}, {43,11}); bounds safety of every load (`n ≥ 32`, `n ≥ 8`, `p + n - 8` with `p + n == data + len`, tail shift ≤ 48); all rotation amounts in 1..63; `len = 0` terminates and returns a fixed nonzero token.

Pre-committed falsification rules, so the result is not re-interpretable after the fact:

| observation | conclusion | the one fix I would make |
|---|---|---|
| speedup ≥ 8 | prediction holds; multiply-free turning is the win | none |
| speedup 4–8 at large sizes | memory-bound, not latency-bound | none — the metaphor forbids more spheres, so this is the honest ceiling |
| speedup < 3 | `TURN64` did not become `rol`, or the loads are not being hoisted ahead of the `h` chain | inspect the asm; widen to a 64-byte block |
| avalanche 0.47–0.53 | 6 grinds suffice; multiplication was not needed for avalanche | none |
| avalanche 0.40–0.47 | under-rounded finalizer | add grinds 7–9 (+3 cycles per hash, ~0 at large `len`) |
| avalanche < 0.40 | the claim fails: multiply-free ARX cannot replace the multiply at this budget | report it as a refutation of the seed, not patch it with an `imul` |

# VERDICT

The native's idea translates with almost no slack, and it is a real idea, not a re-skin: *a mason shapes stone by turning and striking it.* Rotate-and-press replaces multiply-and-xor, and the replacement is cheap where FNV is expensive — 2 cycles per 8 bytes instead of 4 cycles per 1 byte — because a 1-cycle `rol` is doing the work of a 3-cycle `imul`. The second seed pays for the first: because the trail's stalks are sparse and every absorb step is invertible, no avalanche work has to happen per byte at all, and the whole avalanche budget can be spent once, in the final crack, where its cost is O(1). That inversion — cheap body, one expensive ending — is what makes the multiply removable, and it came from the metaphor, not from the textbook.

Named risks, and how each is discharged rather than left dangling:

- **Wide 8-byte ingest needs a full face** → guarded by `n ≥ 8`, falling back to crumb gathering, itself guarded by `len ≥ 8` for the backed-up full face.
- **The 32-byte coil block needs a full turn** → guarded by `n ≥ 32`, falling back to the simpler face-at-a-time path. Both regimes converge on the identical finalizer, so quality does not depend on which path ran.
- **Thread parallelism would help on huge buffers** → dropped entirely, not shipped-and-hoped: the metaphor's unit of work is one sphere, so there is no second work unit, and an unguarded fork would be exactly the "mechanism whose own risk I didn't address" I was told not to ship.
- **The ~30-cycle finalizer is a fixed cost** → crossover with FNV at `len ≈ 9` bytes. I did *not* guard this with an FNV fallback for `len < 9`, and I want the reason on record rather than buried: below the crossover the "simpler path" fails the problem's own primary requirement — FNV over 1–4 bytes has ~0 avalanche in its low bits — so it is not a valid fallback, it is a different, worse answer. Trading the stated objective for a throughput point at 4 bytes would be cheating the benchmark, not beating it. The honest form of this is a disclosed regime boundary, and it is disclosed.

Where this kernel is genuinely worse than the known way, said plainly: xxHash's four independent accumulators exploit instruction-level parallelism that my single serial chain cannot, so on L1-resident large buffers xxHash should beat this by up to ~4×; SEED 1 forbids me the lanes and I did not smuggle them in. The body is also only weakly nonlinear (carries from `h + w` are the sole source), so an adversary can solve for colliding inputs — the same weakness FNV has, and not something the avalanche metric will catch. And the result is endianness-dependent, as xxHash's is.

The one thing that would make me call this a failure rather than a trade-off is avalanche below 0.40: that would mean six ARX grinds cannot substitute for a multiply within a 30-cycle budget, and the right response would be to report the refutation, not to quietly reintroduce an `imul` and keep the metaphor as decoration.