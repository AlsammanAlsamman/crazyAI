# MAPPING

Common vocabulary first (every world-object → one computational object, nothing left over):

| World object | Problem object |
|---|---|
| pile of marks, in their given order | the byte buffer `data[0..len-1]`, read in order |
| the desk's numbered groove | the index line `0,1,2,…` |
| "a mark read as a *weight*, the way the physicist reads a flat thing about to be invested into three" | a byte read not as a character but as a numeric weight, and a **flat row of 8 bytes lifted into one wider word**: `uint64_t w` (little-endian load) |
| the **bone-colored die-stone** | the state: a *cube*, not a number — 6 faces × 64 bits = 384 bits of state |
| the 4 faces that pass through the groove's seated position | the **rate** lanes `a0,a1,a2,a3` (256 bits) — the only lanes data is ever pressed into |
| the 2 axis caps (poles), which never seat under the groove | the **capacity** lanes `c0,c1` (128 bits) — never touched by data directly |
| "turn it a quarter through the chalk-bank groove" | advance which face is seated: the absorb index steps `a0→a1→a2→a3` (free, resolved at compile time) |
| "drop the next mark's weight into the same seated place" | `a_k += w_k` — always the one seated slot; the stone has rotated, so the *face* differs |
| press depth bent by "the stone's own memory of the last press" | **`+=` not `^=`**: integer carries make the perturbation depend nonlinearly on what callus is already in that face |
| "fold the stone" / a full revolution | one invertible add-rotate-xor round over the 4 equatorial faces (`TURN`) |
| "tipping the stone onto its caps" | `TIP`: the poles take up the callus and bend the faces back |
| "never lifting it fully off the desk" | the state lives in registers, never re-initialised, start to end |
| "no two folding-paths that started differently ever walk the same last step" | every line of `TURN`/`TIP` modifies exactly one lane as an invertible function of the others ⇒ the round is a **bijection** on 384 bits; distinct prefixes can never collapse |
| "the wooden numbered keeps mounted at the desk's edge" | fixed finalisation constants / initial seating |
| "read off its final seated position… that reading, and only that reading, is the token" | squeeze: `a0^a1^a2^a3^c0^c1` → 64 bits |
| "groove-dust, intermediate turns, chalk residue — swept off and thrown away" | 320 of 384 state bits are discarded; no intermediate value is ever emitted |
| "centuries of marks fold down just the same as a handful" | O(1) state, single streaming pass, no size-dependent strategy switch for big piles |

### Seed 1 — the stone never resets; every fold carries the callus of all before
| World | Problem |
|---|---|
| one stone for the whole pile | one state, carried across the whole buffer |
| callus accumulating | state is never zeroed or re-seeded mid-stream |

Assumption broken: **"mixing one byte requires a multiplication."** The callus is a *press and a turn* — carries and rotations, never a multiply. (It does **not** break the single-accumulator assumption; FNV also never resets.)

### Seed 2 — each mark is pressed into the stone's *already-turned* position, not onto a clean face
| World | Problem |
|---|---|
| the stone is a **die** (a cube with faces), not a pebble | state is a 6-lane vector, not one value |
| the seated place is fixed in the *desk's* frame | the absorb slot is one fixed position |
| the quarter turn means a *different face* is seated each time | consecutive words land in different lanes `a0..a3` |
| early marks bend how later marks land | the fold between revolutions makes lane contents path-dependent |
| "the next fold multiplies it, and the one after multiplies it again" | per-round差 amplification through adds + rotations; the round is bijective so differences never cancel back out |

Assumption broken: **"the state is a single accumulator updated in place, one value."**

### Seed 3 — only the final seated reading leaves the desk
| World | Problem |
|---|---|
| final seated number against the keeps | truncated squeeze of the state to 64 bits |
| residue swept away | 320 state bits discarded; capacity lanes never exposed |

Assumption broken: **"more mixing rounds always means better mixing"** — it says the *width* you keep hidden does the work, not the number of passes; the cheap body + a fixed final read is enough.

# CHOSEN SEED

**Seed 2.** It is the only one of the three that breaks the preferred assumption, and its mapping is the most literal: a *die*-stone is a cube, a cube has faces, a quarter turn moves which face is seated, and "the same seated place" with a rotated stone means consecutive marks are pressed into *different faces of one never-reset body*. That is a multi-lane state with rotating absorption and an invertible whole-body fold — structurally the opposite of one accumulator times a prime.

# ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."**
Also broken, as a consequence: *mixing one byte requires a multiplication* (there is not a single `imul` anywhere in the kernel), and *each byte must be mixed into the running state before the next byte is read* (32 bytes are pressed into four faces before the body is folded).

**Arrival at a validated technique, not an invention.** Honoring step 4: what Seed 2 literally describes is a **sponge** — a wide never-reset permutation state, split into a rate (the 4 faces that seat) and a capacity (the 2 poles that never do), absorbed into at fixed slots, truncated on output. So I let the mechanism land on the real, validated construction rather than a new one: the *turn* is literally **SipRound** (SipHash's add-rotate-xor round, constants 13/32/16/21/17/32, a known bijection with well-studied diffusion), the rate/capacity split and zero-padding-plus-length-keep are **Keccak/sponge** practice, and the `+=` absorb with data-dependent carries is the classic ARX "depth of press" coupling. Nothing here is untested cryptanalytically; what is novel is only the *shape* the metaphor forces — 6 lanes as a cube, 32-byte rate, poles tipped once per four revolutions.

**Regime handling.** The known-way section describes two regimes (a buffer long enough to stream vs. a short one). The metaphor recognises exactly those two and refuses a third: a pile long enough to complete revolutions, and *"a handful"* too short to turn the stone all the way round — handled by an explicit runtime `len >= 32` / `len >= 128` check with the handful path as the fallback (one zero-padded press, no per-byte folds, O(1) cost). The metaphor explicitly declines a large-pile special case — *"centuries of marks fold down just the same as a handful"* — and I honor that: **no OpenMP, no multiple stones.** Thread-parallelising would require several independent stones, which destroys mechanism #1 (one never-reset body carrying every callus); the serial fold chain is also only ~6 cycles per 32 bytes, far below any thread-launch cost at benchmark sizes. The risk my own verdict names is the constant finalisation cost on tiny inputs; it is guarded by the `len < 32` branch, which routes a handful away from the revolution loop entirely and does zero per-byte work, so the small-input cost is flat rather than growing.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 *  THE DESK
 *
 *  state = one bone-coloured die-stone: a cube, 6 faces of 64 bits.
 *    a0..a3  the four faces that pass through the groove's seated
 *            position -- the only places a mark's weight is ever
 *            pressed (the rate).
 *    c0,c1   the two axis caps, which never seat under the groove.
 *            No mark ever touches them; they only take up callus,
 *            and they are swept away at the end (the capacity).
 *
 *  The stone is never lifted off the desk and never washed clean.
 * ------------------------------------------------------------------ */

#define ROTL(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

/* the wooden numbered keeps mounted at the desk's edge */
#define KEEP0 0x736f6d6570736575ULL
#define KEEP1 0x646f72616e646f6dULL
#define KEEP2 0x6c7967656e657261ULL
#define KEEP3 0x7465646279746573ULL
#define KEEP4 0x243f6a8885a308d3ULL
#define KEEP5 0x13198a2e03707344ULL

/* ONE FULL REVOLUTION of the stone -- four quarter turns through the
 * chalk-bank groove, the whole body deforming at once.
 * Every single line bends exactly one face as an invertible function
 * of the others, so the revolution is a bijection on the stone: two
 * folding-paths that entered differently can never leave together.
 * This is SipRound -- the validated add-rotate-xor turn. */
#define TURN()                                                        \
    do {                                                             \
        a0 += a1; a1 = ROTL(a1, 13); a1 ^= a0; a0 = ROTL(a0, 32);     \
        a2 += a3; a3 = ROTL(a3, 16); a3 ^= a2;                        \
        a0 += a3; a3 = ROTL(a3, 21); a3 ^= a0;                        \
        a2 += a1; a1 = ROTL(a1, 17); a1 ^= a2; a2 = ROTL(a2, 32);     \
    } while (0)

/* TIPPING THE STONE onto its axis caps: once it has come all the way
 * round the groove four times, the two faces that never seat take up
 * the callus too, and bend the seating faces back. Also a bijection:
 * c0 = rotr(c0',43) - (a0^a2), etc. */
#define TIP()                                                         \
    do {                                                             \
        c0 += a0 ^ a2; c0 = ROTL(c0, 43);                             \
        c1 += a1 ^ a3; c1 = ROTL(c1, 23);                             \
        a1 ^= c0;                                                     \
        a3 ^= c1;                                                     \
        a0 += c1;                                                     \
        a2 += c0;                                                     \
    } while (0)

/* reading a mark as a weight: a flat row of eight marks invested
 * into one wider thing */
static inline uint64_t weight(const unsigned char *m)
{
    uint64_t w;
    memcpy(&w, m, 8);          /* -O3: a single unaligned load */
    return w;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict m = data;
    size_t n = len;

    /* seat the stone against the keeps; the caps remember how long
     * the pile runs */
    uint64_t a0 = KEEP0, a1 = KEEP1, a2 = KEEP2, a3 = KEEP3;
    uint64_t c0 = KEEP4, c1 = KEEP5 ^ (uint64_t)len;

    /* ---- regime 1: a pile long enough to turn the stone right round
     *      four times over. Four revolutions, then a tip. ---- */
    while (n >= 128) {
        a0 += weight(m +   0); a1 += weight(m +   8);
        a2 += weight(m +  16); a3 += weight(m +  24); TURN();
        a0 += weight(m +  32); a1 += weight(m +  40);
        a2 += weight(m +  48); a3 += weight(m +  56); TURN();
        a0 += weight(m +  64); a1 += weight(m +  72);
        a2 += weight(m +  80); a3 += weight(m +  88); TURN();
        a0 += weight(m +  96); a1 += weight(m + 104);
        a2 += weight(m + 112); a3 += weight(m + 120); TURN();
        TIP();
        m += 128;
        n -= 128;
    }

    /* ---- regime 2: whole revolutions. A quarter turn per mark means
     *      the four marks of a revolution seat on four different
     *      faces; the press is an ADD, so how deep each one goes is
     *      bent by the callus already in that face. ---- */
    while (n >= 32) {
        a0 += weight(m +  0); a1 += weight(m +  8);
        a2 += weight(m + 16); a3 += weight(m + 24);
        TURN();
        m += 32;
        n -= 32;
    }

    /* ---- regime 3 (fallback): a handful, too few marks to carry the
     *      stone all the way round. Lay them in the groove as they
     *      are, press once, and let the last keep record how many
     *      there were. No per-mark folding: flat cost. ---- */
    {
        uint64_t w[4] = { 0, 0, 0, 0 };
        if (n) memcpy(w, m, n);              /* zero-padded handful */
        a0 += w[0]; a1 += w[1]; a2 += w[2]; a3 += w[3];
        a3 ^= ((uint64_t)n) << 56;
    }

    /* ---- lift the stone: let it come fully around against the
     *      wooden keeps, tipping it onto its caps twice. ---- */
    TIP();  TURN(); TURN();
    TIP();  TURN(); TURN(); TURN();

    /* ---- the reading, and only the reading, leaves the desk:
     *      384 bits of stone, 64 bits of token, 320 bits of
     *      groove-dust swept off and thrown away. ---- */
    return a0 ^ a1 ^ a2 ^ a3 ^ c0 ^ c1;
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 12

Stated before any measurement, with the reasoning so it can be falsified:

- **Baseline (FNV-1a).** Serial chain per byte = `xor` (1 cycle) + `imul` (3 cycles latency) ≈ 4 cycles/byte ⇒ ~0.25 B/cycle.
- **This kernel.** Critical path per 32-byte revolution = one press (`+=`, 1 cycle; the four presses are independent) + `TURN`, whose dependency chain is about 5 serial add/rotate/xor ops ⇒ ~6 cycles per 32 bytes ⇒ ~5.3 B/cycle. The `TIP` every 128 bytes adds ~4 cycles ⇒ ~4.6 B/cycle sustained.
- Ratio ≈ 18× if ALU-bound; I predict **12×** because large buffers will be partly memory-bound and small-input cases pay the fixed ~40-cycle read-off at the keeps.
- **Avalanche prediction: 0.49–0.50.** Every byte is pressed into the stone before any fold; the fold is a bijection so differences cannot cancel; the last-pressed word still receives 5 revolutions + 2 tips before the reading, which is at or above SipHash-2-4's finalisation budget, and the reading discards 320 of 384 bits.
- **Failure modes I expect, named in advance:** (a) throughput below ~6× would mean the `TURN` chain is longer than my 5-cycle estimate or the loads are not folding into the adds; (b) an avalanche score below ~0.45 would mean one `TURN` per 32-byte revolution is too thin for the interior and the fix is a second `TURN` per revolution at ~2× the cycle cost, not a different mechanism.

# MEASUREMENT

**Not measured — reported plainly as a failure of this session, not papered over.** No tools were available in this run: `hash_bench` and `hash_contract` could not be invoked, so the four permitted improvement rounds never happened and the numbers above remain a prediction, not a result. What I can state honestly:

- Compile-correctness checked only by reading: C99, `<stdint.h>/<stddef.h>/<string.h>` only, exact contract `uint64_t kernel(const unsigned char *, size_t)`, no reads past `data[len-1]` (the 128/32 loops are bounded and the handful uses `memcpy(w, m, n)` with `n < 32`), `len == 0` well-defined.
- Invertibility of `TURN` and `TIP` verified by hand inversion (each statement updates one lane via `+`, `^`, or a rotation of itself, so the inverse is written mechanically).
- Throughput and avalanche: **unverified.** The pipeline's measurement is the real result; if it contradicts the prediction above, the prediction was wrong.

# VERDICT

The core of this kernel **is** Seed 2, translated literally: a cube-shaped never-reset state whose quarter turn changes which of four faces receives the next weight, pressed with an add so the callus already there bends how deep the press goes, folded by a bijection so differing paths never re-merge, and read out once through fixed keeps with 320 of 384 bits swept away. There is no accumulator and no multiply anywhere — which is exactly the previous attempt's failure, since that one was xxHash with extra lanes.

What the metaphor bought, concretely: a 32-byte rate instead of a 1-byte one (four presses per fold instead of one multiply per byte), which is where the predicted ~12× comes from; and a 384-bit hidden body truncated to 64 bits, which is where the avalanche comes from.

Risks, and how each is addressed rather than left standing:

1. *Fixed read-off cost dominates tiny piles.* Guarded by the `len < 32` branch: the handful path does one `memcpy`, one press, and no per-byte folding, so cost is flat in `len` below 32. I deliberately did **not** add an FNV fallback for `len < 8` — it would be faster by a few cycles and would wreck the avalanche score, which FNV does not provide at short lengths.
2. *One `TURN` per revolution may be thin for interior blocks.* Named above with its exact remedy (a second `TURN` per revolution, halving throughput) so the measurement can decide; I did not pre-pay for it.
3. *Thread parallelism.* Refused, and the refusal is the metaphor's own — one stone, centuries folding down like a handful. Several stones would break mechanism #1. The fold chain is ~6 cycles per 32 bytes, so there is no thread-sized unit of work here at benchmark sizes.
4. *Endianness.* `weight()` assumes little-endian, which the stated `gcc -march=native` x86 target satisfies; on a big-endian host the token would differ (still a valid hash, not a bug, but worth saying out loud).