## DICTIONARY

| Recipe object | Computational object |
|---|---|
| the huge sheet, pink or orange | the 64-bit word space the state lives in; "pink or orange" = which constant family — irrelevant to the result, so one fixed family is chosen |
| spiral / snail-shell markings running corner to corner | the spiral of positions on the sheet: shell `k` is reached from a point by `k` applications of the step `x ↦ rotl(x,7) + SPIRAL_D`. Closed form for `k` steps: `rotl(x,7k) + k·SPIRAL_D` |
| a crease | a **line segment**, i.e. a pair of 64-bit words `(A,B)` = (where it began, where it ended). This is the running state |
| the four wire birds along the left edge | four 64-bit accumulator lanes `w0..w3`, each with its own fixed beak constant |
| a beak meeting the paper | the lane is in contact, i.e. non-zero after the touch. A beak "misses" iff its lane lands on 0 |
| the pile of marks, face down, top only | the input buffer `data[0..len-1]`, read strictly forward, one byte, no lookahead, no buffering, single pass |
| one mark | one input byte |
| strokes in a mark | `popcount(byte)` — the *angle*; the mark's full 8-bit identity is still the thing being folded in (see Ambiguity 1) |
| the seed crease | fixed input-independent init: `A = BIRD0`, `B = FAR_SHELL` |
| walking `n` shells from where the last crease ended | `P = rotl(B, 7n) + n·SPIRAL_D` |
| folding from that shell toward where the last crease began, pressed flat | `E = rotl(P ^ A, 23) * PRESS_K` |
| "where it stops is where the next one starts" | `A ← P`, `B ← E` |
| the new fold's corner | `corner = E ^ (E >> 29)` |
| touching the corner to a beak | lane update (add / xor / rotate-xor / rotate-add — four genuinely different touches so four lanes carry four different views) |
| unfolding that one crease only | algebraic inverse of the four touches (each is a bijection in the lane), applied to the *current* crease only — nothing earlier is ever revisited |
| pulling a half-shell tighter | `corner = rotl(corner,3) + HALF_D` (half of `rotl(·,7)+SPIRAL_D`) |
| throwing the mark in the mud at the tide line | `mark++` and never re-reading it; no second pass, no comparison buffer |
| one crease per mark, count equal exactly | exactly `len` rounds, no padding round, no extra round, no round without a byte |
| right hand closes on nothing | `remaining == 0` |
| holding under water, thin → thinner → one dense corner, for one fixed breath | finalization: collapse the 6 state words to one, then a **fixed** (data-independent) 3-stage xorshift-multiply press |
| the coin / the token | the returned `uint64_t` |
| the mud where the tide-boundary closes | all locals cleared; nothing survives the call, no statics, no side effects |
| step 12 side-by-side check | the kernel is a pure deterministic function of `(data,len)`: re-running reproduces the token bit-for-bit |

**Ambiguity 1 (step 3 vs step 4 — the one place literal reading would give a *wrong* hash).** Step 4 reads *the angle* from the stroke count. If the byte entered **only** through `popcount`, the hash would be blind to bit permutations inside a byte (`0x01` and `0x80` identical) — 8 bits collapsed to 3.17. Step 3 is explicit that "the mark and the crease you made last — **those two** together" decide the fold, so the mark is present as a whole object, not merely its stroke count. The smallest change: the angle stays `popcount`, and the mark additionally lies on **its own shell** of the sheet, `(m+1)·SPIRAL_C`, added at the point the fold starts. No step added, step 4 unchanged in structure.

**Ambiguity 2 (step 5).** "Repeat until all four agree" is unbounded. At most 4 corner values out of 2⁶⁴ can fail, and tightening is a bijection, so the loop terminates with probability 1 — but not provably. I cap at 8 tightenings (unreachable in practice: P ≈ 2⁻⁶² per byte) so the kernel cannot hang. This is the only deviation.

**Ambiguity 3 (parallelism).** Steps 3, 6, 7 forbid it outright: each crease needs the previous crease, marks are destroyed after one read, one crease per mark in pile order. So **no OpenMP, no SIMD over bytes**. The recipe is a strictly serial byte-at-a-time chain and I am not going to quietly replace it with a block-parallel hash. This is the dominant cost and I expect it to lose to the baseline.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- *
 *  the sheet (the spiral of snail shells), the birds, the press     *
 * ---------------------------------------------------------------- */
#define ROTL64(x,r) (((x) << ((r) & 63)) | ((x) >> ((64 - (r)) & 63)))

#define SHELL_STEP  7                        /* one shell along a crease: rotl(.,7)+D   */
#define SPIRAL_D    0xD1B54A32D192ED03ULL    /* additive part of one shell step         */
#define SPIRAL_C    0x9E3779B97F4A7C15ULL    /* which shell a mark itself lies on       */
#define PRESS_K     0xBF58476D1CE4E5B9ULL    /* pressing a fold flat                    */
#define HALF_D      0x94D049BB133111EBULL    /* half a shell, for tightening            */
#define FAR_SHELL   0x452821E638D01377ULL    /* the farthest snail shell on the sheet   */

static const uint64_t BEAK[4] = {            /* the four wire birds, left edge          */
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL
};

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the sheet flat; set the four wire birds along its left edge,
       beaks toward the paper; stack the marks face down at the right hand, so
       only the top one is ever reachable and only in arrival order.          */
    uint64_t w0 = BEAK[0], w1 = BEAK[1], w2 = BEAK[2], w3 = BEAK[3];
    const unsigned char *mark = data;   /* top of the face-down pile */
    size_t remaining = len;

    /* step 2: before any mark is read, the seed crease — from the first bird's
       beak to the farthest snail shell. Input-independent, identical for every
       sheet, so two folders of the same pile begin identically.              */
    uint64_t A = BEAK[0];               /* where the last crease began */
    uint64_t B = FAR_SHELL;             /* where the last crease ended */

    while (remaining) {
        /* step 3: take the top mark off the pile; do not look at the one
           beneath it. This mark and the crease made last — those two, and
           nothing else, decide the next fold.                               */
        unsigned m = *mark;

        /* step 4: count the strokes in the mark; walk that many snail shells
           along the last crease from where the last crease ENDED; fold from
           that shell toward the point where the last crease BEGAN; press the
           fold flat. Where it stops is where the next one starts.
           (n shells = n applications of rotl(.,7)+SPIRAL_D, in closed form;
            the mark itself lies on its own shell (m+1)*SPIRAL_C.)           */
        unsigned n   = (unsigned)__builtin_popcount(m);
        uint64_t off = (uint64_t)n * SPIRAL_D + ((uint64_t)m + 1u) * SPIRAL_C;
        uint64_t P   = ROTL64(B, SHELL_STEP * n) + off;     /* the shell walked to */
        uint64_t E   = ROTL64(P ^ A, 23) * PRESS_K;         /* folded toward A, pressed */
        A = P;                                              /* next crease begins here */
        B = E;                                              /* ...and ends here        */

        /* step 5: carry the new fold's corner to each of the four beaks in
           turn. If all four meet the paper, the crease is set. If any beak
           misses (its lane lands on nothing), unfold THAT ONE CREASE ONLY —
           by inverting exactly those four touches — pull it a half-shell
           tighter, and touch again. Nothing earlier is ever unfolded.       */
        uint64_t corner = E ^ (E >> 29);
        unsigned tries  = 0;
        for (;;) {
            w0 += corner;
            w1 ^= corner;
            w2  = ROTL64(w2, 19) ^ corner;
            w3  = ROTL64(w3, 41) + corner;
            int all_four_meet = ((w0 != 0) & (w1 != 0) & (w2 != 0) & (w3 != 0));
            if (__builtin_expect(all_four_meet, 1) || tries >= 8) break;
            w0 -= corner;                                   /* unfold this touch only */
            w1 ^= corner;
            w2  = ROTL64(w2 ^ corner, 64 - 19);
            w3  = ROTL64(w3 - corner, 64 - 41);
            corner = ROTL64(corner, 3) + HALF_D;            /* a half shell tighter   */
            ++tries;
        }

        /* step 6: discard the mark into the mud at the tide line. It is never
           read a second time and is never kept anywhere it could be compared
           against later — the pointer only ever moves forward, single pass.  */
        ++mark;

        /* step 7: return to step 3 with the next top mark — once for every
           mark in the pile, one mark one crease, no crease without a mark.   */
        --remaining;
    }

    /* step 8: the marks are finished exactly when the right hand closes on
       nothing. No crease is added to tidy the wad and none is skipped because
       the sheet grew stiff: the loop above ran exactly len times, so
       creases == marks exactly. (Invariant, not an operation.)              */

    /* step 9: take the wad to the water's edge and press — thin, then
       thinner, then one hard dense corner no bigger than a coin — for one
       held breath, the same measure every time (fixed, data-independent).   */
    uint64_t t = w0 ^ ROTL64(w1, 17) ^ ROTL64(w2, 31) ^ ROTL64(w3, 47)
                    ^ ROTL64(A, 11) ^ B;                    /* thin  */
    t ^= t >> 33;  t *= 0xFF51AFD7ED558CCDULL;              /* thinner */
    t ^= t >> 29;  t *= 0xC4CEB9FE1A85EC53ULL;              /* one hard dense corner */
    t ^= t >> 32;                                           /* takes no more pressing */

    /* step 10: lift out the corner. That is the token — it holds its shape
       dry, so it is simply carried out as it stands.                        */
    uint64_t token = t;

    /* step 11: everything else goes into the mud where the tide-boundary
       closes — the birds, the last crease, the failed half-shell refolds,
       the wet remainder of the sheet. Nothing but the coin leaves.          */
    w0 = w1 = w2 = w3 = 0; A = B = 0; remaining = 0; mark = 0;

    /* step 12: to check a pile against a token later, fold it again from a
       fresh sheet and compare the two corners. This holds because the kernel
       is a pure function of (data,len): no statics, no clocks, no addresses,
       no retained state between calls.                                      */
    return token;
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 0.45**

Stated before any measurement, with the reasoning so it is falsifiable:

- The recipe is strictly serial (one crease per mark, each crease from the previous one). Critical path per byte: `rotl(B,7n)` (1c) → `+off` (1c) → `^A` (1c) → `rotl` (1c) → `imul` (3c) ≈ **7 cycles/byte**, ≈0.14 B/cycle, ≈0.4–0.5 GB/s at 3.5 GHz. The `popcount`, the two multiplies forming `off`, the four beak touches and the four zero-tests all sit off the critical path and should be absorbed by the out-of-order window (~3 multiplies/byte on one multiply port, below the 7c path bound).
- I assume `dp` is the simple straightforward byte-at-a-time reference (FNV-1a-class: `xor` + `imul` ≈ 4–4.5 c/byte). 4.5/7 ≈ **0.64** optimistically; I discount to **0.45** for the retry-branch and beak-test pressure and for imperfect scheduling.
- **If instead `dp` is a block-wide reference** (32 bytes/iteration, ~0.3 c/byte), the honest expectation is ~**0.05**, and I will not pretend otherwise — the recipe forbids the block parallelism that would be needed to compete.
- Avalanche: predicted **0.49–0.51**. Every byte enters through a rotate-amount *and* its own multiplicative shell offset, then through a full 64×64 multiply on the crease chain; the final press is a 2-round xorshift-multiply over all six state words, which is what carries the *last* byte (the weakest case) to full diffusion.
- Predicted weak spot if any: not avalanche but distribution on very short inputs (len ≤ 2), where only 1–2 multiplies precede the press — the press should still cover it.

## MEASUREMENT

**Not measured.** No tools were available in this session — `hash_bench`, `hash_contract` and the symbolic/unconventional helpers were all absent, so I could neither verify the signature against `hash_contract` nor obtain throughput or an avalanche score. I therefore performed **0 of the 4 permitted improvement iterations**, because iterating without a measurement is just guessing dressed as tuning.

What I did verify by hand, since it was all I could do honestly:

- Signature matches the fixed contract exactly: `uint64_t kernel(const unsigned char *data, size_t len)`.
- No UB in the rotates: `ROTL64` masks both shift counts, so `r == 0` yields `x >> 0`, not a 64-bit shift.
- `len == 0` is well defined: no rounds, token = press of the seed state.
- The step-5 retry is provably bounded (cap 8) and its unfold is the exact algebraic inverse of the four touches — I checked each of the four inverses individually.
- Steps appear once each, in order, none added, none substituted.

Everything in the PREDICTION section is a claim the pipeline can now refute.

## VERDICT

The recipe translates cleanly and almost without strain: a crease really is a line segment (two words), the four birds really are four accumulator lanes with a contact test, the snail spiral really is a closed-form `rotl(x,7k)+k·D` walk, and "one crease per mark, no crease without a mark" really is a byte-at-a-time round function with no padding. Two places needed a decision rather than a reading: the mark had to enter the fold as a whole object and not only as its stroke count (step 3 says so explicitly; without it the hash is blind to bit permutations within a byte and is simply broken), and the "repeat until all four agree" loop needed a termination cap it will never reach.

The honest cost is speed. Steps 3, 6 and 7 together are an explicit prohibition on parallelism — one mark at a time, no lookahead, no second pass, and each crease depends on the one before — so this cannot be a wide hash, and I did not make it one. I expect it to be roughly half the speed of a simple serial reference and an order of magnitude slower than a block-based one, while being *better* mixed than either. If the measured avalanche comes back near 0.50 and the speedup near 0.45, the recipe is a sound but deliberately serial hash. If avalanche comes back low, the fault is most likely the final byte's short path into the press, and step 9 — not the round — is where I would look first.