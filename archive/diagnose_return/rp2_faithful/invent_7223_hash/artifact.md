## MAPPING

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| world object | problem object |
|---|---|
| pile of marks, given order | `data[0..len-1]`, read in index order |
| the sphere | one carried register — the entire state |
| the sphere's face | the bit positions the mark is pressed into (XOR site) |
| "my only memory" | no side table, no per-position array, no second accumulator |
| "the old face I let shrink and go" | right xorshift: bits fall off the low end |

Breaks: **nothing** — it *ratifies* "the state is a single accumulator updated in place." It is actually a *stronger* constraint than the known way: it forbids xxHash's four independent lanes. (This matters below: my four "faces" are cross-coupled every revolution, so they are one sphere, not four accumulators.)

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| world object | problem object |
|---|---|
| one tumble / turn of the coil | one **rotation** of the 64-bit state |
| the mason (sacbe) trail, coiled, fixed pitch | a fixed rotation constant, the coil's pitch |
| coil circumference = one turn's worth of marks | the load width: 8 marks = one 64-bit word |
| striking a stalk, "sobs once" | one mixing step per turn: `s ^= rot(s,pitch) ^ rot(s,angle)` |
| "cracking a hairline into itself" | XOR of the state with rotations **of itself** |
| "the angle of that crack as the next weight to carry forward" | the rotation amount for the *next* stalk is read out of the current state → **data-dependent rotation** (the sole nonlinearity) |
| "a fixed count, no more, no fewer" | a fixed, small, non-tunable round count |
| "a footprint at a stalk is not my business, never look back" | no lookup table, no memo array, strictly forward streaming |
| "one wrong mark and every stalk downstream sobs differently" | avalanche: one bit flip changes every later angle, so the whole trajectory diverges |

Breaks: **"mixing one byte requires a multiplication"** (the mixing primitive is rotate+XOR+shift; *zero* multiply instructions anywhere in the mix), and also **"more mixing rounds always means better mixing"** ("no more, no fewer": one crack per turn, a fixed finalizer).

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token; intermediate cracks and dust swept away."**

| world object | problem object |
|---|---|
| the sphere at the end | the streaming state — *discarded*, not returned |
| the last, smallest crack in the final stalk | the finalizer's output: the returned 64-bit token |
| stone dust, discarded footprints, eggshells | intermediate states / partial words — never stored |
| "swept into the jungle's open plumbing" (a cenote) | nothing is kept live; O(1) space |

Breaks: the *"in place, one value"* half of "the state is a single accumulator updated in place" — in FNV the accumulator **is** the answer; here the answer is a distinct token squeezed out of the state at the end, and the state itself is thrown away. That licenses concentrating mixing effort at the end instead of spreading it uniformly.

## CHOSEN SEED

**SEED 2.** It is the one that breaks "mixing one byte requires a multiplication" (the instructed preference), and it is the most mechanically literal: "tumble" = rotate, "crack a hairline into itself" = XOR with rotations of itself, "the angle of the crack carried forward" = data-dependent rotation amount. SEED 1 and SEED 3 are kept as *constraints on* SEED 2's implementation (one sphere; the output is a token, not the sphere), not as separate mechanisms.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** The kernel contains no multiply in the mixing path at all — only XOR, fixed rotates, one data-dependent rotate, and one right xorshift per stalk. Secondarily: **"more mixing rounds always means better mixing"** — the count of cracks is fixed by the coil's geometry ("no more, no fewer"), and the *only* place extra rounds are spent is the final token.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ============ THE NARROW COIL: one sphere, one face ============ */

/* the sphere turns on the trail: a rotation -- never a multiply */
static inline uint64_t sturn(uint64_t x, unsigned r) {
    return (x << (r & 63u)) | (x >> ((64u - r) & 63u));
}

/* the angle of the hairline crack, read off the sphere's high face.
   Forced odd (1..63) so it can never coincide with the coil's even pitch:
   s ^ rot(s,pitch) ^ rot(s,angle) is then always a weight-3 rotation
   polynomial, hence a bijection -- the sphere loses no marks. */
static inline unsigned cangle(uint64_t s) {
    return (unsigned)((((s) >> 58) & 31u) << 1) | 1u;
}

#define PITCH 26u                 /* the coil's own pitch: fixed, even */

/* A STALK: the sphere sobs once, cracks a hairline into itself at the angle
   it carried here, reads the new angle to carry forward, and lets the old
   face shrink and go. */
#define STALK(s, a) do {                                  \
    (s) ^= sturn((s), PITCH) ^ sturn((s), (a));           \
    (a)  = cangle((s));                                   \
    (s) ^= (s) >> 28;                                     \
} while (0)

/* THE LAST, SMALLEST CRACK: the sphere is not kept, only this token.
   The shrink distance is itself read from the crack (bits 60..63 survive a
   right xorshift of >=17, so the step stays invertible). */
#define FROUND(s, a, pit) do {                            \
    (s) ^= sturn((s), (pit)) ^ sturn((s), (a));           \
    (s) ^= (s) >> (17u + (unsigned)((s) >> 60));          \
    (a)  = cangle((s));                                   \
} while (0)

static inline uint64_t last_crack(uint64_t s, unsigned a, uint64_t len) {
    s ^= sturn(len, 40);                  /* the tally of marks, at the final stalk */
    FROUND(s, a, 14u);
    FROUND(s, a, 22u);
    FROUND(s, a, 30u);
    FROUND(s, a,  6u);
    FROUND(s, a, 18u);
    return s;
}

#if defined(__AVX2__)
/* ============ THE WIDE COIL: one sphere, four faces ============ */

#define VTURN(x, r) _mm256_or_si256(_mm256_slli_epi64((x), (r)),           \
                                    _mm256_srli_epi64((x), 64 - (r)))

static inline __m256i vturnv(__m256i x, __m256i r) {   /* per-face crack angle */
    return _mm256_or_si256(
        _mm256_sllv_epi64(x, r),
        _mm256_srlv_epi64(x, _mm256_sub_epi64(_mm256_set1_epi64x(64), r)));
}
static inline __m256i vangle(__m256i s) {
    return _mm256_or_si256(
        _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(s, 58),
                                           _mm256_set1_epi64x(31)), 1),
        _mm256_set1_epi64x(1));
}

/* ONE TURN of the wide coil: a full turn's worth of marks (32) is pressed
   onto the sphere's four faces at once, then each face cracks at its own
   carried angle and its old face shrinks and goes. */
#define WTURN(S, A, q) do {                                               \
    __m256i W_ = _mm256_loadu_si256((const __m256i *)(q));                \
    (S) = _mm256_xor_si256((S), W_);                                      \
    (S) = _mm256_xor_si256((S), _mm256_xor_si256(VTURN((S), 26),          \
                                                 vturnv((S), (A))));      \
    (A) = vangle((S));                                                    \
    (S) = _mm256_xor_si256((S), _mm256_srli_epi64((S), 28));              \
} while (0)

/* THE STALK after a fixed count of turns (one full revolution, four faces):
   every face cracks into the others.  I + P_neighbour + P_opposite has odd
   weight in the group algebra, so this too is a bijection -- no face is lost.
   This is what makes the four faces ONE sphere and not four accumulators. */
static inline __m256i wstalk(__m256i S) {
    __m256i u = _mm256_shuffle_epi32(S, _MM_SHUFFLE(1, 0, 3, 2)); /* neighbours */
    __m256i v = _mm256_permute2x128_si256(S, S, 0x01);            /* opposites  */
    return _mm256_xor_si256(S, _mm256_xor_si256(u, v));
}
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t n = len;
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;  /* sphere at the high mouth */
    unsigned a = 29u;                                    /* the mouth's own angle    */

#if defined(__AVX2__)
    /* REGIME TEST, in world terms: does the pile fill whole turns of the WIDE
       coil?  If not, the wide coil is not worth walking -- take the narrow one. */
    if (n >= 64) {
        __m256i S = _mm256_xor_si256(
            _mm256_set_epi64x((int64_t)0x452821E638D01377ULL,
                              (int64_t)0xBE5466CF34E90C6CULL,
                              (int64_t)0xC0AC29B7C97C50DDULL,
                              (int64_t)0x9E3779B97F4A7C15ULL),
            _mm256_set1_epi64x((int64_t)len));
        __m256i A = _mm256_set_epi64x(45, 19, 7, 29);    /* odd mouth angles */
        uint64_t f[4];

        while (n >= 128) {           /* a fixed count of turns: four, then a stalk */
            WTURN(S, A, p);
            WTURN(S, A, p + 32);
            WTURN(S, A, p + 64);
            WTURN(S, A, p + 96);
            S = wstalk(S);
            p += 128; n -= 128;
        }
        while (n >= 32) {            /* the coil's last short revolutions */
            WTURN(S, A, p);
            S = wstalk(S);
            p += 32; n -= 32;
        }
        _mm256_storeu_si256((__m256i *)f, S);   /* the sphere is set down: */
        s = sturn(f[0], 1) ^ sturn(f[1], 17)    /* its four faces crack into one */
          ^ sturn(f[2], 33) ^ sturn(f[3], 49);
        a = cangle(s);
    }
#endif

    while (n >= 8) {                 /* the narrow coil: one turn = 8 marks */
        uint64_t w;
        memcpy(&w, p, 8);
        s ^= w;                      /* press the turn's marks into the face */
        STALK(s, a);
        p += 8; n -= 8;
    }
    while (n != 0) {                 /* the last marks, pressed one by one */
        s ^= (uint64_t)(*p++);
        STALK(s, a);
        n--;
    }
    return last_crack(s, a, (uint64_t)len);  /* hand over only the hairline */
}
```

**Which code implements which part of the native's mechanism**

| native's words | code |
|---|---|
| "a sphere at the trail's high mouth… my only memory" | `s` (narrow) / `S` (wide, one YMM register = one sphere, four faces) |
| "press its shape into the sphere's face" | `s ^= w` / `S = xor(S, W)` |
| "it tumbles a fixed count of turns down the mason trail" | `sturn` / `VTURN(S,26)` — rotation by the coil's fixed pitch |
| "where it strikes a stalk it sobs once, cracking a hairline into itself" | `s ^= sturn(s,PITCH) ^ sturn(s,a)` (weight-3 ⇒ bijective) |
| "I read the angle of that crack as the next weight to carry forward" | `a = cangle(s)` / `A = vangle(S)` — data-dependent rotation, carried to the *next* stalk (which is also why it is off the critical path) |
| "the old face I let shrink and go" | `s ^= s >> 28` |
| "a footprint at a stalk is not my business, I never look back" | no table, no scratch array, no second accumulator; single forward pass |
| "no more, no fewer" | one crack per turn; one cross-face stalk per revolution of four; a fixed 5-round token |
| "one wrong mark and every stalk downstream sobs differently" | every later `a` depends on the changed state ⇒ the two runs rotate by different amounts ⇒ trajectories diverge nonlinearly |
| "I don't keep the sphere, only the last hairline crack" | `last_crack()`; the streaming state is never returned |
| "dust, footprints, eggshells swept into the open plumbing" | nothing retained: O(1) space, no allocation |
| the coiled trail's two gauges (wide/narrow) | the `n >= 64` regime test + the 8-byte and 1-byte fallbacks |

**Regime handling (requirement 5).** The known way spans small-key and large-buffer regimes. The native recognizes the regime by the coil's geometry: a pile that does not fill whole turns of the *wide* coil (`len < 64`, or no wide coil on this machine — `#if !defined(__AVX2__)`) is walked on the *narrow* coil, 8 marks per turn; the trailing marks that fill no turn at all are pressed one by one. All three paths converge on the same stalk rule and the same final crack.

## PREDICTION

PREDICTION: speedup_vs_dp = 8.0

Reasoning stated before any measurement: FNV-1a's recurrence is latency-bound at `xor(1) + imul(3) ≈ 4` cycles **per byte**. The wide coil's recurrence is ≈ 7 cycles per 32-byte turn plus ≈ 5 cycles per 4-turn stalk → ≈ 33 cycles per 128 bytes ≈ **0.26 c/B** (≈ 15× on pure compute); the narrow coil is ≈ 6 cycles per 8 bytes ≈ **0.75 c/B** (≈ 5×). I discount to 8.0 for memory bandwidth on large buffers, the fixed ~35-cycle token cost on short ones, and the possibility that variable-count `rol` costs 2 cycles rather than 1 on the host. Avalanche: I predict a good score (per-bit flip probability within ~1% of 0.5) because the 5-round finalizer runs on *every* path and each round's rotation amount differs between the two runs as soon as the difference reaches the top bits — i.e. the mechanism is not linear, unlike a table/CRC-style rotate-XOR chain, which would score badly here. Weakest point of the prediction: rotate-only mixing has no carry propagation, so any residual bias would show up as a small but systematic avalanche deficit rather than a random one.

## MEASUREMENT

**Not measured.** `hash_bench` / `hash_contract` were not available in this session (no tools were exposed), so the four allowed improvement rounds could not be run. Stating this plainly rather than reporting numbers I did not obtain: the prediction above is unverified, and the artifact is the first version, not a tuned one. What I *did* verify, by hand, is the algebra that the quality claim rests on:

- `s ^ rot(s,pitch) ^ rot(s,angle)` with `pitch` even and `angle` odd ⇒ three distinct rotation exponents ⇒ odd-weight polynomial ⇒ non-zero at `x=1` ⇒ coprime to `x⁶⁴+1 = (x+1)⁶⁴` over GF(2) ⇒ **bijective every time**. (The tempting `s ^= rot(s,a)` is *never* bijective — it always collapses onto the even-parity half. I avoided it for that reason.)
- `wstalk` = `I + P_neighbour + P_opposite`; Gaussian elimination on the 4×4 GF(2) matrix gives rank 4 ⇒ bijective. (The obvious `S ^= permute4x64(S, cyclic)` is 2⁶⁴-to-1 — an entropy leak of 64 bits per iteration. Avoided.)
- `s ^= s >> (17 + (s>>60))`: the shift is ≥ 17, so bits 60–63 are untouched, so the shift distance is recoverable ⇒ invertible despite being data-dependent.
- `sturn(x,0)` and `sturn(x,64)` are safe (no `>> 64` UB), and the vector rotate is safe for counts 0…64 because AVX2 variable shifts saturate to zero rather than being undefined.

## VERDICT

The native's mechanism translates without any textbook substitution: **the multiplication is simply gone**, replaced by the tumble (rotation) and the crack whose angle is read out of the sphere itself (data-dependent rotation). That data-dependent angle is the whole nonlinearity budget, and it is exactly what the native described — not something I added to rescue the idea. Two places where taking the metaphor literally forced a *better* design than the obvious reading would have: "cracking a hairline **into itself**" had to become a three-term rotation XOR, because the two-term version silently throws away a bit on every stalk; and "the sphere tumbles, the faces turn into one another" had to become an odd-weight face permutation XOR, because the natural cyclic version leaks 64 bits per iteration. Honest cost of the mechanism: the sphere is a single carried memory, so throughput is capped by one dependency chain — it cannot reach the ~0.05 c/B of a four-independent-lane SIMD hash, and it is not supposed to, because SEED 1 forbids independent lanes.

Risks I named, and how each is closed in the shipped code rather than left open:

1. *SIMD is only worth it on a long pile.* Closed by the `n >= 64` runtime regime test, with the narrow-coil scalar path as the fallback, plus the `#if defined(__AVX2__)` compile-time fallback so the kernel still builds and still wins ~5× on a machine with no wide coil.
2. *A fixed multi-round finalizer could dominate very short inputs.* Trimmed to 5 rounds (~35 cycles) — under FNV's own cost for anything from ~10 bytes up — and it is unconditional, so no input length gets weak avalanche.
3. *Pure rotate/XOR could be too linear.* Closed by making every stalk's angle data-dependent (streaming, wide, tail *and* finalizer) and by adding a data-dependent shrink distance in the finalizer; no path is purely linear.
4. *Thread parallelism.* Deliberately not used: one sphere is the metaphor's own unit of work, splitting it into per-thread spheres would be four accumulators, and at these buffer sizes OpenMP fork cost would swamp a kernel running at ~0.26 c/B.

Nothing in this verdict names a condition I have not guarded. The one thing I cannot claim is the number: with no bench available, the 8.0 and the avalanche estimate stand as predictions to be falsified by the pipeline, and if the avalanche score comes back low, the honest fix inside this mechanism is more stalk-angle entropy (read the angle from a bit field that the xorshift has already stirred) rather than reintroducing a multiply.

---

*Note: the `claude.ai PubMed` MCP server in this environment is unauthorized, and this session is non-interactive so it can't be authorized here — if you need it, authorize it from your claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session). It wasn't needed for this task.*