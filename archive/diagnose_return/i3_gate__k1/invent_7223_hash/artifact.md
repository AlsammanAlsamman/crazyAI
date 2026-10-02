## MAPPING

I take every object in the account as naming something concrete, and read the account twice: once for what the objects *are*, once for what the grammar *forbids*.

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| World object | Problem object |
|---|---|
| mark | one input byte of `data` |
| the pile, in its given order | the buffer `data[0..len)`, read forward |
| the sphere | the one state object carried in registers, nothing else live |
| "press its shape into the sphere's **face**" | the mark lands on *whichever face is up* — a sphere rolling down a trail presents a different face to each successive mark |
| "only the sphere's new face matters, the old face I let shrink and go" | a face's old value is never re-read except through that same face: faces do not consult each other while the pile is being laid |
| "a footprint at a stalk is a woman's business, I never look back" | zero scratch arrays, zero stores inside the loop; register-resident only |

Breaks **"the state is a single accumulator updated in place, one value."** The sphere is *one* memory but it has *many faces*; successive marks go to different faces, and faces are mutually independent during absorption. One object, several lanes.

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| World object | Problem object |
|---|---|
| **tumble / "turns"** | **a bit rotation** — the trail is *coiled*, so every turn is circular: `rotl` |
| the coiled limestone trail, "the shadow already runs white and arching" | the rotation schedule is fixed before the first mark: compile-time constants, no table, no memory traffic |
| stalk | one mixing site, with its own fixed angle |
| "it sobs once, **cracking a hairline into itself**" | the state is XORed with rotations *of itself*: `x ^= rotl(x,a) ^ rotl(x,b)` |
| "I read **the angle** of that crack as the next **weight** to carry forward" | angle = rotation amount; **weight = an additive quantity, carried** → carry-propagating addition `x += mark` |
| "a fixed count of turns, **no more, no fewer**" | exactly 2 crack-sites per 8 bytes, a compile-time constant, unrolled |

Breaks **"mixing one byte requires a multiplication."** The only verbs on this trail are *turn* (rotate), *carry weight* (add), and *press in* (xor). There is no multiplier anywhere in the jungle.

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token; all intermediate cracks and dust are swept away."**

| World object | Problem object |
|---|---|
| "I don't keep the sphere itself" | the carried state is **not** the hash; it is wider than the hash and is discarded |
| "the last, smallest crack in the **final** stalk" | a separate finalization: one last stalk, struck harder (more turns, paid once), collapsing all faces to 64 bits |
| "the eggshells the trail sheds along the way" | the trailing bytes that don't fill a face, plus the pile's own size |
| "sweep off into the jungle's open plumbing and forget on purpose" | intermediate per-stalk values are dead; nothing is retained |

Breaks **"more mixing rounds always means better mixing."** The trail gets a *small* fixed count; the extra turns are spent once, at the last stalk, where they cost nothing per byte. Asymmetric round budget, not a uniform one.

## CHOSEN SEED

**SEED 2.** It is the one that breaks the preferred assumption ("mixing one byte requires a multiplication"), its mapping is the most literal of the three — *turn* → `rotl`, *weight carried* → `+`, *press in* → `^`, *crack into itself* → `x ^= rotl(x,·)` — and it is maximally distant from FNV-1a/xxHash, whose entire engine is the multiply. SEEDs 1 and 3 are kept as the structure that carries SEED 2's primitive (faces for width, a separate final stalk for the token), because the native states all three in one breath and they are not separable.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** Replaced by Add–Rotate–Xor.

Two things make this honest rather than wishful:

1. *No entropy is lost without a multiplier.* An odd multiplier is a bijection, which is why FNV works. My crack operator `x ↦ x ^ rotl(x,a) ^ rotl(x,b)` is the polynomial `1 + z^a + z^b` in `GF(2)[z]/(z^64+1)`, and `z^64+1 = (z+1)^64` over GF(2), so that ring is local with maximal ideal `(z+1)`: an element is a unit **iff it has an odd number of terms**. Three terms → always a bijection. Every stalk on the trail is invertible, for free, for any angles.
2. *Rotation and xor alone are GF(2)-linear*, so a pure-rotation trail would have deterministic (maximally biased) per-bit avalanche. The native supplies the fix himself: the sphere's **running weight, carried forward**. Carry is the one thing a rotation can never fake. `+` is the sole nonlinearity, and it is enough.

Per step 4, I let this arrive at a validated technique instead of inventing one: ARX with a fixed small per-word round count and a longer finalization, output = XOR-fold of the state, length pressed into the last partial word — that is **SipHash-c-d**, and I use **SipHash's exact `SIPROUND` and its 4-round finalization as the final stalk** (SipHash-1-3/2-4 are the default hashers in CPython 3.11+ and Rust). The chosen angles `(13,37)` and `(19,47)` are picked so the composed polynomial has 9 *distinct* exponents mod 64 — `{0,13,19,20,32,37,47,56,60}`, no XOR cancellation (the naive `(17,41),(23,53)` loses two terms to `41+23≡0` and would be weaker).

Regime recognition, which the metaphor must supply itself (step 5): the native counts the pile at the trail's mouth. A pile too small to turn the sphere onto a second face uses one face; a pile big enough splits onto eight faces on the wide trail (AVX2) and is gathered again. The wide trail is **guarded at `len >= 256`** with the narrow 4-face trail as the fallback, because splitting and re-gathering the sphere is a cost that only a big pile repays — that is the one condition under which my own mechanism could lose, so it is guarded rather than asserted. **No thread parallelism**: the metaphor's unit of work is one sphere-tumble over 8 bytes, far too small to pay for a thread, and at benchmark sizes this kernel is memory-bandwidth-bound long before it is core-bound. Vectorization only.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the trail is coiled, so every turn is circular ---------------- */
#define TURN(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the mason's stalk angles, set before the first mark is laid: compile-time
   constants, so the trail costs no memory traffic.  Chosen so the composed
   diffusion polynomial (1+z^13+z^37)(1+z^19+z^47) has 9 distinct exponents
   mod 64 - nothing cancels.                                            */
#define A1 13
#define B1 37
#define A2 19
#define B2 47

/* the sphere's eight faces, as the mason left them */
#define C0 0x736f6d6570736575ULL
#define C1 0x646f72616e646f6dULL
#define C2 0x6c7967656e657261ULL
#define C3 0x7465646279746573ULL
#define C4 0x9e3779b97f4a7c15ULL
#define C5 0xbf58476d1ce4e5b9ULL
#define C6 0x94d049bb133111ebULL
#define C7 0xd6e8feb86659fd93ULL

static inline uint64_t ld64(const unsigned char *q) {
    uint64_t v; memcpy(&v, q, 8); return v;
}

/* one strike against a stalk: the sphere cracks a hairline into itself along
   the stalk's angle.  XOR of an odd number of rotations is always a bijection
   on 64 bits (1+z^a+z^b is a unit in GF(2)[z]/(z+1)^64), so nothing is lost
   anywhere down the trail - no multiplier needed to stay injective.      */
static inline uint64_t stalk(uint64_t x, unsigned a, unsigned b) {
    return x ^ TURN(x, a) ^ TURN(x, b);
}

/* the sphere takes the mark's weight and carries it forward (this add is the
   only nonlinearity on the whole trail - carry is what a rotation can never
   fake), then tumbles a fixed count of two turns.  No more, no fewer.    */
static inline uint64_t tumble(uint64_t face, uint64_t mark) {
    uint64_t x = face + mark;
    x = stalk(x, A1, B1);
    x = stalk(x, A2, B2);
    return x;
}

/* the final stalk, struck harder because it is struck only once: SipRound,
   verbatim, the validated ARX round this metaphor arrives at.            */
#define SIPROUND(v0,v1,v2,v3) do {                                  \
    v0 += v1; v1 = TURN(v1,13); v1 ^= v0; v0 = TURN(v0,32);         \
    v2 += v3; v3 = TURN(v3,16); v3 ^= v2;                           \
    v0 += v3; v3 = TURN(v3,21); v3 ^= v0;                           \
    v2 += v1; v1 = TURN(v1,17); v1 ^= v2; v2 = TURN(v2,32);         \
} while (0)

#if defined(__AVX2__)
#define VTURN(x, r) _mm256_or_si256(_mm256_slli_epi64((x),(r)),           \
                                    _mm256_srli_epi64((x),64-(r)))
#define VTUMBLE(x, m) do {                                                 \
    __m256i t_ = _mm256_add_epi64((x), (m));                               \
    t_   = _mm256_xor_si256(_mm256_xor_si256(t_, VTURN(t_,A1)),            \
                            VTURN(t_,B1));                                 \
    (x)  = _mm256_xor_si256(_mm256_xor_si256(t_, VTURN(t_,A2)),            \
                            VTURN(t_,B2));                                 \
} while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    uint64_t f0 = C0, f1 = C1, f2 = C2, f3 = C3;
    size_t i = 0;

#if defined(__AVX2__)
    /* THE WIDE TRAIL.  Eight faces, 64 marks a roll.  Splitting the sphere
       and gathering it again is a real cost, so it is only set up when the
       pile is big enough to repay it; otherwise the narrow trail below runs
       as the fallback.  Runtime regime check, the native counting the pile
       at the trail's mouth.                                             */
    if (len >= 256) {
        __m256i s0 = _mm256_set_epi64x((long long)C3, (long long)C2,
                                       (long long)C1, (long long)C0);
        __m256i s1 = _mm256_set_epi64x((long long)C7, (long long)C6,
                                       (long long)C5, (long long)C4);
        for (; i + 64 <= len; i += 64) {
            __m256i m0 = _mm256_loadu_si256((const __m256i *)(p + i));
            __m256i m1 = _mm256_loadu_si256((const __m256i *)(p + i + 32));
            VTUMBLE(s0, m0);
            VTUMBLE(s1, m1);
        }
        uint64_t g[8];
        _mm256_storeu_si256((__m256i *)g,       s0);
        _mm256_storeu_si256((__m256i *)(g + 4), s1);
        f0 = g[0] ^ TURN(g[4], 13);     /* gather the eight faces back */
        f1 = g[1] ^ TURN(g[5], 29);     /* onto four, offset so no two */
        f2 = g[2] ^ TURN(g[6], 47);     /* faces can cancel each other */
        f3 = g[3] ^ TURN(g[7],  7);
    }
#endif

    /* THE NARROW TRAIL.  Four faces, 32 marks a roll.  The faces never
       consult each other here - the old face shrinks and goes - which is
       exactly what lets four tumbles overlap in flight.                 */
    for (; i + 32 <= len; i += 32) {
        f0 = tumble(f0, ld64(p + i));
        f1 = tumble(f1, ld64(p + i +  8));
        f2 = tumble(f2, ld64(p + i + 16));
        f3 = tumble(f3, ld64(p + i + 24));
    }

    /* the sphere is turned onto a new face only while marks remain: a pile
       of fewer than 32 uses as many faces as it has words, and no more.  */
    if (i + 8 <= len) { f0 = tumble(f0, ld64(p + i)); i += 8; }
    if (i + 8 <= len) { f1 = tumble(f1, ld64(p + i)); i += 8; }
    if (i + 8 <= len) { f2 = tumble(f2, ld64(p + i)); i += 8; }

    /* the eggshells the trail sheds: 0..7 trailing marks, and the size of
       the pile itself, pressed into the top of the same last face.       */
    {
        uint64_t tail = 0;
        size_t r = len - i, k;
        for (k = 0; k < r; k++) tail |= (uint64_t)p[i + k] << (8 * k);
        tail |= (uint64_t)(len & 0xff) << 56;
        f3 = tumble(f3, tail);
    }

    /* THE FINAL STALK.  The sphere itself is not kept; only the last,
       smallest crack it leaves here is handed over.                      */
    f0 ^= (uint64_t)len;
    f2 ^= 0xffULL;
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    return f0 ^ f1 ^ f2 ^ f3;
}
```

## PREDICTION

Stated before any measurement, and no measurement was possible in this session — see MEASUREMENT.

**PREDICTION: speedup_vs_dp = 12**

Reasoning behind the number. The FNV-1a reference is one dependent `imul` per byte: ~4 cycles/byte, ≈0.25 B/cycle. The wide trail is 2 stalks × 8 faces per 64 bytes = ~18 vector ops with a serial chain of ~8 cycles across two independent accumulators, i.e. ~6–8 B/cycle while the data is cache-resident; the narrow trail is ~40 uops per 32 bytes, ~3 B/cycle. That is 12–30× on a large buffer and roughly 1.5–3× on 16–32-byte inputs, where the 4-round final stalk (~25–30 cycles, paid once) dominates. 12 is my estimate for a mixed or bandwidth-limited benchmark; if the harness hashes a single L1/L2-resident multi-KB buffer I expect it to come in above that, and if it hashes short keys, below.

Avalanche prediction: **≈0.50 with small per-bit bias**, at every length. I am confident here for a specific reason, and it is not confidence in my own trail: the trail only has to accumulate injectively, and every stalk on it is provably a bijection. The avalanche guarantee is carried entirely by the final stalk, which is SipHash's own 4-round finalization over the full 256-bit state — a construction with a decade of public cryptanalysis behind it. If avalanche comes out poor, the fault will be in my gather step letting two faces cancel, not in the finalizer.

## MEASUREMENT

**Not measured. Reporting this plainly rather than dressing it up:** this session was invoked with no tools available — `hash_bench` and `hash_contract` could not be called, so the prediction above is unmeasured and the "improve it at most four times" loop never ran. Zero of the four permitted revisions were spent on evidence; the artifact is a first draft that has never been compiled. Everything in PREDICTION is arithmetic on instruction latencies and an appeal to SipHash's published analysis, not a number I observed.

What I *can* assert without the bench, because it is structural rather than empirical:

- Every stalk is invertible (the odd-number-of-terms unit argument above), so no entropy is dropped on the trail.
- The composed per-word diffusion spreads each bit to 9 distinct positions mod 64, verified by hand on the exponent sums; the naive angle choice would have cancelled two of them.
- The tail never collides with the length byte: after the whole-word absorbs, `r ∈ [0,7]`, so tail bytes occupy shifts 0–48 and the length sits at 56.
- All rotation amounts lie in 1..63, so no shift-by-64 UB; all 8-byte loads go through `memcpy`, so unaligned input is safe and nothing is read past `data+len`.
- The path taken depends only on `len`, so the function is deterministic — a prerequisite the harness will check implicitly.

The pipeline should treat throughput and avalanche as open questions.

## VERDICT

The native's claim is substantive and it is correct: **a multiplier is not required to mix a byte.** Turn, carry, press — rotate, add, xor — suffice, and the invertibility that FNV buys with an odd prime is available for free from any XOR of an odd number of rotations. That is the real content of SEED 2, and it is why this is not a decorative reskin of the known way: the multiply is gone entirely, including from the finalizer.

Where I expect this to win: large piles. Independent faces convert a latency-bound serial chain into a throughput-bound one, which is the whole arithmetic of the 12×.

Where it could lose, and what I did about it rather than just noting it:

- *Splitting the sphere onto eight faces costs setup and a gather.* Guarded at `len >= 256`, with the 4-face narrow trail as the fallback. Stated risk, addressed in code.
- *The final stalk is a fixed ~25–30 cycle toll.* Below roughly 16 bytes this makes the kernel no faster than FNV-1a, possibly slower. I deliberately did **not** guard this with an FNV fallback, and this is the one place I knowingly traded speed for quality: FNV-1a on a 1-byte input flips only ~7 of 64 output bits, so a small-input fallback to the known way would have bought a few cycles by wrecking exactly the metric the task asks for. Paying the toll at every length is the choice; if the harness hashes short keys, this is where the predicted 12 will fail, and it will fail for a reason I accepted in advance rather than one I overlooked.
- *Thread parallelism.* Dropped, not guarded — the metaphor's unit of work is 8 bytes, and a `#pragma omp parallel` over a trail that fits in registers would be pure overhead at any size this benchmark plausibly uses. Per step 4, the risky part is dropped rather than shipped with a hopeful comment.

The honest residual worry is not throughput, it is the 8→4 face gather in the AVX2 path. It is mine, not SipHash's, and it is the only step in the kernel I cannot prove is free of cancellation. If the measured avalanche for `len >= 256` comes out materially worse than for `len < 256`, that gather is the suspect, and the first revision should be to feed the eight faces through two SipRounds before folding instead of trusting rotated XOR to keep them apart.