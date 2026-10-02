No tools were available in this session (the MCP servers listed require authorization, and `hash_bench` / `hash_contract` were not reachable), so below I state the prediction up front and report plainly that the measurement step did not execute. Everything else — the mapping, the choice, the kernel — is complete and compilable as given.

---

# MAPPING

### SEED 1 — "The stone never resets between marks; every fold carries the callus of all folds before it."

| World object | Problem object |
|---|---|
| die-stone | the hash state, live across the whole buffer |
| "never resets" | state is not re-initialized per byte/block |
| callus of all prior presses | current state value is a function of all prior bytes |
| fold | one state-update step |

**Breaks:** nothing. This *is* FNV-1a's defining property (one carried accumulator, no reset). Seed 1 restates the known way.

### SEED 2 — "Each mark's weight is pressed into the already-turned position, not onto a clean face; early marks bend how later marks land."

| World object | Problem object |
|---|---|
| mark's weight | byte (or word) value, read as a number not a character |
| already-turned position | current state, used as an operand not just a target |
| "bends how deep this one goes" | update is *non-linear in the state*: `f(state, word)`, e.g. `(state+w·P)*Q` |
| chalk-dust hair's width → multiplied twice → never the same last step | per-step multiplicative divergence = avalanche/collision argument |

**Breaks:** nothing structural — it is the argument *for* `h = (h ^ b) * P`. It confirms the per-step multiply, it does not question it.

### SEED 3 — "Only the stone's final seated number against the wooden keeps ever leaves the desk; every intermediate turn and residue is swept away and discarded."

| World object | Problem object |
|---|---|
| the desk, its numbered groove, starlight (constant light, any hour) | the scratch register file / deterministic, input-order-indexed layout; no hour-dependence = no seed/randomness |
| a *die*-stone turned **a quarter** per fold | the state is a *multi-faced* object: a quarter turn exposes one of **four** faces, so mark *i* lands on face *i mod 4* → **4 (and, widened, 8) parallel accumulator lanes inside one state** |
| press into the *same seated place* on a turned stone | lane-local update: `acc[i] = f(acc[i], word_i)` |
| groove-dust, intermediate turns, chalk residue — swept off | intermediate states are **never observed**, so they need not be well-mixed |
| the wooden **numbered keeps** at the desk's edge (fixed, not part of the stone) | the **finalizer**: fixed odd constants that fold the lanes together and avalanche once, at the end |
| "that reading, and only that reading, is the token" | the 64-bit return value is produced *by the keeps*, not by the state |
| "centuries of marks fold down just the same as a handful" | the kernel must be correct and fast at both extremes → two regimes |

**Breaks:** "each byte must be mixed into the running state before the next byte is read" and "more mixing rounds always means better mixing" — and, derivatively, "the state is a single accumulator updated in place, one value".

---

# CHOSEN SEED

**Seed 3.**

Honest statement on the preference you asked for: **none of the three seed sentences literally names a multi-valued state.** Seed 1 and Seed 2 are both pure restatements of FNV-1a. Seed 3 is the only one that *breaks* that assumption, and it does so indirectly but rigorously: it severs the state from the output. The single-accumulator assumption exists only because the textbook silently identifies "the state" with "the hash". Once only the *final reading against the keeps* leaves the desk, the thing on the desk is unconstrained in width and shape — and the native's own object is not a counter, it is a **die**, turned **a quarter** per fold, i.e. an object with four faces presenting in rotation. So Seed 3 is both the most literal (every noun maps to a register or a constant) and the most different from the known way.

---

# ASSUMPTION BROKEN

> *"each byte must be mixed into the running state before the next byte is read"* (and with it: *"the state is a single accumulator updated in place, one value"*, and *"more mixing rounds always means better mixing"*).

The sweep-it-off clause is the proof: if the intermediate state is discarded, per-byte mixing quality is **unobservable and therefore worthless**. All that is required is (a) each byte reaches *some* lane injectively-enough, cheaply, and (b) the keeps diffuse once, hard, at the end. That converts a latency-bound serial multiply chain (FNV: ~4 cycles/byte, loop-carried) into a throughput-bound set of independent lanes, plus one fixed-cost finalizer.

Per your step 4, I deliberately let the mechanism **land on validated real-world technique rather than invent**: four faces × 8-byte presses with `rotl(acc + w·P2, 31)·P1` and a merge+avalanche tail is exactly **xxHash64**; the widened eight-face desk (`acc[i] += lo32(k)·hi32(k)`, sibling-lane add, periodic scramble, `mul128-fold` merge) is exactly **XXH3's accumulate/scramble/mergeAccs**. Both are SMHasher-validated. The only deviation I add is the native's own **"desk's numbered groove"**: a per-stripe tweak, because a pure sum over stripes is commutative and two swapped stripes would seat identically — the native forbids that ("two different piles essentially never seat the same way twice").

**Regime recognition (step 5).** The known_way section describes two regimes ("the whole buffer read once start to end" vs. tiny inputs; `centuries of marks` vs. `a handful`). The metaphor supplies the test itself: *a handful of marks never completes a full turn of the stone through the groove.* So at runtime, `len < 256` → the narrow desk (scalar four-face xxHash64, which is exact and fast for 0…255 bytes, including `len == 0` with no loads); `len >= 256` → the wide desk (eight faces, AVX2). This is the guard for the risk my own verdict names (vector setup + merge overhead only pays on long piles), with a fallback to the simpler path.

**No thread parallelism.** The native has **one stone on one desk**. A second desk is not in the world, and at these sizes OpenMP fork/join would dominate. Vectorization hints only (`restrict`, 64-byte stripes, contiguous forward reads, overlapping last stripe instead of a branchy tail).

---

# ARTIFACT

```c
/* The Desk: one die-stone, four faces narrow / eight faces wide, read once
   against the wooden numbered keeps.  Contract: uint64_t kernel(const unsigned
   char *data, size_t len); */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* the wooden numbered keeps: fixed odd constants, never on the stone */
#define KP1 11400714785074694791ULL
#define KP2 14029467366897019727ULL
#define KP3  1609587929392839161ULL
#define KP4  9650029242287828579ULL
#define KP5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* one fold: the mark's weight dropped into the face's already-turned position */
static inline uint64_t press(uint64_t face, uint64_t w) {
    face += w * KP2;
    face  = rotl64(face, 31);
    return face * KP1;
}
/* a wooden keep: pull one face's final seat into the reading */
static inline uint64_t keep(uint64_t h, uint64_t face) {
    h ^= press(0, face);
    return h * KP1 + KP4;
}
/* the reading itself: the only thing that leaves the desk */
static inline uint64_t read_off(uint64_t h) {
    h ^= h >> 33; h *= KP2;
    h ^= h >> 29; h *= KP3;
    h ^= h >> 32;
    return h;
}

/* ---------- narrow desk: a handful of marks, the stone never completes a turn
   ---------- (exactly xxHash64, seed 0: validated, exact for every len >= 0) */
static uint64_t desk_narrow(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    if (len >= 32) {
        const unsigned char *const limit = end - 32;
        uint64_t f0 = KP1 + KP2, f1 = KP2, f2 = 0, f3 = 0 - KP1; /* four faces */
        do {                                 /* a quarter turn per mark-weight */
            f0 = press(f0, rd64(p)); p += 8;
            f1 = press(f1, rd64(p)); p += 8;
            f2 = press(f2, rd64(p)); p += 8;
            f3 = press(f3, rd64(p)); p += 8;
        } while (p <= limit);
        h = rotl64(f0, 1) + rotl64(f1, 7) + rotl64(f2, 12) + rotl64(f3, 18);
        h = keep(h, f0); h = keep(h, f1); h = keep(h, f2); h = keep(h, f3);
    } else {
        h = KP5;
    }
    h += (uint64_t)len;

    while (p + 8 <= end) { h ^= press(0, rd64(p)); h = rotl64(h, 27) * KP1 + KP4; p += 8; }
    if    (p + 4 <= end) { h ^= (uint64_t)rd32(p) * KP1; h = rotl64(h, 23) * KP2 + KP3; p += 4; }
    while (p < end)      { h ^= (uint64_t)(*p) * KP5; h = rotl64(h, 11) * KP1; p++; }

    return read_off(h);
}

#if defined(__AVX2__)
/* ---------- wide desk: centuries of marks, eight faces seated at once -------
   ---------- (exactly XXH3's accumulate / scramble / mul128-fold merge) ----- */
static inline uint64_t fold128(uint64_t a, uint64_t b) {
    unsigned __int128 pr = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)pr ^ (uint64_t)(pr >> 64);
}
static inline __m256i scrape(__m256i a, __m256i key) {   /* XXH3_scrambleAcc */
    const __m256i pr = _mm256_set1_epi64x(0x9E3779B1LL);
    a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 47));
    a = _mm256_xor_si256(a, key);
    return _mm256_add_epi64(_mm256_mul_epu32(a, pr),
                            _mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), pr), 32));
}

#define FOLD_STRIPE(S) do {                                                       \
    __m256i d0 = _mm256_loadu_si256((const __m256i *)(const void *)(S));          \
    __m256i d1 = _mm256_loadu_si256((const __m256i *)(const void *)((S) + 32));   \
    tw = _mm256_add_epi64(tw, step);          /* the groove is numbered */        \
    __m256i k0 = _mm256_xor_si256(_mm256_xor_si256(d0, sec0), tw);                \
    __m256i k1 = _mm256_xor_si256(_mm256_xor_si256(d1, sec1), tw);                \
    a0 = _mm256_add_epi64(a0, _mm256_mul_epu32(k0, _mm256_srli_epi64(k0, 32)));   \
    a1 = _mm256_add_epi64(a1, _mm256_mul_epu32(k1, _mm256_srli_epi64(k1, 32)));   \
    a0 = _mm256_add_epi64(a0, _mm256_shuffle_epi32(d0, 0x4E)); /* sibling face */ \
    a1 = _mm256_add_epi64(a1, _mm256_shuffle_epi32(d1, 0x4E));                    \
} while (0)

static uint64_t desk_wide(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const __m256i sec0 = _mm256_setr_epi64x((long long)0xb8fe6c3923a44bbeULL,
                                            (long long)0x7c01812cf721ad1cULL,
                                            (long long)0xded46de9839097dbULL,
                                            (long long)0x7240a4a4b7b3671fULL);
    const __m256i sec1 = _mm256_setr_epi64x((long long)0xcb79e64eccc0e578ULL,
                                            (long long)0x825ad07dccff7221ULL,
                                            (long long)0xb4323668eb9dcbb6ULL,
                                            (long long)0x9e3779b97f4a7c15ULL);
    const __m256i scr0 = _mm256_setr_epi64x((long long)0xd6d018f5a7f1b9e3ULL,
                                            (long long)0x8a51e04bcdc4e9adULL,
                                            (long long)0xc2b2ae3d27d4eb4fULL,
                                            (long long)0x165667b19e3779f9ULL);
    const __m256i scr1 = _mm256_setr_epi64x((long long)0x27d4eb2f165667c5ULL,
                                            (long long)0x85ebca77c2b2ae63ULL,
                                            (long long)0x9e3779b185ebca87ULL,
                                            (long long)0xff51afd7ed558ccdULL);
    const __m256i step = _mm256_set1_epi64x((long long)0x9e3779b97f4a7c15ULL);
    __m256i tw = _mm256_setzero_si256();
    /* eight faces of one stone (XXH3 init accumulators) */
    __m256i a0 = _mm256_setr_epi64x((long long)0xC2B2AE3DULL, (long long)KP1,
                                    (long long)KP2,           (long long)KP3);
    __m256i a1 = _mm256_setr_epi64x((long long)KP4,           (long long)0x85EBCA77ULL,
                                    (long long)KP5,           (long long)0x9E3779B1ULL);

    const size_t nstripe = len >> 6;                 /* 64-byte stripes */
    for (size_t i = 0; i < nstripe; i++) {
        FOLD_STRIPE(p + (i << 6));
        if ((i & 15) == 15) { a0 = scrape(a0, scr0); a1 = scrape(a1, scr1); }
    }
    if (len & 63) FOLD_STRIPE(p + len - 64);         /* overlapping last stripe */

    uint64_t f[8];
    _mm256_storeu_si256((__m256i *)(void *)f,       a0);
    _mm256_storeu_si256((__m256i *)(void *)(f + 4), a1);

    uint64_t h = (uint64_t)len * KP1;                /* the wooden keeps */
    h += fold128(f[0] ^ 0x9e3779b97f4a7c15ULL, f[1] ^ 0xc2b2ae3d27d4eb4fULL);
    h += fold128(f[2] ^ 0x165667b19e3779f9ULL, f[3] ^ 0x85ebca77c2b2ae63ULL);
    h += fold128(f[4] ^ 0x27d4eb2f165667c5ULL, f[5] ^ 0xff51afd7ed558ccdULL);
    h += fold128(f[6] ^ 0xd6d018f5a7f1b9e3ULL, f[7] ^ 0x8a51e04bcdc4e9adULL);
    return read_off(h);
}
#endif /* __AVX2__ */

uint64_t kernel(const unsigned char *data, size_t len) {
#if defined(__AVX2__)
    if (len >= 256) return desk_wide(data, len);   /* centuries of marks */
#endif
    return desk_narrow(data, len);                 /* a handful of marks */
}
```

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 18.0**

(Stated before any measurement. Reasoning: FNV-1a's loop-carried `xor`+`imul` chain is ~4 cycles/byte ≈ 0.25 B/cycle. The narrow desk removes the dependency chain into 4 lanes at 8 bytes/press → ~4–6 B/cycle, i.e. ~16–24×. The wide desk issues ~17 uops per 64 bytes → ~12–14 B/cycle in L1/L2, i.e. 50×+, falling back toward DRAM bandwidth (~3–5 B/cycle effective) on multi-MB buffers. A size-mixed harness should land in the teens; a short-input-only harness nearer 5×. Avalanche: I predict ~0.50 bit-flip fraction, score ≥0.99 of ideal, since the final reading is a full xxHash64 avalanche over a `mul128`-folded merge and every single-bit input change perturbs at least the sibling-lane additive term.)

---

# MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were not callable in this session (no tools exposed; the claude.ai PubMed / connector servers also require authorization, which cannot be completed non-interactively). I will not report numbers I did not take.

What I can offer instead, flagged as *analysis, not measurement*:

| Quantity | Status | Basis |
|---|---|---|
| throughput, len < 256 | unmeasured; est. 4–6 B/cycle | xxHash64 is this exact code path; published ~5–8 GB/s |
| throughput, len ≥ 256 | unmeasured; est. 12–14 B/cycle cached | uop count 17/64 B; XXH3 published ~30+ GB/s |
| avalanche | unmeasured; est. 0.500 | finalizer is SMHasher-validated xxHash64 avalanche over a 128-bit-fold merge |
| `len == 0`, `len == 1` safety | argued, not measured | narrow path performs zero loads at `len == 0`; wide path never entered below 256, so `p + len - 64` is always in-bounds |

To be authorized by you if you want the real numbers: this needs `hash_bench` on the block above, unchanged.

---

# VERDICT

The native's mechanism is **real and it is not FNV**. The decisive sentence is the throwaway one — *only the last seated number ever leaves the desk* — because it is a statement about **observability**, and observability is what licenses cheap intermediate work. Two of the three seeds (1 and 2) are faithful descriptions of the known way and I say so rather than torturing them into novelty; only Seed 3 breaks an assumption, and it breaks the right one.

Taken literally, Seed 3 plus the die-stone's **quarter turn** forces a multi-faced state, and the **wooden keeps mounted at the desk's edge** — outside the stone — force a separate finalizer. That pair is not a new invention: it is the xxHash64 / XXH3 architecture, which is exactly what I shipped, deliberately, per your step 4. The metaphor's one genuine contribution beyond the textbook is the **numbered groove** — position entering each press — which fixes a real defect that a naive "sum the lanes" reading would have had (stripe-order commutativity ⇒ trivial collisions on permuted blocks), and which the native's own clause about two piles never seating alike demanded.

Where this could be **worse** than the known way, and what I did about it:
- *Vector setup and an 8-lane merge cost ~40–60 cycles regardless of length* — so below ~256 bytes the wide desk loses to the narrow one, and both lose relative advantage against FNV as `len → 1`. **Addressed, not hand-waved:** a hard `len >= 256` guard with the narrow desk as the fallback, and the narrow desk itself is exact and loadless at `len == 0`.
- *Non-AVX2 hardware* — **addressed:** the entire wide path is `#if defined(__AVX2__)`; without it the kernel is pure, portable, still-fast scalar.
- *Thread parallelism* — **dropped on principle, not forgotten.** The world contains one stone and one desk; OpenMP would be a second desk the native never built, and at any size a hash benchmark plausibly tests, fork/join would cost more than it buys.

The honest residual risk is that I am predicting a number I could not verify. If `hash_bench` comes back below ~8× on a size-mixed harness, the likely culprit is small-input dominance, not the mechanism — and the fix would be a *third*, narrower desk (a single press-and-read for `len <= 16`), not a change of idea.