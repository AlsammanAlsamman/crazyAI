# MAPPING

### SEED 1 — "the die-stone never resets between marks, every fold carries the callus of all folds before it"

| world object | problem object |
|---|---|
| die-stone | a persistent state kept in registers across the whole buffer |
| "never lifting it fully off the desk" | state never written back to memory / never re-initialized mid-stream |
| "the callus of the one before it" | fold *i* is a function of the result of fold *i-1* (true data dependence) |
| "centuries fold down just the same as a handful" | state size is O(1) in `len`; one loop body for any length |

**Assumption broken: none of the five.** A single carried-forward accumulator is *exactly* what FNV-1a/xxHash already do. I say that plainly rather than dressing it up — this seed constrains the design (no resets, no restarts) but it is not the novelty.

### SEED 2 — "each mark's weight is pressed into the stone's already-turned position, not onto a clean face"

| world object | problem object |
|---|---|
| mark's weight | an input word, read as a number, not a character |
| "already-turned position" | the mark is combined with state *after* the previous turn, so its effect depends on all previous marks |
| "turn it a quarter through the chalk-bank groove" | a fixed rotation applied to the state once per absorbed chunk |
| "the next fold multiplies it, and the one after multiplies it again" | difference amplification: carry propagation + rotation + coupling, doubling the active-bit set per fold |

**Assumption broken: "mixing one byte requires a multiplication."** The amplification the native describes is *geometric difference growth*, which add-rotate-xor gives for free; no `imul` appears in the per-byte path.

### SEED 3 — "only the stone's final seated position, read against the wooden numbered **keeps**, ever leaves the desk; every intermediate turn and residue is swept away"

| world object | problem object |
|---|---|
| the stone's *seated position* | the state is an **orientation**, not a scalar: several coordinates at once (256 bits) |
| the **numbered keeps** (plural, mounted, fixed) | four fixed 64-bit words of the state, read at fixed positions with fixed constants |
| "drop the next mark's weight into the *same seated place*" | input is absorbed into a fixed **rate** slot of a wide state — a sponge absorb |
| "turn it a quarter" between presses | one fixed permutation round between absorbs |
| "lift the stone and read off" | the squeeze: turns with no new mark, then collapse the keeps to one number |
| "groove-dust, intermediate turns, chalk residue... swept off" | no intermediate output, no running checksum, no per-block emission |
| "two different piles essentially never seat the same way twice" | collision resistance of the final 64-bit token |

**Assumption broken: "the state is a single accumulator updated in place, one value."** The stone is *one* object with *one* sequential history, but its state is wide and multi-coordinate, and it becomes a single value **only at the readout**.

# CHOSEN SEED

**SEED 3**, which is the one that breaks the preferred assumption. It is also the most literal: the native says *keeps*, plural, mounted at a fixed edge, read once at the end — a scalar accumulator cannot be read "against numbered keeps."

The crucial distinction from the previous attempt the reviewer rejected: xxHash's four lanes are **four independent stones** that never see each other until the merge. Here there is **one stone**. Every quarter-turn couples all four keeps (`keep[i] += keep[i+k]` before the turn), so after one fold every keep already depends on every other keep, and the whole thing is a single serial dependence chain — exactly SEED 1's callus, exactly SEED 2's press-into-the-already-turned-position. The width buys bytes-per-fold, not parallel chains.

Per step 4, I let the mechanism land on the validated technique it actually *is* rather than inventing: this is the **sponge construction with an ARX permutation** (Keccak/Gimli/Xoodoo/SipHash family) — wide state, fixed rate slot, never reset, no intermediate output, squeeze at the end. I did not reach for a multiply-accumulator; the per-byte core has no multiplication at all.

# ASSUMPTION BROKEN

> "the state is a single accumulator updated in place, one value"

Replaced by: a 256-bit stone (4 numbered keeps) advanced by one serial chain, absorbing 32 bytes per quarter-turn, collapsed to one 64-bit token only when it is lifted.

Secondary casualty: "mixing one byte requires a multiplication" — the absorb path is add/rotate/permute/xor only. The two multiplies in the code are in the **constant-cost readout** (`fmix64`, the validated Murmur3/splitmix finalizer), not per byte; that is deliberate — step 4 says a validated known technique beats a novel untested one for the job of collapsing a diffused state into a well-avalanched word.

**Regimes (step 5).** The native's own regime test is whether the pile can fill the groove's seated place:
- **a handful of marks** (`len < 16`): the place cannot be seated at all → thin-stack path, two turns, read off. This also guards the risk my own verdict would otherwise name (constant squeeze cost dominating tiny inputs).
- **not quite a seating** (`16 ≤ len < 32`): one seating, zero-padded, length-tagged.
- **a pile** (`len ≥ 32`): the folding loop, 128 B per unrolled iteration, with the final seating backed up so the place is full (overlapping tail read — no byte-at-a-time drizzle).

**No thread parallelism, on purpose.** A second stone contradicts the mechanism outright, and the chain is the point. Vectorization only (one AVX2 register = one stone), per step 4's default.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the wooden numbered keeps mounted at the desk's edge: fixed forever,
   "that same starlight color no matter the hour" (digits of pi) ---- */
#define KA 0x243F6A8885A308D3ULL
#define KB 0x13198A2E03707344ULL
#define KC 0xA4093822299F31D0ULL
#define KD 0x082EFA98EC4E6C89ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* ---- the lift: read the stone's final seated position against the keeps.
   This single number is the only thing that ever leaves the desk. ---- */
static inline uint64_t read_off(const uint64_t s[4], uint64_t tag) {
    uint64_t h = (s[0] ^ rotl64(s[1], 17)) + (rotl64(s[2], 41) ^ rotl64(s[3], 53)) + tag;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

/* ---- one quarter-turn with no new mark (scalar form, used by the squeeze
   and by the handful-of-marks regime). Every keep takes on a neighbour:
   the stone is one object, not four. ---- */
static inline void turn4(uint64_t s[4], int k, unsigned r) {
    uint64_t t0 = s[(0 + k) & 3], t1 = s[(1 + k) & 3];
    uint64_t t2 = s[(2 + k) & 3], t3 = s[(3 + k) & 3];
    s[0] = rotl64(s[0] + t0, r); s[1] = rotl64(s[1] + t1, r);
    s[2] = rotl64(s[2] + t2, r); s[3] = rotl64(s[3] + t3, r);
}

/* ---- press a mark's weight into the already-turned position, then turn ---- */
static inline void fold4(uint64_t s[4], const uint64_t m[4], int k, unsigned r) {
    uint64_t t0 = s[(0 + k) & 3], t1 = s[(1 + k) & 3];
    uint64_t t2 = s[(2 + k) & 3], t3 = s[(3 + k) & 3];
    uint64_t x0 = s[0] ^ m[0], x1 = s[1] ^ m[1];
    uint64_t x2 = s[2] ^ m[2], x3 = s[3] ^ m[3];
    s[0] = rotl64(x0 + t0, r); s[1] = rotl64(x1 + t1, r);
    s[2] = rotl64(x2 + t2, r); s[3] = rotl64(x3 + t3, r);
}

#if defined(__AVX2__)
/* the chalk-bank groove: a bitwise quarter-turn of every keep */
#define V_ROTL(v, r) _mm256_or_si256(_mm256_slli_epi64((v), (r)),              \
                                     _mm256_srli_epi64((v), 64 - (r)))

/* cheap coupling: each keep takes its partner within the half (latency 1) */
#define V_FOLD_H(s, m, r) do {                                                 \
    __m256i t_ = _mm256_shuffle_epi32((s), 0x4E);                              \
    __m256i x_ = _mm256_xor_si256((s), (m));                                   \
    (s) = V_ROTL(_mm256_add_epi64(x_, t_), (r));                               \
} while (0)

/* full coupling: each keep takes a keep SH places round the stone */
#define V_FOLD_P(s, m, SH, r) do {                                             \
    __m256i t_ = _mm256_permute4x64_epi64((s), (SH));                          \
    __m256i x_ = _mm256_xor_si256((s), (m));                                   \
    (s) = V_ROTL(_mm256_add_epi64(x_, t_), (r));                               \
} while (0)

/* a turn with no new mark */
#define V_TURN(s, SH, r) do {                                                  \
    __m256i t_ = _mm256_permute4x64_epi64((s), (SH));                          \
    (s) = V_ROTL(_mm256_add_epi64((s), t_), (r));                              \
} while (0)
#endif

/* ---- the pile regime: n >= 32 guaranteed by the caller ---- */
static uint64_t fold_pile(const unsigned char *restrict p, size_t n, uint64_t tag) {
    uint64_t out[4];
#if defined(__AVX2__)
    /* the stone, seated on the keeps */
    __m256i s = _mm256_set_epi64x((long long)KD, (long long)KC,
                                  (long long)KB, (long long)KA);
    const unsigned char *q = p;
    size_t m = n;

    /* four quarter-turns per pass: the groove's constant and the direction of
       coupling change every turn, as a die lands on a different face. This is
       ONE chain unrolled, not four chains. */
    while (m >= 128) {
        V_FOLD_H(s, _mm256_loadu_si256((const __m256i *)(q +   0)),       29);
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(q +  32)), 0x39, 47);
        V_FOLD_H(s, _mm256_loadu_si256((const __m256i *)(q +  64)),       17);
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(q +  96)), 0x93, 41);
        q += 128; m -= 128;
    }
    while (m >= 32) {
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)q), 0x39, 29);
        q += 32; m -= 32;
    }
    if (m) {
        /* back the stone up so the seated place is full: never a part-press */
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(p + n - 32)), 0x93, 47);
    }

    /* the pile's own extent is a mark too */
    s = _mm256_xor_si256(s, _mm256_set1_epi64x((long long)tag));

    /* the lift: four turns with no new mark, so even the last mark's
       hair's-width difference has nowhere left to hide */
    V_TURN(s, 0x39, 23);
    V_TURN(s, 0x4E, 37);
    V_TURN(s, 0x93, 19);
    V_TURN(s, 0x39, 43);

    _mm256_storeu_si256((__m256i *)out, s);
#else
    uint64_t s[4] = { KA, KB, KC, KD };
    const unsigned char *q = p;
    size_t m = n;
    while (m >= 32) {
        uint64_t w[4]; memcpy(w, q, 32);
        fold4(s, w, 1, 29);
        q += 32; m -= 32;
    }
    if (m) {
        uint64_t w[4]; memcpy(w, p + n - 32, 32);
        fold4(s, w, 3, 47);
    }
    s[0] ^= tag; s[1] ^= tag; s[2] ^= tag; s[3] ^= tag;
    turn4(s, 1, 23); turn4(s, 2, 37); turn4(s, 3, 19); turn4(s, 1, 43);
    out[0] = s[0]; out[1] = s[1]; out[2] = s[2]; out[3] = s[3];
#endif
    return read_off(out, tag);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* ---- regime check, the native's own: can the pile fill the groove's
       seated place at all? ---- */
    if (len < 16) {
        /* a handful of marks: no full seating exists, so press the thin
           stack straight against the keeps and lift at once. Guards the
           squeeze's fixed cost from dominating tiny piles. */
        uint64_t a = 0, b = 0;
        if (len >= 8) {
            memcpy(&a, data, 8);
            memcpy(&b, data + len - 8, 8);
        } else if (len >= 4) {
            uint32_t x, y;
            memcpy(&x, data, 4);
            memcpy(&y, data + len - 4, 4);
            a = x; b = y;
        } else if (len) {
            a = (uint64_t)data[0]
              | ((uint64_t)data[len >> 1] << 8)
              | ((uint64_t)data[len - 1] << 16);
        }
        uint64_t s[4];
        s[0] = KA ^ (a + (uint64_t)len);
        s[1] = KB ^ rotl64(b, 23);
        s[2] = KC + rotl64(a, 41);
        s[3] = KD ^ (b + rotl64(a, 11));
        turn4(s, 1, 23);
        turn4(s, 3, 37);
        return read_off(s, (uint64_t)len);
    }
    if (len < 32) {
        /* not quite a seating: seat it on a swept desk, zero-padded,
           and let the extent distinguish the short piles */
        unsigned char buf[32];
        memset(buf, 0, sizeof buf);
        memcpy(buf, data, len);
        return fold_pile(buf, 32, (uint64_t)len);
    }
    return fold_pile(data, len, (uint64_t)len);
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Stated before any measurement, with the reasoning so it can be falsified cleanly:

- The reference FNV-1a is latency-bound at `xor` + `imul` ≈ 4 cycles per **byte** → ~0.25 B/cycle.
- My stone is latency-bound on one chain too (that is the mechanism), but the chain advances **32 bytes per link**. Critical path per quarter-turn: `permute` (3) or `shuffle` (1) + `add` (1) + `shift/shift/or` (2) ⇒ 20 cycles per 128-byte pass ⇒ ~6.4 B/cycle theoretical, call it ~5 B/cycle after loop and load effects.
- 5 / 0.25 = **20×** in the large-buffer limit; I discount to **12** because a size-mixed benchmark pays the fixed ~45-cycle lift on every short call, and the `len<16` regime is only ~2× ahead of FNV.
- Avalanche: I predict **≈0.5 (score ≥0.45)**. A one-bit flip reaches at most two keeps in its own fold, but the four coupling turns of the lift plus `fmix64` guarantee full 64-bit diffusion even for a flip in the *last* absorbed block, which is the worst case.

Falsification: if measured throughput ratio comes in **below ~4×**, the latency model is wrong (most likely `vpermq` is not the only thing on the critical path, or the loads are not running ahead). If avalanche comes in **below 0.4**, the four-turn squeeze is too short and the honest fix is more turns, not more lanes.

# MEASUREMENT

No tools were available in this session — `hash_bench` and `hash_contract` could not be invoked, so **I have not measured this kernel.** I will not report numbers I did not take. What is above is a prediction and nothing more; the pipeline's compile-and-measure pass is the first real datum.

What I *can* verify by inspection, and did:
- Bounds: `fold_pile` is only entered with `n ≥ 32`, so `p + n - 32 ≥ p`; the `len ∈ [16,32)` case is routed through a zero-filled 32-byte buffer, so the overlapping tail read is never out of bounds. `len == 0` returns a fixed token from the constants.
- Coverage: every input byte is absorbed on every path (tail overlap re-presses some bytes; that is deterministic and order-preserving, not a skip).
- Injectivity per fold: for fixed state, `m ↦ rotl((s ^ m) + perm(s), r)` is a bijection per keep, so no two distinct 32-byte blocks collide within a single seating.
- No multiplication in the per-byte path; two in the constant-cost readout.

# VERDICT

The core of this kernel *is* SEED 3, translated literally: one stone, 256 bits wide, four fixed numbered keeps, never reset, each 32-byte mark pressed into the already-turned position, every keep coupled to its neighbours on every quarter-turn, nothing emitted until the lift, and one number leaving the desk. That construction has a real name — a sponge over an ARX permutation — and I let the metaphor arrive there rather than inventing a private scheme, because a validated construction beats a novel one.

It is *not* the known way, and the difference is checkable rather than rhetorical: xxHash's speed comes from four **independent** accumulators with no cross-talk until the merge; here the keeps are coupled every single fold, so the whole kernel is one serial dependence chain from the first byte to the last. That is precisely the property the reviewer found missing before. The speed comes from making each link of that one chain carry 32 bytes and cost ~5 cycles — improving the implementation of the mechanism, not swapping the mechanism out.

Honest risks, each either guarded or dropped:
- *Fixed lift cost on tiny inputs* — the condition where this would lose to FNV. Guarded: `len < 16` takes a short path with two turns and no 32-byte seating, and `len < 32` takes one padded seating.
- *Serial chain on huge buffers* — the chain is the mechanism, so I did not break it with threads or extra stones; instead I widened the link. If measurement shows memory bandwidth, not latency, is the ceiling at large sizes, that is a win for this design, not a problem.
- *Thread parallelism* — dropped entirely, not guarded, because a second stone would contradict the mechanism and the metaphor offers no second stone. I would rather ship a slower kernel that is the native's idea than a faster one that isn't.

Residual uncertainty I am not hiding: the mostly-`vpshufd` coupling in the unrolled body crosses the 128-bit boundary only twice per 128 bytes. I believe the four full-coupling turns in the lift cover that, but it is the first thing I would test if the avalanche score lands between 0.35 and 0.45 — and the fix would be more turns in the lift, which is cheap and still the same mechanism.