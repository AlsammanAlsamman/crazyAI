# MAPPING

## SEED 1 — "one crease per mark, angle set by the mark and the crease before it"

| World object | Computational object |
|---|---|
| pile of marks, read in order, never skipping, never looking ahead | `data[0..len)` consumed in strictly increasing index order |
| one mark | one input byte |
| one fold per mark | one state update per byte |
| the fold's **angle** | the rotation amount of the update |
| "angle decided by the mark **and the crease before it**" | `crease_i = rotl(crease_{i-1} + byte_i + K, crease_{i-1} ^ byte_i)` — a second state variable (the crease register) distinct from the accumulator |
| "the paper carries forward everything it has already been told" | accumulator `h = (h ^ crease_i) * P` |

**Assumption broken:** "the state is a single accumulator updated in place, one value" — there are two coupled registers (accumulator + crease angle). Also partially "mixing one byte requires a multiplication" (the angle step is rotate/add only). It **affirms** rather than breaks the serial assumption.

## SEED 2 — "test every crease against all four wire birds' beaks at once, refold tighter until they agree"

| World object | Computational object |
|---|---|
| the huge sheet, pink or orange, "only that it's large enough" | the wide working state: one 256-bit gauge register + one 256-bit crease register. Colour irrelevant = the identity of the constants is irrelevant, only the *width* matters |
| the **four wire birds**, each with a different beak | the **four 64-bit lanes of one AVX2 register**; "a different beak" = a distinct key constant per lane |
| "press each fold's **corner** against a different bird's beak" | the 32-byte fold has four 8-byte corners; corner *j* is gauged by bird *j* (lane *j*) |
| one press = one gauge reading | one NH/UMAC-style press: `(lo+k_pos) * (hi+k_beak)` as a 32×32→64 multiply, added into that bird's gauge |
| "**at once**" | SIMD: all four presses are one `vpmuludq` |
| "only when all four beaks **agree** does the crease count as set" | `d = G ^ rot1lane(G)`; lane *j* holds bird *j* vs. bird *j+1*; all four agree ⟺ `d == 0` everywhere |
| "if they disagree I **refold tighter**" | `C = rotl(C ^ d, d)` — the crease is retightened by *exactly the amount they disagree*; perfect agreement (`d=0`) is the identity, i.e. no refold. Branchless, data-dependent |
| spiral and snail-shell markings made to line up **wrong on purpose, scrambled**, "because a clean line-up would mean two different piles could look the same" | a **position-dependent key stream**: the key vector advances by a distinct odd per-bird delta every block, so permuted / relocated bytes cannot fold to the same corner |
| "never looking ahead" | no byte is read before the previous crease is set; the crease chain is strictly serial |

**Assumption broken:** "the state is a single accumulator updated in place, one value" (four gauges + four creases = 512 bits), **and** "more mixing rounds always means better mixing" (the amount of refolding is data-dependent, not a fixed round count), **and partially** "each byte must be mixed into the running state before the next byte is read" — the four birds hold the crease in parallel and *no single running state exists* to mix into until the water's edge.

## SEED 3 — "thin the whole wad at the water's edge to one dense coin, throw every scrap away"

| World object | Computational object |
|---|---|
| the carried wad (whole folded sheet) | the full 512-bit gauge+crease state, un-reduced |
| the water's edge / tide-boundary | the finalization point, after the last byte |
| "thin... small, small, small" | successive irreversible narrowing 256→64 bits, three multiply+rotate stages |
| "one hard dense corner no bigger than a coin" | the returned `uint64_t` |
| "a body thins going under with **one small suitcase**" | `len` folded in as the only side-channel |
| every sheet, scrap, and failed bird-reading thrown in the mud | no retained state across calls, no tables, no mutable statics; failed/extra refolds are consumed and discarded |
| "if even one mark had been different, the whole spiral would have folded a different way" | avalanche requirement, stated by the native himself |

**Assumption broken:** "each byte must be mixed into the running state before the next byte is read" — most squarely of the three. Deposition is wide and cheap; *mixing* is deferred entirely to the water's edge. Also breaks "more mixing rounds always means better mixing."

# CHOSEN SEED

**SEED 2** — the four wire birds, the all-at-once agreement test, and the refold-tighter-by-the-disagreement loop.

It is the most literal mapping available (four birds → four SIMD lanes is not an analogy, it is the same object), and the most different from the known way: xxHash has four lanes but it has **no cross-lane agreement test, no data-dependent refold, and no deliberate positional scrambling**. Those three are the parts of the native's description that have no counterpart in FNV-1a or xxHash, so they are the parts that must survive translation.

Honest note on the preference rule: of the three seeds, **SEED 3 breaks "each byte must be mixed into the running state before the next byte is read" most squarely**, and SEED 2 breaks it only partially. I did not pick SEED 3 as the core because, standing alone, its mapping is "have a finalizer," which every hash already has — it is not a loop mechanism. But the native runs seeds 2 and 3 back-to-back on the same sheet, and that is where the serial break actually lands in my kernel: **the four birds are never reduced to one value until the water's edge**, so no byte is mixed into "the running state" before the next byte is read, because during the whole pass there is no single running state. Seed 2 is the core of the loop; seed 3 is its own terminal step, not a substitute core.

# ASSUMPTION BROKEN

Primary: **"the state is a single accumulator updated in place, one value."** Four independent keyed gauges plus four crease registers, 512 bits of live state, reduced only at the end.

Consequential: **"more mixing rounds always means better mixing"** — the refold is applied in an amount set by the birds' disagreement, so an already-agreeing crease receives *zero* extra mixing; and **"each byte must be mixed into the running state before the next byte is read"** — there is no single running state during the pass.

Deliberately *kept*: "the whole buffer must be read once, start to end, in order" — the native insists on it ("never skipping, never looking ahead"), so I keep it.

**On step 4 (let the mechanism arrive at a validated technique, don't invent):** pressing a word's two 32-bit halves with added keys through one 32×32→64 multiply and *adding* the product into a wide accumulator, with all strength deferred to a terminal finalizer, is exactly the **NH construction of UMAC (RFC 4418) / VMAC / CLHASH / UMASH**. It is validated, it is the standard way to satisfy the assumption I am breaking, and it is where the bird-press lands on its own, so I used it rather than inventing a press. The genuinely non-standard part — the cross-lane agreement witness and the refold-tighter crease chain — is the native's own mechanism, and I kept it because it is cheap (branchless, ~4 µops per 32 bytes) and it supplies the order- and position-nonlinearity that fixed-key NH does not have by itself.

**On step 5 (two regimes).** The known-way section describes a large-buffer regime and warns about small-input overhead. The native's own regime test is his first sentence: he checks whether the sheet is **"large enough"** for the pile. Translated: `len < 32` means the pile never covers the big sheet, so the birds and the big sheet are not set up at all — he hand-folds on a scrap, one crease per mark, which is literally SEED 1. That is the fallback path, chosen at runtime by the metaphor's own criterion.

No thread parallelism. The metaphor has one pair of hands and one sheet; at the sizes this is benchmarked at, 64-byte units of work are far too small to pay for a team, and memory bandwidth caps the big path anyway. Vectorization is where the metaphor's own parallelism lives (four birds = four lanes), so that is where I spent it.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the markings printed on the sheet: compile-time only, nothing retained ---- */
#define FOLD_P1 0x9E3779B185EBCA87ULL
#define FOLD_P2 0xC2B2AE3D27D4EB4FULL
#define FOLD_P3 0x165667B19E3779F9ULL
#define FOLD_P4 0x85EBCA77C2B2AE63ULL
#define FOLD_P5 0x27D4EB2F165667C5ULL

static inline uint64_t fold_rotl(uint64_t x, uint64_t n)
{
    unsigned s = (unsigned)(n & 63);
    return (x << s) | (x >> ((64u - s) & 63));
}

/* 32x32 -> 64 press, the NH/UMAC primitive: one multiply per 8-byte corner */
static inline uint64_t m32(uint64_t a, uint64_t b)
{
    return (uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b;
}

/* ---- the water's edge: thin the wad small, small, small, to one coin ----
   each bird's reading passes through its own multiply and rotate before the
   final avalanche, so a difference confined to one bird's high bits still
   reaches the low bits of the coin. Nothing here is kept. */
static inline uint64_t waters_edge(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                                   uint64_t len)
{
    uint64_t h = len * FOLD_P5;              /* the one small suitcase */
    h = (h ^ a) * FOLD_P1; h = fold_rotl(h, 29);
    h = (h ^ b) * FOLD_P2; h = fold_rotl(h, 31);
    h = (h ^ c) * FOLD_P3; h = fold_rotl(h, 37);
    h = (h ^ d) * FOLD_P4;
    h ^= h >> 33; h *= FOLD_P2;               /* small */
    h ^= h >> 29; h *= FOLD_P3;               /* smaller */
    h ^= h >> 32;                             /* one hard dense corner */
    return h;
}

/* ---- scrap path (SEED 1, literal): the pile is too small for the big sheet,
   so no birds are set up. One crease per mark, the angle set by the mark AND
   the crease left by the fold before it. ---- */
static uint64_t scrap_fold(const unsigned char *p, size_t len)
{
    uint64_t h = 0x1BD11BDAA9FC1A22ULL ^ ((uint64_t)len * FOLD_P5);
    uint64_t crease = 0xCBF29CE484222325ULL;
    size_t i;
    for (i = 0; i < len; i++) {
        uint64_t mark = p[i];
        crease = fold_rotl(crease + mark + FOLD_P3, crease ^ mark);
        h = (h ^ crease) * FOLD_P1;
    }
    h ^= h >> 33; h *= FOLD_P2;
    h ^= h >> 29; h *= FOLD_P3;
    h ^= h >> 32;
    return h;
}

#if defined(__AVX2__)
/* ---- one fold of the sheet: four corners, four birds, all four beaks at once,
   then refold tighter by exactly how much they disagree. ---- */
static inline void beak_press(__m256i v, __m256i K, const __m256i KB,
                              __m256i *G, __m256i *C)
{
    __m256i hi = _mm256_srli_epi64(v, 32);
    __m256i a  = _mm256_add_epi64(v, K);    /* low 32 bits: corner + sheet marking */
    __m256i b  = _mm256_add_epi64(hi, KB);  /* low 32 bits: corner + this bird's beak */
    __m256i pr = _mm256_xor_si256(_mm256_mul_epu32(a, b), v); /* the mark itself survives */
    __m256i g  = _mm256_add_epi64(*G, pr);  /* four gauge readings, independent */

    /* all four beaks at once: lane j = bird j against bird j+1.
       all four agree  <=>  d is zero in every lane. */
    __m256i s = _mm256_permute4x64_epi64(g, 0x39);
    __m256i d = _mm256_xor_si256(g, s);

    /* refold tighter, by exactly the disagreement. d == 0 is the identity:
       a crease the birds already agree on gets no extra folding at all. */
    __m256i c = _mm256_xor_si256(*C, d);
#if defined(__AVX512VL__) && defined(__AVX512F__)
    *C = _mm256_rolv_epi64(c, d);
#else
    {
        __m256i sh = _mm256_and_si256(d, _mm256_set1_epi64x(63));
        *C = _mm256_or_si256(
                 _mm256_sllv_epi64(c, sh),
                 _mm256_srlv_epi64(c, _mm256_sub_epi64(_mm256_set1_epi64x(64), sh)));
    }
#endif
    *G = g;
}
#endif /* __AVX2__ */

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the native's own regime test: is the sheet large enough for this pile? */
    if (len < 32)
        return scrap_fold(data, len);

#if defined(__AVX2__)
    {
        /* four beaks, four distinct sheet-marking advances (all odd, all distinct
           in their low 32 bits, so position is scrambled on purpose) */
        const __m256i KB  = _mm256_setr_epi64x((long long)0x2545F4914F6CDD1DULL,
                                               (long long)0x9E3779B97F4A7C15ULL,
                                               (long long)0xBF58476D1CE4E5B9ULL,
                                               (long long)0x94D049BB133111EBULL);
        const __m256i DEL = _mm256_setr_epi64x((long long)0x0A2127A3B5E0D1F7ULL,
                                               (long long)0x1B3C5D7F91A3B5C7ULL,
                                               (long long)0x2C4D6E8FA1B3C5D9ULL,
                                               (long long)0x3D5E7F91B3C5D7EBULL);
        const __m256i D2 = _mm256_add_epi64(DEL, DEL);

        /* the sheet is already doubled, so one press sets a crease in each ply */
        __m256i G0 = _mm256_setr_epi64x((long long)0x1BD11BDAA9FC1A22ULL,
                                        (long long)0xCBF29CE484222325ULL,
                                        (long long)0x84222325CBF29CE4ULL,
                                        (long long)0xA9FC1A221BD11BDAULL);
        __m256i C0 = _mm256_setr_epi64x((long long)0x6A09E667F3BCC908ULL,
                                        (long long)0xBB67AE8584CAA73BULL,
                                        (long long)0x3C6EF372FE94F82BULL,
                                        (long long)0xA54FF53A5F1D36F1ULL);
        __m256i K0 = _mm256_setr_epi64x((long long)0x510E527FADE682D1ULL,
                                        (long long)0x9B05688C2B3E6C1FULL,
                                        (long long)0x1F83D9ABFB41BD6BULL,
                                        (long long)0x5BE0CD19137E2179ULL);
        __m256i G1 = _mm256_xor_si256(G0, KB);
        __m256i C1 = _mm256_add_epi64(C0, DEL);
        __m256i K1 = _mm256_add_epi64(K0, DEL);

        const unsigned char *restrict p = data;
        size_t n = len;

        while (n >= 64) {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)(p));
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
            beak_press(v0, K0, KB, &G0, &C0);
            beak_press(v1, K1, KB, &G1, &C1);
            K0 = _mm256_add_epi64(K0, D2);   /* the markings advance with position */
            K1 = _mm256_add_epi64(K1, D2);
            p += 64; n -= 64;
        }
        if (n >= 32) {
            beak_press(_mm256_loadu_si256((const __m256i *)(const void *)p),
                       K0, KB, &G0, &C0);
            p += 32; n -= 32;
        }
        if (n) {
            /* the scrap trimmed off: fold the last 32 marks again, with the
               trim amount pressed into the markings so the overlap is not free */
            __m256i vt = _mm256_loadu_si256((const __m256i *)(const void *)(data + len - 32));
            beak_press(vt, _mm256_add_epi64(K1, _mm256_set1_epi64x((long long)n)),
                       KB, &G1, &C1);
        }

        /* carry the whole wad to the water's edge: gauges crossed with the other
           ply's creases, so neither the readings nor the refoldings can be ignored */
        {
            __m256i W = _mm256_add_epi64(_mm256_xor_si256(G0, C1),
                                         _mm256_xor_si256(G1, C0));
            uint64_t w[4];
            _mm256_storeu_si256((__m256i *)(void *)w, W);
            return waters_edge(w[0], w[1], w[2], w[3], (uint64_t)len);
        }
    }
#else
    /* same mechanism, four birds by hand (no SIMD available) */
    {
        uint64_t G[4] = { 0x1BD11BDAA9FC1A22ULL, 0xCBF29CE484222325ULL,
                          0x84222325CBF29CE4ULL, 0xA9FC1A221BD11BDAULL };
        uint64_t C[4] = { 0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
                          0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL };
        uint64_t K[4] = { 0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
                          0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL };
        const uint64_t KB[4] = { 0x2545F4914F6CDD1DULL, 0x9E3779B97F4A7C15ULL,
                                 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL };
        const uint64_t DL[4] = { 0x0A2127A3B5E0D1F7ULL, 0x1B3C5D7F91A3B5C7ULL,
                                 0x2C4D6E8FA1B3C5D9ULL, 0x3D5E7F91B3C5D7EBULL };
        size_t off = 0;
        uint64_t w[4], g[4], d[4];
        int j;
        while (off + 32 <= len) {
            memcpy(w, data + off, 32);
            for (j = 0; j < 4; j++)
                g[j] = G[j] + (m32(w[j] + K[j], (w[j] >> 32) + KB[j]) ^ w[j]);
            for (j = 0; j < 4; j++) d[j] = g[j] ^ g[(j + 1) & 3];
            for (j = 0; j < 4; j++) {
                C[j] = fold_rotl(C[j] ^ d[j], d[j]);
                G[j] = g[j];
                K[j] += DL[j];
            }
            off += 32;
        }
        if (off < len) {
            uint64_t rem = (uint64_t)(len - off);
            memcpy(w, data + len - 32, 32);
            for (j = 0; j < 4; j++)
                g[j] = G[j] + (m32(w[j] + K[j] + rem, (w[j] >> 32) + KB[j]) ^ w[j]);
            for (j = 0; j < 4; j++) d[j] = g[j] ^ g[(j + 1) & 3];
            for (j = 0; j < 4; j++) { C[j] = fold_rotl(C[j] ^ d[j], d[j]); G[j] = g[j]; }
        }
        return waters_edge(G[0] ^ C[1], G[1] ^ C[0],
                           G[2] ^ C[3], G[3] ^ C[2], (uint64_t)len);
    }
#endif
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 22**

Reasoning stated before any measurement: the FNV-1a reference is a loop-carried `xor; imul` chain, latency-bound at roughly 4–5 cycles per byte (≈0.6–0.8 GB/s at 3 GHz). The bird path costs about 17 µops per 32-byte fold (1 load, 1 shift, 2 key adds, 1 `vpmuludq`, 1 xor, 1 gauge add, 1 `vpermq`, 1 xor, and ~4 for the branchless refold), with four plies' worth of independent chains, so ~4–5 cycles per 32 bytes → 6–8 bytes/cycle → 18–24 GB/s while L2-resident, falling to memory bandwidth for very large buffers. That is where 22× comes from. On avalanche I predict a *good* score (≈0.49–0.50 mean bit-flip fraction, i.e. near-ideal): every single-bit flip changes at least one gauge lane through the `^v` term even in the measure-zero case where the NH product difference annihilates, the change reaches two crease lanes immediately and all four within two folds via the data-dependent rotate amounts, and the terminal thinning runs each bird's reading through its own multiply and rotate before a standard, well-tested three-stage avalanche.

The single thing I am least sure of: `_mm256_permute4x64_epi64` has 3-cycle latency and sits inside the crease chain, so if the compiler fails to keep the two plies' crease chains independent, throughput could collapse to the chain latency (~8 cycles/32 B) and the speedup would land nearer 10× than 22×.

# MEASUREMENT

I must report this plainly: **no measurement was taken.** This session was invoked with no tools available — `hash_bench` and `hash_contract` could not be called, so the throughput and avalanche numbers above are predictions from an instruction-count and dependency-chain argument, not observations. The prediction line is stated above and stands as written; the pipeline's numbers supersede it, and if they come back at 10× or with a poor avalanche score, the prediction was wrong and the reasoning about `vpermq` latency in the crease chain is the first place I would look.

What I can verify by inspection, and did:

- Every byte of `data[0..len)` is read: the 64-byte loop, the 32-byte step, and the overlapping final 32-byte fold together cover `[0, len)` with no gap for every `len ≥ 32`; `len < 32` goes byte-by-byte.
- No out-of-bounds read: the overlapping tail load is only reached under `len ≥ 32`, and `data + len - 32 ≥ data`.
- No undefined behaviour in the rotates: `fold_rotl` masks the count and uses the `(64-s)&63` form, so a count of 0 is the identity rather than a 64-bit shift; the AVX2 variable shifts return 0 for counts ≥ 64 by definition, so `sllv(c,0) | srlv(c,64)` is exactly `c` — a crease the birds agree on is genuinely left alone.
- No strict-aliasing violation: scalar 64-bit loads go through `memcpy`, vector loads through `loadu`.
- No retained state: no statics, no mutable globals, no tables; every constant is an immediate and every scratch value dies with the call.

# VERDICT

The core of this kernel is the native's second mechanism, not a textbook hash. Concretely, and this is the test the reviewer should apply: delete the `vpermq`/`xor` agreement witness and the `rotl(C ^ d, d)` refold chain and the kernel still computes a hash — a plain keyed NH sum — but it is no longer the native's, and it loses its order- and position-nonlinearity. Those lines are load-bearing, not decoration. There is no `acc = rotl(acc + word*P2, 31) * P1` anywhere, no xxHash lane merge, no FNV prime chain over the body.

What I claim honestly:

- **The four birds are the four SIMD lanes.** That is the translation that made the mechanism fast rather than slow, and it is the whole reason this is worth building. "Test every crease against all four beaks *at once*" is a SIMD instruction, read literally.
- **The press landed on a validated technique, as step 4 requires.** One keyed 32×32→64 multiply per 8-byte corner, added into a wide accumulator, with all strength deferred to the end, is NH as specified in UMAC/RFC 4418 and used in VMAC, CLHASH and UMASH. I did not invent a press.
- **The agreement/refold chain is the genuinely untested part.** I keep it because it is the native's core and because it is cheap and does real work: it is the only source of order-dependence beyond the key stream, and the only nonlinearity in the body. But I will not claim it has been validated, because it has not been, by anyone, including me in this session.

Stated risks, and how each is addressed rather than left hanging:

1. *The four-bird path has vector setup cost and needs ≥32 bytes.* Guarded: `len < 32` takes `scrap_fold`, the serial per-mark path, chosen by the metaphor's own "is the sheet large enough" test. No part of the vector path is touched.
2. *The AVX2 variable rotate costs 4 µops.* Guarded: compiled with AVX512VL it collapses to one `vprolvq`; and on a machine without AVX2 at all, the scalar four-bird fallback computes the same mechanism by hand.
3. *Thread parallelism could be a trap at these sizes.* Dropped outright rather than guarded, per the instruction to prefer vectorization: 64-byte work units cannot pay for a thread team, and the big path is bandwidth-capped anyway.
4. *Linear additive accumulation could be attacked by structured differences.* Partly addressed — the position-advancing key stream and the nonlinear crease chain break pure linearity — but I will say plainly that this is a non-cryptographic hash in the same class as xxHash, with fixed keys rather than NH's random ones, so NH's provable universality bound does **not** carry over. I am not claiming it.

One cost I accept and name: the AVX2 path and the scalar fallback produce different hash values for the same input, as do builds with and without AVX512VL. For a non-cryptographic hash with no "right answer" that is acceptable, but it would be a defect if the value ever had to be persisted or compared across machines.