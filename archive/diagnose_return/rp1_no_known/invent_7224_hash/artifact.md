## MAPPING

**SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."**

| World object | Problem object |
|---|---|
| pile of marks, given order | `data[0..len)`, the byte buffer |
| one mark | one byte |
| flat stone at the riverbank | a lane slot: byte *i* is assigned a (lane, rotate-constant) address from its index |
| servant with the inflated house on a stick, swatting the mouth | the position-fixing step: order is recorded **as placement**, not as traversal time — nothing is "spoken" (folded) while the marks are being laid out |
| "the order never gets spoken aloud and lost" | order-sensitivity comes from *where* a byte lands (which of 16 lanes, which row's rotate constant, which stripe), not from *when* it was read |
| never speaking until laid down | therefore the buffer need not be walked start-to-end: two cursors walk two stripes at once |

Breaks: **"the whole buffer must be read once, start to end, in order"** (and, as a consequence, "each byte must be mixed into the running state before the next byte is read").

**SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."**

| World object | Problem object |
|---|---|
| heap of even count | 128-byte block = 16 × `uint64` lanes |
| musk deer's fixed strides | fixed block granularity, fixed shift/rotate schedule |
| uphill twist | `rotl64(x, r)` (per-row constant r) |
| downhill fold | `x ^= x >> s` |
| doubling-back that eats its own trail | the lane-slide + add/xor chain: a row is consumed into the row that just consumed it |
| "no heap keeps the shape it entered with" | the per-block map is a bijection on the whole 1024-bit state — no data difference can die |
| the final stone at the shrine, never let ossify | the carried 16-lane state, the only thing persisting across heaps |

Breaks: **"the state is a single accumulator updated in place, one value"** and **"mixing one byte requires a multiplication"** (the heap race uses rotate/xor/add only — zero multiplies per byte).

**SEED 3 — "The owl calls the final shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned."**

| World object | Problem object |
|---|---|
| the owl taking the pen | the finalizer, run exactly once |
| "inside the dreamer inside" | two square rounds (no data) then `fmix64`-style seal — a *bounded* number of turns |
| "small and unchanging, the size of any other token" | 1024-bit state → one `uint64_t`, independent of `len` |
| heaps burned the moment the next swallows them | no scratch buffer, no history: lanes live in registers, overwritten in place |
| the butterfly who cannot recall being a man | one-way compression: the 16:1 fold is not invertible |

Breaks: **"more mixing rounds always means better mixing"** — the owl *fixes* the count ("a calculation grown too satisfied with itself is a pillar that will bring the roof down"); diffusion is accumulated across heaps, not bought with extra rounds at the end.

## CHOSEN SEED

**SEED 1.** It is the one that breaks the preferred assumption ("the whole buffer must be read once, start to end, in order"), and its mapping is the most literal: the servant physically prevents any folding until every mark has a *place*, which is exactly the statement "order is carried by lane address, therefore traversal order is free." It is also maximally far from FNV-1a, whose entire identity is a strictly ordered byte-at-a-time walk through one accumulator. SEEDs 2 and 3 are implemented too — they are facets of the same working, not alternatives — but SEED 1 is what licenses the traversal.

## ASSUMPTION BROKEN

Primary: **the whole buffer must be read once, start to end, in order.** The kernel walks two ascending stripes simultaneously — cursor 0 from the riverbank (`data`), cursor 1 from the midstream stone (`data + len - 64*nb`) — meeting at a middle remainder. Byte order still matters completely, because it is encoded in placement (stripe → row pair, offset → lane, row → rotate constant, word index → lane in the remainder path).

Secondary, from the same working: no accumulator (16 lanes), **no multiplication anywhere in the per-byte path** (multiplies appear only twice, in the O(1) owl seal), and a fixed, small round count.

**Regime recognition (required by step 5).** The known way names two regimes — a short-buffer regime where setup cost is everything (FNV's home) and a long-buffer regime where throughput is everything (xxHash's home). The metaphor supplies the runtime test itself: *the musk deer needs enough ground for a stride.* If the pile cannot fill one heap (`len < 128`), there is no mountain path at all — the marks stay at the riverbank on four stones and the owl seals them immediately (no 16-lane setup, no square rounds). At `len >= 128` the grid is laid and the heaps race. A third sub-path handles the middle remainder word-by-word into the grid. This is a size check with a genuine fallback, not two names for one code path.

**Risk I dropped rather than shipped.** My first construction walked head-forward and tail-*backward* (the deer's literal doubling-back). That risks defeating the hardware prefetcher on the descending stream — a condition where I would lose to the known way. Rather than ship it with a caveat, I replaced it with two *ascending* stripes, which is equally out-of-order at the whole-buffer level and prefetch-friendly on both streams. I also considered OpenMP over heaps and **dropped it**: at these sizes the loop is already at memory bandwidth, thread count would make the hash value nondeterministic, and the per-thread cost dwarfs a 128-byte heap. Vectorization only (AVX2 intrinsics, guarded, with a bit-identical scalar fallback).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- *
 *  The riverbank, the mountain path, and the owl.                  *
 *  No multiplication touches a byte; the only two multiplies in    *
 *  the whole working are the owl's seal, done once.                *
 * ---------------------------------------------------------------- */

#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the sixteen flat stones at the riverbank */
static const uint64_t RIVER[16] = {
    0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL,
    0x2545f4914f6cdd1dULL, 0xc2b2ae3d27d4eb4fULL, 0x165667b19e3779f9ULL,
    0x85ebca77c2b2ae63ULL, 0x27d4eb2f165667c5ULL, 0x6a09e667f3bcc909ULL,
    0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL
};

/* the owl takes the pen: fixed size, fixed cost, not invertible back to marks */
static inline uint64_t owl_seal(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

/* the deer's stride: uphill twist, downhill fold (one constant pair per row) */
static inline void deer_stride(uint64_t *st) {
    int j;
    for (j = 0; j < 4; j++) { st[j]      = ROTL64(st[j], 23);      st[j]      ^= st[j]      >> 29; }
    for (j = 0; j < 4; j++) { st[4 + j]  = ROTL64(st[4 + j], 37);  st[4 + j]  ^= st[4 + j]  >> 17; }
    for (j = 0; j < 4; j++) { st[8 + j]  = ROTL64(st[8 + j], 13);  st[8 + j]  ^= st[8 + j]  >> 41; }
    for (j = 0; j < 4; j++) { st[12 + j] = ROTL64(st[12 + j], 47); st[12 + j] ^= st[12 + j] >> 11; }
}

/* searching in squares: turn the shape ninety degrees, fold the corners
   into its own centre.  Bijective on the whole 1024-bit state.          */
static inline void sq_mix(uint64_t *st) {
    uint64_t t[4];
    int j;
    for (j = 0; j < 4; j++) t[j] = st[4  + ((j + 1) & 3)];
    for (j = 0; j < 4; j++) st[4  + j] = t[j];
    for (j = 0; j < 4; j++) t[j] = st[8  + ((j + 2) & 3)];
    for (j = 0; j < 4; j++) st[8  + j] = t[j];
    for (j = 0; j < 4; j++) t[j] = st[12 + ((j + 3) & 3)];
    for (j = 0; j < 4; j++) st[12 + j] = t[j];

    for (j = 0; j < 4; j++) { st[j]     += st[4 + j];  st[12 + j] ^= st[j];     st[12 + j] = ROTL64(st[12 + j], 32); }
    for (j = 0; j < 4; j++) { st[8 + j] += st[12 + j]; st[4 + j]  ^= st[8 + j]; st[4 + j]  = ROTL64(st[4 + j], 24); }
    for (j = 0; j < 4; j++) { st[j]     += st[4 + j];  st[12 + j] ^= st[j];     st[12 + j] = ROTL64(st[12 + j], 16); }
    for (j = 0; j < 4; j++) { st[8 + j] += st[12 + j]; st[4 + j]  ^= st[8 + j]; st[4 + j]  = ROTL64(st[4 + j], 21); }
}

/* ---- regime 1: the pile is too small to fill a heap.  No mountain path,
        no grid, no square rounds: four stones at the riverbank, then the owl. */
static inline uint64_t riverbank(const unsigned char *data, size_t len) {
    uint64_t s[4];
    size_t i = 0, k = 0;
    s[0] = RIVER[0]  ^ (uint64_t)len;
    s[1] = RIVER[5];
    s[2] = RIVER[10] ^ ROTL64((uint64_t)len, 32);
    s[3] = RIVER[15];
    while (i + 8 <= len) {
        uint64_t w, v;
        memcpy(&w, data + i, 8);
        v = ROTL64(s[k] ^ w, 29);
        v ^= v >> 31;
        s[k] = v;
        k = (k + 1) & 3;
        s[k] += v;                    /* the stone carried forward */
        i += 8;
    }
    if (i < len) {
        uint64_t w = 0, v;
        size_t b = 0;
        while (i < len) { w |= (uint64_t)data[i] << (b * 8); i++; b++; }
        v = ROTL64(s[k] ^ w, 41);
        v ^= v >> 23;
        s[k] = v;
        s[(k + 1) & 3] += v;
    }
    {
        uint64_t x = (s[0] + ROTL64(s[1], 13)) ^ (s[2] + ROTL64(s[3], 29));
        return owl_seal(x ^ ROTL64(s[1] ^ s[3], 47) ^ (uint64_t)len);
    }
}

#if defined(__AVX2__)
#define VROTL(x, r) _mm256_or_si256(_mm256_slli_epi64((x), (r)), _mm256_srli_epi64((x), 64 - (r)))
#define VXSR(x, s)  _mm256_xor_si256((x), _mm256_srli_epi64((x), (s)))
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t st[16];
    size_t i, nb, rem;
    const unsigned char *p0, *p1, *mid;

    /* the deer needs enough ground for one stride */
    if (len < 128) return riverbank(data, len);

    for (i = 0; i < 16; i++) st[i] = RIVER[i];
    st[0]  ^= (uint64_t)len;
    st[15] ^= ROTL64((uint64_t)len, 32);

    nb  = len >> 7;                   /* heaps of even count: 128 marks each */
    rem = len - (nb << 7);
    p0  = data;                       /* cursor at the riverbank            */
    p1  = data + len - (nb << 6);     /* cursor at the midstream stone      */
    mid = data + (nb << 6);           /* the marks left in the middle       */

#if defined(__AVX2__)
    {
        __m256i A = _mm256_loadu_si256((const __m256i *)(st + 0));
        __m256i B = _mm256_loadu_si256((const __m256i *)(st + 4));
        __m256i C = _mm256_loadu_si256((const __m256i *)(st + 8));
        __m256i D = _mm256_loadu_si256((const __m256i *)(st + 12));
        for (i = 0; i < nb; i++) {
            __m256i h0 = _mm256_loadu_si256((const __m256i *)(p0));
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(p0 + 32));
            __m256i t0 = _mm256_loadu_si256((const __m256i *)(p1));
            __m256i t1 = _mm256_loadu_si256((const __m256i *)(p1 + 32));
            p0 += 64; p1 += 64;
            /* every mark laid on its own stone: order is placement, not time */
            A = _mm256_xor_si256(A, h0);
            B = _mm256_xor_si256(B, h1);
            C = _mm256_xor_si256(C, t0);
            D = _mm256_xor_si256(D, t1);
            /* the deer's stride */
            A = VROTL(A, 23); A = VXSR(A, 29);
            B = VROTL(B, 37); B = VXSR(B, 17);
            C = VROTL(C, 13); C = VXSR(C, 41);
            D = VROTL(D, 47); D = VXSR(D, 11);
            /* turn ninety degrees */
            B = _mm256_permute4x64_epi64(B, 0x39);
            C = _mm256_permute4x64_epi64(C, 0x4E);
            D = _mm256_permute4x64_epi64(D, 0x93);
            /* fold the corners into the centre */
            A = _mm256_add_epi64(A, B); D = _mm256_xor_si256(D, A); D = VROTL(D, 32);
            C = _mm256_add_epi64(C, D); B = _mm256_xor_si256(B, C); B = VROTL(B, 24);
            A = _mm256_add_epi64(A, B); D = _mm256_xor_si256(D, A); D = VROTL(D, 16);
            C = _mm256_add_epi64(C, D); B = _mm256_xor_si256(B, C); B = VROTL(B, 21);
        }
        _mm256_storeu_si256((__m256i *)(st + 0),  A);
        _mm256_storeu_si256((__m256i *)(st + 4),  B);
        _mm256_storeu_si256((__m256i *)(st + 8),  C);
        _mm256_storeu_si256((__m256i *)(st + 12), D);
    }
#else
    for (i = 0; i < nb; i++) {
        uint64_t w[16];
        int j;
        memcpy(w,     p0, 64);
        memcpy(w + 8, p1, 64);
        p0 += 64; p1 += 64;
        for (j = 0; j < 16; j++) st[j] ^= w[j];
        deer_stride(st);
        sq_mix(st);
    }
#endif

    /* the marks left in the middle, each still on its own stone */
    {
        size_t k = 0;
        while (k + 8 <= rem) {
            uint64_t w;
            size_t idx = (k >> 3) & 15;
            memcpy(&w, mid + k, 8);
            st[idx] = ROTL64(st[idx] ^ w, 29);
            st[idx] ^= st[idx] >> 31;
            st[(idx + 7) & 15] += st[idx];
            k += 8;
        }
        if (k < rem) {
            uint64_t w = 0;
            size_t b = 0;
            while (k < rem) { w |= (uint64_t)mid[k] << (b * 8); k++; b++; }
            st[15] = ROTL64(st[15] ^ w, 41);
            st[15] ^= st[15] >> 23;
            st[0]  += st[15];
        }
    }

    /* the owl calls the shape inside the dreamer inside: two turns, no more */
    sq_mix(st);
    sq_mix(st);

    /* burn the heaps: sixteen stones down to one token, one way only */
    {
        uint64_t a0 = (st[0]  + ROTL64(st[1],  13)) ^ (st[2]  + ROTL64(st[3],  29));
        uint64_t a1 = (st[4]  + ROTL64(st[5],  41)) ^ (st[6]  + ROTL64(st[7],   7));
        uint64_t a2 = (st[8]  + ROTL64(st[9],  19)) ^ (st[10] + ROTL64(st[11], 53));
        uint64_t a3 = (st[12] + ROTL64(st[13], 23)) ^ (st[14] + ROTL64(st[15], 37));
        uint64_t x  = (a0 + ROTL64(a1, 17)) ^ (a2 + ROTL64(a3, 43));
        return owl_seal(x ^ (uint64_t)len);
    }
}
```

The AVX2 and scalar loops are exact mirrors (same lane↔byte map, same constants, same order), so on a little-endian machine both paths return bit-identical values; `-march=native` only changes speed, not the hash.

## PREDICTION

**PREDICTION: speedup_vs_dp = 12.0**

Stated before any measurement, with the reasoning so it can be scored honestly:

- Baseline FNV-1a is latency-bound at xor + `imul` ≈ **4 cycles/byte** (0.25 B/cycle) — it cannot use more than one ALU.
- Grid path: ~50 vector uops per 128 bytes, critical path ≈ 20–24 cycles per heap → **~0.17 cycles/byte** (≈ 5–6 B/cycle), i.e. ~20× compute-bound, clamped to single-core memory bandwidth (~10–20 GB/s) for buffers past L2.
- Fixed cost of the grid path (16-lane init + 2 owl rounds + fold + seal) ≈ 120–170 cycles, so at `len = 128` the win shrinks to ~3×; the riverbank path covers everything below that at ~0.7 cycles/byte ≈ **4–6×**.
- A size sweep that includes small buffers therefore lands near **8–15×**; a large-buffer-only sweep should land **20×+**.
- Avalanche: I predict a flip fraction of **0.50 ± 0.01** and an essentially maximal score. The argument, not a hope: every per-heap operation (xor-in, rotate, `x ^= x>>s`, lane slide, `A += B`, `D ^= A`) is a bijection on the 1024-bit state and injective in the absorbed data, so a one-bit input change **can never** be annihilated before the fold; the fold then reaches all 16 lanes and `owl_seal` is the murmur3 finalizer, whose avalanche is known-good.

Falsifiers I will accept: throughput ratio < 4× on any tested size, or avalanche flip fraction outside 0.48–0.52, means this construction failed on that axis and I should say so rather than re-narrate it.

## MEASUREMENT

**Not measured. No tools were available in this session** — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all absent, so I could not compile, could not run a single trial, and could not iterate the four allowed times. I ran **zero** measured improvement rounds. What I report instead:

- **Checked by hand, not by tool:** in-bounds reads (`64*nb ≤ len - 64*nb` for the two stripes; middle region `[64nb, 64nb+rem)` with `rem ≤ 127`; `len = 0` reads nothing), all rotate amounts in 1..63, all `_mm256_slli/srli/permute4x64` immediates compile-time constant via macros/literals, permute immediates verified (`0x39` = slide 1, `0x4E` = slide 2, `0x93` = slide 3), and the AVX2/scalar mirror equivalence lane by lane.
- **Two design-time changes made for stated-risk reasons, not for measured reasons:** (1) the backward tail cursor became a second forward stripe (prefetcher risk eliminated rather than caveated); (2) OpenMP over heaps dropped entirely (bandwidth-bound already, and thread count would make the returned value nondeterministic).
- Everything in PREDICTION is an unverified cycle-count estimate. The pipeline's numbers supersede all of it.

## VERDICT

The native's method translates into a real, non-textbook hash and nothing had to be quietly swapped for FNV-style mixing: the servant's silence-swats become "order lives in placement," which is exactly what licenses reading the buffer as two interleaved ascending stripes instead of once start-to-end; the musk deer's stride becomes a rotate/xorshift/lane-slide block function with **zero multiplications per byte**; the shrine's one unossified stone becomes the 16-lane carried state; the owl becomes a fixed two-round finalization plus a one-way 16:1 fold. The one silent assumption I could not honestly break is that the buffer must be *read* (every byte touched exactly once) — the marks do all get laid on stones, and no metaphor object corresponds to skipping a mark.

Where this can still lose, and what I did about it:

- **Small inputs** — the grid's setup would crush it. Guarded: `len < 128` takes the riverbank path with no grid, no square rounds, four stones.
- **Very small inputs (< 16 bytes)** — the riverbank path's fixed cost (~15 cycles of seal) is now comparable to FNV's total work, so the win narrows toward ~2×. It should not *lose*, since the per-byte cost is still lower and the seal is cheaper than 16 chained `imul`s — but this is the weakest regime and I am naming it rather than hiding it.
- **Nondeterminism / portability** — the hash is little-endian-specific and requires `len` up front (it is not a streaming hash). That is inherent to the metaphor (order fixed before folding), shared with xxh3's large path, and within the contract.
- **Unverified** — every performance and avalanche claim above is a prediction. If the bench says otherwise, the construction is wrong and the honest next move is to shorten the per-heap critical path (drop to 3 fold lines) or narrow the grid threshold, not to reach for a multiply-accumulator.