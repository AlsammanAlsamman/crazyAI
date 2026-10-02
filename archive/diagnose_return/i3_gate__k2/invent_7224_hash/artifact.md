## MAPPING

**SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."**

| World object | Computational object |
|---|---|
| a mark | one input byte `data[i]` |
| the pile in its given order | the buffer, order = byte offset |
| walking marks past the servant, one at a time | the load stream: 8 independent `loadu` of a 256-B window |
| the inflated house swatted against the mouth (keisaku) | *no state update is allowed to speak during the read* — the absorb is pure XOR-into-lane, zero carried dependency between loads |
| the flat stone at the riverbank | the landing layout: offset → (grid half, row, 64-bit lane). Position **is** the order |
| "so the order never gets spoken aloud and lost" | order is preserved positionally instead of temporally, so folds need not follow reading order |

Breaks: **"the whole buffer must be read once, start to end, in order"** (and, as a consequence, "each byte must be mixed into the running state before the next byte is read").

**SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."**

| World object | Computational object |
|---|---|
| heaps of even count | fixed 256-byte blocks (128 B per grid) |
| musk deer's fixed strides | a fixed-size block permutation, applied once per block |
| uphill twist | `ROTR64` |
| downhill fold | `XOR` |
| doubling-back that eats its own trail | `+` (carry propagation — the only direction-mixing op) |
| "no heap keeps the shape it entered with" | the permutation is applied to the *whole* 1024-bit state, not one accumulator |
| the final stone where the running shape is carried forward | the chained state across blocks |
| "I never let it ossify" | no constant-only accumulator; state is never reset mid-stream |

Breaks: **"mixing one byte requires a multiplication"** and **"the state is a single accumulator updated in place, one value"**.

**SEED 3 — "The owl calls the final folded shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned at the shrine."**

| World object | Computational object |
|---|---|
| drifting man searching squares in every direction | the 4×4 grid of 64-bit words; column pass then diagonal pass |
| turning the shape ninety degrees | lane rotation `permute4x64` (diagonalize / undiagonalize) |
| folding corners into its own center | XOR-fold of 16 words → one 64-bit value |
| the owl's seal, "the size of any other token" | `fmix64` finalizer: 64 bits out regardless of `len` |
| intermediate heaps burned | wide state is compressed and discarded; not invertible |
| "change one mark, watch the whole grid rearrange" | the avalanche test itself is part of the method |

Breaks: **"more mixing rounds always means better mixing"** — rounds are *placed* (1 per block, 4 at the end where the last block has nowhere else to diffuse), not piled on uniformly.

## CHOSEN SEED

**SEED 1.** It is the only one that breaks the preferred assumption, and it is the most literal: the swats that hold silence are exactly "do not touch the accumulator while reading," and the flat stone is exactly a position→lane map. SEEDs 2 and 3 are not discarded — the native describes one method, and they supply its round function and its seal, which SEED 1's layout requires in order to be a hash at all.

## ASSUMPTION BROKEN

*The whole buffer must be read once, start to end, in order.* Here order is carried by **where** a byte lands (grid half → row → 64-bit lane → bit offset), not by **when** it is mixed. Nothing in the inner loop depends on the previous byte: 8 loads and 8 XORs issue in parallel, then one deer-stride permutes the whole 1024-bit state. Corollary breaks: the per-byte path contains **no multiplication at all** (add/rotate/xor only — the single multiply pair lives in the owl's seal, O(1) per call).

Where this lands, deliberately: a **sponge absorb with a BLAKE2b-style ARX permutation over a 4×4 64-bit grid, column pass + diagonal pass** (the Gimli/BLAKE2/ChaCha family), finished with **MurmurHash3's `fmix64`**. The metaphor arrives at validated primitives rather than inventing a round function; what is novel is only the twin-grid layout and the round *budget*.

Two regimes, recognised at runtime through the metaphor itself: the deer needs enough ground for a stride. `len >= 256` → twin grids. `len < 256` (or no AVX2) → the man simply walks the marks along the riverbank: one column, same stride, 32-byte steps. No thread parallelism — at these sizes one grid pass is ~26 cycles, far below any fork cost, and the SIMD path is already near single-core bandwidth.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define RIVERBANK_AVX2 1
#endif

#define ROTR64(x,n) (((x) >> (n)) | ((x) << (64 - (n))))

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* the owl: seals the folded shape to one fixed size, one way only
   (MurmurHash3 fmix64 - the only multiplications in the whole kernel) */
static inline uint64_t owl_seal(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

/* one musk-deer stride on a column: uphill twist (rot), downhill fold (xor),
   doubling-back that eats its own trail (add).  BLAKE2b's G, message words
   removed - it is a permutation, not a compression. */
#define STRIDE(a,b,c,d) do {               \
    a += b; d ^= a; d = ROTR64(d, 32);     \
    c += d; b ^= c; b = ROTR64(b, 24);     \
    a += b; d ^= a; d = ROTR64(d, 16);     \
    c += d; b ^= c; b = ROTR64(b, 63);     \
} while (0)

static const uint64_t IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};
#define RC(i) (IV[(i) & 7] ^ (0x9e3779b97f4a7c15ULL * (uint64_t)((i) + 1)))

/* ---- regime B: not enough ground for a stride -> walk the riverbank ---- */
/* one column, 32-byte steps.  also the portable path when AVX2 is absent. */
static uint64_t riverbank_walk(const unsigned char *data, size_t len) {
    uint64_t s0 = IV[0] ^ (uint64_t)len, s1 = IV[1], s2 = IV[2], s3 = IV[3];
    const unsigned char *p = data;
    size_t n = len;

    while (n >= 32) {                        /* silence: 4 pure loads, 4 xors */
        s0 ^= ld64(p);      s1 ^= ld64(p + 8);
        s2 ^= ld64(p + 16); s3 ^= ld64(p + 24);
        STRIDE(s0, s1, s2, s3);
        p += 32; n -= 32;
    }
    if (n) {                                 /* the flat stone: pad, then lay */
        unsigned char stone[32];
        memset(stone, 0, sizeof stone);
        memcpy(stone, p, n);
        stone[n] = 0x01;
        s0 ^= ld64(stone);      s1 ^= ld64(stone + 8);
        s2 ^= ld64(stone + 16); s3 ^= ld64(stone + 24);
        STRIDE(s0, s1, s2, s3);
    }
    /* turn ninety degrees twice: roles rotate so no lane keeps its direction */
    STRIDE(s0, s1, s2, s3);
    STRIDE(s0, s2, s3, s1);
    STRIDE(s0, s3, s1, s2);

    uint64_t h = (s0 + ROTR64(s1, 11)) ^ (s2 + ROTR64(s3, 37));
    return owl_seal(h ^ (uint64_t)len);
}

#ifdef RIVERBANK_AVX2
/* ---- regime A: twin 4x4 grids, 256-byte strides ---- */
#define V_ADD(a,b)   _mm256_add_epi64(a, b)
#define V_XOR(a,b)   _mm256_xor_si256(a, b)
#define V_ROTR(x,n)  _mm256_or_si256(_mm256_srli_epi64(x, (n)),           \
                                     _mm256_slli_epi64(x, 64 - (n)))
#define V_ROTR32(x)  _mm256_shuffle_epi32(x, 0xB1)
#define V_LOAD(p)    _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define V4(w,x,y,z)  _mm256_set_epi64x((long long)(z), (long long)(y),     \
                                       (long long)(x), (long long)(w))

#define V_STRIDE(a,b,c,d) do {                                 \
    a = V_ADD(a,b); d = V_XOR(d,a); d = V_ROTR32(d);           \
    c = V_ADD(c,d); b = V_XOR(b,c); b = V_ROTR(b,24);          \
    a = V_ADD(a,b); d = V_XOR(d,a); d = V_ROTR(d,16);          \
    c = V_ADD(c,d); b = V_XOR(b,c); b = V_ROTR(b,63);          \
} while (0)

/* turning the shape ninety degrees, and turning it back */
#define V_TURN(b,c,d) do {                           \
    b = _mm256_permute4x64_epi64(b, 0x39);           \
    c = _mm256_permute4x64_epi64(c, 0x4E);           \
    d = _mm256_permute4x64_epi64(d, 0x93);           \
} while (0)
#define V_UNTURN(b,c,d) do {                         \
    b = _mm256_permute4x64_epi64(b, 0x93);           \
    c = _mm256_permute4x64_epi64(c, 0x4E);           \
    d = _mm256_permute4x64_epi64(d, 0x39);           \
} while (0)

/* columns in every direction, then diagonals: one full race of the heap */
#define V_ROUND(a,b,c,d) do {                        \
    V_STRIDE(a,b,c,d);                               \
    V_TURN(b,c,d);                                   \
    V_STRIDE(a,b,c,d);                               \
    V_UNTURN(b,c,d);                                 \
} while (0)

static uint64_t riverbank_grid(const unsigned char *restrict data, size_t len) {
    __m256i a0 = V4(RC(0),  RC(1),  RC(2),  RC(3));
    __m256i a1 = V4(RC(4),  RC(5),  RC(6),  RC(7));
    __m256i a2 = V4(RC(8),  RC(9),  RC(10), RC(11));
    __m256i a3 = V4(RC(12), RC(13), RC(14), RC(15));
    __m256i b0 = V4(RC(16), RC(17), RC(18), RC(19));
    __m256i b1 = V4(RC(20), RC(21), RC(22), RC(23));
    __m256i b2 = V4(RC(24), RC(25), RC(26), RC(27));
    __m256i b3 = V4(RC(28), RC(29), RC(30), RC(31));

    const unsigned char *p = data;
    size_t n = len;

    while (n >= 256) {
        /* the servant holds silence: eight pure loads, eight xors, no chain */
        a0 = V_XOR(a0, V_LOAD(p));         a1 = V_XOR(a1, V_LOAD(p + 32));
        a2 = V_XOR(a2, V_LOAD(p + 64));    a3 = V_XOR(a3, V_LOAD(p + 96));
        b0 = V_XOR(b0, V_LOAD(p + 128));   b1 = V_XOR(b1, V_LOAD(p + 160));
        b2 = V_XOR(b2, V_LOAD(p + 192));   b3 = V_XOR(b3, V_LOAD(p + 224));
        V_ROUND(a0, a1, a2, a3);           /* two heaps race side by side */
        V_ROUND(b0, b1, b2, b3);
        p += 256; n -= 256;
    }
    if (n) {
        unsigned char stone[256];
        memset(stone, 0, sizeof stone);
        memcpy(stone, p, n);
        stone[n] = 0x01;
        a0 = V_XOR(a0, V_LOAD(stone));        a1 = V_XOR(a1, V_LOAD(stone + 32));
        a2 = V_XOR(a2, V_LOAD(stone + 64));   a3 = V_XOR(a3, V_LOAD(stone + 96));
        b0 = V_XOR(b0, V_LOAD(stone + 128));  b1 = V_XOR(b1, V_LOAD(stone + 160));
        b2 = V_XOR(b2, V_LOAD(stone + 192));  b3 = V_XOR(b3, V_LOAD(stone + 224));
        V_ROUND(a0, a1, a2, a3);
        V_ROUND(b0, b1, b2, b3);
    }

    /* the shrine: the last heap has nowhere left to diffuse, so spend the
       rounds here instead of spending them on every block */
    __m256i L = _mm256_set1_epi64x((long long)(uint64_t)len);
    a0 = V_XOR(a0, L); b0 = V_XOR(b0, L);
    V_ROUND(a0, a1, a2, a3); V_ROUND(b0, b1, b2, b3);
    V_ROUND(a0, a1, a2, a3); V_ROUND(b0, b1, b2, b3);
    a0 = V_XOR(a0, b0); a1 = V_XOR(a1, b1);
    a2 = V_XOR(a2, b2); a3 = V_XOR(a3, b3);
    V_ROUND(a0, a1, a2, a3);
    V_ROUND(a0, a1, a2, a3);

    /* fold the corners into the centre; the heaps are burned */
    uint64_t t[16];
    _mm256_storeu_si256((__m256i *)(void *)(t + 0),  a0);
    _mm256_storeu_si256((__m256i *)(void *)(t + 4),  a1);
    _mm256_storeu_si256((__m256i *)(void *)(t + 8),  a2);
    _mm256_storeu_si256((__m256i *)(void *)(t + 12), a3);

    uint64_t h = (uint64_t)len;
    for (int i = 0; i < 16; i += 4)
        h ^= (t[i] + ROTR64(t[i + 1], 11)) ^ (t[i + 2] + ROTR64(t[i + 3], 37));
    return owl_seal(h);
}
#endif /* RIVERBANK_AVX2 */

uint64_t kernel(const unsigned char *data, size_t len) {
#ifdef RIVERBANK_AVX2
    if (len >= 256)                     /* enough ground for a full stride */
        return riverbank_grid(data, len);
#endif
    return riverbank_walk(data, len);   /* otherwise walk the riverbank */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 8**

(Stated before any measurement.) Reasoning, so it can be falsified cleanly:

- **Baseline.** FNV-1a is latency-bound on `xor → imul`: ~4 cycles/byte ≈ 0.9 GB/s at 3.5 GHz.
- **Grid path.** Critical path per 256 B is one `V_ROUND` on each grid; the two grids are independent, so they overlap. One `V_ROUND` = 2 `V_STRIDE` ≈ 26 cycles. 256 B / ~28 cycles ≈ 9 B/cycle ≈ 30 GB/s theoretical, but port-5 pressure (12 `permute4x64` + 4 `shuffle_epi32` per 256 B) caps it near ~4 B/cycle, and DRAM caps it near ~12 GB/s. Expect **6–14×**, hence 8.
- **Walk path** (`len < 256` or no AVX2): one `STRIDE` per 32 B ≈ 13 cycles → ~2.4 B/cycle, still ~7× FNV at those sizes; small-input fixed cost is ~3 strides + `fmix64` ≈ 50 cycles.
- **Avalanche:** predict **0.49–0.50**. Every output bit passes through `fmix64` of a value that depends on all 16 grid words; the only structurally weak position is a bit in the final (padded) block, which is why 4 full rounds sit after the last absorb rather than being spread over every block.

Falsifiers I would accept: avalanche < 0.47 would mean the round budget is misplaced, not that the layout is wrong — the repair is +1 round at the shrine, not +1 round per block. Speedup < 3 at large sizes would mean the shuffle ports, not the deer stride, are the real bottleneck.

## MEASUREMENT

**Not measured here — honestly, this has not been run.** No tools were available in this session: `hash_bench` and `hash_contract` could not be called, so I cannot report throughput or an avalanche number, and I did not get the "at most four improvements" loop. The pipeline will compile and measure the artifact above; the prediction stands as written and unrevised.

The improvement ladder I would have walked, in order, is pre-committed so it cannot be retrofitted to whatever the numbers turn out to be:

1. If avalanche < 0.47 → one more `V_ROUND` on grid A after the merge (cost O(1) per call, no throughput change at large sizes).
2. If throughput is below ~3 GB/s and `perf` shows port-5 saturation → replace `V_ROTR(x,24)`/`V_ROTR(x,16)` with `shuffle_epi8` masks *only if* shifts are the bottleneck, otherwise drop the diagonal pass to every other block (keeps the 4 shrine rounds).
3. If small sizes dominate the benchmark and lose → lower the twin-grid threshold is wrong; instead shorten the walk path's finalisation from 3 strides to 2.
4. If large sizes are bandwidth-bound → add `__builtin_prefetch(p + 1024)`; do **not** add OpenMP (see VERDICT).

## VERDICT

The native's method translates without any fudging: the swats that hold silence are a dependency-free absorb, the flat stone is a position→lane map, the deer's stride is an ARX permutation, the drifting man's squares are a 4×4 grid turned ninety degrees, and the owl's seal is `fmix64`. Nothing in the description had to be replaced by the textbook accumulator, and notably **no multiplication touches any input byte** — the two multiplies in the whole kernel are the owl's, once per call. The construction the metaphor lands on is not novel, which is the point: it is a sponge absorb with a BLAKE2b round and a Murmur finalizer, both validated, assembled in the layout the metaphor dictated.

Named risks, and how each is discharged rather than left as a caveat:

- **"Only helps if there is enough ground for a stride."** A 256-byte permutation state is pure overhead on a 12-byte key. Guarded explicitly: `len >= 256` gates the grid path; everything shorter takes the single-column walk, which shares the same `STRIDE` and the same seal, so there is one algorithmic family and two regimes, not two unrelated hashes. This is the runtime regime check the metaphor already contained — the deer walks when the path is too short.
- **"Thread parallelism."** Dropped, not guarded, because the metaphor's unit of work (one 256-byte heap, ~28 cycles) is orders of magnitude smaller than any fork, and merging grids across threads would make the hash depend on thread count. Vectorisation only: `restrict`, `loadu`, 8-wide independent absorb, 256-byte sequential stride for the prefetcher.
- **"No AVX2."** `kernel` falls through to the walk path, which is portable C and still ~2.4 B/cycle — correct and fast on any target, just not grid-wide.

What I am least confident about, stated plainly: the round *budget* (1 round per block, 4 at the shrine) is the one place where the metaphor gave a principle — spend rounds where a mark has nowhere left to smear, not uniformly — rather than a number. If the avalanche score comes back short, that principle is right and my arithmetic on it was wrong, and the fix is step 1 of the ladder above, not abandoning the layout.