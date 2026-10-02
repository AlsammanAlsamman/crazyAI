## MAPPING

### SEED 1 — "the servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded"

| world object | problem object |
|---|---|
| pile of marks, in their given order | the input buffer `data[0..len)` |
| walking marks past the servant, one at a time | the *load* stream — bytes are fetched, nothing else |
| servant with the inflated house on a stick | the barrier between loading and mixing |
| the swat holding my mouth shut | **no arithmetic is permitted while marks are being laid** — zero mixing during absorption |
| the flat stone at the riverbank | a 128-byte block laid into 16 persistent 64-bit state words |
| "the order itself never gets spoken aloud and lost" | byte order is recorded by **position** (which word, which byte of it) rather than by **time** (when it was folded) |
| folding only begins once the stone is full | exactly one permutation per block, after the whole block is XOR-absorbed |

Breaks: *"each byte must be mixed into the running state before the next byte is read."* And as a direct consequence it breaks *"the whole buffer must be read once, start to end, in order"* — once order is carried positionally, the 16 words (4 SIMD lanes × 4 rows) are consumed **simultaneously**, and the final partial heap is read as an *overlapping* last-128-bytes window rather than in sequence.

### SEED 2 — "each heap races the musk deer's mountain stride"

| world object | problem object |
|---|---|
| musk deer, fixed strides, gathering ground beneath it | fixed 128-byte stride over the buffer |
| uphill twist | `rotl64` (32, 24, 16, 63 — BLAKE2b's constants) |
| downhill fold | `xor` |
| doubling-back that eats its own trail | `a += b; d ^= a;` — state overwritten by a function of itself, **no multiply** |
| no heap keeps the shape it entered with | full diffusion inside the block |
| the final stone at the shrine, carried forward, never allowed to ossify | the chaining word `s[15]`, into which a block counter is added so the state can never settle into a fixed point |

Breaks: *"mixing one byte requires a multiplication"* and *"the state is a single accumulator updated in place, one value."*

### SEED 3 — "the owl calls the final shape inside the dreamer inside; every intermediate heap is burned"

| world object | problem object |
|---|---|
| the owl taking the pen | the finalizer, run once |
| the dreamer inside the dreamer | nested mixing: wide ARX double-rounds, then a 64-bit multiply-xorshift mixer nested inside them (`fmix64`'s two stages) |
| one fixed size regardless of how many marks | the 64-bit return value |
| intermediate heaps burned the moment the next swallows them | state overwritten in place; no scratch buffer kept; 16→4→1 lossy fold |
| the man who cannot recall being a man | non-injective compression — unrecoverable from the token |

Breaks: *"more mixing rounds always means better mixing"* — per-heap mixing is deliberately kept to the **minimum** (one double round), and mixing is *concentrated* at the shrine.

Two further pieces of the telling are not in the seeds but are load-bearing, and I take them literally: **"I search the result in squares, the drifting man searching squares in every direction, turning the shape ninety degrees and folding its corners into its own center"** = a **4×4 square of 64-bit words, alternating column rounds and diagonal rounds** — this is exactly the ChaCha/BLAKE2b permutation, and the 90° turn is literally ChaCha's diagonalization. And **"I test this by changing one mark and watching the whole grid rearrange itself"** = the avalanche test itself.

## CHOSEN SEED

**SEED 1.** It is the one that breaks the preferred assumption ("read once, start to end, in order") — not by reading bytes twice, but by making order *positional instead of temporal*, which is what licenses lane-parallel consumption and an overlapping tail window. It is also the most literal: the swat-induced silence maps onto a single mechanical rule ("no ALU op between the first and last byte of a block"), which is precisely the rule FNV-1a violates on every byte. SEEDs 2 and 3 are not discarded — they are the round and the shrine of the same working, and the kernel implements all three.

## ASSUMPTION BROKEN

Primary: **each byte must be mixed into the running state before the next byte is read.** Consequent: **the whole buffer must be read once, start to end, in order** — it is read as 16 positional slots at a time, and the ragged end is read as an overlapping window, not in sequence. Secondary (SEED 2): **mixing one byte requires a multiplication** — there are zero multiplies in the per-byte path; the only multiplies are O(1), in the owl's seal. Secondary (SEED 3): **more rounds always means better mixing** — one double round per heap, mixing concentrated at the shrine.

Where the mechanism lands on a validated technique, I let it land there rather than invent: the square with column+diagonal ARX rounds **is** ChaCha/BLAKE2b's permutation (rotations 32/24/16/63 are BLAKE2b's; the diagonal indices are ChaCha's); the owl's seal **is** the splitmix64 finalizer; the short-input tiering with overlapping head/tail reads **is** xxHash3/wyhash's small-key strategy; the constants are the SHA-512/BLAKE2b IVs.

**Regime recognition, in-metaphor** (required, because known_way spans byte-at-a-time FNV and block-wise xxHash): *"I gather the marks into small heaps of even count"* — so the first thing the native does at the riverbank is see whether the pile can fill a heap at all. Three gatherings: a **handful** (`len < 16` — too few to race, carried straight to the owl), a **partial heap** (`16 ≤ len < 128` — the single-column stone, 32-byte strides), **full heaps** (`len ≥ 128` — the 4×4 stone, SIMD). This is also the guard demanded by step 4: the wide-square machinery, which is the part that could lose to FNV on tiny inputs, never runs on tiny inputs. No thread parallelism: the metaphor's own unit of work is one 128-byte heap, which is orders of magnitude below a thread's worth, and a thread-count-dependent merge would make the hash non-deterministic. Vectorization only.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* nothing-up-my-sleeve constants: SHA-512 / BLAKE2b / SHA-384 IVs */
static const uint64_t IVC[16] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL,
    0xCBBB9D5DC1059ED8ULL, 0x629A292A367CD507ULL, 0x9159015A3070DD17ULL, 0x152FECD8F70E5939ULL,
    0x67332667FFC00B31ULL, 0x8EB44A8768581511ULL, 0xDB0C2E0D64F98FA7ULL, 0x47B5481DBEFA4FA4ULL
};
#define GOLD 0x9E3779B97F4A7C15ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* the owl's seal: splitmix64 finalizer -- the dreamer inside the dreamer */
static inline uint64_t fmix64(uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31; return z;
}

/* the musk deer's stride: uphill twist (rotl), downhill fold (xor),
   doubling-back that eats its own trail (state <- f(state)).  No multiply. */
#define QR(a,b,c,d) do {                              \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 32);     \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 24);     \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 16);     \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 63);     \
} while (0)

/* searching the result in squares: column round, then the ninety-degree
   turn that folds the corners into the centre (ChaCha's diagonals). */
#define DR(s) do {                            \
    QR((s)[0],(s)[4],(s)[ 8],(s)[12]);        \
    QR((s)[1],(s)[5],(s)[ 9],(s)[13]);        \
    QR((s)[2],(s)[6],(s)[10],(s)[14]);        \
    QR((s)[3],(s)[7],(s)[11],(s)[15]);        \
    QR((s)[0],(s)[5],(s)[10],(s)[15]);        \
    QR((s)[1],(s)[6],(s)[11],(s)[12]);        \
    QR((s)[2],(s)[7],(s)[ 8],(s)[13]);        \
    QR((s)[3],(s)[4],(s)[ 9],(s)[14]);        \
} while (0)

/* diagonal fold of the 4x4 stone into one column, then the seal */
static inline uint64_t seal(uint64_t *s, size_t len) {
    uint64_t a, b, c, d;
    s[0] ^= (uint64_t)len;
    s[7] += (uint64_t)len;
    DR(s); DR(s); DR(s);                 /* mixing concentrated at the shrine */
    a = s[0] ^ s[5] ^ s[10] ^ s[15];
    b = s[1] ^ s[6] ^ s[11] ^ s[12];
    c = s[2] ^ s[7] ^ s[ 8] ^ s[13];
    d = s[3] ^ s[4] ^ s[ 9] ^ s[14];
    QR(a, b, c, d);
    return fmix64((a ^ rotl64(c, 32)) + (b ^ rotl64(d, 19)));
}

/* ---- regime: a partial heap (16 <= len < 128): the single-column stone ---- */
static uint64_t small_square(const unsigned char *restrict d, size_t len) {
    uint64_t a = IVC[0] ^ (uint64_t)len;
    uint64_t b = IVC[1];
    uint64_t c = IVC[2];
    uint64_t e = IVC[3] ^ ((uint64_t)len << 32);
    size_t i = 0;
    while (len - i >= 32) {                     /* lay 4 marks, then race once */
        a ^= ld64(d + i);      b ^= ld64(d + i + 8);
        c ^= ld64(d + i + 16); e ^= ld64(d + i + 24);
        QR(a, b, c, e); QR(b, c, e, a);
        i += 32;
    }
    if (len - i >= 16) { a ^= ld64(d + i); b ^= ld64(d + i + 8); }
    c ^= ld64(d + len - 16);                    /* overlapping tail window */
    e ^= ld64(d + len - 8);
    QR(a, b, c, e); QR(b, c, e, a); QR(c, e, a, b); QR(e, a, b, c);
    return fmix64((a ^ rotl64(c, 32)) + (b ^ rotl64(e, 19)));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    uint64_t s[16];
    size_t nb, b;
    int i;

    /* ---- at the riverbank: can this pile fill a heap at all? ---- */
    if (len < 16) {                 /* a handful -- straight to the owl */
        uint64_t x;
        if (len >= 8) {
            x = (ld64(p) ^ IVC[0]) + rotl64(ld64(p + len - 8) ^ IVC[1], 32);
        } else if (len >= 4) {
            x = ((uint64_t)ld32(p) << 32) | (uint64_t)ld32(p + len - 4);
        } else if (len) {
            x = (uint64_t)p[0] | ((uint64_t)p[len >> 1] << 8)
              | ((uint64_t)p[len - 1] << 16);
        } else {
            x = 0;
        }
        return fmix64(x ^ ((uint64_t)len * GOLD));
    }
    if (len < 128) return small_square(p, len);

    for (i = 0; i < 16; i++) s[i] = IVC[i];
    s[0] ^= (uint64_t)len;
    nb = len >> 7;

#if defined(__AVX2__)
    {
        const __m256i r24 = _mm256_setr_epi8(5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12,
                                             5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12);
        const __m256i r16 = _mm256_setr_epi8(6,7,0,1,2,3,4,5, 14,15,8,9,10,11,12,13,
                                             6,7,0,1,2,3,4,5, 14,15,8,9,10,11,12,13);
        const __m256i inc = _mm256_setr_epi64x(0, 0, 0, 1);
        __m256i ctr = _mm256_setzero_si256();
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(s + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(s + 4));
        __m256i v2 = _mm256_loadu_si256((const __m256i *)(s + 8));
        __m256i v3 = _mm256_loadu_si256((const __m256i *)(s + 12));
#define VR32(x) _mm256_shuffle_epi32((x), 0xB1)
#define VR24(x) _mm256_shuffle_epi8((x), r24)
#define VR16(x) _mm256_shuffle_epi8((x), r16)
#define VR63(x) _mm256_or_si256(_mm256_slli_epi64((x),63), _mm256_srli_epi64((x),1))
#define VQR(a,b,c,d) do {                                                   \
    a = _mm256_add_epi64(a,b); d = _mm256_xor_si256(d,a); d = VR32(d);      \
    c = _mm256_add_epi64(c,d); b = _mm256_xor_si256(b,c); b = VR24(b);      \
    a = _mm256_add_epi64(a,b); d = _mm256_xor_si256(d,a); d = VR16(d);      \
    c = _mm256_add_epi64(c,d); b = _mm256_xor_si256(b,c); b = VR63(b);      \
} while (0)
        for (b = 0; b < nb; b++) {
            const unsigned char *q = p + (b << 7);
            /* silence: the whole heap is laid on the stone before any racing */
            v0 = _mm256_xor_si256(v0, _mm256_loadu_si256((const __m256i *)(q +  0)));
            v1 = _mm256_xor_si256(v1, _mm256_loadu_si256((const __m256i *)(q + 32)));
            v2 = _mm256_xor_si256(v2, _mm256_loadu_si256((const __m256i *)(q + 64)));
            v3 = _mm256_xor_si256(v3, _mm256_loadu_si256((const __m256i *)(q + 96)));
            ctr = _mm256_add_epi64(ctr, inc);   /* the stone that never ossifies */
            v3  = _mm256_add_epi64(v3, ctr);
            VQR(v0, v1, v2, v3);                            /* columns */
            v1 = _mm256_permute4x64_epi64(v1, 0x39);        /* ninety degrees */
            v2 = _mm256_permute4x64_epi64(v2, 0x4E);
            v3 = _mm256_permute4x64_epi64(v3, 0x93);
            VQR(v0, v1, v2, v3);                            /* diagonals */
            v1 = _mm256_permute4x64_epi64(v1, 0x93);
            v2 = _mm256_permute4x64_epi64(v2, 0x4E);
            v3 = _mm256_permute4x64_epi64(v3, 0x39);
        }
        _mm256_storeu_si256((__m256i *)(s +  0), v0);
        _mm256_storeu_si256((__m256i *)(s +  4), v1);
        _mm256_storeu_si256((__m256i *)(s +  8), v2);
        _mm256_storeu_si256((__m256i *)(s + 12), v3);
#undef VQR
#undef VR63
#undef VR16
#undef VR24
#undef VR32
    }
#else
    for (b = 0; b < nb; b++) {
        const unsigned char *q = p + (b << 7);
        for (i = 0; i < 16; i++) s[i] ^= ld64(q + 8 * i);
        s[15] += (uint64_t)(b + 1);
        DR(s);
    }
#endif

    if ((nb << 7) < len) {                 /* ragged end: overlapping last heap */
        const unsigned char *q = p + len - 128;
        for (i = 0; i < 16; i++) s[i] ^= ld64(q + 8 * i);
        s[15] += (uint64_t)(nb + 1);
        DR(s);
    }
    return seal(s, len);
}
```

The scalar and AVX2 block loops are semantically identical (verified by hand: rotating row 1 by one lane, row 2 by two, row 3 by three turns the column quarter-rounds into exactly ChaCha's diagonal set `(0,5,10,15) (1,6,11,12) (2,7,8,13) (3,4,9,14)`), so the kernel returns the same value whether or not AVX2 is present.

## PREDICTION

**PREDICTION: speedup_vs_dp = 18**

Reasoning stated before any measurement: FNV-1a is a serial `imul`→`xor` dependency chain, ≈4–5 cycles/byte (~0.22 B/cycle). The square path issues ≈44 vector µops per 128 bytes with port-5 shuffle pressure as the limit, ≈14–18 cycles/block → ≈7–9 B/cycle in cache, falling to memory bandwidth above L2. That is 30–40× in cache, ~12–25× out of cache, and only ~3–5× on sub-128-byte inputs, so an aggregate near 18 is my honest point estimate. Avalanche I predict at essentially ideal (score ≈0.5 bit-flip probability, bias well under 1%): every input bit passes through at least 4 double rounds (8 ARX rounds — ChaCha reaches full diffusion in 3–4) *and* the splitmix64 seal, whose own avalanche bias is <0.1%. The failure mode I would expect, if any, is throughput on the short tiers, not quality.

## MEASUREMENT

**Not performed.** This session was invoked with no tools available — `hash_bench` and `hash_contract` could not be called, so I have neither a measured throughput, a measured avalanche score, nor any of the four permitted improvement rounds. I am reporting that plainly rather than inventing numbers: everything in PREDICTION is an a-priori estimate from µop counting and from the published diffusion properties of the ChaCha/BLAKE2b round and the splitmix64 finalizer, and none of it has been confirmed on this machine. The pipeline's numbers supersede mine, including if they contradict me.

What I would check first, in order: (1) throughput at 16–127 bytes, where `small_square`'s four finalizing quarter-rounds are a dependency chain and could lose to FNV around len≈24–40 — if so, the fix is to drop to two quarter-rounds below 32 bytes, since the seal already carries the avalanche; (2) whether the shuffle-based rotations (`VR24`/`VR16`) actually beat generic shift-or on this core, since port 5 is the predicted bottleneck; (3) whether one double round per heap is enough, by flipping a bit in the *last* full block specifically — that is the bit with the least mixing and the first place a weak avalanche would show.

## VERDICT

The literal translation holds together without being bent toward the textbook: the swat-enforced silence is a real mechanical rule (zero ALU work between a block's first and last byte), the deer's stride is genuinely multiply-free per byte, the "squares in every direction, turned ninety degrees, corners folded into the centre" is a real 4×4 matrix with real column and diagonal rounds, and the owl's seal is a real one-shot fixed-width compression. The native's method, taken at face value, *is* the ChaCha/BLAKE2b permutation used as a compression function with xxHash3-style size tiering — which is the outcome I wanted: the metaphor arrived at validated techniques instead of producing a novel untested mixer.

Named risk and how it is addressed, not just noted: the wide-square machinery has fixed setup cost (16-word state, three finalizing double rounds) that must lose to byte-at-a-time FNV on short inputs. That risk is guarded structurally, not by hope — `len < 16` never touches the square at all (one multiply plus one `fmix64`, bijective in the bytes-plus-length for `len < 8`, so it is both faster and *better* than FNV there), and `16 ≤ len < 128` uses the single-column stone with no 128-byte block machinery. Thread parallelism is deliberately absent: the metaphor's unit of work is a 128-byte heap, far too small to pay for a thread, and a thread-count-dependent merge would break determinism — the instruction's preference for vectorization before threads coincides here with the metaphor's own grain.

Two honest weaknesses. First, the kernel assumes a little-endian host with byte-addressable unaligned loads (true for the stated `gcc -march=native` x86 target; the `memcpy`-based loads are strict-aliasing-safe but not endian-neutral, so the hash *value* would differ on a big-endian machine, though its quality would not). Second, and more importantly, **the measurement that would make this claim worth anything is missing**. A hash design with a plausible µop count and no benchmark is a hypothesis, not a result. If the pipeline reports speedup materially below ~8, the first suspect is the short/medium tiers, and the second is that the benchmark is bandwidth-bound — in which case no amount of round-tuning will help and the correct report is that the known way was already at the memory wall.