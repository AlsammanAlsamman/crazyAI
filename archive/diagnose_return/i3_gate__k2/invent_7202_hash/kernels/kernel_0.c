#include <stdint.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the stream: water runs over the clay until only ridges remain ----
   splitmix64's validated finalizer (Steele/Lea/Vigna). One O(1) epilogue;
   the per-mark path above it never multiplies. */
static inline uint64_t ridges(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    r &= 63u;
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* ---- one short path of smoke: the chain cannot span, walk marks straight.
   Multiply-free: XOR the mark in, then swing by a distance the mark itself
   sets (its low bits + a fixed short arc). ---- */
static uint64_t walk_short(const unsigned char *data, size_t len) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    for (size_t i = 0; i < len; i++) {
        unsigned char m = data[i];
        h ^= (uint64_t)m;
        /* the mark's own count sets the swing: short chain, bounded arc */
        h = rotl64(h, (unsigned)((m & 31u) + 7u));
        h ^= h >> 17;
    }
    return ridges(h ^ (uint64_t)len);
}

uint64_t kernel(const unsigned char *data, size_t len) {
#if defined(__AVX2__)
    /* smoke has not thinned: too few marks for the chain to span the piazza */
    if (len < 32) return walk_short(data, len);

    const unsigned char *restrict p = data;
    size_t n = len;

    /* the sailor-chain: four knots hung between fire and wall, distinct seeds
       so no two lanes are the same row */
    __m256i v = _mm256_set_epi64x(
        (int64_t)0x2545F4914F6CDD1DULL,
        (int64_t)0x9E3779B97F4A7C15ULL,
        (int64_t)0xC2B2AE3D27D4EB4FULL,
        (int64_t)0x165667B19E3779F9ULL);
    v = _mm256_xor_si256(v, _mm256_set1_epi64x((int64_t)len));

    const __m256i arc = _mm256_set1_epi64x(7);          /* short chain */
    const __m256i mask5 = _mm256_set1_epi64x(31);       /* bounded swing */
    const __m256i c64 = _mm256_set1_epi64x(64);

    while (n >= 32) {
        __m256i m = _mm256_loadu_si256((const __m256i *)p);

        /* the fire falls on the chain: every knot takes the mark */
        v = _mm256_xor_si256(v, m);

        /* the mark's own pebble-count sets the censer's swing, per lane.
           count the strokes of this group of marks, bound the arc. */
        __m256i cnt = _mm256_and_si256(_mm256_srli_epi64(m, 3), mask5);
        __m256i sw  = _mm256_add_epi64(cnt, arc);               /* 7..38 */
        __m256i isw = _mm256_sub_epi64(c64, sw);

        /* rotate: no multiplication anywhere on this path */
        v = _mm256_or_si256(_mm256_sllv_epi64(v, sw),
                            _mm256_srlv_epi64(v, isw));

        /* the whole line hangs together: couple the knots laterally so
           knot i's next swing sees knot i+1's content */
        v = _mm256_xor_si256(v, _mm256_permute4x64_epi64(v, 0x39)); /* 0,3,2,1 */

        /* a second swing whose distance comes from the chain itself, not a
           constant -- keeps the operator data-selected (nonlinear) */
        __m256i c2  = _mm256_and_si256(v, mask5);
        __m256i sw2 = _mm256_add_epi64(c2, arc);
        __m256i is2 = _mm256_sub_epi64(c64, sw2);
        v = _mm256_or_si256(_mm256_sllv_epi64(v, sw2),
                            _mm256_srlv_epi64(v, is2));
        v = _mm256_xor_si256(v, m);

        p += 32; n -= 32;
    }

    /* read the six fixed cracks: fold the hanging chain to one shadow */
    uint64_t lane[4];
    _mm256_storeu_si256((__m256i *)lane, v);
    uint64_t h = lane[0];
    h = rotl64(h, 13) ^ lane[1];
    h = rotl64(h, 29) ^ lane[2];
    h = rotl64(h, 41) ^ lane[3];

    /* the last few marks, walked straight */
    while (n > 0) {
        unsigned char mk = *p++;
        h ^= (uint64_t)mk;
        h = rotl64(h, (unsigned)((mk & 31u) + 7u));
        h ^= h >> 17;
        n--;
    }

    /* throw away the smoke and the cup; keep only the ridges */
    return ridges(h ^ (uint64_t)len);
#else
    return walk_short(data, len);
#endif
}
