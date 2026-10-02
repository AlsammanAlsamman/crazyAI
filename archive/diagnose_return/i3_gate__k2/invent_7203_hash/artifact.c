#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

/* ---- the four chained prisoners: they grind angles, they never multiply ---- */
#define PR0 7u
#define PR1 19u
#define PR2 31u
#define PR3 47u

/* ---- the lean of each anchored wire under one and the same flood ---- */
#define WB0 11u
#define WB1 29u
#define WB2 43u
#define WB3 57u

#define DRAIN     6      /* passes until the shadows stop chittering (chain depth 3 + margin) */
#define FLOOD_MIN 32u    /* below this the river never climbs to the wires */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* The one read at the pit's rim: splitmix64 / fmix64 avalanche, validated,
   used verbatim. Exactly two multiplications in the entire kernel. */
static inline uint64_t rim_read(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* which lean, which stand straight, which cross another -- plus the water line */
static inline uint64_t silhouette(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                                  uint64_t waterline) {
    uint64_t s = (rotl64(a, 17) ^ b) + (rotl64(c, 41) ^ d);
    s ^= rotl64(a + d, 23) ^ rotl64(b + c, 53);
    s ^= waterline * 0x9E3779B97F4A7C15ULL;
    return rim_read(s);
}

/* one stroke of the chain: the mark reaches only the nearest prisoner;
   every other stage receives the ANGLE its upstream neighbour held on the
   PREVIOUS pass -- so the four recurrences run in parallel (2 cycles each)
   instead of forming one long per-byte dependency chain. */
#define GRIND(WRD) do {                           \
    uint64_t _n0 = rotl64(g0, PR0) ^ (uint64_t)(WRD); \
    uint64_t _n1 = rotl64(g1, PR1) ^ g0;          \
    uint64_t _n2 = rotl64(g2, PR2) ^ g1;          \
    uint64_t _n3 = rotl64(g3, PR3) ^ g2;          \
    g0 = _n0; g1 = _n1; g2 = _n2; g3 = _n3;       \
} while (0)

#if defined(__AVX2__)
/* the flood is one event: it bends all four wires at once, each by its own lean.
   The bend is computed off the wires' critical path, so the wire recurrence is
   a single 1-cycle add. */
#  define FLOOD() do {                                                          \
      __m256i _b = _mm256_set1_epi64x((long long)g3);                           \
      V = _mm256_add_epi64(V, _mm256_or_si256(_mm256_sllv_epi64(_b, SHL),       \
                                              _mm256_srlv_epi64(_b, SHR)));     \
  } while (0)
#else
#  define FLOOD() do { uint64_t _a = g3;              \
      v0 += rotl64(_a, WB0); v1 += rotl64(_a, WB1);   \
      v2 += rotl64(_a, WB2); v3 += rotl64(_a, WB3);   \
  } while (0)
#endif

#define PASS(WRD) do { GRIND(WRD); FLOOD(); } while (0)

/* silence still has a texture: the drain is fed fixed non-zero constants so the
   chain cannot settle into a degenerate pattern while it empties. */
static const uint64_t SILENCE[DRAIN] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0xD6E8FEB86659FD93ULL, 0xA3B195354A39B70DULL, 0x1B03738712FAD5C9ULL
};

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t i = 0;
    uint64_t w;

    /* the blank pillar, as the four chained prisoners hold it */
    uint64_t g0 = 0x243F6A8885A308D3ULL;
    uint64_t g1 = 0x13198A2E03707344ULL;
    uint64_t g2 = 0xA4093822299F31D0ULL;
    uint64_t g3 = 0x082EFA98EC4E6C89ULL;

    /* ---------------- REGIME 1: the river never reaches the wires -----------
       Too few marks for a flood to be worth raising: no vector lanes, no
       drain, no broadcast. He grinds the pillar and reads its notches at the
       rim -- through the very same single nonlinear read, so short inputs keep
       full avalanche quality. */
    if (len < FLOOD_MIN) {
        for (; i + 8 <= len; i += 8) {
            memcpy(&w, p + i, 8);
            GRIND(w);
        }
        for (; i < len; i++) {           /* marks are fed singly */
            GRIND((uint64_t)p[i] + 0x9E3779B97F4A7C15ULL);
        }
        return silhouette(g0, g1, g2, g3, (uint64_t)len);
    }

    /* ---------------- REGIME 2: the flood rises ---------------------------- */
#if defined(__AVX2__)
    const __m256i SHL = _mm256_setr_epi64x(WB0, WB1, WB2, WB3);
    const __m256i SHR = _mm256_setr_epi64x(64 - WB0, 64 - WB1, 64 - WB2, 64 - WB3);
    __m256i V = _mm256_setr_epi64x((long long)0x452821E638D01377ULL,
                                   (long long)0xBE5466CF34E90C6CULL,
                                   (long long)0xC0AC29B7C97C50DDULL,
                                   (long long)0x3F84D5B5B5470917ULL);
    uint64_t v0, v1, v2, v3;
#else
    uint64_t v0 = 0x452821E638D01377ULL, v1 = 0xBE5466CF34E90C6CULL,
             v2 = 0xC0AC29B7C97C50DDULL, v3 = 0x3F84D5B5B5470917ULL;
#endif

    /* lead the marks down past the knowing walls */
    for (; i + 32 <= len; i += 32) {
        __builtin_prefetch(p + i + 384, 0, 0);
        memcpy(&w, p + i +  0, 8); PASS(w);
        memcpy(&w, p + i +  8, 8); PASS(w);
        memcpy(&w, p + i + 16, 8); PASS(w);
        memcpy(&w, p + i + 24, 8); PASS(w);
    }
    for (; i + 8 <= len; i += 8) {
        memcpy(&w, p + i, 8); PASS(w);
    }
    for (; i < len; i++) {                 /* the last marks, fed singly */
        PASS((uint64_t)p[i] + 0x9E3779B97F4A7C15ULL);
    }

    /* wait for the recession: never pull the token while the water is up.
       The angle at pass t carries the word from pass t-3, so the chain must be
       emptied before the wires have felt the final marks at all. */
    for (int d = 0; d < DRAIN; d++) {
        PASS(SILENCE[d]);
    }

#if defined(__AVX2__)
    {
        uint64_t vv[4];
        _mm256_storeu_si256((__m256i *)vv, V);
        v0 = vv[0]; v1 = vv[1]; v2 = vv[2]; v3 = vv[3];
    }
#endif

    /* climb to the rim, read the shadow-shape once, keep nothing else */
    return silhouette(v0, v1, v2, v3, (uint64_t)len);
}
