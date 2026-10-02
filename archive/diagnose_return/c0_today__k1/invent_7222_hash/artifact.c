#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------- the chalk-bank groove: 16 odd, high-entropy keeps ------- */
static const uint64_t GROOVE[16] = {
  0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
  0x85EBCA77C2B2AE63ULL, 0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0x9FB21C651E98DF25ULL,
  0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL,
  0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL, 0xEB44ACCAB455D165ULL, 0x4D5A2DA51DE1AA47ULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t rotl64(uint64_t x, int r){ return (x << r) | (x >> (64 - r)); }

/* fold a 128-bit product back onto one keep */
static inline uint64_t fold128(uint64_t a, uint64_t b){
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
}

/* reading the stone's seated position against the wooden numbered keeps
   (Murmur3 fmix64 - validated, bias well under 1%) */
static inline uint64_t keeps(uint64_t x){
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

/* ================= regime A: "a handful" (len < 64) =================
   the stone is seated once; a plain serial press, no wide setup.      */
static uint64_t handful(const unsigned char *d, size_t len){
    uint64_t h = GROOVE[0] ^ ((uint64_t)len * GROOVE[1]);
    if (len >= 8){
        size_t i = 0;
        for (; i + 8 <= len; i += 8){
            h ^= fold128(rd64(d + i) ^ GROOVE[2], h + GROOVE[3]);
            h  = rotl64(h, 27) + GROOVE[4];
        }
        if (i != len)                                   /* overlapping last 8 */
            h ^= fold128(rd64(d + len - 8) ^ GROOVE[5], h + GROOVE[6]);
    } else if (len >= 4){
        uint64_t a = (uint64_t)rd32(d), b = (uint64_t)rd32(d + len - 4);
        h ^= fold128((a + (b << 32)) ^ GROOVE[2], h + GROOVE[3]);
    } else if (len > 0){
        uint64_t k = ((uint64_t)d[0] << 16) | ((uint64_t)d[len >> 1] << 8) | (uint64_t)d[len - 1];
        h ^= fold128(k ^ GROOVE[2], h + GROOVE[3]);
    }
    return keeps(h);
}

#if defined(__AVX2__)
/* one seating of the four-faced stone: 32 marks pressed into the
   already-turned position (callus), one fold, nothing washed clean. */
static inline __m256i seat(__m256i acc, const unsigned char *p,
                           __m256i key, __m256i callus){
    __m256i d    = _mm256_loadu_si256((const __m256i *)p);
    __m256i dk   = _mm256_xor_si256(_mm256_xor_si256(d, key), callus);
    __m256i prod = _mm256_mul_epu32(dk, _mm256_srli_epi64(dk, 32)); /* lo32*hi32 */
    __m256i swap = _mm256_shuffle_epi32(d, 0x4E);   /* face i takes mark i^1 */
    /* one add-latency on the acc chain: prod+swap are off the chain */
    return _mm256_add_epi64(acc, _mm256_add_epi64(prod, swap));
}
#else
static inline void seat_s(uint64_t *v, const unsigned char *p,
                          const uint64_t *key, const uint64_t *callus){
    uint64_t d0 = rd64(p), d1 = rd64(p+8), d2 = rd64(p+16), d3 = rd64(p+24);
    uint64_t a0 = d0 ^ key[0] ^ callus[0], a1 = d1 ^ key[1] ^ callus[1];
    uint64_t a2 = d2 ^ key[2] ^ callus[2], a3 = d3 ^ key[3] ^ callus[3];
    v[0] += (uint64_t)(uint32_t)a0 * (uint32_t)(a0 >> 32) + d1;
    v[1] += (uint64_t)(uint32_t)a1 * (uint32_t)(a1 >> 32) + d0;
    v[2] += (uint64_t)(uint32_t)a2 * (uint32_t)(a2 >> 32) + d3;
    v[3] += (uint64_t)(uint32_t)a3 * (uint32_t)(a3 >> 32) + d2;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* --- regime check: a handful folds on the simple path --- */
    if (len < 64) return handful(data, len);

    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t lane[4];

#if defined(__AVX2__)
    const __m256i k0 = _mm256_loadu_si256((const __m256i *)(GROOVE + 0));
    const __m256i k1 = _mm256_loadu_si256((const __m256i *)(GROOVE + 4));
    const __m256i k2 = _mm256_loadu_si256((const __m256i *)(GROOVE + 8));
    const __m256i k3 = _mm256_loadu_si256((const __m256i *)(GROOVE + 12));

    __m256i acc = _mm256_set_epi64x((long long)GROOVE[3], (long long)GROOVE[2],
                                    (long long)GROOVE[1], (long long)GROOVE[0]);
    __m256i callus = acc, callus_next = acc;

    /* one full revolution of the stone = four quarter-turns = 128 marks */
    while ((size_t)(end - p) >= 128){
        __m256i c = callus;
        acc = seat(acc, p +   0, k0, c);
        acc = seat(acc, p +  32, k1, c);
        acc = seat(acc, p +  64, k2, c);
        acc = seat(acc, p +  96, k3, c);
        callus      = callus_next;                              /* lagged callus */
        callus_next = _mm256_permute4x64_epi64(acc, 0x93);      /* the quarter turn */
        p += 128;
    }
    {
        __m256i c = callus;
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k0, c); p += 32; }
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k1, c); p += 32; }
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k2, c); p += 32; }
        if (p != end)                 acc = seat(acc, end - 32, k3, c); /* overlap */
    }
    _mm256_storeu_si256((__m256i *)lane, acc);
#else
    uint64_t v[4]  = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };
    uint64_t c[4]  = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };
    uint64_t c2[4] = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };

    while ((size_t)(end - p) >= 128){
        seat_s(v, p +   0, GROOVE +  0, c);
        seat_s(v, p +  32, GROOVE +  4, c);
        seat_s(v, p +  64, GROOVE +  8, c);
        seat_s(v, p +  96, GROOVE + 12, c);
        c[0]=c2[0]; c[1]=c2[1]; c[2]=c2[2]; c[3]=c2[3];
        c2[0]=v[3]; c2[1]=v[0]; c2[2]=v[1]; c2[3]=v[2];   /* the quarter turn */
        p += 128;
    }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  0, c); p += 32; }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  4, c); p += 32; }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  8, c); p += 32; }
    if (p != end)                 seat_s(v, end - 32, GROOVE + 12, c);
    lane[0]=v[0]; lane[1]=v[1]; lane[2]=v[2]; lane[3]=v[3];
#endif

    /* lift the stone: only the final seated number leaves the desk.
       192 of the 256 state bits are swept off and discarded. */
    uint64_t h = (uint64_t)len * GROOVE[5];
    h += fold128(lane[0] ^ GROOVE[6],  lane[1] ^ GROOVE[7]);
    h += fold128(lane[2] ^ GROOVE[8],  lane[3] ^ GROOVE[9]);
    return keeps(h);
}
