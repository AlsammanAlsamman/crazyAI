#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- constants: high-entropy digits of pi (Blowfish S-box tail) ---------- */
static const uint64_t RIVER[24] = {
    0xD1310BA698DFB5ACULL, 0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL, 0x0801F2E2858EFC16ULL,
    0x636920D871574E69ULL, 0xA458FEA3F4933D7EULL, 0x0D95748F728EB658ULL,
    0x718BCD5882154AEEULL, 0x7B54A41DC25A59B5ULL, 0x9C30D5392AF26013ULL,
    0xC5D1B023286085F0ULL, 0xCA417918B8DB38EFULL, 0x8E79DCB0603A180EULL,
    0x6C9E0E8BB01E8A3EULL, 0xD71577C1BD314B27ULL, 0x78AF2FDA55605C60ULL,
    0xE65525F3AA55AB94ULL, 0x5748986263E81440ULL, 0x55CA396A2AAB10B6ULL,
    0xB4CC5C341141E8CEULL, 0xA15486AF7C72E993ULL, 0xB3EE1411636FBC2AULL
};
#define K_P64_1 0x9E3779B185EBCA87ULL
#define K_P64_2 0xC2B2AE3D27D4EB4FULL
#define K_P64_3 0x165667B19E3779F9ULL
#define K_P64_4 0x85EBCA77C2B2AE63ULL
#define K_P64_5 0x27D4EB2F165667C5ULL
#define K_P32_1 0x9E3779B1U

/* ---- the flapper: a pure load. no arithmetic while a mark is placed ------ */
static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t rotl64(uint64_t x, unsigned r){ return (x<<r)|(x>>((64-r)&63)); }

/* ---- ninety degrees, corners folded into the center --------------------- */
static inline uint64_t fold128(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t ll=(a&0xFFFFFFFFULL)*(b&0xFFFFFFFFULL);
    uint64_t hl=(a>>32)*(b&0xFFFFFFFFULL);
    uint64_t lh=(a&0xFFFFFFFFULL)*(b>>32);
    uint64_t hh=(a>>32)*(b>>32);
    uint64_t cross=(ll>>32)+(hl&0xFFFFFFFFULL)+lh;
    uint64_t up=(hl>>32)+(cross>>32)+hh;
    uint64_t lo=(cross<<32)|(ll&0xFFFFFFFFULL);
    return lo ^ up;
#endif
}

/* ---- the owl: seals any shape to one fixed size, one way (moremur) ------ */
static inline uint64_t owl(uint64_t h){
    h ^= h >> 27; h *= 0x3C79AC492BA7B653ULL;
    h ^= h >> 33; h *= 0x1C69B3F74AC4AE35ULL;
    h ^= h >> 27;
    return h;
}

/* ---- the musk deer's stride over one heap: 8 lanes, doubling back ------- */
static inline void heap_race(uint64_t *acc, const unsigned char *p, const uint64_t *sec){
    int j;
    for (j = 0; j < 8; j++){
        uint64_t d = rd64(p + 8*j);
        uint64_t v = d ^ sec[j];
        acc[j ^ 1] += d;                                  /* eats its own trail */
        acc[j]     += (uint64_t)(uint32_t)v * (uint64_t)(v >> 32);
    }
}
/* ---- burned at the shrine: no heap keeps the shape it entered with ------ */
static inline void burn(uint64_t *acc, const uint64_t *sec){
    int j;
    for (j = 0; j < 8; j++){
        uint64_t v = acc[j];
        v ^= v >> 47;                 /* downhill fold */
        v ^= sec[j];
        v *= (uint64_t)K_P32_1;       /* uphill twist  */
        acc[j] = v;
    }
}

#if defined(__AVX2__)
static inline void heap_race_v(__m256i *x, const unsigned char *p, const uint64_t *sec){
    int k;
    for (k = 0; k < 2; k++){
        __m256i dv   = _mm256_loadu_si256((const __m256i*)(const void*)(p + 32*k));
        __m256i kv   = _mm256_loadu_si256((const __m256i*)(const void*)(sec + 4*k));
        __m256i dk   = _mm256_xor_si256(dv, kv);
        __m256i dkh  = _mm256_srli_epi64(dk, 32);
        __m256i prod = _mm256_mul_epu32(dk, dkh);
        __m256i swap = _mm256_shuffle_epi32(dv, _MM_SHUFFLE(1,0,3,2));
        x[k] = _mm256_add_epi64(prod, _mm256_add_epi64(x[k], swap));
    }
}
static inline void burn_v(__m256i *x, const uint64_t *sec){
    const __m256i pr = _mm256_set1_epi32((int)K_P32_1);
    int k;
    for (k = 0; k < 2; k++){
        __m256i a  = x[k];
        __m256i dv = _mm256_xor_si256(a, _mm256_srli_epi64(a, 47));
        __m256i kv = _mm256_loadu_si256((const __m256i*)(const void*)(sec + 4*k));
        __m256i dk = _mm256_xor_si256(dv, kv);
        __m256i lo = _mm256_mul_epu32(dk, pr);
        __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(dk, 32), pr);
        x[k] = _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32));
    }
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *__restrict p = data;

    /* ===== regime 1: too few marks to make a heap — straight to the owl == */
    if (len <= 16){
        if (len > 8){
            uint64_t a = rd64(p) ^ RIVER[0];
            uint64_t b = rd64(p + len - 8) ^ RIVER[1];
            return owl(fold128(a,b) + rotl64(a,23) + rotl64(b,41)
                       + (uint64_t)len * K_P64_5);
        }
        if (len >= 4){
            uint64_t a = (uint64_t)rd32(p);
            uint64_t b = (uint64_t)rd32(p + len - 4);
            return owl(((a << 32) | b) ^ RIVER[2] ^ ((uint64_t)len * K_P64_5));
        }
        if (len){
            uint64_t c = ((uint64_t)p[0] << 16)
                       | ((uint64_t)p[len >> 1] << 8)
                       |  (uint64_t)p[len - 1]
                       | ((uint64_t)len << 24);
            return owl(c ^ RIVER[3]);
        }
        return owl(RIVER[4]);
    }

    /* ===== regime 2: a handful of half-heaps (17..128 bytes) ============= */
    if (len <= 128){
        uint64_t h0 = (uint64_t)len * K_P64_1;
        uint64_t h1 = ~(uint64_t)len * K_P64_2;
        size_t nb = len >> 4, k;
        for (k = 0; k < nb; k++){
            const unsigned char *q = p + (k << 4);
            uint64_t lo = rd64(q), hi = rd64(q + 8);
            h0 += fold128(lo ^ RIVER[k], hi ^ RIVER[k + 8]);
            h1 ^= rotl64(lo + hi + RIVER[k + 16], 31) * K_P64_2;
        }
        {   const unsigned char *t = p + len - 16;
            uint64_t lo = rd64(t), hi = rd64(t + 8);
            h0 += fold128(lo ^ RIVER[16], hi ^ RIVER[17]);
            h1 ^= rotl64(lo ^ hi, 17) * K_P64_4;
        }
        return owl(h0 ^ rotl64(h1, 37));
    }

    /* ===== regime 3: the full mountain path — 8 lanes, out of order ====== */
    {
        uint64_t acc[8] = { 0x9E3779B1ULL, K_P64_1, K_P64_2, K_P64_3,
                            K_P64_4, 0xC2B2AE3DULL, K_P64_5, 0x85EBCA77ULL };
        const size_t nbStripes = (len - 1) / 64;   /* last 64B handled apart */
        const size_t SPB = 16;                     /* heaps per shrine-burn  */
        size_t s = 0, run, k;
#if defined(__AVX2__)
        __m256i x[2];
        x[0] = _mm256_loadu_si256((const __m256i*)(const void*)(acc + 0));
        x[1] = _mm256_loadu_si256((const __m256i*)(const void*)(acc + 4));
        while (s < nbStripes){
            run = nbStripes - s; if (run > SPB) run = SPB;
            for (k = 0; k < run; k++) heap_race_v(x, p + (s + k) * 64, RIVER + k);
            s += run;
            if (run == SPB) burn_v(x, RIVER + 16);
        }
        heap_race_v(x, p + len - 64, RIVER + 15);
        _mm256_storeu_si256((__m256i*)(void*)(acc + 0), x[0]);
        _mm256_storeu_si256((__m256i*)(void*)(acc + 4), x[1]);
#else
        while (s < nbStripes){
            run = nbStripes - s; if (run > SPB) run = SPB;
            for (k = 0; k < run; k++) heap_race(acc, p + (s + k) * 64, RIVER + k);
            s += run;
            if (run == SPB) burn(acc, RIVER + 16);
        }
        heap_race(acc, p + len - 64, RIVER + 15);
#endif
        {   /* ninety degrees, corners folded into the center, then the owl */
            uint64_t h = (uint64_t)len * K_P64_1;
            h += fold128(acc[0] ^ RIVER[0], acc[1] ^ RIVER[1]);
            h += fold128(acc[2] ^ RIVER[2], acc[3] ^ RIVER[3]);
            h += fold128(acc[4] ^ RIVER[4], acc[5] ^ RIVER[5]);
            h += fold128(acc[6] ^ RIVER[6], acc[7] ^ RIVER[7]);
            return owl(h);
        }
    }
}
