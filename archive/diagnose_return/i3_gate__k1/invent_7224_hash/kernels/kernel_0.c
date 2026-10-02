#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

/* The eight stones at the shrine: distinct, nonzero.  Nonzero is the
   anti-ossification guard -- x=0 is not a fixed point of a lane step, so a
   buffer of zeros still moves the state every heap. (pi fraction digits) */
static const uint64_t HK[8] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t x; memcpy(&x,p,8); return x; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t x; memcpy(&x,p,4); return (uint64_t)x; }

/* the owl takes the pen: 64 -> 64 avalanche (Murmur3/splitmix finalizer) */
static inline uint64_t owl(uint64_t h){
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

/* one heap raced in the deer's pattern: uphill twist (rotate), downhill fold
   (xor), doubling-back that eats its own trail (add -- carries, so the step is
   not GF(2)-linear).  Eight lanes, eight strides, no multiply, no cross-lane
   traffic: eight independent dependency chains for the out-of-order engine. */
static inline void deer(uint64_t *restrict v, const unsigned char *restrict p){
    uint64_t w0=ld64(p),    w1=ld64(p+8),  w2=ld64(p+16), w3=ld64(p+24);
    uint64_t w4=ld64(p+32), w5=ld64(p+40), w6=ld64(p+48), w7=ld64(p+56);
    v[0] = ROTL64(v[0]^w0,13) + (w0^HK[0]);
    v[1] = ROTL64(v[1]^w1,23) + (w1^HK[1]);
    v[2] = ROTL64(v[2]^w2,29) + (w2^HK[2]);
    v[3] = ROTL64(v[3]^w3,31) + (w3^HK[3]);
    v[4] = ROTL64(v[4]^w4,37) + (w4^HK[4]);
    v[5] = ROTL64(v[5]^w5,41) + (w5^HK[5]);
    v[6] = ROTL64(v[6]^w6,47) + (w6^HK[6]);
    v[7] = ROTL64(v[7]^w7,53) + (w7^HK[7]);
}

/* searching the result in squares: the eight lanes as a 4x2 square -- columns
   (i,i+4), then a ninety-degree turn into the diagonals (i,((i+1)&3)+4).
   Lane graph 0-4-3-7-2-6-1-5-0, diameter 4; three of these reach everywhere. */
static inline void square(uint64_t *restrict v){
    int i;
    for (i = 0; i < 4; i++){
        v[i]   += v[i+4];
        v[i+4]  = ROTL64(v[i+4],17) ^ v[i];
        v[i]    = ROTL64(v[i],43);
    }
    for (i = 0; i < 4; i++){
        int j = ((i+1)&3)+4;
        v[i] += v[j];
        v[j]  = ROTL64(v[j],31) ^ v[i];
        v[i]  = ROTL64(v[i],23);
    }
}

/* the shrine: squares, then the dreamer inside the dreamer (8->4->2->1),
   then the owl's seal.  O(1): all the expensive mixing lives here, none of
   it per byte. */
static inline uint64_t shrine(uint64_t *restrict v, size_t len){
    uint64_t a,b,c,d,x,y;
    v[0] ^= (uint64_t)len;                  /* the order, never lost */
    v[5] += (uint64_t)len + 0x9E3779B97F4A7C15ULL;
    square(v); square(v); square(v);
    a = v[0] ^ ROTL64(v[4],11);
    b = v[1] + ROTL64(v[5],29);
    c = v[2] ^ ROTL64(v[6],47);
    d = v[3] + ROTL64(v[7],19);
    x = a + b;  y = c ^ d;
    x = ROTL64(x,31) + y;
    x ^= ROTL64(y,17);
    return owl(x + (uint64_t)len);
}

/* regime A: a pile too small to form a heap.  Marks laid by position with
   overlapping reads (never outside the buffer), straight to the owl. */
static uint64_t tiny_pile(const unsigned char *p, size_t len){
    uint64_t a,b;
    if (len >= 8){ a = ld64(p) ^ HK[0];        b = ld64(p+len-8) ^ HK[1]; }
    else if (len >= 4){ a = ld32(p) ^ HK[2];   b = ld32(p+len-4) ^ HK[3]; }
    else if (len){ a = (((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8)
                        | (uint64_t)p[len-1]) ^ HK[4];
                   b = HK[5]; }
    else { a = HK[6]; b = HK[7]; }
    a += (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    b ^= ROTL64(a,37);
    a += ROTL64(b,23);
    return owl(a ^ ROTL64(b,41));
}

uint64_t kernel(const unsigned char *data, size_t len){
    uint64_t v[8];
    const unsigned char *p = data;
    size_t n = len;
    int i;

    if (len < 16) return tiny_pile(data, len);      /* regime check + fallback */

    for (i = 0; i < 8; i++) v[i] = HK[i];

#if defined(__AVX2__)
    if (n >= 64){                                   /* regime C: racing heaps */
        __m256i V0 = _mm256_loadu_si256((const __m256i *)(const void *)&v[0]);
        __m256i V1 = _mm256_loadu_si256((const __m256i *)(const void *)&v[4]);
        const __m256i K0 = _mm256_loadu_si256((const __m256i *)(const void *)&HK[0]);
        const __m256i K1 = _mm256_loadu_si256((const __m256i *)(const void *)&HK[4]);
        const __m256i S0 = _mm256_setr_epi64x(13,23,29,31);
        const __m256i S1 = _mm256_setr_epi64x(37,41,47,53);
        const __m256i T0 = _mm256_setr_epi64x(64-13,64-23,64-29,64-31);
        const __m256i T1 = _mm256_setr_epi64x(64-37,64-41,64-47,64-53);
        do {
            /* the flat stone: 64 marks laid down by position, in silence */
            __m256i W0 = _mm256_loadu_si256((const __m256i *)(const void *)p);
            __m256i W1 = _mm256_loadu_si256((const __m256i *)(const void *)(p+32));
            __m256i X0 = _mm256_xor_si256(V0, W0);
            __m256i X1 = _mm256_xor_si256(V1, W1);
            X0 = _mm256_or_si256(_mm256_sllv_epi64(X0,S0), _mm256_srlv_epi64(X0,T0));
            X1 = _mm256_or_si256(_mm256_sllv_epi64(X1,S1), _mm256_srlv_epi64(X1,T1));
            V0 = _mm256_add_epi64(X0, _mm256_xor_si256(W0,K0));
            V1 = _mm256_add_epi64(X1, _mm256_xor_si256(W1,K1));
            p += 64; n -= 64;
        } while (n >= 64);
        _mm256_storeu_si256((__m256i *)(void *)&v[0], V0);
        _mm256_storeu_si256((__m256i *)(void *)&v[4], V1);
    }
#else
    while (n >= 64){ deer(v, p); p += 64; n -= 64; }   /* scalar fallback,
                       bit-identical to the AVX2 path; SLP-vectorizable */
#endif

    {   /* regime B / the tail: the last marks laid on a zeroed stone */
        unsigned char stone[64];
        memset(stone, 0, sizeof stone);
        if (n) memcpy(stone, p, n);
        deer(v, stone);
    }
    return shrine(v, len);
}
