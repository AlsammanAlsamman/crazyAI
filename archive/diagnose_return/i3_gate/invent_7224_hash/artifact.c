#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define RL(x,r) (((x) << (r)) | ((x) >> (64 - (r))))
#define RR(x,r) (((x) >> (r)) | ((x) << (64 - (r))))

/* --- the flat stone at the riverbank: position-addressed, never "spoken" --- */
static inline uint64_t ld64(const unsigned char *p) { uint64_t x; memcpy(&x, p, 8); return x; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t x; memcpy(&x, p, 4); return (uint64_t)x; }

/* --- the owl takes the pen: murmur3 fmix64, the ONLY multiplications spent --- */
static inline uint64_t owl(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

/* distinct, nonzero lane stones (digits of pi) - nothing may ossify to zero */
static const uint64_t STONE_IV[16] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL, 0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
    0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL, 0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL, 0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
};

/* --- the musk deer's stride: uphill twist, downhill fold, doubling back ---
   8 Speck128-style ARX steps over 8 independent lane-pairs. No multiplies.  */
static inline void heap_round(uint64_t *v) {
    int j;
    for (j = 0; j < 4; j++) { v[j]    = RR(v[j], 8)     + v[j+4];  v[j+8]  = RR(v[j+8], 8)   + v[j+12]; }
    for (j = 0; j < 4; j++) { v[j+4]  = RL(v[j+4], 3)   ^ v[j];    v[j+12] = RL(v[j+12], 3)  ^ v[j+8];  }
    for (j = 0; j < 4; j++) { v[j+4]  = RR(v[j+4], 8)   + v[j];    v[j+12] = RR(v[j+12], 8)  + v[j+8];  }
    for (j = 0; j < 4; j++) { v[j]    = RL(v[j], 3)     ^ v[j+4];  v[j+8]  = RL(v[j+8], 3)   ^ v[j+12]; }
}

/* --- the shrine: search the result in squares. BLAKE2b's G on a 4x4 grid,
       turned ninety degrees (columns) then folded corner-to-centre (diagonals) */
#define G(a,b,c,d) do {                        \
    (a) = (a) + (b); (d) = RR((d) ^ (a), 32);  \
    (c) = (c) + (d); (b) = RR((b) ^ (c), 24);  \
    (a) = (a) + (b); (d) = RR((d) ^ (a), 16);  \
    (c) = (c) + (d); (b) = RR((b) ^ (c), 63);  \
} while (0)

#if defined(__AVX2__)
#define VRL(x,r) _mm256_or_si256(_mm256_slli_epi64((x),(r)), _mm256_srli_epi64((x),64-(r)))
#define VRR(x,r) _mm256_or_si256(_mm256_srli_epi64((x),(r)), _mm256_slli_epi64((x),64-(r)))
#endif

/* one runner: race every heap of this stretch, then carry the stone to the shrine */
static uint64_t heap_race(const unsigned char * restrict data, size_t len, uint64_t id) {
    uint64_t v[16];
    uint64_t t = id * 0x9E3779B97F4A7C15ULL;
    const unsigned char * restrict p = data;
    size_t nb = len >> 7, r = len & 127, i;
    int j, rnd;

    for (j = 0; j < 16; j++) v[j] = STONE_IV[j] + t;

#if defined(__AVX2__)
    {
        __m256i A = _mm256_loadu_si256((const __m256i *)(v + 0));
        __m256i B = _mm256_loadu_si256((const __m256i *)(v + 4));
        __m256i C = _mm256_loadu_si256((const __m256i *)(v + 8));
        __m256i D = _mm256_loadu_si256((const __m256i *)(v + 12));
        for (i = 0; i < nb; i++, p += 128) {
            A = _mm256_xor_si256(A, _mm256_loadu_si256((const __m256i *)(p +  0)));
            B = _mm256_xor_si256(B, _mm256_loadu_si256((const __m256i *)(p + 32)));
            C = _mm256_xor_si256(C, _mm256_loadu_si256((const __m256i *)(p + 64)));
            D = _mm256_xor_si256(D, _mm256_loadu_si256((const __m256i *)(p + 96)));
            A = _mm256_add_epi64(VRR(A, 8), B);
            C = _mm256_add_epi64(VRR(C, 8), D);
            B = _mm256_xor_si256(VRL(B, 3), A);
            D = _mm256_xor_si256(VRL(D, 3), C);
            B = _mm256_add_epi64(VRR(B, 8), A);
            D = _mm256_add_epi64(VRR(D, 8), C);
            A = _mm256_xor_si256(VRL(A, 3), B);
            C = _mm256_xor_si256(VRL(C, 3), D);
        }
        _mm256_storeu_si256((__m256i *)(v +  0), A);
        _mm256_storeu_si256((__m256i *)(v +  4), B);
        _mm256_storeu_si256((__m256i *)(v +  8), C);
        _mm256_storeu_si256((__m256i *)(v + 12), D);
    }
#else
    for (i = 0; i < nb; i++, p += 128) {
        for (j = 0; j < 16; j++) v[j] ^= ld64(p + 8 * j);
        heap_round(v);
    }
#endif

    if (r) {                              /* the short heap at the end of the pile */
        unsigned char tail[128];
        memset(tail, 0, sizeof tail);
        memcpy(tail, p, r);
        for (j = 0; j < 16; j++) v[j] ^= ld64(tail + 8 * j);
        heap_round(v);
    }

    for (rnd = 0; rnd < 2; rnd++) {       /* turn ninety degrees, fold the corners in */
        G(v[0], v[4], v[8],  v[12]);
        G(v[1], v[5], v[9],  v[13]);
        G(v[2], v[6], v[10], v[14]);
        G(v[3], v[7], v[11], v[15]);
        G(v[0], v[5], v[10], v[15]);
        G(v[1], v[6], v[11], v[12]);
        G(v[2], v[7], v[8],  v[13]);
        G(v[3], v[4], v[9],  v[14]);
    }

    {                                     /* burn the heaps: 1024 bits -> 64, one way */
        uint64_t x0 = (v[0] ^ v[8])  + RL(v[4]  ^ v[12], 17);
        uint64_t x1 = (v[1] ^ v[9])  + RL(v[5]  ^ v[13], 29);
        uint64_t x2 = (v[2] ^ v[10]) + RL(v[6]  ^ v[14], 41);
        uint64_t x3 = (v[3] ^ v[11]) + RL(v[7]  ^ v[15], 53);
        uint64_t y0 = x0 ^ RL(x1, 23);
        uint64_t y1 = x2 ^ RL(x3, 47);
        return y0 + RL(y1, 31) + (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    }
}

/* a pile too small to make a heap: straight to the shrine (xxh3-style short path) */
static uint64_t riverbank_short(const unsigned char *data, size_t len) {
    uint64_t a, b;
    if (len >= 8) {
        a = ld64(data);
        b = ld64(data + len - 8);
        if (len > 16) { a ^= ld64(data + 8); b ^= ld64(data + len - 16); }
    } else if (len >= 4) {
        a = ld32(data);
        b = ld32(data + len - 4);
    } else if (len > 0) {
        a = (uint64_t)data[0];
        b = ((uint64_t)data[len >> 1] << 8) | (uint64_t)data[len - 1];
    } else {
        a = 0; b = 0;
    }
    a ^= STONE_IV[0] + (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    b ^= STONE_IV[5];
    a += RL(b, 31);
    b ^= RL(a, 17);
    return owl(a ^ owl(b));
}

uint64_t kernel(const unsigned char * restrict data, size_t len) {
    if (len < 32)                        /* regime 1: no heap can be formed */
        return riverbank_short(data, len);

    if (len < ((size_t)4 << 20))         /* regime 2: one runner, SIMD lanes */
        return owl(heap_race(data, len, 0));

    {                                    /* regime 3: the pile needs many runners */
        size_t nch = len >> 20;          /* ~1 MiB of marks per runner, 4..64 runners */
        if (nch > 64) nch = 64;
        {
            size_t cs = (len / nch) & ~(size_t)127;   /* whole heaps only */
            uint64_t stones[64];
            uint64_t acc;
            size_t k;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long ii = 0; ii < (long)nch; ii++) {
                size_t off = (size_t)ii * cs;
                size_t cl  = ((size_t)ii == nch - 1) ? (len - off) : cs;
                stones[ii] = heap_race(data + off, cl, (uint64_t)ii + 1);
            }
            acc = 0x243F6A8885A308D3ULL ^ ((uint64_t)len * 0x9E3779B97F4A7C15ULL);
            for (k = 0; k < nch; k++)    /* the stones folded at the last shrine */
                acc = RL(acc, 27) ^ owl(stones[k] + (uint64_t)k * 0xC2B2AE3D27D4EB4FULL);
            return owl(acc);
        }
    }
}
