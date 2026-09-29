#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define PM 1099511628211ULL          /* FNV prime  : the per-byte "twist"      */
#define OB 14695981039346656037ULL   /* FNV basis  : folded in at the end      */

typedef unsigned long long u64x4 __attribute__((vector_size(32)));
typedef unsigned char      u8x4  __attribute__((vector_size(4)));

static inline uint64_t mix_fin(uint64_t x) {   /* MurmurHash3 fmix64 */
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* zero-extend 4 consecutive bytes into 4 x u64 lanes */
static inline u64x4 ldx(const unsigned char *p) {
#if defined(__AVX2__)
    int t; memcpy(&t, p, 4);
    return (u64x4)_mm256_cvtepu8_epi64(_mm_cvtsi32_si128(t));
#elif defined(__GNUC__) && (__GNUC__ >= 9)
    u8x4 t; memcpy(&t, p, 4);
    return __builtin_convertvector(t, u64x4);
#else
    u64x4 r = { p[0], p[1], p[2], p[3] };
    return r;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* Reference semantics (what this function computes, exactly):
     *     A = 0; for (i) A = (A + data[i]) * PM;   return mix_fin(A ^ tag);
     * Everything below is an exact re-association of that recurrence.      */
    uint64_t A = 0;
    size_t   i = 0;

    if (len >= 64) {
        const uint64_t P2 = PM * PM, P4 = P2 * P2, P8 = P4 * P4,
                       P16 = P8 * P8, Q = P16 * P16;      /* Q = PM^32 */
        const u64x4 vq = { Q, Q, Q, Q };
        u64x4 a0 = { 0, 0, 0, 0 }, a1 = a0, a2 = a0, a3 = a0,
              a4 = a0,             a5 = a0, a6 = a0, a7 = a0;

        const size_t M = len >> 5;                 /* full 32-byte blocks */
        const unsigned char *p = data;

        for (size_t m = 0; m < M; ++m, p += 32) {
            a0 = a0 * vq + ldx(p +  0);
            a1 = a1 * vq + ldx(p +  4);
            a2 = a2 * vq + ldx(p +  8);
            a3 = a3 * vq + ldx(p + 12);
            a4 = a4 * vq + ldx(p + 16);
            a5 = a5 * vq + ldx(p + 20);
            a6 = a6 * vq + ldx(p + 24);
            a7 = a7 * vq + ldx(p + 28);
        }

        /* lane l carries weight PM^(32-l): fold with one Horner pass */
        uint64_t acc[32];
        memcpy(&acc[ 0], &a0, 32); memcpy(&acc[ 4], &a1, 32);
        memcpy(&acc[ 8], &a2, 32); memcpy(&acc[12], &a3, 32);
        memcpy(&acc[16], &a4, 32); memcpy(&acc[20], &a5, 32);
        memcpy(&acc[24], &a6, 32); memcpy(&acc[28], &a7, 32);

        uint64_t s = 0;
        for (int l = 0; l < 32; ++l) s = s * PM + acc[l];
        A = s * PM;
        i = M << 5;
    }

    /* ragged tail: the original serial twist, unchanged */
    for (; i < len; ++i) A = (A + data[i]) * PM;

    return mix_fin(A ^ (OB + (uint64_t)len * 0x9E3779B97F4A7C15ULL));
}
