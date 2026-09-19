#include <immintrin.h>
#include <stdint.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int i;
    int xor_all = 0;

#if defined(__AVX2__)
    __m256i vxor = _mm256_setzero_si256();
    int i8 = 0;
    for (; i8 + 8 <= n; i8 += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i8));
        vxor = _mm256_xor_si256(vxor, v);
    }
    __m128i lo = _mm256_castsi256_si128(vxor);
    __m128i hi = _mm256_extracti128_si256(vxor, 1);
    __m128i x128 = _mm_xor_si128(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(x128, _MM_SHUFFLE(1, 0, 3, 2));
    x128 = _mm_xor_si128(x128, shuf);
    shuf = _mm_shuffle_epi32(x128, _MM_SHUFFLE(2, 3, 0, 1));
    x128 = _mm_xor_si128(x128, shuf);
    xor_all = _mm_cvtsi128_si32(x128);
    for (i = i8; i < n; i++) xor_all ^= piles[i];
#else
    for (i = 0; i < n; i++) xor_all ^= piles[i];
#endif

    if (xor_all == 0) {
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* isolate the highest set bit of xor_all */
    unsigned int mask = (unsigned int)xor_all;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask = mask - (mask >> 1);

#if defined(__AVX2__)
    __m256i vmask = _mm256_set1_epi32((int)mask);
    __m256i vzero = _mm256_setzero_si256();
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + j));
        __m256i t = _mm256_and_si256(v, vmask);
        __m256i eqz = _mm256_cmpeq_epi32(t, vzero); /* all-1s where AND==0 (no match) */
        unsigned int mm = (unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(eqz));
        if (mm != 0xFFu) {
            unsigned int hit = (~mm) & 0xFFu; /* bits where match (AND != 0) */
            int lane = __builtin_ctz(hit);
            int idx = j + lane;
            *out_pile = idx;
            *out_remove = piles[idx] - (piles[idx] ^ xor_all);
            return;
        }
    }
    for (i = j; i < n; i++) {
        if ((unsigned int)piles[i] & mask) {
            *out_pile = i;
            *out_remove = piles[i] - (piles[i] ^ xor_all);
            return;
        }
    }
#else
    for (i = 0; i < n; i++) {
        if ((unsigned int)piles[i] & mask) {
            *out_pile = i;
            *out_remove = piles[i] - (piles[i] ^ xor_all);
            return;
        }
    }
#endif

    /* unreachable when xor_all != 0, but keep the contract satisfied defensively */
    *out_pile = 0;
    *out_remove = 1;
}
