#include <immintrin.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int xor_all = 0;
    int i = 0;

#if defined(__AVX2__)
    __m256i vxor = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        vxor = _mm256_xor_si256(vxor, v);
    }
    {
        __m128i lo  = _mm256_castsi256_si128(vxor);
        __m128i hi  = _mm256_extracti128_si256(vxor, 1);
        __m128i x128 = _mm_xor_si128(lo, hi);
        __m128i x64  = _mm_xor_si128(x128, _mm_srli_si128(x128, 8));
        __m128i x32  = _mm_xor_si128(x64,  _mm_srli_si128(x64, 4));
        xor_all ^= _mm_cvtsi128_si32(x32);
    }
#endif
    for (; i < n; i++) xor_all ^= piles[i];

    if (xor_all == 0) {
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* highest set bit of xor_all: any pile with this bit set works */
    unsigned int hb = 1u << (31 - __builtin_clz((unsigned int)xor_all));

    i = 0;
#if defined(__AVX2__)
    __m256i vhb = _mm256_set1_epi32((int)hb);
    __m256i vzero = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        __m256i t = _mm256_and_si256(v, vhb);
        __m256i cmp = _mm256_cmpeq_epi32(t, vzero);       /* 0xFFFFFFFF where bit NOT set */
        unsigned int mask = (unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        unsigned int hit = (~mask) & 0xFFu;               /* bits where bit IS set */
        if (hit) {
            int idx = __builtin_ctz(hit);
            int pos = i + idx;
            *out_pile = pos;
            *out_remove = piles[pos] - (piles[pos] ^ xor_all);
            return;
        }
    }
#endif
    for (; i < n; i++) {
        if (piles[i] & hb) {
            *out_pile = i;
            *out_remove = piles[i] - (piles[i] ^ xor_all);
            return;
        }
    }
}
