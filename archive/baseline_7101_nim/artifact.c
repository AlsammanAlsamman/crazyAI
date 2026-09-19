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
        int tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, vxor);
        for (int k = 0; k < 8; k++) xor_all ^= tmp[k];
    }
#endif
    for (; i < n; i++) xor_all ^= piles[i];

    if (xor_all == 0) { *out_pile = 0; *out_remove = 1; return; }

    i = 0;
#if defined(__AVX2__)
    {
        __m256i vx = _mm256_set1_epi32(xor_all);
        for (; i + 8 <= n; i += 8) {
            __m256i v      = _mm256_loadu_si256((const __m256i*)(piles + i));
            __m256i target = _mm256_xor_si256(v, vx);
            __m256i cmp    = _mm256_cmpgt_epi32(v, target); /* target < v */
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask != 0) {
                int lane = __builtin_ctz((unsigned)mask);
                int idx  = i + lane;
                int t    = piles[idx] ^ xor_all;
                *out_pile   = idx;
                *out_remove = piles[idx] - t;
                return;
            }
        }
    }
#endif
    for (; i < n; i++) {
        int target = piles[i] ^ xor_all;
        if (target < piles[i]) {
            *out_pile = i;
            *out_remove = piles[i] - target;
            return;
        }
    }
}
