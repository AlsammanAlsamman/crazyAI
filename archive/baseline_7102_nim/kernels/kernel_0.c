#include <immintrin.h>

static inline int highest_bit_mask(unsigned int x) {
    return (int)(1u << (31 - __builtin_clz(x)));
}

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int xor_all = 0;
    int i = 0;

#if defined(__AVX2__)
    __m256i vacc = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        vacc = _mm256_xor_si256(vacc, v);
    }
    int tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vacc);
    for (int k = 0; k < 8; k++) xor_all ^= tmp[k];
#endif
    for (; i < n; i++) xor_all ^= piles[i];

    if (xor_all == 0) {
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    int mask = highest_bit_mask((unsigned int)xor_all);
    int j = 0;

#if defined(__AVX2__)
    __m256i vmask = _mm256_set1_epi32(mask);
    __m256i vzero = _mm256_setzero_si256();
    for (; j + 8 <= n; j += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + j));
        __m256i vand = _mm256_and_si256(v, vmask);
        __m256i vcmp = _mm256_cmpeq_epi32(vand, vzero); /* all-1s lane == bit NOT set */
        int mm = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));
        if (mm != 0xFF) {
            int inv = (~mm) & 0xFF;          /* bits where AND != 0 */
            int lane = __builtin_ctz((unsigned)inv);
            int idx = j + lane;
            int target = piles[idx] ^ xor_all;
            *out_pile = idx;
            *out_remove = piles[idx] - target;
            return;
        }
    }
#endif
    for (; j < n; j++) {
        if (piles[j] & mask) {
            int target = piles[j] ^ xor_all;
            *out_pile = j;
            *out_remove = piles[j] - target;
            return;
        }
    }
}
