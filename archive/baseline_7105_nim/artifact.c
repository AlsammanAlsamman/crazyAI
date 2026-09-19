#include <immintrin.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int xor_all = 0;

#if defined(__AVX2__)
    int i = 0;
    __m256i vxor = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        vxor = _mm256_xor_si256(vxor, v);
    }
    int tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vxor);
    for (int k = 0; k < 8; k++) xor_all ^= tmp[k];
    for (; i < n; i++) xor_all ^= piles[i];
#else
    for (int i = 0; i < n; i++) xor_all ^= piles[i];
#endif

    if (xor_all == 0) {
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* highest set bit of xor_all: a ^ xor_all < a  <=>  (a & h) != 0 */
    unsigned h = 1u << (31 - __builtin_clz((unsigned)xor_all));

#if defined(__AVX2__)
    __m256i vh = _mm256_set1_epi32((int)h);
    __m256i vzero = _mm256_setzero_si256();
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + j));
        __m256i a = _mm256_and_si256(v, vh);
        __m256i cmp = _mm256_cmpeq_epi32(a, vzero); /* all-1s lane where a==0 */
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        unsigned hitmask = (~(unsigned)mask) & 0xFFu; /* lanes where a != 0 */
        if (hitmask) {
            int lane = __builtin_ctz(hitmask);
            int idx = j + lane;
            *out_pile = idx;
            *out_remove = piles[idx] - (piles[idx] ^ xor_all);
            return;
        }
    }
    for (; j < n; j++) {
        if ((unsigned)piles[j] & h) {
            *out_pile = j;
            *out_remove = piles[j] - (piles[j] ^ xor_all);
            return;
        }
    }
#else
    for (int j = 0; j < n; j++) {
        if ((unsigned)piles[j] & h) {
            *out_pile = j;
            *out_remove = piles[j] - (piles[j] ^ xor_all);
            return;
        }
    }
#endif
    /* unreachable if xor_all != 0, per Bouton's theorem */
}
