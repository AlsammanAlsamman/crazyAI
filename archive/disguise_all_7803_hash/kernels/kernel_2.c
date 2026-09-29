#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* per-position key schedule: key(block j, lane l) = KB[l] + j*KS[l] */
static const uint64_t KB[4] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0xD6E8FEB86659FD93ULL };
static const uint64_t KS[4] = {
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
    0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL };

static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* leaf: multiply-free, invertible, nonlinear (shift-and-add carries) */
static inline uint64_t leaf64(uint64_t x, uint64_t key) {
    x ^= key;
    x ^= x >> 29;
    x += x << 13;
    x ^= x >> 31;
    x += x << 17;
    return x;
}

/* O(1) strong mixer -- the only place a multiply appears */
static inline uint64_t fmix64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31; return x;
}

#if defined(__AVX2__)
#define VLEAF(x) do {                                        \
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 29));       \
    x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 13));       \
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 31));       \
    x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 17));       \
} while (0)
#endif

/* absorb 32-byte blocks [j0,j1) as independent leaves; XOR-fold into out[4].
   XOR is associative+commutative, so this is exactly a tournament bracket
   over those blocks, in any order, from any number of workers. */
static void absorb_range(const unsigned char *data, size_t j0, size_t j1, uint64_t out[4])
{
    uint64_t k[4];
    int l;
    for (l = 0; l < 4; l++) k[l] = KB[l] + (uint64_t)j0 * KS[l];
    const unsigned char *p = data + (j0 << 5);
    size_t n = j1 - j0, i = 0;

#if defined(__AVX2__)
    const __m256i S1 = _mm256_loadu_si256((const __m256i *)KS);
    const __m256i S2 = _mm256_add_epi64(S1, S1);
    const __m256i S3 = _mm256_add_epi64(S2, S1);
    const __m256i S4 = _mm256_add_epi64(S2, S2);
    __m256i base = _mm256_loadu_si256((const __m256i *)k);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;

    for (; i + 4 <= n; i += 4, p += 128) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p      )), base);
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 32)), _mm256_add_epi64(base, S1));
        __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 64)), _mm256_add_epi64(base, S2));
        __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 96)), _mm256_add_epi64(base, S3));
        VLEAF(x0); VLEAF(x1); VLEAF(x2); VLEAF(x3);
        a0 = _mm256_xor_si256(a0, x0);
        a1 = _mm256_xor_si256(a1, x1);
        a2 = _mm256_xor_si256(a2, x2);
        a3 = _mm256_xor_si256(a3, x3);
        base = _mm256_add_epi64(base, S4);
    }
    for (; i < n; i++, p += 32) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)p), base);
        VLEAF(x0);
        a0 = _mm256_xor_si256(a0, x0);
        base = _mm256_add_epi64(base, S1);
    }
    /* pair, press, pair, press */
    a0 = _mm256_xor_si256(a0, a1);
    a2 = _mm256_xor_si256(a2, a3);
    a0 = _mm256_xor_si256(a0, a2);
    {
        uint64_t t[4];
        _mm256_storeu_si256((__m256i *)t, a0);
        out[0] ^= t[0]; out[1] ^= t[1]; out[2] ^= t[2]; out[3] ^= t[3];
    }
#else
    /* scalar fallback: identical lane/key mapping => identical output */
    uint64_t b0 = k[0], b1 = k[1], b2 = k[2], b3 = k[3];
    uint64_t A0 = 0, A1 = 0, A2 = 0, A3 = 0, B0 = 0, B1 = 0, B2 = 0, B3 = 0;
    for (; i + 2 <= n; i += 2, p += 64) {
        A0 ^= leaf64(ld64(p      ), b0);
        A1 ^= leaf64(ld64(p +  8 ), b1);
        A2 ^= leaf64(ld64(p + 16 ), b2);
        A3 ^= leaf64(ld64(p + 24 ), b3);
        B0 ^= leaf64(ld64(p + 32 ), b0 + KS[0]);
        B1 ^= leaf64(ld64(p + 40 ), b1 + KS[1]);
        B2 ^= leaf64(ld64(p + 48 ), b2 + KS[2]);
        B3 ^= leaf64(ld64(p + 56 ), b3 + KS[3]);
        b0 += 2 * KS[0]; b1 += 2 * KS[1]; b2 += 2 * KS[2]; b3 += 2 * KS[3];
    }
    for (; i < n; i++, p += 32) {
        A0 ^= leaf64(ld64(p      ), b0);
        A1 ^= leaf64(ld64(p +  8 ), b1);
        A2 ^= leaf64(ld64(p + 16 ), b2);
        A3 ^= leaf64(ld64(p + 24 ), b3);
        b0 += KS[0]; b1 += KS[1]; b2 += KS[2]; b3 += KS[3];
    }
    out[0] ^= A0 ^ B0; out[1] ^= A1 ^ B1; out[2] ^= A2 ^ B2; out[3] ^= A3 ^ B3;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t acc[4];

    if (len >= 32) {
        size_t nb = len >> 5;                 /* full 32-byte leaf blocks */
        acc[0] = acc[1] = acc[2] = acc[3] = 0;

#ifdef _OPENMP
        if (len >= ((size_t)4 << 20) && omp_get_max_threads() > 1) {
            uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            #pragma omp parallel reduction(^ : r0, r1, r2, r3)
            {
                int nt = omp_get_num_threads(), id = omp_get_thread_num();
                size_t per = (nb + (size_t)nt - 1) / (size_t)nt;
                per = (per + 3) & ~(size_t)3;        /* keep 4-block groups whole */
                size_t b0 = (size_t)id * per;
                size_t b1 = b0 + per;
                if (b0 > nb) b0 = nb;
                if (b1 > nb) b1 = nb;
                uint64_t loc[4] = { 0, 0, 0, 0 };
                if (b1 > b0) absorb_range(data, b0, b1, loc);
                r0 ^= loc[0]; r1 ^= loc[1]; r2 ^= loc[2]; r3 ^= loc[3];
            }
            acc[0] = r0; acc[1] = r1; acc[2] = r2; acc[3] = r3;
        } else
#endif
        absorb_range(data, 0, nb, acc);

        /* final 32 bytes (may overlap): permuted lanes + rotation so the
           duplicated region can never cancel the main absorption */
        {
            const unsigned char *t = data + len - 32;
            uint64_t tk = 0x2545F4914F6CDD1DULL + (uint64_t)len;
            uint64_t v0 = leaf64(ld64(t      ), KB[0] ^ tk);
            uint64_t v1 = leaf64(ld64(t +  8 ), KB[1] ^ tk);
            uint64_t v2 = leaf64(ld64(t + 16 ), KB[2] ^ tk);
            uint64_t v3 = leaf64(ld64(t + 24 ), KB[3] ^ tk);
            acc[1] ^= ROTL64(v0, 11);
            acc[2] ^= ROTL64(v1, 23);
            acc[3] ^= ROTL64(v2, 37);
            acc[0] ^= ROTL64(v3, 53);
        }
    } else if (len >= 16) {
        uint64_t tk = 0x2545F4914F6CDD1DULL + (uint64_t)len;
        acc[0] = leaf64(ld64(data          ), KB[0] ^ tk);
        acc[1] = leaf64(ld64(data + 8      ), KB[1] ^ tk);
        acc[2] = leaf64(ld64(data + len - 16), KB[2] ^ tk);
        acc[3] = leaf64(ld64(data + len - 8 ), KB[3] ^ tk);
    } else {
        uint64_t a, b, x, y;
        if (len >= 8)      { a = ld64(data); b = ld64(data + len - 8); }
        else if (len >= 4) { a = ld32(data); b = ld32(data + len - 4); }
        else if (len)      { a = ((uint64_t)data[0] << 16) |
                                 ((uint64_t)data[len >> 1] << 8) |
                                  (uint64_t)data[len - 1]; b = 0; }
        else               { a = 0; b = 0; }
        x = a ^ (KB[0] + (uint64_t)len);
        y = b ^ KB[1];
        x += ROTL64(y, 31);
        y ^= ROTL64(x, 19);
        x += y;
        return fmix64(x);
    }

    /* root of the bracket: pair, press, pair, press -- each lane enters
       through a bijection, so any single-bit input delta reaches the root */
    {
        uint64_t t0 = fmix64(acc[0] + ROTL64(acc[1], 29) + (uint64_t)len);
        uint64_t t1 = fmix64(acc[2] + ROTL64(acc[3], 41) + 0x9E3779B97F4A7C15ULL);
        return fmix64(t0 ^ ROTL64(t1, 17));
    }
}
