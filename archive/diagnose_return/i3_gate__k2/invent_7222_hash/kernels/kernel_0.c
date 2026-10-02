#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the wooden numbered keeps mounted at the desk's edge: fixed forever,
   "that same starlight color no matter the hour" (digits of pi) ---- */
#define KA 0x243F6A8885A308D3ULL
#define KB 0x13198A2E03707344ULL
#define KC 0xA4093822299F31D0ULL
#define KD 0x082EFA98EC4E6C89ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* ---- the lift: read the stone's final seated position against the keeps.
   This single number is the only thing that ever leaves the desk. ---- */
static inline uint64_t read_off(const uint64_t s[4], uint64_t tag) {
    uint64_t h = (s[0] ^ rotl64(s[1], 17)) + (rotl64(s[2], 41) ^ rotl64(s[3], 53)) + tag;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

/* ---- one quarter-turn with no new mark (scalar form, used by the squeeze
   and by the handful-of-marks regime). Every keep takes on a neighbour:
   the stone is one object, not four. ---- */
static inline void turn4(uint64_t s[4], int k, unsigned r) {
    uint64_t t0 = s[(0 + k) & 3], t1 = s[(1 + k) & 3];
    uint64_t t2 = s[(2 + k) & 3], t3 = s[(3 + k) & 3];
    s[0] = rotl64(s[0] + t0, r); s[1] = rotl64(s[1] + t1, r);
    s[2] = rotl64(s[2] + t2, r); s[3] = rotl64(s[3] + t3, r);
}

/* ---- press a mark's weight into the already-turned position, then turn ---- */
static inline void fold4(uint64_t s[4], const uint64_t m[4], int k, unsigned r) {
    uint64_t t0 = s[(0 + k) & 3], t1 = s[(1 + k) & 3];
    uint64_t t2 = s[(2 + k) & 3], t3 = s[(3 + k) & 3];
    uint64_t x0 = s[0] ^ m[0], x1 = s[1] ^ m[1];
    uint64_t x2 = s[2] ^ m[2], x3 = s[3] ^ m[3];
    s[0] = rotl64(x0 + t0, r); s[1] = rotl64(x1 + t1, r);
    s[2] = rotl64(x2 + t2, r); s[3] = rotl64(x3 + t3, r);
}

#if defined(__AVX2__)
/* the chalk-bank groove: a bitwise quarter-turn of every keep */
#define V_ROTL(v, r) _mm256_or_si256(_mm256_slli_epi64((v), (r)),              \
                                     _mm256_srli_epi64((v), 64 - (r)))

/* cheap coupling: each keep takes its partner within the half (latency 1) */
#define V_FOLD_H(s, m, r) do {                                                 \
    __m256i t_ = _mm256_shuffle_epi32((s), 0x4E);                              \
    __m256i x_ = _mm256_xor_si256((s), (m));                                   \
    (s) = V_ROTL(_mm256_add_epi64(x_, t_), (r));                               \
} while (0)

/* full coupling: each keep takes a keep SH places round the stone */
#define V_FOLD_P(s, m, SH, r) do {                                             \
    __m256i t_ = _mm256_permute4x64_epi64((s), (SH));                          \
    __m256i x_ = _mm256_xor_si256((s), (m));                                   \
    (s) = V_ROTL(_mm256_add_epi64(x_, t_), (r));                               \
} while (0)

/* a turn with no new mark */
#define V_TURN(s, SH, r) do {                                                  \
    __m256i t_ = _mm256_permute4x64_epi64((s), (SH));                          \
    (s) = V_ROTL(_mm256_add_epi64((s), t_), (r));                              \
} while (0)
#endif

/* ---- the pile regime: n >= 32 guaranteed by the caller ---- */
static uint64_t fold_pile(const unsigned char *restrict p, size_t n, uint64_t tag) {
    uint64_t out[4];
#if defined(__AVX2__)
    /* the stone, seated on the keeps */
    __m256i s = _mm256_set_epi64x((long long)KD, (long long)KC,
                                  (long long)KB, (long long)KA);
    const unsigned char *q = p;
    size_t m = n;

    /* four quarter-turns per pass: the groove's constant and the direction of
       coupling change every turn, as a die lands on a different face. This is
       ONE chain unrolled, not four chains. */
    while (m >= 128) {
        V_FOLD_H(s, _mm256_loadu_si256((const __m256i *)(q +   0)),       29);
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(q +  32)), 0x39, 47);
        V_FOLD_H(s, _mm256_loadu_si256((const __m256i *)(q +  64)),       17);
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(q +  96)), 0x93, 41);
        q += 128; m -= 128;
    }
    while (m >= 32) {
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)q), 0x39, 29);
        q += 32; m -= 32;
    }
    if (m) {
        /* back the stone up so the seated place is full: never a part-press */
        V_FOLD_P(s, _mm256_loadu_si256((const __m256i *)(p + n - 32)), 0x93, 47);
    }

    /* the pile's own extent is a mark too */
    s = _mm256_xor_si256(s, _mm256_set1_epi64x((long long)tag));

    /* the lift: four turns with no new mark, so even the last mark's
       hair's-width difference has nowhere left to hide */
    V_TURN(s, 0x39, 23);
    V_TURN(s, 0x4E, 37);
    V_TURN(s, 0x93, 19);
    V_TURN(s, 0x39, 43);

    _mm256_storeu_si256((__m256i *)out, s);
#else
    uint64_t s[4] = { KA, KB, KC, KD };
    const unsigned char *q = p;
    size_t m = n;
    while (m >= 32) {
        uint64_t w[4]; memcpy(w, q, 32);
        fold4(s, w, 1, 29);
        q += 32; m -= 32;
    }
    if (m) {
        uint64_t w[4]; memcpy(w, p + n - 32, 32);
        fold4(s, w, 3, 47);
    }
    s[0] ^= tag; s[1] ^= tag; s[2] ^= tag; s[3] ^= tag;
    turn4(s, 1, 23); turn4(s, 2, 37); turn4(s, 3, 19); turn4(s, 1, 43);
    out[0] = s[0]; out[1] = s[1]; out[2] = s[2]; out[3] = s[3];
#endif
    return read_off(out, tag);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* ---- regime check, the native's own: can the pile fill the groove's
       seated place at all? ---- */
    if (len < 16) {
        /* a handful of marks: no full seating exists, so press the thin
           stack straight against the keeps and lift at once. Guards the
           squeeze's fixed cost from dominating tiny piles. */
        uint64_t a = 0, b = 0;
        if (len >= 8) {
            memcpy(&a, data, 8);
            memcpy(&b, data + len - 8, 8);
        } else if (len >= 4) {
            uint32_t x, y;
            memcpy(&x, data, 4);
            memcpy(&y, data + len - 4, 4);
            a = x; b = y;
        } else if (len) {
            a = (uint64_t)data[0]
              | ((uint64_t)data[len >> 1] << 8)
              | ((uint64_t)data[len - 1] << 16);
        }
        uint64_t s[4];
        s[0] = KA ^ (a + (uint64_t)len);
        s[1] = KB ^ rotl64(b, 23);
        s[2] = KC + rotl64(a, 41);
        s[3] = KD ^ (b + rotl64(a, 11));
        turn4(s, 1, 23);
        turn4(s, 3, 37);
        return read_off(s, (uint64_t)len);
    }
    if (len < 32) {
        /* not quite a seating: seat it on a swept desk, zero-padded,
           and let the extent distinguish the short piles */
        unsigned char buf[32];
        memset(buf, 0, sizeof buf);
        memcpy(buf, data, len);
        return fold_pile(buf, 32, (uint64_t)len);
    }
    return fold_pile(data, len, (uint64_t)len);
}
