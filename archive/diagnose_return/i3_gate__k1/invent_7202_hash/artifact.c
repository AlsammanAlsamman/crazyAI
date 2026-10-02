#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= the wall's six fixed cracks: the same six every time ===== */
#define CR0 0x9E3779B97F4A7C15ULL
#define CR1 0xBB67AE8584CAA73BULL
#define CR2 0x3C6EF372FE94F82BULL
#define CR3 0xA54FF53A5F1D36F1ULL
#define CR4 0x510E527FADE682D1ULL
#define CR5 0x1F83D9ABFB41BD6BULL

/* ============ the six resting angles of the brick-incense chain ============ */
#define A0  7u
#define A1 19u
#define A2 31u
#define A3 43u
#define A4 11u
#define A5 53u

static inline uint64_t rotl64(uint64_t v, unsigned r) {
    unsigned s = r & 63u;                       /* the swing is a closed arc */
    return (v << s) | (v >> ((64u - s) & 63u));
}

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint32_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}

/* ===== the stream: water over the clay stub until only the ridges remain.
   Thomas Wang's multiplication-free 64-bit finalizer, with an xor-rotate
   rinse at each end.  There is no '*' operator anywhere in this file.   ===== */
static inline uint64_t ridges(uint64_t k) {
    k ^= rotl64(k, 49) ^ rotl64(k, 24);
    k  = (~k) + (k << 21);
    k ^= k >> 24;
    k  = k + (k << 3) + (k << 8);
    k ^= k >> 14;
    k  = k + (k << 2) + (k << 4);
    k ^= k >> 28;
    k  = k + (k << 31);
    k ^= k >> 30;
    return k;
}

/* one full row of six knots: six marks walk past the fire, each one's
   pebble-count setting its own swing, every lane taking the hang of the
   whole chain as it stood a moment before.                               */
#define ROW(q)                                                            \
    do {                                                                  \
        uint64_t w0 = ld64((q)),       w1 = ld64((q) +  8),               \
                 w2 = ld64((q) + 16),  w3 = ld64((q) + 24),               \
                 w4 = ld64((q) + 32),  w5 = ld64((q) + 40);               \
        unsigned c0 = (unsigned)__builtin_popcountll(w0);                 \
        unsigned c1 = (unsigned)__builtin_popcountll(w1);                 \
        unsigned c2 = (unsigned)__builtin_popcountll(w2);                 \
        unsigned c3 = (unsigned)__builtin_popcountll(w3);                 \
        unsigned c4 = (unsigned)__builtin_popcountll(w4);                 \
        unsigned c5 = (unsigned)__builtin_popcountll(w5);                 \
        s0 = rotl64(s0 + w0 + hang, (c0 << 1) + A0);                      \
        s1 = rotl64(s1 + w1 + hang, (c1 << 1) + A1);                      \
        s2 = rotl64(s2 + w2 + hang, (c2 << 1) + A2);                      \
        s3 = rotl64(s3 + w3 + hang, (c3 << 1) + A3);                      \
        s4 = rotl64(s4 + w4 + hang, (c4 << 1) + A4);                      \
        s5 = rotl64(s5 + w5 + hang, (c5 << 1) + A5);                      \
        ctr += 48u;                       /* the row only ever lengthens */\
        hang = ((s0 ^ s2) ^ s4) + rotl64((s1 ^ s3) ^ s5, 29) + ctr;       \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* ============ regime 1: a short rope hangs TAUT — read it whole ======= */
    if (len < 48) {
        if (len == 0) return ridges(CR0);

        if (len < 8) {                    /* not even one whole mark to walk */
            uint64_t a, b;
            if (len >= 4) {
                a = ((uint64_t)ld32(p + len - 4) << 32) | (uint64_t)ld32(p);
                b = rotl64(a, 27) + (uint64_t)len;
            } else {
                a = ((uint64_t)p[0] << 16)
                  | ((uint64_t)p[len >> 1] << 8)
                  |  (uint64_t)p[len - 1];
                b = a + ((uint64_t)len << 5);
            }
            unsigned c = (unsigned)__builtin_popcountll(a);
            uint64_t x = rotl64(a + CR0, (c << 1) + 11u) + (b ^ CR1) + (uint64_t)len;
            return ridges(x);
        }

        uint64_t x = CR0 ^ (uint64_t)len;
        uint64_t y = CR3 + (uint64_t)len;
        size_t n = len;
        while (n >= 8) {
            uint64_t w = ld64(p);
            unsigned c = (unsigned)__builtin_popcountll(w);
            x = rotl64(x + w + y, (c << 1) + A0);
            y = (y ^ w) + rotl64(x, 29);
            p += 8; n -= 8;
        }
        if (n) {                               /* last mark, read overlapping */
            uint64_t w = ld64(data + len - 8);
            unsigned c = (unsigned)__builtin_popcountll(w);
            x = rotl64(x + w + y, (c << 1) + A2);
            y = (y ^ w) + rotl64(x, 37);
        }
        return ridges(x + rotl64(y, 23) + (uint64_t)len);
    }

    /* ===== regime 2: the rope SAGS across the piazza — six knots a swing === */
    uint64_t s0 = CR0 ^ (uint64_t)len, s1 = CR1, s2 = CR2,
             s3 = CR3, s4 = CR4,       s5 = CR5 + (uint64_t)len;
    uint64_t hang = CR2 + (uint64_t)len;
    uint64_t ctr  = 0;
    size_t   n    = len;

    while (n >= 48) {
        ROW(p);
        p += 48; n -= 48;
    }
    if (n) {                     /* the last row, re-hung over the chain's end */
        const unsigned char *q = data + len - 48;
        hang ^= (uint64_t)n;
        ROW(q);
    }

    /* the wet clay takes the print of the whole settled row, then the stream */
    uint64_t fold = (s0 + rotl64(s1, 11)) ^ (s2 + rotl64(s3, 29));
    fold += (s4 + rotl64(s5, 47)) ^ hang;
    return ridges(fold + (uint64_t)len);
}
