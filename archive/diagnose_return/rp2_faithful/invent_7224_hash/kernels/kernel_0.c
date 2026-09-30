#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the flat stone at the riverbank: sixteen fixed places for marks ---- */
static const uint64_t RIVERBANK[16] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL
};

/* uphill twist */
static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* the musk deer's stride: uphill twist, downhill fold, doubling-back that
   eats its own trail.  No multiplication anywhere. */
#define QR(a,b,c,d) do {                                     \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 32);           \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 24);           \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 16);           \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 63);           \
    (a) ^= (a) >> 29;                                        \
    (c) ^= (c) >> 37;                                        \
} while (0)

/* searching the square: straight on ... */
#define COLROUND(s) do {                                     \
    QR((s)[0],(s)[4],(s)[ 8],(s)[12]);                       \
    QR((s)[1],(s)[5],(s)[ 9],(s)[13]);                       \
    QR((s)[2],(s)[6],(s)[10],(s)[14]);                       \
    QR((s)[3],(s)[7],(s)[11],(s)[15]);                       \
} while (0)

/* ... then turned ninety degrees, corners folded into the centre */
#define DIAGROUND(s) do {                                    \
    QR((s)[0],(s)[5],(s)[10],(s)[15]);                       \
    QR((s)[1],(s)[6],(s)[11],(s)[12]);                       \
    QR((s)[2],(s)[7],(s)[ 8],(s)[13]);                       \
    QR((s)[3],(s)[4],(s)[ 9],(s)[14]);                       \
} while (0)

#define TURN90(s) do { COLROUND(s); DIAGROUND(s); } while (0)

/* the owl's seal */
static inline uint64_t owl(uint64_t h) {
    h ^= h >> 32;
    h += rotl64(h, 27);
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *restrict data, size_t len)
{
    if (len < 128) {
        /* ---- short ground: the deer gathers a smaller square ---- */
        uint64_t a = RIVERBANK[0] ^ (uint64_t)len;
        uint64_t b = RIVERBANK[1];
        uint64_t c = RIVERBANK[2];
        uint64_t d = RIVERBANK[3] ^ rotl64((uint64_t)len, 32);
        uint64_t t[4];
        uint64_t blk = 0;
        size_t i = 0, rem;

        while (len - i >= 32) {
            memcpy(t, data + i, 32);          /* silence until all four are laid */
            a ^= t[0]; b ^= t[1]; c ^= t[2]; d ^= t[3];
            c += blk++;                       /* order fixed by placement */
            QR(a, b, c, d);
            QR(a, c, d, b);                   /* the ninety-degree turn */
            i += 32;
        }

        rem = len - i;
        t[0] = t[1] = t[2] = t[3] = 0;
        if (rem) memcpy(t, data + i, rem);
        a ^= t[0]; b ^= t[1]; c ^= t[2]; d ^= t[3];
        b ^= (uint64_t)rem + 0x9E3779B97F4A7C15ULL;
        d ^= (uint64_t)len;
        c += blk;

        QR(a, b, c, d); QR(a, c, d, b);       /* three square-rounds at the shrine */
        QR(a, b, c, d); QR(a, c, d, b);
        QR(a, b, c, d); QR(a, c, d, b);

        return owl((a + rotl64(b, 21)) ^ (c + rotl64(d, 43)));
    } else {
        /* ---- the mountain path: heaps of 128 marks, one square per heap ---- */
        uint64_t s[16];
        uint64_t m[16];
        uint64_t blk = 0;
        size_t i = 0, rem;
        int j;

        for (j = 0; j < 16; j++) s[j] = RIVERBANK[j];
        s[0] ^= (uint64_t)len;
        s[7] ^= rotl64((uint64_t)len, 32);

        while (len - i >= 128) {
            memcpy(m, data + i, 128);         /* the swat: whole heap laid, nothing spoken */
            for (j = 0; j < 16; j++) s[j] ^= m[j];
            s[12] += blk++;                   /* the order, written as place */
            TURN90(s);                        /* the heap races; the shrine carries it forward */
            i += 128;
        }

        rem = len - i;
        memset(m, 0, sizeof m);
        if (rem) memcpy(m, data + i, rem);
        for (j = 0; j < 16; j++) s[j] ^= m[j];
        s[12] += blk;
        s[13] ^= (uint64_t)rem + 0x9E3779B97F4A7C15ULL;
        s[15] ^= (uint64_t)len;

        /* again and again, until one changed mark has smeared across every square */
        TURN90(s); TURN90(s); TURN90(s); TURN90(s);

        {   /* the owl calls the shape inside the dreamer inside: 16 -> 8 -> 4 -> 2 -> 1 */
            uint64_t x0 = s[0] ^ rotl64(s[ 8], 13);
            uint64_t x1 = s[1] ^ rotl64(s[ 9], 23);
            uint64_t x2 = s[2] ^ rotl64(s[10], 31);
            uint64_t x3 = s[3] ^ rotl64(s[11], 41);
            uint64_t x4 = s[4] ^ rotl64(s[12], 47);
            uint64_t x5 = s[5] ^ rotl64(s[13], 53);
            uint64_t x6 = s[6] ^ rotl64(s[14],  7);
            uint64_t x7 = s[7] ^ rotl64(s[15], 17);
            uint64_t y0 = x0 + rotl64(x4, 21);
            uint64_t y1 = x1 + rotl64(x5, 29);
            uint64_t y2 = x2 + rotl64(x6, 37);
            uint64_t y3 = x3 + rotl64(x7, 43);
            uint64_t z0 = y0 ^ rotl64(y2, 25);
            uint64_t z1 = y1 ^ rotl64(y3, 51);
            return owl(z0 + rotl64(z1, 33));
        }
    }
}
