#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* The inkwell that never runs dry: eight odd drops. */
static const uint64_t INK[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL,
    0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL
};
#define DROP   0x9E3779B97F4A7C15ULL
#define STORM  16   /* the storm passes once -- O(1), not O(n) */

static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* THE BRIDGE: one reduced round. A pluck on any string shivers into the rest.
   Each (a[j],b[j]) step is an invertible Feistel-like ARX pair, so a difference
   can never die; the re-stringing (lower half turns by 1, upper by 2) changes
   the pairing every round, so all eight strings couple within ~4 rounds. */
static inline void bridge(uint64_t a[4], uint64_t b[4], uint64_t rc)
{
    int j;
    uint64_t t0, t1, t2, t3;
    for (j = 0; j < 4; j++) {
        a[j] += b[j];
        b[j]  = ROTL64(b[j], 23) ^ a[j];
        a[j]  = ROTL64(a[j], 41);
    }
    t0 = a[1]; t1 = a[2]; t2 = a[3]; t3 = a[0];
    a[0] = t0 ^ rc; a[1] = t1; a[2] = t2; a[3] = t3;
    t0 = b[2]; t1 = b[3]; t2 = b[0]; t3 = b[1];
    b[0] = t0; b[1] = t1; b[2] = t2; b[3] = t3;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t a[4], b[4], w[8], ctr, h;
    size_t i, n, rem;
    int j, r;

    /* ---- REGIME 1: the pile does not cover the bars (len < 64).
       Raising the whole instrument would cost a fixed 16-round storm for a
       handful of marks. One bar, one mark at a time, each with a fresh drop. */
    if (len < 64) {
        uint64_t k = 0;
        h = INK[0] ^ ((uint64_t)len * DROP);
        for (i = 0; i + 8 <= len; i += 8) {
            k += DROP;                    /* a feather never used twice */
            h ^= rd8(p + i) + k;
            h *= INK[1];
            h ^= h >> 29;
        }
        if (i < len) {                    /* the last 1..7 marks */
            uint64_t t;
            size_t rr = len - i;
            if (len >= 8)     t = rd8(p + len - 8);                        /* overlapping read */
            else if (rr >= 4) t = rd4(p + i) | (rd4(p + len - 4) << 32);
            else              t = ((uint64_t)p[0] << 16)
                                | ((uint64_t)p[rr >> 1] << 8)
                                | ((uint64_t)p[len - 1]);
            k += DROP;
            h ^= t + k;
            h *= INK[2];
            h ^= h >> 32;
        }
        h *= INK[3]; h ^= h >> 29;        /* a brief gust, not the full storm */
        h *= INK[4]; h ^= h >> 32;
        return h;
    }

    /* ---- REGIME 2: the full instrument. Eight strings over the counting-bars. */
    for (j = 0; j < 4; j++) { a[j] = INK[j]; b[j] = INK[j + 4]; }
    a[0] ^= (uint64_t)len * DROP;
    b[3] ^= ROTL64((uint64_t)len + 1, 32);

    ctr = 0;
    n = len >> 6;
    for (i = 0; i < n; i++) {
        ctr += DROP;                                  /* fresh drop from the inkwell */
        memcpy(w, p, 64);                             /* a chord of eight marks */
        for (j = 0; j < 4; j++) a[j] ^= w[j]     + (ctr ^ INK[j]);     /* weight, not shape */
        for (j = 0; j < 4; j++) b[j] ^= w[j + 4] + (ctr ^ INK[j + 4]);
        bridge(a, b, ctr);                            /* caught mid-tremor: ONE round */
        p += 64;
    }
    rem = len & 63;
    if (rem) {
        unsigned char tail[64];
        memset(tail, 0, 64);
        memcpy(tail, p, rem);
        memcpy(w, tail, 64);
        ctr += DROP;
        for (j = 0; j < 4; j++) a[j] ^= w[j]     + (ctr ^ INK[j]);
        for (j = 0; j < 4; j++) b[j] ^= w[j + 4] + (ctr ^ INK[j + 4]);
        bridge(a, b, ctr ^ (uint64_t)rem);
    }

    /* ---- THE STORM: hands off, one pass over the strings. O(1), never per byte. */
    for (r = 0; r < STORM; r++)
        bridge(a, b, INK[r & 7] + (uint64_t)(r * 0x9E3779B1u));

    /* ---- Eight small notes, no more: each string collapses to one byte,
       every bit of it folded in; the other 448 bits go to the pickers. */
    h = 0;
    for (j = 0; j < 4; j++) {
        uint64_t v = a[j], u = b[j];
        v ^= v >> 32; v ^= v >> 16; v ^= v >> 8;
        u ^= u >> 32; u ^= u >> 16; u ^= u >> 8;
        h |= (v & 0xFFULL) << (8 * j);
        h |= (u & 0xFFULL) << (8 * (j + 4));
    }
    return h;
}
