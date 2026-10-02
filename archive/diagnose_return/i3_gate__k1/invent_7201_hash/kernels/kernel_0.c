#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the slate wheel: it only ever turns forward, never back ---- */
static inline uint64_t wheel(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* ---- the wax: a mark is never read clean, only as an 8-byte smear ---- */
static inline uint64_t wax(const unsigned char *p) {
    uint64_t w;
    memcpy(&w, p, 8);          /* -O3 emits one unaligned mov */
    return w;
}

/* ---- the brine basin: frost-flowers 2..6, no two alike.
       splitmix64 / Stafford Mix13 finalizer, used verbatim, not invented.  ---- */
static inline uint64_t frost(uint64_t h) {
    h ^= h >> 30;                       /* flower 2 */
    h *= 0xBF58476D1CE4E5B9ULL;         /* flower 3 */
    h ^= h >> 27;                       /* flower 4 */
    h *= 0x94D049BB133111EBULL;         /* flower 5 */
    h ^= h >> 31;                       /* flower 6 */
    return h;
}

/* one swarm biting the wire: xor the smear in, turn the wheel forward,
   add the smear back (a bite cannot be walked off), add the jar's hunger. */
#define BITE(a, word, R, H) \
    do { uint64_t _w = (word); (a) = wheel((a) ^ _w, (R)) + _w + (uint64_t)(H); } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ---- regime test: is the pile thick enough to give all seven jars a bite? ---- */
    if (len < 56) {                      /* too thin: open one jar, bite the marks directly */
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        size_t i = 0;
        for (; i + 8 <= len; i += 8) BITE(h, wax(p + i), 11u, 0x27220A95u);
        uint64_t t = 0;
        for (; i < len; i++) t = (t << 8) | (uint64_t)p[i];
        BITE(h, t, 23u, 0x165667B1u);
        h ^= (uint64_t)len;
        return frost(h);                 /* same brine, never skipped */
    }

    /* ---- seven jars, seven hungers, seven bite-counts, seven rhythms ---- */
    uint64_t a0 = 0x9E3779B97F4A7C15ULL, a1 = 0xBF58476D1CE4E5B9ULL,
             a2 = 0x94D049BB133111EBULL, a3 = 0xD6E8FEB86659FD93ULL,
             a4 = 0xA0761D6478BD642FULL, a5 = 0xE7037ED1A0B428DBULL,
             a6 = 0x8EBC6AF09C88C6E3ULL;

    size_t rem = len;
    while (rem >= 56) {                  /* reads at most p[0..49]; 56 keeps a safe margin */
        BITE(a0, wax(p +  0),  7u, 0x27220A95u);
        BITE(a1, wax(p +  7), 11u, 0x165667B1u);
        BITE(a2, wax(p + 14), 19u, 0x05EBCA77u);
        BITE(a3, wax(p + 21), 23u, 0x42B2AE3Du);
        BITE(a4, wax(p + 28), 29u, 0x27D4EB2Fu);
        BITE(a5, wax(p + 35), 37u, 0x165667C5u);
        BITE(a6, wax(p + 42), 43u, 0x1E3779B1u);
        p   += 49;                       /* stride 7 x 7 jars: the wax never cools clean */
        rem -= 49;
    }

    /* ---- the last marks, round-robin so no jar is starved ---- */
    uint64_t lane[7] = { a0, a1, a2, a3, a4, a5, a6 };
    static const unsigned char RT[7] = { 7u, 11u, 19u, 23u, 29u, 37u, 43u };
    static const uint32_t      HT[7] = { 0x27220A95u, 0x165667B1u, 0x05EBCA77u,
                                         0x42B2AE3Du, 0x27D4EB2Fu, 0x165667C5u,
                                         0x1E3779B1u };
    unsigned j = 0;
    size_t i = 0;
    for (; i + 8 <= rem; i += 8) {
        BITE(lane[j], wax(p + i), RT[j], HT[j]);
        if (++j == 7) j = 0;
    }
    uint64_t t = 0;
    for (; i < rem; i++) t = (t << 8) | (uint64_t)p[i];
    BITE(lane[j], t, RT[j], HT[j]);

    /* ---- flower 1: the wire comes out of the wax; every jar thrown back in ---- */
    uint64_t h = lane[0]
               + wheel(lane[1],  9) + wheel(lane[2], 18) + wheel(lane[3], 27)
               + wheel(lane[4], 36) + wheel(lane[5], 45) + wheel(lane[6], 54);
    h ^= (uint64_t)len;

    /* ---- plunge once, wait for the frost to stop spreading, sketch to tin ---- */
    return frost(h);
}
