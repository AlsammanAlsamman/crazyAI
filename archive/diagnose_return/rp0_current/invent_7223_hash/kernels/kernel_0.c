#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t x;
    memcpy(&x, p, sizeof x);   /* unaligned word load; -O3 emits one mov */
    return x;
}

/* ONE TUMBLE.  The sphere turns a fixed count, strikes a stalk, sobs once,
   and cracks a hairline into itself: every step is add / rotate / xor of the
   state against a displaced copy of itself.  No multiplication exists on this
   trail.  This permutation is SipRound (Aumasson & Bernstein, 2012). */
#define TUMBLE(v0, v1, v2, v3)                                            \
    do {                                                                  \
        v0 += v1;  v1 = rotl64(v1, 13);  v1 ^= v0;  v0 = rotl64(v0, 32);  \
        v2 += v3;  v3 = rotl64(v3, 16);  v3 ^= v2;                        \
        v0 += v3;  v3 = rotl64(v3, 21);  v3 ^= v0;                        \
        v2 += v1;  v1 = rotl64(v1, 17);  v1 ^= v2;  v2 = rotl64(v2, 32);  \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* The one sphere, four faces, set at the trail's high mouth.
       (SipHash initial state with a zero key.) */
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;

    size_t i = 0;

    /* ---- REGIME TEST: is the pile longer than one turn of the coil? ---- */
    if (len >= 32u) {
        /* WIDE COIL.  The sphere is rolling fast enough that four marks land
           on four different faces between one stalk and the next: absorb 32
           bytes, then the same fixed two tumbles -- no more, no fewer. */
        size_t nblk = len & ~(size_t)31;
        for (; i < nblk; i += 32) {
            uint64_t m0 = ld64(p + i);
            uint64_t m1 = ld64(p + i +  8);
            uint64_t m2 = ld64(p + i + 16);
            uint64_t m3 = ld64(p + i + 24);
            v0 ^= m0; v1 ^= m1; v2 ^= m2; v3 ^= m3;
            TUMBLE(v0, v1, v2, v3);
            TUMBLE(v0, v1, v2, v3);
        }
    }

    /* NARROW COIL.  The short pile, and the eggshells the wide coil shed:
       one mark-word per stalk, two tumbles per stalk.  For len < 32 nothing
       above ran and this is verbatim SipHash-2-4 with a zero key. */
    {
        size_t end = len & ~(size_t)7;
        for (; i < end; i += 8) {
            uint64_t m = ld64(p + i);
            v3 ^= m;
            TUMBLE(v0, v1, v2, v3);
            TUMBLE(v0, v1, v2, v3);
            v0 ^= m;
        }
    }

    /* The last eggshell: the leftover marks, with the pile's own length
       pressed into the top face so a short pile can never wear a long
       pile's token. */
    {
        uint64_t b = ((uint64_t)len) << 56;
        switch (len & 7u) {
            case 7: b |= (uint64_t)p[i + 6] << 48; /* fall through */
            case 6: b |= (uint64_t)p[i + 5] << 40; /* fall through */
            case 5: b |= (uint64_t)p[i + 4] << 32; /* fall through */
            case 4: b |= (uint64_t)p[i + 3] << 24; /* fall through */
            case 3: b |= (uint64_t)p[i + 2] << 16; /* fall through */
            case 2: b |= (uint64_t)p[i + 1] <<  8; /* fall through */
            case 1: b |= (uint64_t)p[i + 0];       /* fall through */
            case 0: break;
        }
        v3 ^= b;
        TUMBLE(v0, v1, v2, v3);
        TUMBLE(v0, v1, v2, v3);
        v0 ^= b;
    }

    /* Only the last, smallest crack in the final stalk is kept.  Four closing
       tumbles; then the sphere, the dust and the footprints all go into the
       cenote and only the token is handed over. */
    v2 ^= 0xffu;
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
