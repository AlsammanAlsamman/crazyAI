#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============ the wall's six fixed cracks, the same six every time ============
   One reading of the shadow = six rotations at fixed places
   (13, 32, 16, 21, 17, 32) over a four-knot row.  This is SipRound.
   There is no multiplication anywhere in this world.                         */

static inline uint64_t bend(uint64_t x, unsigned r) {
    return (x << (r & 63)) | (x >> ((64u - r) & 63));   /* r==64 == r==0 */
}

#define CRACKS(a,b,c,d) do {                                               \
    (a) += (b); (b) = bend((b),13); (b) ^= (a); (a) = bend((a),32);        \
    (c) += (d); (d) = bend((d),16); (d) ^= (c);                            \
    (a) += (d); (d) = bend((d),21); (d) ^= (a);                            \
    (c) += (b); (b) = bend((b),17); (b) ^= (c); (c) = bend((c),32);        \
} while (0)

static inline uint64_t mark8(const unsigned char *p) {
    uint64_t m; memcpy(&m, p, sizeof m); return m;      /* one mov */
}

/* the small cup: this mark's own strokes, counted as pebbles,
   decide how far the brick-incense swings on its short chain      */
#define CUP(m) ((unsigned)__builtin_popcountll(m))

/* knot the mark onto the row, then let the smoke bend the throw   */
#define KNOT(v,m) do { uint64_t _m = (m); (v) ^= _m; (v) = bend((v), CUP(_m)); } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t n = len;

    /* the row as first hung between fire and wall */
    uint64_t v0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL ^ (uint64_t)len;

    /* ---- REGIME A: a long pile.  The row is strung across the whole mouth
            of the piazza and four marks walk abreast, one to each quarter.
            Guarded: only when the pile is big enough to pay for stringing
            it; otherwise fall straight through to the short span below. */
    if (n >= 256) {
        uint64_t a0=v0, a1=v1, a2=v2, a3=v3;
        uint64_t b0=v0^0x9e3779b97f4a7c15ULL, b1=v1^0x9e3779b97f4a7c15ULL, b2=v2, b3=v3;
        uint64_t c0=v0^0xbf58476d1ce4e5b9ULL, c1=v1^0xbf58476d1ce4e5b9ULL, c2=v2, c3=v3;
        uint64_t d0=v0^0x94d049bb133111ebULL, d1=v1^0x94d049bb133111ebULL, d2=v2, d3=v3;

        do {
            KNOT(a0, mark8(p +   0)); KNOT(a1, mark8(p +   8));
            KNOT(a2, mark8(p +  16)); KNOT(a3, mark8(p +  24));
            KNOT(b0, mark8(p +  32)); KNOT(b1, mark8(p +  40));
            KNOT(b2, mark8(p +  48)); KNOT(b3, mark8(p +  56));
            KNOT(c0, mark8(p +  64)); KNOT(c1, mark8(p +  72));
            KNOT(c2, mark8(p +  80)); KNOT(c3, mark8(p +  88));
            KNOT(d0, mark8(p +  96)); KNOT(d1, mark8(p + 104));
            KNOT(d2, mark8(p + 112)); KNOT(d3, mark8(p + 120));
            CRACKS(a0,a1,a2,a3);
            CRACKS(b0,b1,b2,b3);
            CRACKS(c0,c1,c2,c3);
            CRACKS(d0,d1,d2,d3);
            p += 128; n -= 128;
        } while (n >= 128);

        /* knot the four quarters back into one row: every quarter must
           hang on every other, or the line does not hang at all       */
        v0 = a0 ^ b1 ^ c2 ^ d3;
        v1 = a1 ^ b2 ^ c3 ^ d0;
        v2 = a2 ^ b3 ^ c0 ^ d1;
        v3 = a3 ^ b0 ^ c1 ^ d2;
        CRACKS(v0,v1,v2,v3);
        CRACKS(v0,v1,v2,v3);
    }

    /* ---- REGIME B: the short pile -- and every long pile's remainder.
            One short span; each mark walks past the fire alone and the
            whole chain is read once for it.                           */
    while (n >= 8) {
        uint64_t m = mark8(p);
        v3 ^= m;                    /* the new knot            */
        v3 = bend(v3, CUP(m));      /* the cup bends the throw */
        CRACKS(v0,v1,v2,v3);        /* shadow of the whole row */
        v0 ^= m;
        p += 8; n -= 8;
    }

    /* the last short mark, with the pile's own length knotted in, so that
       moving a mark changes every bend after it and the length is not free */
    {
        uint64_t t = ((uint64_t)(len & 0xff)) << 56;
        switch (n) {
            case 7: t |= (uint64_t)p[6] << 48; /* fall through */
            case 6: t |= (uint64_t)p[5] << 40; /* fall through */
            case 5: t |= (uint64_t)p[4] << 32; /* fall through */
            case 4: t |= (uint64_t)p[3] << 24; /* fall through */
            case 3: t |= (uint64_t)p[2] << 16; /* fall through */
            case 2: t |= (uint64_t)p[1] <<  8; /* fall through */
            case 1: t |= (uint64_t)p[0];       /* fall through */
            default: break;
        }
        v3 ^= t;
        v3 = bend(v3, CUP(t));
        CRACKS(v0,v1,v2,v3);
        v0 ^= t;
    }

    /* ---- the mouth of the stream: the clay stub held under running water
            until only the hardened ridges remain.  Four full readings with
            no mark feeding in, then the whole row folded into one fist --
            lossy, so no hooded reader walks back from print to pile.    */
    v2 ^= 0xff;
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
