#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= the world, made literal =========================
   marks              -> input bytes, in their given order
   chained prisoners  -> 8 lanes; each holds one mark of the stripe in flight
   "the angle"        -> rotate-then-add. No prisoner multiplies.
                         No prisoner looks at the mark (no compare, no branch).
   angle passed on    -> lane j adds lane j-1's PREVIOUS angle; lane 0 <- lane 7,
                         a closed chain, so the first mark still trembles in the last hand
   blank pillar       -> the lane bank; never read whole inside the loop
   the flood          -> one in-order sweep; does not recede until every mark is through
   wire shapes        -> the 8 anchored lanes: same angle in, own rotation + own key,
                         so each wire leans differently
   THE SILHOUETTE     -> ONE multiply-fold + ONE avalanche, after the flood, read once.
                         This is the only mixing round in the entire hash.
   shallow water      -> len < 64: the flood never reaches the wires, so the pit
                         holds only two of the chained (same algebra, 2 lanes)
   ================================================================== */

static const uint64_t K[16] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0x2545F4914F6CDD1DULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
    0x8EBC6AF09C88C6E3ULL, 0x589965CBA07BB1F9ULL, 0x2D358DCCAA6C78A5ULL,
    0x8BB84B93962EACC9ULL, 0x4B33A62ED433D4A3ULL, 0x4D5A2DA51DE1AA47ULL,
    0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
    0x27D4EB2F165667C5ULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, unsigned r){ return (x << r) | (x >> (64u - r)); }

/* "which cross another" - two wires casting one shadow.
   64x64->128 multiply, high half folded onto low. (xxh3's mul128_fold64.) */
static inline uint64_t crossw(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)prod ^ (uint64_t)(prod >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32;
    uint64_t bl = (uint32_t)b, bh = b >> 32;
    uint64_t m0 = al*bl, m1 = ah*bl, m2 = al*bh, m3 = ah*bh;
    uint64_t mid = (m0 >> 32) + (uint32_t)m1 + (uint32_t)m2;
    uint64_t lo  = (m0 & 0xFFFFFFFFULL) | (mid << 32);
    uint64_t hi  = m3 + (m1 >> 32) + (m2 >> 32) + (mid >> 32);
    return lo ^ hi;
#endif
}

/* the shadow-shape itself, drawn once the flood is gone. (MurmurHash3 fmix64.) */
static inline uint64_t silhouette(uint64_t h)
{
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

/* one pass of the flood over the eight wires: 8 loads, 8 xors, 8 rotates, 8 adds.
   all eight updates read only the PREVIOUS pass's angles -> 8 independent chains,
   latency-free, issue-bound. No multiply. Nothing is read whole. */
#define FLOOD(k0,k1,k2,k3,k4,k5,k6,k7, r0,r1,r2,r3,r4,r5,r6,r7, q)   \
    do {                                                             \
        uint64_t t0=a0,t1=a1,t2=a2,t3=a3,t4=a4,t5=a5,t6=a6,t7=a7;    \
        a0 = rotl64(t0 ^ (ld64((q) +  0) ^ (k0)), (r0)) + t7;        \
        a1 = rotl64(t1 ^ (ld64((q) +  8) ^ (k1)), (r1)) + t0;        \
        a2 = rotl64(t2 ^ (ld64((q) + 16) ^ (k2)), (r2)) + t1;        \
        a3 = rotl64(t3 ^ (ld64((q) + 24) ^ (k3)), (r3)) + t2;        \
        a4 = rotl64(t4 ^ (ld64((q) + 32) ^ (k4)), (r4)) + t3;        \
        a5 = rotl64(t5 ^ (ld64((q) + 40) ^ (k5)), (r5)) + t4;        \
        a6 = rotl64(t6 ^ (ld64((q) + 48) ^ (k6)), (r6)) + t5;        \
        a7 = rotl64(t7 ^ (ld64((q) + 56) ^ (k7)), (r7)) + t6;        \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;

    /* ---------- regime test: does the flood reach the wires' anchors? ---------- */
    if (len < 64) {
        /* shallow water: only two of the chained. Same angle algebra, no wire bank,
           and a cheaper silhouette (one cross), so short piles are never penalised
           by a finalizer sized for eight wires. */
        uint64_t h0 = K[12] ^ ((uint64_t)len * K[13]);
        uint64_t h1 = K[14] ^ rotl64((uint64_t)len + 1u, 32);
        const unsigned char *q = p;
        size_t n = len;
        while (n >= 16) {
            uint64_t u0 = h0, u1 = h1;
            h0 = rotl64(u0 ^ (ld64(q    ) ^ K[0]), 27) + u1;
            h1 = rotl64(u1 ^ (ld64(q + 8) ^ K[1]), 41) + u0;
            q += 16; n -= 16;
        }
        if (n >= 8) { h0 = rotl64(h0 ^ (ld64(q) ^ K[2]), 23) + h1; q += 8; n -= 8; }
        if (n >= 4) { h1 = rotl64(h1 ^ (ld32(q) ^ K[3]), 31) + h0; q += 4; n -= 4; }
        while (n) { h0 = rotl64(h0 ^ ((uint64_t)(*q++) ^ K[4]), 11) + h1; --n; }
        return silhouette(crossw(h0 ^ K[5], h1 ^ K[6]) ^ rotl64(h0 + h1, 32));
    }

    /* ---------- the flood: eight wires anchored in the riverbed ---------- */
    uint64_t a0 = K[0] ^ ((uint64_t)len * K[8]);
    uint64_t a1 = K[1], a2 = K[2], a3 = K[3];
    uint64_t a4 = K[4], a5 = K[5], a6 = K[6], a7 = K[7];

    size_t nstripe = len >> 6;
    const unsigned char *q = p;
    for (size_t b = 0; b < nstripe; ++b, q += 64)
        FLOOD(K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],
              13,29,41,7,53,19,37,59, q);

    /* the water does not recede until EVERY mark has gone through: the final
       stripe is re-read overlapping the tail, with the other anchoring and the
       rotations reversed, so no mark is left dry and nothing cancels. */
    FLOOD(K[8],K[9],K[10],K[11],K[12],K[13],K[14],K[15],
          59,37,19,53,7,41,29,13, p + (len - 64));

    /* ---------- the silhouette: read once, kept alone ---------- */
    uint64_t s = (uint64_t)len * K[14];
    s ^= crossw(a0 ^ K[0], a1 ^ K[1]);             /* which lean       */
    s += crossw(a2 ^ K[2], a3 ^ K[3]);
    s ^= crossw(a4 ^ K[4], a5 ^ K[5]);
    s += crossw(a6 ^ K[6], a7 ^ K[7]);
    s ^= crossw((a0 + a4) ^ K[9],  (a2 + a6) ^ K[10]);  /* which cross another */
    s += crossw((a1 + a5) ^ K[11], (a3 + a7) ^ K[12]);
    return silhouette(s);                          /* the shadow-shape */
}
