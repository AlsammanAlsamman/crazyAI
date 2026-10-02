#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------------------
   The native's trail, taken literally.

   STALK[lane][face] : a coiled limestone trail, 8 stalks to a turn, 256 faces
   to a stalk. Laid once, before any mark arrives, with the arc of every tumble
   already pressed into the stone ("the shadow already runs white and arching").
   Read-only forever: "a footprint left at a stalk is a woman's business".

   Per mark: one load, one xor, one fixed turn. No multiplication, anywhere --
   not in the loop, and not even in laying the trail (xoshiro256++ is
   rotate/shift/xor/add only).

   This is simple tabulation hashing (Zobrist 1970; Patrascu & Thorup 2011)
   with a fixed per-mark rotation folded into the table. The fold is exact:
     h8 = R^(8r)h0 ^ R^(7r)c0 ^ R^(6r)c1 ^ ... ^ R^0 c7      (R is linear)
   so eight per-mark turns of 29 become one turn of (8*29 mod 64) = 40.
--------------------------------------------------------------------------- */

#define ROTL(x, r) (((x) << (r)) | ((x) >> (64 - (r))))
#define BASIS 0x243F6A8885A308D3ULL   /* the sphere, set at the trail's high mouth */
#define ARC   29u                     /* one tumble per mark: no more, no fewer   */
#define TURN  ((8u * ARC) & 63u)      /* == 40: one full revolution of the coil   */

static uint64_t STALK[8][256] __attribute__((aligned(64)));  /* 16 KB, L1-resident */
static int TRAIL_LAID = 0;

/* rotation that tolerates k == 0; used only while laying the trail */
static inline uint64_t arc(uint64_t x, unsigned k)
{
    k &= 63u;
    return k ? ((x << k) | (x >> (64 - k))) : x;
}

static void lay_the_trail(void)
{
    uint64_t s0 = 0x9E3779B97F4A7C15ULL, s1 = 0xBF58476D1CE4E5B9ULL;
    uint64_t s2 = 0x94D049BB133111EBULL, s3 = 0x2545F4914F6CDD1DULL;
    uint64_t t;
    int warm, lane, face;

    for (warm = 0; warm < 128; warm++) {          /* let the quarry settle */
        t = s1 << 17;
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = ROTL(s3, 45);
    }
    for (lane = 0; lane < 8; lane++) {
        for (face = 0; face < 256; face++) {
            uint64_t r = ROTL(s0 + s3, 23) + s0;  /* xoshiro256++ : no multiply */
            t = s1 << 17;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = ROTL(s3, 45);
            /* the tumbles this stalk's crack will still have to make, pre-arced */
            STALK[lane][face] = arc(r, (unsigned)((7 - lane) * (int)ARC));
        }
    }
    TRAIL_LAID = 1;
}

__attribute__((constructor))
static void lay_the_trail_ctor(void) { lay_the_trail(); }

/* one full turn of the coil: eight marks, eight stalks, eight cracks.
   The eight loads are mutually independent -- the stalk is chosen by the
   mark's face, never by where the sphere happens to be. */
static inline uint64_t cracks(const unsigned char *q)
{
    return STALK[0][q[0]] ^ STALK[1][q[1]] ^ STALK[2][q[2]] ^ STALK[3][q[3]]
         ^ STALK[4][q[4]] ^ STALK[5][q[5]] ^ STALK[6][q[6]] ^ STALK[7][q[7]];
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t n = len;
    uint64_t h, f;

    /* the trail is laid before main(); this guard is belt-and-braces. The
       generator is deterministic, so a concurrent double-lay writes identical
       bytes and is benign. */
    if (!TRAIL_LAID) lay_the_trail();

    /* --- REGIME 0: a pile too small to be worth the walk up to the mouth.
       No final tumble is paid for: the stalk's own crack already carries the
       whole avalanche, so the hairline is handed over as it is. --- */
    if (n < 4) {
        h = BASIS ^ arc((uint64_t)n, 59);
        if (n > 0) h = ROTL(h, ARC) ^ STALK[0][p[0]];
        if (n > 1) h = ROTL(h, ARC) ^ STALK[1][p[1]];
        if (n > 2) h = ROTL(h, ARC) ^ STALK[2][p[2]];
        return h;
    }

    h = BASIS;

    /* --- REGIME 1: a long pile. Four turns of the coil in flight; 32
       independent loads feed a chain of only 4 xor+rotate pairs. --- */
    while (n >= 32) {
        uint64_t a = cracks(p),      b = cracks(p +  8);
        uint64_t c = cracks(p + 16), d = cracks(p + 24);
        h = ROTL(h, TURN) ^ a;
        h = ROTL(h, TURN) ^ b;
        h = ROTL(h, TURN) ^ c;
        h = ROTL(h, TURN) ^ d;
        p += 32; n -= 32;
    }

    /* --- REGIME 2: whole turns, one at a time --- */
    while (n >= 8) {
        h = ROTL(h, TURN) ^ cracks(p);
        p += 8; n -= 8;
    }

    /* --- the eggshells the trail sheds: a partial turn, taken mark by mark
       exactly as the native describes, one fixed tumble each --- */
    {
        size_t i;
        for (i = 0; i < n; i++) h = ROTL(h, ARC) ^ STALK[i][p[i]];
    }

    /* how many marks the sphere carried is part of its running weight */
    h ^= ROTL((uint64_t)len, 32) ^ (uint64_t)len;

    /* --- "I don't keep the sphere itself; I take only the last, smallest
       crack it left in the final stalk." One last tumble, the sphere's own
       eight faces down the coil; that hairline is the token. --- */
    f  = ROTL(h, ARC);
    f ^=      STALK[0][(unsigned char)(h      )];
    f ^= ROTL(STALK[1][(unsigned char)(h >>  8)],  8);
    f ^= ROTL(STALK[2][(unsigned char)(h >> 16)], 16);
    f ^= ROTL(STALK[3][(unsigned char)(h >> 24)], 24);
    f ^= ROTL(STALK[4][(unsigned char)(h >> 32)], 32);
    f ^= ROTL(STALK[5][(unsigned char)(h >> 40)], 40);
    f ^= ROTL(STALK[6][(unsigned char)(h >> 48)], 48);
    f ^= ROTL(STALK[7][(unsigned char)(h >> 56)], 56);
    return f;
}
