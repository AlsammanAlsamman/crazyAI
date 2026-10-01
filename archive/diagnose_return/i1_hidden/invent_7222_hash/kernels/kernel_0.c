#include <stdint.h>
#include <stddef.h>

/* the four faces of the stone: four distinct odd multipliers.
   odd => x |-> (x ^ w) * M is a bijection on 2^64, so no two
   folding-paths that started differently collapse onto one. */
#define M0 0x9E3779B97F4A7C15ULL
#define M1 0xBF58476D1CE4E5B9ULL
#define M2 0x94D049BB133111EBULL
#define M3 0xD6E8FEB86659FD93ULL

static inline uint64_t rotl64(uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

/* one press: the weight is dropped into the seated face's *current*
   position, never onto a clean face (SEED 2). */
#define PRESS(s, M, w) ((s) = ((s) ^ (uint64_t)(w)) * (M))

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the bone-coloured die-stone: ONE object, four faces, 256 bits.
       Seeded once, never reset between marks (SEED 1). */
    uint64_t s0 = 0x243F6A8885A308D3ULL ^ (uint64_t)len;
    uint64_t s1 = 0x13198A2E03707344ULL;
    uint64_t s2 = 0xA4093822299F31D0ULL;
    uint64_t s3 = 0x082EFA98EC4E6C89ULL;

    size_t i = 0;

    /* SIZE GUARD (see VERDICT, risk R3): the cross-face callus is hoisted
       to once per 64 marks. Done once per revolution it would sit on the
       dependency chain and cost ~1.5 cycles/byte instead of ~1.0. Long
       piles therefore take this path; short piles skip it entirely so
       they never pay for a block they do not fill. */
    if (len >= 64) {
        size_t blocks = len >> 6;                 /* 16 revolutions each */
        for (size_t b = 0; b < blocks; ++b) {
            const unsigned char *p = data + i;
#pragma GCC unroll 16
            for (int k = 0; k < 64; k += 4) {     /* one full quarter-cycle */
                PRESS(s0, M0, p[k + 0]);
                PRESS(s1, M1, p[k + 1]);
                PRESS(s2, M2, p[k + 2]);
                PRESS(s3, M3, p[k + 3]);
            }
            /* it is one stone: the callus travels across the faces. */
            {
                uint64_t t = s0;
                s0 ^= rotl64(s1, 17);
                s1 ^= rotl64(s2, 31);
                s2 ^= rotl64(s3, 43);
                s3 ^= rotl64(t,  53);
            }
            i += 64;
        }
    }

    /* the rest of the pile: still one press per mark, still quarter-turning */
    for (; i + 4 <= len; i += 4) {
        PRESS(s0, M0, data[i + 0]);
        PRESS(s1, M1, data[i + 1]);
        PRESS(s2, M2, data[i + 2]);
        PRESS(s3, M3, data[i + 3]);
    }
    /* TAIL GUARD (risk R1): resolved with one branch total, not a
       per-byte switch on (i & 3). */
    switch (len - i) {
        case 3: PRESS(s2, M2, data[i + 2]); /* fall through */
        case 2: PRESS(s1, M1, data[i + 1]); /* fall through */
        case 1: PRESS(s0, M0, data[i + 0]); /* fall through */
        default: break;
    }

    /* Lift the stone and read it against the wooden numbered keeps.
       Everything else -- the other faces, the groove-dust, every
       intermediate turn -- is swept off and thrown away. This is the
       only 256 -> 64 projection, and it happens exactly once.
       SHORT-INPUT GUARD (risk R2): len < 4 leaves faces unpressed, so
       len is folded in and all four faces are read unconditionally. */
    {
        uint64_t a = (s0 ^ rotl64(s1, 29)) * M2;
        uint64_t b = (s2 ^ rotl64(s3, 47)) * M3;
        uint64_t h = (a ^ rotl64(b, 31)) + (uint64_t)len * M0;

        h ^= h >> 27; h *= 0x3C79AC492BA7B653ULL;   /* moremur finisher */
        h ^= h >> 33; h *= 0x1C69B3F74AC4AE35ULL;
        h ^= h >> 27;
        return h;
    }
}
