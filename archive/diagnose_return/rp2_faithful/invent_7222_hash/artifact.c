#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===========================================================================
   THE DESK.  The ship's cycle-light is the same starlight colour no matter the
   hour: one fixed constant schedule, never varying with position or round.
   These are also the wooden numbered keeps the stone is finally read against.
   =========================================================================== */
#define KEEP0 0x243F6A8885A308D3ULL
#define KEEP1 0x13198A2E03707344ULL
#define KEEP2 0xA4093822299F31D0ULL
#define KEEP3 0x082EFA98EC4E6C89ULL
#define KEEP4 0x452821E638D01377ULL
#define KEEP5 0xBE5466CF34E90C6CULL
#define KEEP6 0xC0AC29B7C97C50DDULL
#define KEEP7 0x9216D5D98979FB1BULL

static inline uint64_t turn(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

/* THE QUARTER TURN.  One 90-degree turn of the stone about one axis: four
   corners move, the other four stay.  Add-rotate-xor only -- the mark's weight
   is pressed into the corner's ALREADY-TURNED value, and the corner's own
   memory (the carry out of +=) bends how deep the press goes.  No multiply. */
#define QTURN(a,b,c,d) do {                        \
        (a) += (b); (d) = turn((d) ^ (a), 32);     \
        (c) += (d); (b) = turn((b) ^ (c), 24);     \
        (a) += (b); (d) = turn((d) ^ (a), 16);     \
        (c) += (d); (b) = turn((b) ^ (c), 63);     \
    } while (0)

/* reading a mark as a WEIGHT, never as a word */
static inline uint64_t weight64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t weight32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;

    /* --- SEED 1: the stone is seated ONCE and never reset between marks. --- */
    uint64_t s0 = KEEP0, s1 = KEEP1, s2 = KEEP2, s3 = KEEP3;
    uint64_t s4 = KEEP4, s5 = KEEP5, s6 = KEEP6, s7 = KEEP7;

    /* how far down the desk's numbered groove the pile reaches is pressed in
       first -- and it is also what the native glances at to choose a groove. */
    s0 ^= (uint64_t)len;
    s7 ^= turn((uint64_t)len | 1ULL, 32);

    /* ======================= REGIME 1: THE WIDE GROOVE =======================
       A long pile.  The groove is a full stone-width, so eight marks-worth lie
       side by side in one seating.  Two axes of the stone turn at once; they
       touch disjoint corners, so the second turn is free in latency.  The
       pairing of corners ALTERNATES between seatings -- the stone rolling one
       place along the groove -- which is what carries a difference sideways
       into corners it did not first land on.
       GUARDED: entered only for long piles; short piles fall straight through
       to the narrow groove, so the wide path can never cost a short pile. */
    if (n >= 128) {
        do {
            s0 ^= weight64(p +   0); s1 ^= weight64(p +   8);
            s2 ^= weight64(p +  16); s3 ^= weight64(p +  24);
            s4 ^= weight64(p +  32); s5 ^= weight64(p +  40);
            s6 ^= weight64(p +  48); s7 ^= weight64(p +  56);
            QTURN(s0, s2, s4, s6);
            QTURN(s1, s3, s5, s7);

            s0 ^= weight64(p +  64); s1 ^= weight64(p +  72);
            s2 ^= weight64(p +  80); s3 ^= weight64(p +  88);
            s4 ^= weight64(p +  96); s5 ^= weight64(p + 104);
            s6 ^= weight64(p + 112); s7 ^= weight64(p + 120);
            QTURN(s0, s3, s4, s7);      /* rolled one place: new pairing */
            QTURN(s1, s2, s5, s6);

            p += 128; n -= 128;
        } while (n >= 128);
    }

    /* ====================== REGIME 2: THE NARROW GROOVE ======================
       A handful of marks -- or the leftover of a long pile.  Four marks-worth
       per seating, one quarter turn.  The chalk residue creeps into the far
       corners off the critical path so that a medium pile still uses all eight
       corners of the stone. */
    while (n >= 32) {
        s0 ^= weight64(p +  0); s1 ^= weight64(p +  8);
        s2 ^= weight64(p + 16); s3 ^= weight64(p + 24);
        QTURN(s0, s1, s2, s3);
        s4 += s1;  s5 ^= turn(s2, 19);  s6 += s3;  s7 ^= turn(s0, 43);
        p += 32; n -= 32;
    }

    /* --- the last, short seating: overlapping presses, so every mark of the
       remainder is read without a byte-at-a-time loop.  Overlap is harmless
       because the groove number (len) is already pressed in. --- */
    {
        uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        if      (n >= 24) { t0 = weight64(p); t1 = weight64(p+8); t2 = weight64(p+16); t3 = weight64(p+n-8); }
        else if (n >= 16) { t0 = weight64(p); t1 = weight64(p+8); t2 = weight64(p+n-8); }
        else if (n >=  8) { t0 = weight64(p); t1 = weight64(p+n-8); }
        else if (n >=  4) { t0 = weight32(p); t1 = weight32(p+n-4); }
        else if (n)       { t0 = (uint64_t)p[0]
                               | ((uint64_t)p[n >> 1] << 16)
                               | ((uint64_t)p[n - 1]  << 32); }
        s0 ^= t0; s1 ^= t1; s2 ^= t2; s3 ^= t3;
        QTURN(s0, s1, s2, s3);
    }

    /* =================== THE LIFT (SEED 3) ===================
       Nothing new is dropped in.  The stone turns twice more with both axes
       moving and the pairing alternating, so a difference from the last
       seating reaches all eight corners; then the eight corners are read down
       against the wooden numbered keeps to ONE number.  Every intermediate
       turn, all the groove-dust and chalk residue, is swept off and thrown
       away -- only this last seated number ever leaves the desk. */
    QTURN(s0, s2, s4, s6);  QTURN(s1, s3, s5, s7);
    QTURN(s0, s3, s4, s7);  QTURN(s1, s2, s5, s6);

    {
        uint64_t a = s0 + turn(s4, 11);
        uint64_t b = s1 ^ turn(s5, 29);
        uint64_t c = s2 + turn(s6, 47);
        uint64_t d = s3 ^ turn(s7,  7);
        QTURN(a, b, c, d);                          /* the final seating */
        return (a + turn(b, 13)) ^ (c + turn(d, 41)); /* read off the keeps */
    }
}
