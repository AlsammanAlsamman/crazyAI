#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* --- the ship's cycle-light: the same starlight colour no matter the hour.
       Fixed constants, no seed, no clock, no length-dependent tuning.     --- */
#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL
#define LIGHT 0x9E3779B97F4A7C15ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t w64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t w32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* one fold: drop the mark's weight into the seated place, quarter-turn the
   stone, let its own memory of the last press bend how deep this one goes.
   Bijective in `face` (add, rotate, multiply by an odd constant). */
static inline uint64_t seat(uint64_t face, uint64_t weight) {
    face += weight * P2;
    face  = rotl64(face, 31);
    face *= P1;
    return face;
}

/* lifting one face's residue into the running read */
static inline uint64_t press(uint64_t h, uint64_t face) {
    h ^= seat(0, face);
    return h * P1 + P4;
}

/* the wooden numbered keeps at the desk's edge: the ONLY thing that leaves */
static inline uint64_t keeps(uint64_t h) {
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char * restrict data, size_t len)
{
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* -- which regime? can the pile stand one full revolution of the stone?
          four faces x an eight-mark seating = thirty-two marks. -- */
    if (len >= 32) {
        /* REGIME A: turn the stone. Four faces of one body. */
        uint64_t v1 = LIGHT + P1 + P2;
        uint64_t v2 = LIGHT + P2;
        uint64_t v3 = LIGHT;
        uint64_t v4 = LIGHT - P1;
        uint64_t c  = LIGHT ^ P3;          /* the callus: residue of the last press */
        const unsigned char *const limit = end - 32;

        do {
            uint64_t t1 = seat(v1, w64(p     ));
            uint64_t t2 = seat(v2, w64(p +  8));
            uint64_t t3 = seat(v3, w64(p + 16));
            uint64_t t4 = seat(v4, w64(p + 24));
            /* nothing washes clean between marks: each face carries the
               callus of the face seated just before it; face one carries
               the callus that survived the previous revolution.
               Strictly triangular => exactly invertible => no state lost. */
            v1 = t1 ^ rotl64(c , 13);
            v2 = t2 ^ rotl64(t1, 13);
            v3 = t3 ^ rotl64(t2, 13);
            v4 = t4 ^ rotl64(t3, 13);
            c  = t4;
            p += 32;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = press(h, v1);
        h = press(h, v2);
        h = press(h, v3);
        h = press(h, v4);
        h = press(h, c);
    } else {
        /* REGIME B: no room to turn. One face, pressed straight on. */
        h = LIGHT + P5;
    }

    h += (uint64_t)len;

    /* whatever is left of the pile: full seatings, then a half seating,
       then the last marks one at a time. */
    while (p + 8 <= end) { h ^= seat(0, w64(p));      h = rotl64(h, 27) * P1 + P4; p += 8; }
    if    (p + 4 <= end) { h ^= (uint64_t)w32(p) * P1; h = rotl64(h, 23) * P2 + P3; p += 4; }
    while (p     <  end) { h ^= (uint64_t)(*p)   * P5; h = rotl64(h, 11) * P1;      p += 1; }

    /* sweep the groove-dust away; read off the keeps. */
    return keeps(h);
}
