#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the five numbered keeps' constants (xxHash64 primes - validated) */
#define K1 0x9E3779B185EBCA87ULL
#define K2 0xC2B2AE3D27D4EB4FULL
#define K3 0x165667B19E3779F9ULL
#define K4 0x85EBCA77C2B2AE63ULL
#define K5 0x27D4EB2F165667C5ULL

/* a quarter turn through the chalk-bank groove */
static inline uint64_t turn(uint64_t x, int q) {
    return (x << q) | (x >> (64 - q));
}
static inline uint64_t load8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t load4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold: the mark's weight dropped into a keep's ALREADY-TURNED seat,
   the keep's memory of the last press bending how deep this one goes.
   Never lifted, never washed clean. */
static inline uint64_t press(uint64_t seat, uint64_t w) {
    seat += w * K2;
    seat  = turn(seat, 31);
    seat *= K1;
    return seat;
}

/* lifting one keep's seated position into the single reading */
static inline uint64_t lift(uint64_t h, uint64_t seat) {
    seat *= K2;
    seat  = turn(seat, 31);
    seat *= K1;
    h ^= seat;
    h = h * K1 + K4;
    return h;
}

/* reading the stone off against the wooden numbered keeps: the only thing
   that ever leaves the desk. Everything else is swept away. */
static inline uint64_t read_keeps(uint64_t h) {
    h ^= h >> 33; h *= K2;
    h ^= h >> 29; h *= K3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME CHECK: does the pile reach far enough along the groove for the
       stone to complete a full turn across all eight keeps? */
    if (len >= 64) {
        /* --- rack regime: "centuries of marks" --- */
        uint64_t s0 = K1 + K2,            s1 = K2;
        uint64_t s2 = 0,                  s3 = (uint64_t)0 - K1;
        uint64_t s4 = K3 + K4,            s5 = K4;
        uint64_t s6 = K5,                 s7 = (uint64_t)0 - K3;
        const unsigned char *const limit = end - 64;

        /* the seat is fixed in DESK coordinates while the stone turns, so
           mark i lands in keep (i mod 8): eight independent fold-paths,
           branch-free, the cycle-light the same colour at every hour. */
        do {
            s0 = press(s0, load8(p +  0));
            s1 = press(s1, load8(p +  8));
            s2 = press(s2, load8(p + 16));
            s3 = press(s3, load8(p + 24));
            s4 = press(s4, load8(p + 32));
            s5 = press(s5, load8(p + 40));
            s6 = press(s6, load8(p + 48));
            s7 = press(s7, load8(p + 56));
            p += 64;
        } while (p <= limit);

        /* lift the stone: collapse the rack to one reading */
        h = turn(s0,  1) + turn(s1,  7) + turn(s2, 12) + turn(s3, 18)
          + turn(s4, 23) + turn(s5, 27) + turn(s6, 31) + turn(s7, 37);
        h = lift(h, s0); h = lift(h, s1); h = lift(h, s2); h = lift(h, s3);
        h = lift(h, s4); h = lift(h, s5); h = lift(h, s6); h = lift(h, s7);
    } else {
        /* --- single-seat regime: "a handful" --- no rack, no merge cost.
           Identical to the validated scalar short path. */
        h = K5;
    }

    /* the stone turned once per mark, so the turn count is part of where it
       seats */
    h += (uint64_t)len;

    /* sweeping the last few marks in by hand: 8 -> 4 -> 1 */
    while (p + 8 <= end) {
        uint64_t k = load8(p);
        k *= K2; k = turn(k, 31); k *= K1;
        h ^= k;
        h = turn(h, 27) * K1 + K4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)load4(p) * K1;
        h = turn(h, 23) * K2 + K3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * K5;
        h = turn(h, 11) * K1;
        p += 1;
    }

    return read_keeps(h);
}
