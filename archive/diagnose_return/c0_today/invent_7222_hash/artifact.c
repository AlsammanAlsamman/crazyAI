#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four wooden numbered keeps: fixed constants, "same starlight color no
   matter the hour" -- no data-dependent branches, no per-position tables. */
#define KP1 0x9E3779B185EBCA87ULL
#define KP2 0xC2B2AE3D27D4EB4FULL
#define KP3 0x165667B19E3779F9ULL
#define KP4 0x85EBCA77C2B2AE63ULL
#define KP5 0x27D4EB2F165667C5ULL

static inline uint64_t stone_turn(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* "a flat thing about to be invested into three": eight flat marks lifted into
   one wide weight. memcpy compiles to a single unaligned load at -O3. */
static inline uint64_t weight64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t weight32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* One fold: the mark's weight is dropped into the seated place, the stone is
   turned through the chalk-bank groove, and the turned position is pressed --
   never onto a clean face. */
static inline uint64_t fold(uint64_t acc, uint64_t in) {
    acc += in * KP2;
    acc  = stone_turn(acc, 31);
    acc *= KP1;
    return acc;
}

/* Reading one keep off against the desk's edge at the end. */
static inline uint64_t read_keep(uint64_t acc, uint64_t val) {
    val = fold(0, val);
    acc ^= val;
    acc  = acc * KP1 + KP4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME CHECK: does the pile fill all four keeps? */
    if (len >= 32) {
        /* Four-keep desk: the stone turns a quarter between marks, so four
           folding-paths advance at once and no intermediate is ever read. */
        const unsigned char *const limit = end - 32;
        uint64_t k1 = KP1 + KP2;
        uint64_t k2 = KP2;
        uint64_t k3 = 0;
        uint64_t k4 = 0ULL - KP1;
        do {
            k1 = fold(k1, weight64(p));      p += 8;
            k2 = fold(k2, weight64(p));      p += 8;
            k3 = fold(k3, weight64(p));      p += 8;
            k4 = fold(k4, weight64(p));      p += 8;
        } while (p <= limit);

        /* Lift the stone: the four keeps are read off once, together. */
        h = stone_turn(k1, 1) + stone_turn(k2, 7)
          + stone_turn(k3, 12) + stone_turn(k4, 18);
        h = read_keep(h, k1);
        h = read_keep(h, k2);
        h = read_keep(h, k3);
        h = read_keep(h, k4);
    } else {
        /* FALLBACK REGIME: a handful of marks never sets up four stones. */
        h = KP5;
    }

    h += (uint64_t)len;

    /* The pile's ragged end: 8 marks at a time, then 4, then one by one. */
    while (p + 8 <= end) {
        h ^= fold(0, weight64(p));
        h  = stone_turn(h, 27) * KP1 + KP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)weight32(p) * KP1;
        h  = stone_turn(h, 23) * KP2 + KP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * KP5;
        h  = stone_turn(h, 11) * KP1;
        p++;
    }

    /* Only the last seated number leaves the desk: the groove-dust is swept
       off by a final avalanche so one hair's width of chalk moves half the
       reading. */
    h ^= h >> 33;
    h *= KP2;
    h ^= h >> 29;
    h *= KP3;
    h ^= h >> 32;
    return h;
}
