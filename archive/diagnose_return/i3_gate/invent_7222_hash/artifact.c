#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the chalk-bank constants: the cycle-light is the same starlight color
   no matter the hour -- fixed, input-independent. */
#define KP1 11400714785074694791ULL
#define KP2 14029467366897019727ULL
#define KP3  1609587929392839161ULL
#define KP4  9650029242287828579ULL
#define KP5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
/* read a mark's weight off the desk's numbered groove (order as given) */
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ONE FOLD: drop the mark's weight into the stone's already-turned seat,
   quarter-turn it, and let the callus of the last press bend this one.
   Nothing washes clean: acc enters and leaves. */
static inline uint64_t fold(uint64_t acc, uint64_t w) {
    acc += w * KP2;
    acc  = rotl64(acc, 31);
    acc *= KP1;
    return acc;
}
/* read one face against the wooden numbered keeps at the desk's edge */
static inline uint64_t keep(uint64_t h, uint64_t face) {
    h ^= fold(0, face);
    h  = h * KP1 + KP4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME TEST: does the pile fill at least one full revolution of the
       stone?  Four faces, eight bytes a face = 32 marks. */
    if (len >= 32) {
        /* the four faces of the die-stone, seated differently to begin with
           so no two folding-paths start alike */
        uint64_t v1 = KP1 + KP2;
        uint64_t v2 = KP2;
        uint64_t v3 = 0ULL;
        uint64_t v4 = 0ULL - KP1;

        /* two revolutions per pass: same four faces, less groove overhead.
           The stone is never lifted fully off the desk -- v1..v4 stay in
           registers for the whole pile. */
        if (len >= 64) {
            const unsigned char *const lim64 = end - 64;
            while (p <= lim64) {
                v1 = fold(v1, ld64(p     ));  v2 = fold(v2, ld64(p +  8));
                v3 = fold(v3, ld64(p + 16));  v4 = fold(v4, ld64(p + 24));
                v1 = fold(v1, ld64(p + 32));  v2 = fold(v2, ld64(p + 40));
                v3 = fold(v3, ld64(p + 48));  v4 = fold(v4, ld64(p + 56));
                p += 64;
            }
        }
        {   /* len >= 32, so end - 32 is in-bounds */
            const unsigned char *const lim32 = end - 32;
            while (p <= lim32) {
                v1 = fold(v1, ld64(p     ));  v2 = fold(v2, ld64(p +  8));
                v3 = fold(v3, ld64(p + 16));  v4 = fold(v4, ld64(p + 24));
                p += 32;
            }
        }

        /* lift the stone: collapse the four faces against the keeps.
           Everything about how wide the stone was is swept away here. */
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = keep(h, v1);
        h = keep(h, v2);
        h = keep(h, v3);
        h = keep(h, v4);
    } else {
        /* FALLBACK: too short to turn the stone once -- press onto one face */
        h = KP5;
    }

    h += (uint64_t)len;

    /* the last few marks, pressed one face at a time */
    while (p + 8 <= end) {
        h ^= fold(0, ld64(p));
        h  = rotl64(h, 27) * KP1 + KP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)ld32(p) * KP1;
        h  = rotl64(h, 23) * KP2 + KP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * KP5;
        h  = rotl64(h, 11) * KP1;
        p++;
    }

    /* THE FINAL SEATED NUMBER -- the only thing that leaves the desk.
       A hair's width of chalk-dust is multiplied, and multiplied again,
       until no two folding-paths walk the same last step. */
    h ^= h >> 33;  h *= KP2;
    h ^= h >> 29;  h *= KP3;
    h ^= h >> 32;
    return h;
}
