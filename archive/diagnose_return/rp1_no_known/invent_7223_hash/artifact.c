/* The mason trail: a hash built only from turns, strikes and pressed weight.
   No multiplication anywhere, including in the constants.                     */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the mason's only two tools */
#define TURN64(x, p) (((uint64_t)(x) << (p)) | ((uint64_t)(x) >> (64 - (p))))
#define TURN32(x, p) ((uint32_t)(((uint32_t)(x) << (p)) | ((uint32_t)(x) >> (32 - (p)))))

/* one turn of the coil holds 32 marks = four faces of the sphere; the coil's
   slope turns as you go round it, so each quarter has its own pitch, and one
   stalk stands at the end of every turn.                                     */
#define PITCH_A 23
#define PITCH_B 41
#define PITCH_C 13
#define PITCH_D 46          /* 23+41+13+46 = 123 = 59 (mod 64): odd net drift */
#define PITCH_1 29          /* a lone face, off the coil                      */
#define PITCH_0 17          /* a handful of crumbs                            */
#define CRACK   31          /* the angle of the hairline a stalk opens        */

/* press a face of marks into the sphere, then let it take one turn.
   invertible in h for fixed w, and the carries of + are the crack that
   propagates through the stone (the only nonlinearity on the trail).        */
#define ROLL(h, w, p) ((h) = TURN64((h) + (uint64_t)(w), (p)))

/* a stalk: the sphere sobs once -- the old face shrinks (>>) and goes.       */
#define SOB(h) ((h) ^= (h) >> CRACK)

/* one circumference of the sphere = 8 marks. memcpy is a single mov at -O3
   and is safe for unaligned data; the kernel never writes memory, so there
   is no aliasing hazard to annotate away.                                   */
static inline uint64_t face(const unsigned char *p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

/* ------------------------------------------------------------------------ *
 * THE FINAL STALK.  The sphere itself is not kept.  The last stalk cracks
 * it clean in two; the halves grind against each other a fixed count of
 * times, and the hairline left behind -- small as a token -- is what is
 * handed over.  Every stage below is a bijection of 64 bits, so no output
 * entropy is lost:
 *   h ^= TURN(h,a) ^ TURN(h,b)  is invertible because 1 + x^a + x^b has odd
 *   weight, hence is not divisible by (x+1), hence is coprime to
 *   x^64 - 1 = (x+1)^64 over GF(2);
 *   h + K, TURN, h ^= h>>s and the ARX grind (add/rotate/xor on the two
 *   halves) are each invertible.
 * Grinding on 32-bit halves with add-carries is where all the nonlinearity
 * comes from -- this is the multiply's replacement, and it is paid once per
 * hash, not once per byte.                                                  *
 * ------------------------------------------------------------------------ */
static inline uint64_t final_crack(uint64_t h) {
    uint32_t x, y;

    /* the crack opens across the whole sphere, not a sliver of it */
    h ^= TURN64(h, 49) ^ TURN64(h, 24);
    h  = TURN64(h + 0x9E3779B97F4A7C15ULL, 32);

    x = (uint32_t)(h >> 32); y = (uint32_t)h;
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 1 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 2 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 3 */
    h = ((uint64_t)x << 32) | (uint64_t)y;

    h ^= TURN64(h, 17) ^ TURN64(h, 47);               /* re-cross the halves */

    x = (uint32_t)(h >> 32); y = (uint32_t)h;
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 4 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 5 */
    x = TURN32(x, 24) + y; y = TURN32(y, 3) ^ x;      /* grind 6 */
    h = ((uint64_t)x << 32) | (uint64_t)y;

    /* the hairline itself */
    h ^= TURN64(h, 43) ^ TURN64(h, 11);
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;
    /* the sphere at the trail's high mouth, weighed against the pile */
    uint64_t h = 0x243F6A8885A308D3ULL + (uint64_t)len;

    /* REGIME 1 -- the pile reaches round the coil: four faces, one stalk. */
    while (n >= 32) {
        uint64_t w0 = face(p), w1 = face(p + 8);
        uint64_t w2 = face(p + 16), w3 = face(p + 24);
        ROLL(h, w0, PITCH_A);
        ROLL(h, w1, PITCH_B);
        ROLL(h, w2, PITCH_C);
        ROLL(h, w3, PITCH_D);
        SOB(h);
        p += 32; n -= 32;
    }

    /* REGIME 2 -- shorter than a turn of the coil, but at least one face. */
    while (n >= 8) {
        ROLL(h, face(p), PITCH_1);
        SOB(h);
        p += 8; n -= 8;
    }

    /* REGIME 3 -- the eggshells the trail sheds.  A sphere cannot press half
       a face, so it backs up onto ground it has already rolled over and
       presses one full face; only a pile smaller than a single face has to
       be swept up mark by mark.                                            */
    if (n) {
        uint64_t w;
        if (len >= 8) {
            w = face(p + n - 8);          /* always in bounds: p+n == data+len */
        } else {
            size_t i;
            w = 0;
            for (i = 0; i < n; i++) w |= (uint64_t)p[i] << (8 * i);
        }
        ROLL(h, w, PITCH_0);
        SOB(h);
    }

    return final_crack(h);   /* the token: only the last, smallest crack */
}
