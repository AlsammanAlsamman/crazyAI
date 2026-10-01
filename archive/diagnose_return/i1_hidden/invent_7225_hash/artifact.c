#include <stdint.h>
#include <stddef.h>

/* ---- the cup that is never pure ---- */
#define CUP  0x243F6A8885A308D3ULL  /* the wine is already cut before the first mark */
#define WINE 0x9E3779B97F4A7C15ULL  /* the sickened wine: odd, never pure           */
#define LEES 0x6A09E667F3BCC909ULL  /* dregs: no drop ever enters the cup pure      */
#define TILT 23                     /* the cup tips between pulls (gcd(23,64)==1)   */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* ONE PULL: one mark enters the cup, the cup's colour changes, and that
   colour is exactly what the next pull starts from. The wine stains the
   drop (multiply, off the critical path); the cup remembers every drop
   added to it (add, 1 cycle); the cup tips (rotate, 1 cycle).            */
static inline uint64_t pull(uint64_t s, unsigned char b) {
    return rotl64(s, TILT) + ((uint64_t)b * WINE + LEES);
}

/* SEVEN ORGANS: bend, then throw away half of what came before,
   keeping only what refuses to sit still. Exactly seven, size-independent. */
static inline uint64_t organs(uint64_t x) {
    x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;  /* 1 */
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 29;  /* 2 */
    x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;  /* 3 */
    x *= 0x94d049bb133111ebULL; x ^= x >> 30;  /* 4 */
    x *= 0xd6e8feb86659fd93ULL; x ^= x >> 31;  /* 5 */
    x *= 0xa0761d6478bd642fULL; x ^= x >> 32;  /* 6 */
    x *= 0x2545f4914f6cdd1dULL; x ^= x >> 28;  /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t s = CUP;
    size_t i = 0;

    /* size guard: the unrolled pour only where the loop overhead it removes
       is worth its setup. Same arithmetic, same single chain, no lanes.    */
    if (len >= 32) {
        size_t n = len & ~(size_t)7;
        for (; i < n; i += 8) {
            s = pull(s, data[i + 0]);
            s = pull(s, data[i + 1]);
            s = pull(s, data[i + 2]);
            s = pull(s, data[i + 3]);
            s = pull(s, data[i + 4]);
            s = pull(s, data[i + 5]);
            s = pull(s, data[i + 6]);
            s = pull(s, data[i + 7]);
        }
    }
    for (; i < len; ++i)          /* the tail of the pile, same single cup */
        s = pull(s, data[i]);

    return organs(s);             /* carry the last colour through the body */
}
