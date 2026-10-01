#include <stdint.h>
#include <stddef.h>

/* The coiled limestone trail: one sphere, one press and one crack per mark,
   no multiplication anywhere, only the final crack is handed over. */

#define ANGLE  11                          /* the crack's angle; fixed, odd, never 0 */
#define MOUTH  0x9E3779B97F4A7C15ULL       /* the sphere's shape at the trail's high mouth */

/* press the mark into the sphere's face, then strike one stalk and read
   the angle of the crack. carry out of the += is "the weight carried forward". */
static inline uint64_t tumble(uint64_t s, unsigned char mark)
{
    s += (uint64_t)mark;
    return (s << ANGLE) | (s >> (64 - ANGLE));   /* single rol; ANGLE is a nonzero constant */
}

/* the final stalk: the last, smallest crack, kept as the token.
   Thomas Wang's 64-bit mix - shift/add/xor only, no multiply. */
static inline uint64_t last_crack(uint64_t k)
{
    k = (~k) + (k << 21);
    k ^=  k >> 24;
    k =  k + (k << 3) + (k << 8);
    k ^=  k >> 14;
    k =  k + (k << 2) + (k << 4);
    k ^=  k >> 28;
    k =  k + (k << 31);
    return k;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t s = MOUTH;                 /* the sole carried memory */
    size_t i = 0;

    /* unrolled only to keep the loop bookkeeping off the 2-cycle critical
       path; the marks are still taken one by one, in their given order. */
    if (len >= 8) {
        size_t n8 = len & ~(size_t)7;
        for (; i < n8; i += 8) {
            s = tumble(s, data[i + 0]);
            s = tumble(s, data[i + 1]);
            s = tumble(s, data[i + 2]);
            s = tumble(s, data[i + 3]);
            s = tumble(s, data[i + 4]);
            s = tumble(s, data[i + 5]);
            s = tumble(s, data[i + 6]);
            s = tumble(s, data[i + 7]);
        }
    }
    for (; i < len; ++i)                /* the tail marks, same trail */
        s = tumble(s, data[i]);

    return last_crack(s);               /* the sphere itself is not kept */
}
