#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* one odd 64-bit "twist" constant; powers folded at compile time */
#define M1 0x9E3779B97F4A7C15ULL
#define M2 (M1*M1)
#define M3 (M2*M1)
#define M4 (M3*M1)
#define M5 (M4*M1)
#define M6 (M5*M1)
#define M7 (M6*M1)
#define M8 (M7*M1)

static inline uint64_t ld8(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t ld4(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

/* the final hard twist-stir */
static inline uint64_t stir(uint64_t x){
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t h = 0;
    size_t i = 0;

    if (len >= 64) {
        /* 8 interleaved lanes of the SAME ordered Horner fold, base Q = M^8 */
        uint64_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        const unsigned char *p = data;
        size_t nb = len >> 6, b;
        for (b = 0; b < nb; ++b) {
            a0 = (a0 + ld8(p     )) * M8;
            a1 = (a1 + ld8(p +  8)) * M8;
            a2 = (a2 + ld8(p + 16)) * M8;
            a3 = (a3 + ld8(p + 24)) * M8;
            a4 = (a4 + ld8(p + 32)) * M8;
            a5 = (a5 + ld8(p + 40)) * M8;
            a6 = (a6 + ld8(p + 48)) * M8;
            a7 = (a7 + ld8(p + 56)) * M8;
            p += 64;
        }
        /* recombine with the shelf-position weights M^7 .. M^0 */
        h = a0*M7 + a1*M6 + a2*M5 + a3*M4 + a4*M3 + a5*M2 + a6*M1 + a7;
        i = nb << 6;
    }

    for (; i + 8 <= len; i += 8)
        h = (h + ld8(data + i)) * M1;

    {
        size_t r = len - i;                 /* 0..7 */
        if (r) {
            const unsigned char *q = data + i;
            uint64_t t;
            if (r >= 4) t = ld4(q) | (ld4(q + r - 4) << 32);
            else        t = (uint64_t)q[0]
                          | ((uint64_t)q[r >> 1] << 16)
                          | ((uint64_t)q[r - 1] << 40);
            h = (h + t) * M1;
        }
    }

    h ^= (uint64_t)len ^ 0xCBF29CE484222325ULL;
    return stir(h);
}
