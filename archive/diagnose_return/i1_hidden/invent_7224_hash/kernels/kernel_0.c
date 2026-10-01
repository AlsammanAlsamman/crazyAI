/* kernel.c — the riverbank stone, the musk deer's stride, the owl's seal.
   Contract: uint64_t kernel(const unsigned char *data, size_t len);
   gcc -O3 -march=native -fopenmp -lm                                    */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define C1  0xBF58476D1CE4E5B9ULL
#define C2  0x94D049BB133111EBULL
#define C3  0xFF51AFD7ED558CCDULL
#define PHI 0x9E3779B97F4A7C15ULL

/* eight stones of the riverbank: one per lane */
static const uint64_t KK[8] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0x27D4EB2F165667C5ULL, 0xD6E8FEB86659FD93ULL,
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* the owl takes the pen: one-way, sealed to one fixed size */
static inline uint64_t owl(uint64_t h)
{
    h ^= h >> 32; h *= C3;
    h ^= h >> 29; h *= C2;
    h ^= h >> 32; h *= C1;
    h ^= h >> 31;
    return h;
}

/* the drifting man searches squares: turn the 2x4 grid ninety degrees,
   fold its corners into its own centre, then let each cell eat its trail */
static inline void square_turn(uint64_t *v)
{
    uint64_t t0=v[0], t1=v[4], t2=v[1], t3=v[5], t4=v[2], t5=v[6], t6=v[3], t7=v[7];
    t0 += ROTL64(t7, 13); t7 ^= ROTL64(t0, 41);
    t1 += ROTL64(t6, 19); t6 ^= ROTL64(t1, 47);
    t2 += ROTL64(t5, 29); t5 ^= ROTL64(t2, 53);
    t3 += ROTL64(t4, 37); t4 ^= ROTL64(t3, 59);
    v[0]=t0*C1; v[1]=t1*C2; v[2]=t2*C3; v[3]=t3*C1;
    v[4]=t4*C2; v[5]=t5*C3; v[6]=t6*C1; v[7]=t7*C2;
}

/* one run of heaps. n >= 64 required. blk0 = this run's global block offset:
   the order, laid on the stone as data, never spoken as a dependency.      */
static uint64_t heap_run(const unsigned char *p, size_t n, uint64_t blk0, uint64_t tag)
{
    uint64_t v[8];
    const unsigned char *q = p;
    size_t m = n;
    uint64_t blk = blk0;

    v[0]=KK[0]^(tag*PHI); v[1]=KK[1]^blk0; v[2]=KK[2]; v[3]=KK[3];
    v[4]=KK[4];           v[5]=KK[5];      v[6]=KK[6]; v[7]=KK[7]^tag;

    while (m >= 64) {
        uint64_t kb = blk * PHI;                 /* the position, as data */
        v[0] = ROTL64((v[0] ^ (rd64(q+ 0) + (KK[0]^kb))) * C1, 23);
        v[1] = ROTL64((v[1] ^ (rd64(q+ 8) + (KK[1]^kb))) * C1, 27);
        v[2] = ROTL64((v[2] ^ (rd64(q+16) + (KK[2]^kb))) * C1, 31);
        v[3] = ROTL64((v[3] ^ (rd64(q+24) + (KK[3]^kb))) * C1, 35);
        v[4] = ROTL64((v[4] ^ (rd64(q+32) + (KK[4]^kb))) * C1, 39);
        v[5] = ROTL64((v[5] ^ (rd64(q+40) + (KK[5]^kb))) * C1, 43);
        v[6] = ROTL64((v[6] ^ (rd64(q+48) + (KK[6]^kb))) * C1, 47);
        v[7] = ROTL64((v[7] ^ (rd64(q+56) + (KK[7]^kb))) * C1, 51);
        v[0] += ROTL64(v[7], 29);   /* the one stone carried to the next heap */
        q += 64; m -= 64; blk++;
    }

    if (m) {   /* last heap, read overlapping its predecessor: always in bounds */
        const unsigned char *r = p + n - 64;
        uint64_t kb = blk * PHI, x;
        x = rd64(r+ 0) + (KK[7]^kb); x ^= x >> 31; v[0] = (v[0] ^ x) * C2;
        x = rd64(r+ 8) + (KK[6]^kb); x ^= x >> 31; v[1] = (v[1] ^ x) * C2;
        x = rd64(r+16) + (KK[5]^kb); x ^= x >> 31; v[2] = (v[2] ^ x) * C2;
        x = rd64(r+24) + (KK[4]^kb); x ^= x >> 31; v[3] = (v[3] ^ x) * C2;
        x = rd64(r+32) + (KK[3]^kb); x ^= x >> 31; v[4] = (v[4] ^ x) * C2;
        x = rd64(r+40) + (KK[2]^kb); x ^= x >> 31; v[5] = (v[5] ^ x) * C2;
        x = rd64(r+48) + (KK[1]^kb); x ^= x >> 31; v[6] = (v[6] ^ x) * C2;
        x = rd64(r+56) + (KK[0]^kb); x ^= x >> 31; v[7] = (v[7] ^ x) * C2;
    }

    square_turn(v); square_turn(v); square_turn(v); square_turn(v);

    uint64_t h = tag * PHI + blk0;
    h += (v[0] ^ ROTL64(v[1], 17)) * C1;
    h ^= (v[2] + ROTL64(v[3], 29)) * C2;
    h += (v[4] ^ ROTL64(v[5], 41)) * C3;
    h ^= (v[6] + ROTL64(v[7], 53)) * C1;
    return h;                       /* the heaps are burned here */
}

/* short buffers: a two-by-two square, so the eight-lane fold's fixed cost
   is never charged to an input too small to amortise it                  */
static uint64_t short_run(const unsigned char *p, size_t n)
{
    uint64_t a = KK[0] ^ (n * PHI), b = KK[1] ^ n, c = KK[2], d = KK[3];

    if (n >= 16) {
        const unsigned char *e = p + n - 16;
        a ^= rd64(p+0) + KK[4];
        b ^= rd64(p+8) + KK[5];
        c ^= rd64(e+0) + KK[6];
        d ^= rd64(e+8) + KK[7];
        if (n > 32) {                       /* 33..63: cover the middle too */
            const unsigned char *m1 = p + 16, *m2 = p + n - 32;
            a += rd64(m1+0) ^ KK[5];
            b += rd64(m1+8) ^ KK[4];
            c += rd64(m2+0) ^ KK[7];
            d += rd64(m2+8) ^ KK[6];
        }
    } else if (n >= 8) {
        a ^= rd64(p)         + KK[4];
        c ^= rd64(p + n - 8) + KK[6];
    } else if (n) {
        uint64_t w;
        if (n >= 4) w = rd32(p) | (rd32(p + n - 4) << 32);
        else        w = ((uint64_t)p[0])
                      | ((uint64_t)p[n >> 1] << 24)
                      | ((uint64_t)p[n - 1] << 48);
        a ^= w + KK[4];
        c ^= ROTL64(w, 32) + KK[6];
    }

    for (int i = 0; i < 3; i++) {
        uint64_t s0=a, s1=c, s2=b, s3=d;    /* turn the square ninety degrees */
        s0 += ROTL64(s3, 17); s3 ^= ROTL64(s0, 43);
        s1 += ROTL64(s2, 31); s2 ^= ROTL64(s1, 53);
        a = s0*C1; b = s1*C2; c = s2*C3; d = s3*C1;
    }

    uint64_t h = n * PHI;
    h += (a ^ ROTL64(b, 27)) * C1;
    h ^= (c + ROTL64(d, 45)) * C2;
    return owl(h);
}

#define PAR_MIN ((size_t)1 << 22)   /* 4 MB: below this, threads cost more than they buy */
#define CHUNK   ((size_t)1 << 20)   /* fixed chunking => value independent of thread count */

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (len < 64)      return short_run(data, len);
    if (len < PAR_MIN) return owl(heap_run(data, len, 0, (uint64_t)len));

    size_t nc = len / CHUNK;                 /* >= 4, every chunk >= 64 bytes */
    uint64_t acc = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:acc) schedule(static)
#endif
    for (size_t i = 0; i < nc; i++) {
        size_t off = i * CHUNK;
        size_t sz  = (i + 1 == nc) ? (len - off) : CHUNK;
        acc += heap_run(data + off, sz, (uint64_t)(off / 64), (uint64_t)len) * C1;
    }
    return owl(acc ^ ((uint64_t)len * PHI));
}
