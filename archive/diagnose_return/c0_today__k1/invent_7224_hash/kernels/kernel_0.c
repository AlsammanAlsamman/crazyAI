#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL32(x,n) ((uint32_t)(((uint32_t)(x) << (n)) | ((uint32_t)(x) >> (32 - (n)))))
#define ROTL64(x,n) ((uint64_t)(((uint64_t)(x) << (n)) | ((uint64_t)(x) >> (64 - (n)))))

/* fixed stones at the shrine */
static const uint32_t SIGMA[4] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u};
static const uint32_t TAU[4]   = {0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 0x165667b1u};

static inline uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* THE OWL: calls the shape inside the dreamer inside, seals it to one fixed size. */
static inline uint64_t owl(uint64_t z) {
    z ^= z >> 33; z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 29; z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 32;
    return z;
}

/* one lane-wide step over both squares at once (gcc -O3 -march=native -> one AVX2 op) */
#define LANE8(stmt) do { for (j_ = 0; j_ < 8; j_++) { stmt; } } while (0)

/* THE DEER'S STRIDE: uphill twist (rot), downhill fold (xor), doubling-back (add).
   No multiplication anywhere in the stride. */
#define STRIDE()                                   \
    LANE8(a[j_] += b[j_]);                         \
    LANE8(d[j_] ^= a[j_]);                         \
    LANE8(d[j_] = ROTL32(d[j_], 16));              \
    LANE8(c[j_] += d[j_]);                         \
    LANE8(b[j_] ^= c[j_]);                         \
    LANE8(b[j_] = ROTL32(b[j_], 12));              \
    LANE8(a[j_] += b[j_]);                         \
    LANE8(d[j_] ^= a[j_]);                         \
    LANE8(d[j_] = ROTL32(d[j_], 8));               \
    LANE8(c[j_] += d[j_]);                         \
    LANE8(b[j_] ^= c[j_]);                         \
    LANE8(b[j_] = ROTL32(b[j_], 7))

/* TURNING THE SQUARE NINETY DEGREES: rotate lanes inside each 4-lane square,
   so corners fold into the centre and the next stride crosses the diagonals. */
#define SPIN(v, s) do {                                                        \
    uint32_t q0 = v[((s)  )&3], q1 = v[((s)+1)&3],                             \
             q2 = v[((s)+2)&3], q3 = v[((s)+3)&3];                             \
    uint32_t r0 = v[4+(((s)  )&3)], r1 = v[4+(((s)+1)&3)],                     \
             r2 = v[4+(((s)+2)&3)], r3 = v[4+(((s)+3)&3)];                     \
    v[0]=q0; v[1]=q1; v[2]=q2; v[3]=q3;                                        \
    v[4]=r0; v[5]=r1; v[6]=r2; v[7]=r3;                                        \
} while (0)

/* ONE HEAP: 128 marks laid on the flat stone in silence, raced, then burned down
   to the single stone that survives at the shrine. */
static inline uint64_t absorb_heap(const unsigned char *restrict p,
                                   uint64_t chain, uint64_t tag)
{
    uint32_t a[8], b[8], c[8], d[8];
    uint32_t k0 = (uint32_t)chain, k1 = (uint32_t)(chain >> 32);
    uint32_t g0 = (uint32_t)tag,   g1 = (uint32_t)(tag   >> 32);
    uint64_t f = 0;
    int j_;

    /* SEED 1: the whole heap is laid down first, in silence. No mixing between
       byte reads; the order of the marks is held by lane position alone.
       Each row is one contiguous 32-byte load. */
    for (j_ = 0; j_ < 8; j_++) a[j_] = rd32(p +  0 + 4*j_) ^ SIGMA[j_ & 3];
    for (j_ = 0; j_ < 8; j_++) b[j_] = rd32(p + 32 + 4*j_) ^ (k0 + 0x9e3779b9u * (uint32_t)j_);
    for (j_ = 0; j_ < 8; j_++) c[j_] = rd32(p + 64 + 4*j_) ^ (k1 ^ TAU[j_ & 3]);
    /* the stone that never ossifies: the heap counter, different for every heap */
    for (j_ = 0; j_ < 8; j_++) d[j_] = rd32(p + 96 + 4*j_) ^ (g0 + (uint32_t)j_)
                                                           ^ (g1 + TAU[(j_ + 1) & 3]);

    /* Two double-strides: column, diagonal, column, diagonal. One double-stride
       already touches all sixteen words of each square; two give bit-level
       smearing. The owl finishes the rest -- more strides here would only cost
       throughput, which is why the method is not raced ten times over. */
    STRIDE(); SPIN(b,1); SPIN(c,2); SPIN(d,3);
    STRIDE(); SPIN(b,3); SPIN(c,2); SPIN(d,1);
    STRIDE(); SPIN(b,1); SPIN(c,2); SPIN(d,3);
    STRIDE(); SPIN(b,3); SPIN(c,2); SPIN(d,1);

    /* BURNED AT THE SHRINE: 512 bits of heap collapse to 64 and are gone.
       Rotations differ per lane so no two lanes can cancel each other. */
    for (j_ = 0; j_ < 8; j_++) {
        uint64_t t1 = ((uint64_t)a[j_] << 32) | (uint64_t)b[j_];
        uint64_t t2 = ((uint64_t)c[j_] << 32) | (uint64_t)d[j_];
        f += ROTL64(t1, (5  * j_ + 1) & 63);
        f ^= ROTL64(t2, (11 * j_ + 7) & 63);
    }

    /* the running shape carried forward into the next heap's racing;
       non-commutative, so heap order matters */
    return ROTL64(chain ^ f, 27) + (f >> 7) + 0x9e3779b97f4a7c15ULL;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t chain;
    size_t rem, ctr;

    /* REGIME 1 -- too few marks to make a heap: never leave the riverbank. */
    if (len < 32) {
        uint64_t x = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0xbf58476d1ce4e5b9ULL);
        uint64_t y = 0x94d049bb133111ebULL + (uint64_t)len;
        if (len >= 8) {
            size_t i = 0;
            uint64_t v;
            do {
                v = rd64(data + i);
                x ^= v;  x = ROTL64(x, 29) + 0x9e3779b97f4a7c15ULL;
                y += ROTL64(v ^ x, 17);
                i += 8;
            } while (i + 8 <= len);
            v = rd64(data + len - 8);      /* walk back over marks already laid */
            x ^= ROTL64(v, 31);
            y ^= ROTL64(v + x, 13);
        } else if (len > 0) {
            uint64_t v = (uint64_t)len;    /* every mark read; 8*7+8 bits fit exactly */
            size_t i;
            for (i = 0; i < len; i++) v = (v << 8) | (uint64_t)data[i];
            x ^= v;
            y ^= ROTL64(v, 19);
        }
        return owl(x ^ owl(y));            /* the dreamer inside */
    }

    chain = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0x9ddfea08eb382d69ULL);

    /* REGIME 2 -- enough for one heap only: pad to a single heap, length in the tag. */
    if (len < 128) {
        unsigned char buf[128];
        memset(buf, 0, sizeof buf);
        memcpy(buf, data, len);
        chain = absorb_heap(buf, chain, 0x165667b19e3779f9ULL ^ (uint64_t)len);
        return owl(chain ^ ((uint64_t)len << 32));
    }

    /* REGIME 3 -- many heaps: stride the mountain path, one shrine stone surviving. */
    rem = len;
    ctr = 0;
    {
        const unsigned char *p = data;
        while (rem >= 128) {
            chain = absorb_heap(p, chain,
                        (uint64_t)ctr * 0x9e3779b97f4a7c15ULL + 0x2545f4914f6cdd1dULL);
            p   += 128;
            rem -= 128;
            ctr += 1;
        }
    }
    if (rem) {
        /* leftover that cannot fill a heap: re-walk the last 128 marks laid,
           rather than invent marks that were never there. No padding, no
           branch inside the hot loop. */
        chain = absorb_heap(data + len - 128, chain,
                    (uint64_t)rem * 0xff51afd7ed558ccdULL ^ 0x9e3779b97f4a7c15ULL);
    }
    return owl(chain ^ ((uint64_t)len * 0xc4ceb9fe1a85ec53ULL));
}
