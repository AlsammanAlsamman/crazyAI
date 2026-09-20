#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read_le64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char * const end = data + len;
    uint64_t h;

    /* Fallback: too small for the servant to have anything to fix order
       across (fewer than one full heap-of-four) -- plain single
       accumulator, walked mark by mark. */
    if (len < 32) {
        h = PRIME5 ^ (uint64_t)len;
        while (p < end) {
            h ^= (uint64_t)(*p++);
            h *= PRIME1;
            h ^= h >> 33;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    /* Fix each heap's identity (order) before any folding happens: four
       lanes, each seeded distinctly, so their position is recorded up
       front instead of depending on strict single-accumulator sequencing. */
    uint64_t acc0 = PRIME1 + PRIME2;
    uint64_t acc1 = PRIME2;
    uint64_t acc2 = 0;
    uint64_t acc3 = (uint64_t)0 - PRIME1;

    /* Each heap races its own mountain stride -- uphill twist (rotl),
       downhill fold (multiply-add), doubling back (self-multiply) -- with
       no data dependency between lanes, so the four chains overlap on the
       CPU instead of forming one long serial chain. */
    size_t nblocks = len / 32;
    for (size_t b = 0; b < nblocks; b++) {
        acc0 += read_le64(p +  0) * PRIME2; acc0 = rotl64(acc0, 31); acc0 *= PRIME1;
        acc1 += read_le64(p +  8) * PRIME2; acc1 = rotl64(acc1, 31); acc1 *= PRIME1;
        acc2 += read_le64(p + 16) * PRIME2; acc2 = rotl64(acc2, 31); acc2 *= PRIME1;
        acc3 += read_le64(p + 24) * PRIME2; acc3 = rotl64(acc3, 31); acc3 *= PRIME1;
        p += 32;
    }

    /* The four running heaps are folded onto the one final stone. */
    h = rotl64(acc0, 1) + rotl64(acc1, 7) + rotl64(acc2, 12) + rotl64(acc3, 18);
    h ^= h >> 29; h *= PRIME3;
    h ^= (uint64_t)len;

    /* Remaining tail marks, walked one at a time, in order, into the now
       single running shape. */
    while (p + 8 <= end) {
        uint64_t k = read_le64(p) * PRIME2;
        k = rotl64(k, 31); k *= PRIME1;
        h ^= k;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
    }

    /* The owl calls the final shape inside the dreamer inside: a nested
       avalanche finalizer sealing the token to one fixed 64-bit size.
       No intermediate heap (acc0..3, tail words) survives past here. */
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}
