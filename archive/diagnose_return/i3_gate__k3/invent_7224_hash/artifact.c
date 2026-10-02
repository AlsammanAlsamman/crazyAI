#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* "the owl takes the pen and calls the final shape inside the dreamer inside --
   fixes it, small and unchanging, the size of any other token."
   MurmurHash3 fmix64: a validated finalizer, O(1) for the whole buffer.
   These are the only two multiplies in the design. */
static inline uint64_t owl_fix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/* the flat stone: one distinct odd mark per lane, so lanes can never
   become equal to each other and the all-zero state cannot persist
   ("a calculation grown too satisfied with itself"). */
static const uint64_t STONE[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD1B54A32D192ED03ULL, 0xA5CB3B6F2E1D4C87ULL,
    0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL
};
/* the uphill twist: a different climb for every lane */
static const unsigned TWIST[8] = { 7u, 11u, 17u, 23u, 29u, 37u, 43u, 53u };

/* one heap = 64 bytes laid on the stone, order fixed by position (lane i
   <- bytes 8i..8i+7), then raced: twist, then a stride-counter mark added
   so a heap of zeros still changes the running shape. Invertible in s. */
static inline void race(uint64_t *s, const unsigned char *p, uint64_t ctr) {
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t w;
        memcpy(&w, p + (size_t)8 * i, 8);
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
    }
}

/* "turning the shape ninety degrees and folding its corners into its own
   center" -- a hypercube butterfly over the 8 lanes. d = 1,2,4 are the
   three turns; three turns reach every lane from every lane. Each pair
   step is an ARX quarter-round: invertible, so nothing can collapse. */
static inline void fold_square(uint64_t *s, int d, unsigned r1, unsigned r2) {
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t a, b;
        if (i & d) continue;
        a = s[i];
        b = s[i | d];
        a += b;  b = ROTL64(b, r1) ^ a;
        b += a;  a = ROTL64(a, r2) ^ b;
        s[i] = a;
        s[i | d] = b;
    }
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *end = data + len;
    uint64_t s[8];
    uint64_t h, ctr;
    int i;

    /* REGIME TEST, in the metaphor's own terms: the carry-forward only
       means anything once there is a second heap to carry into. Too few
       marks for a pair of heaps -> the deer takes no stride; the marks go
       from the flat stone straight to the owl, through one accumulator.
       This is the small-input fallback that pays no 8-lane setup and no
       end-of-run square folds. */
    if (len < 128) {
        h = STONE[0] ^ ((uint64_t)len * 0xBF58476D1CE4E5B9ULL);
        while (p + 8 <= end) {
            uint64_t w;
            memcpy(&w, p, 8);
            h = ROTL64(h ^ w, 31) + 0x9E3779B97F4A7C15ULL;
            h ^= h >> 27;
            p += 8;
        }
        if (p < end) {                 /* the stone, zero-padded: position
                                          still encodes order, no byte loop */
            uint64_t w = 0;
            memcpy(&w, p, (size_t)(end - p));
            h = ROTL64(h ^ w, 31) + 0x9E3779B97F4A7C15ULL;
        }
        return owl_fix(h);
    }

    /* --- the deer's run: heaps of even count, fixed stride --- */
    for (i = 0; i < 8; i++) s[i] = STONE[i] ^ (uint64_t)len;
    ctr = 1;

    /* two heaps per step, each followed by a different ninety-degree turn,
       so the stride schedule needs no branch */
    while ((size_t)(end - p) >= 128) {
        race(s, p, ctr);
        fold_square(s, 1, 13u, 29u);
        race(s, p + 64, ctr + 1);
        fold_square(s, 2, 23u, 47u);
        p += 128;
        ctr += 2;
    }
    if ((size_t)(end - p) >= 64) {
        race(s, p, ctr);
        fold_square(s, 1, 13u, 29u);
        fold_square(s, 4, 19u, 41u);
        p += 64;
        ctr++;
    }
    /* the last partial heap: whole words to their own lanes, then the stone */
    i = 0;
    while (p + 8 <= end) {
        uint64_t w;
        memcpy(&w, p, 8);
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
        p += 8; i++; ctr++;
    }
    if (p < end) {
        uint64_t w = 0;
        memcpy(&w, p, (size_t)(end - p));
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
    }

    /* "searching in squares ... again and again, until even a single mark
       changed at the start has smeared itself across every square" --
       two full passes of the three turns: >= 6 ARX stages after the last
       byte is absorbed, every lane reachable from every lane twice over. */
    fold_square(s, 1, 13u, 29u);
    fold_square(s, 2, 23u, 47u);
    fold_square(s, 4, 19u, 41u);
    fold_square(s, 1, 41u, 19u);
    fold_square(s, 2, 47u, 23u);
    fold_square(s, 4, 29u, 13u);

    /* the heaps are burned: 8 lanes collapse to one token. This is the
       only non-invertible step in the design -- deliberately so. */
    h = (uint64_t)len ^ 0x6A09E667F3BCC909ULL;
    for (i = 0; i < 8; i++) h = ROTL64(h ^ s[i], 29) + s[i];
    return owl_fix(h);
}
