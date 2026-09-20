STORY

I take the pile of marks the way a child at the chalk grid takes her stone: I do not read the whole procession at once, I throw and watch where it skids. First I ask how long the pile runs. If it is short — a handful of marks, no longer than a breath — I do not call anyone else to the game; one small stone, one hop, one landing square is enough, and calling in more hands would only trip everyone over each other's feet on a board too small to hold them. So for a short pile I walk it alone, eight marks at a step like a figure climbing one particular staircase: I twist what I'm carrying the way the upside-down man on the flight above twists his own weight, multiply it against a fixed, ugly, deliberately irregular number so it can never settle into a lazy rhythm, fold it into what I already carry, and keep climbing until the marks run out, the last few odd ones swept in one at a time like dropped chalk dust. Only when the pile is long — long enough that no single walker could cross it before the candle guttered — do I open the many doors onto the many sideways gardens at once: I cut the pile into shares, one for each hand that answers, each hand seeded not only with what it holds but with which garden-door it walked through, so that two identical shares landing in different places still come out unlike each other, the way the same word means something else spoken by a different mouth in a different room. Every hand climbs its own staircase in parallel, arrives at its own small twisted number, and throws it back to me. I do not just heap these together — I fold them in strict order, one after another as the doors were numbered, each fold twisting the running total hard enough that an early share and a late share of the same value still leave different marks on it, the way the moth's circling is never the same circle twice even when the candle looks unchanged. What I throw away at the end is everything but the final small stone: not the path, not the doors, not the staircases, only the token, cold and complete, barely larger than a coin, such that changing one grain of chalk anywhere upstream sends the stone skidding into an entirely different square, and no two unlike processions, however patiently arranged, come to rest in the same one.

ARTIFACT

```c
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

#define K_PRIME1 0x9E3779B185EBCA87ULL
#define K_PRIME2 0xC2B2AE3D27D4EB4FULL
#define K_PRIME3 0x165667B19E3779F9ULL
#define K_PRIME4 0x85EBCA77C2B2AE63ULL
#define K_PRIME5 0x27D4EB2F165667C5ULL
#define K_OFFSET 1469598103934665603ULL

/* walk one contiguous stretch of marks, seeded so both its content and
   its position in the whole procession shape the outcome */
static uint64_t hash_block(const unsigned char *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ K_PRIME5;
    size_t i = 0;
    while (i + 8 <= n) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        w *= K_PRIME2;
        w = rotl64(w, 31);
        w *= K_PRIME1;
        h ^= w;
        h = rotl64(h, 27) * K_PRIME1 + K_PRIME4;
        i += 8;
    }
    while (i < n) {
        h ^= (uint64_t)p[i] * K_PRIME5;
        h = rotl64(h, 11) * K_PRIME1;
        i++;
    }
    h ^= (uint64_t)n;
    return mix64(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* below this, one walker beats the cost of opening every door */
    const size_t SMALL = (size_t)1 << 20; /* 1 MiB */

    int nthreads = 1;
#ifdef _OPENMP
    if (len >= SMALL) {
        nthreads = omp_get_max_threads();
        if (nthreads < 1) nthreads = 1;
    }
#endif

    if (len < SMALL || nthreads <= 1) {
        return hash_block(data, len, K_OFFSET);
    }

    size_t base = (len / (size_t)nthreads) & ~((size_t)7);
    if (base == 0) base = 8;
    size_t nblocks = len / base;
    if (nblocks > (size_t)nthreads) nblocks = (size_t)nthreads;
    if (nblocks < 1) nblocks = 1;

    uint64_t *partial = (uint64_t *)malloc(nblocks * sizeof(uint64_t));
    if (!partial) {
        return hash_block(data, len, K_OFFSET);
    }

    #pragma omp parallel for schedule(static) num_threads((int)nblocks)
    for (long b = 0; b < (long)nblocks; b++) {
        size_t start = (size_t)b * base;
        size_t end = ((size_t)b == (long)nblocks - 1) ? len : start + base;
        uint64_t seed = K_OFFSET ^ ((uint64_t)b * K_PRIME3 + 0x9E3779B97F4A7C15ULL);
        partial[b] = hash_block(data + start, end - start, seed);
    }

    uint64_t h = K_OFFSET ^ (uint64_t)len;
    for (size_t b = 0; b < nblocks; b++) {
        h ^= partial[b];
        h = rotl64(h, 29) * K_PRIME2 + partial[b];
        h = mix64(h);
    }

    free(partial);
    return h;
}
```

PREDICTION: 3.2