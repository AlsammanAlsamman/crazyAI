STORY

I take the pile of marks as marbles poured down a chute in the order they fell, and I never let that order slip. I set a first marble in my palm — a weight from the old parchment count, one nine four six... — and it never changes twice the same way twice. Then, eight marbles at a time, I let electricity run them through the pipes: each eight-clutch gets folded through a panther's forefinger-thumb — squeeze, spill sideways, squeeze again, thrice — until no one glint remembers which marble it came from. I multiply the clutch against the old prime weight, twist it through a half-turn on the tensed-limb wheel so the top marbles fall to the bottom, and roll on. Whatever marbles are left at the end, too few to make a clutch, I do not discard — I stack them thumb-high, mark how many there are so an empty pile can never wear the same face as a pile of pure zeroes, and fold that stack in same as the rest. Last I stir in a mark for the whole pile's length, because two piles of different length must never rest at the same point on the sky's dial. Now — a small pile, one arcade's worth or less, I work with my own two hands alone; calling the other hands over for a few dozen marbles wastes more breath than it saves, the way you don't summon the whole shooter-crew to sweep one step. But past a certain weight — past what a single lap around the arcades can carry — I cut the pile into eight walks, hand one to each of eight hands standing at the panther-bars, let them each fold their walk down exactly as I would alone, seeded so no two walks ever echo each other, and when they're done I gather the eight small tokens and fold those down once more, in their fixed order, into the one token that goes to the strangers.

ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static uint64_t hash_chunk(const unsigned char *data, size_t len, uint64_t seed) {
    uint64_t h = seed ^ 0x9E3779B97F4A7C15ULL;
    const uint64_t prime = 0x100000001B3ULL;
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t k;
        __builtin_memcpy(&k, data + i, 8);
        h ^= mix64(k);
        h *= prime;
        h = (h << 31) | (h >> 33);
    }
    uint64_t tail = 0;
    size_t rem = len - i;
    for (size_t j = 0; j < rem; j++) {
        tail |= (uint64_t)data[i + j] << (8 * j);
    }
    if (rem) {
        h ^= mix64(tail ^ ((uint64_t)rem << 56));
        h *= prime;
    }
    h ^= (uint64_t)len;
    return mix64(h);
}

#define NCHUNKS 8
#define PARALLEL_THRESHOLD ((size_t)1 << 20) /* 1 MiB: below this, one hand alone */

uint64_t kernel(const unsigned char *data, size_t len) {
    const uint64_t OFFSET = 1469598103934665603ULL;

    if (len < PARALLEL_THRESHOLD) {
        return hash_chunk(data, len, OFFSET);
    }

    size_t bounds[NCHUNKS + 1];
    size_t base = len / NCHUNKS;
    for (int c = 0; c <= NCHUNKS; c++) {
        bounds[c] = (c == NCHUNKS) ? len : (size_t)c * base;
    }

    uint64_t partial[NCHUNKS];

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < NCHUNKS; c++) {
        uint64_t seed = OFFSET ^ ((uint64_t)c * 0x9E3779B97F4A7C15ULL + 0xBF58476D1CE4E5B9ULL);
        partial[c] = hash_chunk(data + bounds[c], bounds[c + 1] - bounds[c], seed);
    }

    uint64_t h = OFFSET ^ (uint64_t)len;
    for (int c = 0; c < NCHUNKS; c++) {
        h ^= mix64(partial[c] + (uint64_t)c * 0x100000001B3ULL);
        h *= 0x100000001B3ULL;
    }
    return mix64(h);
}
```

PREDICTION: 5