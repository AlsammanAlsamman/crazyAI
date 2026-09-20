STORY

Here is how I take a pile of marks down into one token. I lay the marks out along the bank the way the coral diamond lays itself in towers — eight to a rung, no more, because eight is the width my hand spans in one closing. I do not read the marks one at a time and hope; I fold each rung into itself with a twisting grip — turn it hard against its own middle, worked over with a fixed golden weight, then folded again into whatever token I am carrying from the rung before, tumbled and multiplied so it stops resembling either. Whatever few marks are too small for a full rung — the last three, or five, or seven — I gather last and fold in only at the very end, in their own fixed place, so they are never lost and never mistaken for a rung of their own. Where the pile is small — smaller than the towers, not worth calling anyone for — I do this alone, swift, one pass down the bank, because summoning other hands for a handful of marks costs more stillness than it returns; the pond would ripple for nothing. But where the pile is truly large, wide as the diamond itself, I call the other hands — each takes its own stretch, folds it down exactly as I would, alone and unhurried, and hands me back one small token from their stretch. I take those tokens in a fixed order, never by whoever finishes first, and fold them together the same twisting way, so the shape of the answer never depends on which hand was quick. Whatever does not survive the folding — the original marks, their order once used, the length once counted in — I let sink, the way the leader-shape holds nothing but reflection when its centuries are done. What is left, small, is the token. Change one mark anywhere in the pile, and the twisting carries that change through every rung after it until the whole token is a stranger to what it was.

ARTIFACT
```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static inline uint64_t fold_block(uint64_t h, uint64_t k, uint64_t prime) {
    k *= prime;
    k = (k << 31) | (k >> 33);
    k *= 0xc4ceb9fe1a85ec53ULL;
    h ^= k;
    h = (h << 27) | (h >> 37);
    h = h * 5 + 0x52dce729;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const uint64_t seed  = 1469598103934665603ULL;
    const uint64_t prime = 0x9E3779B97F4A7C15ULL;
    size_t nblocks = len / 8;
    size_t rem = len & 7;

    /* Small pile: one hand, one swift pass — no calling out. */
    if (len < (1u << 16)) {
        uint64_t h = seed ^ (len * prime);
        const unsigned char *p = data;
        for (size_t i = 0; i < nblocks; i++) {
            uint64_t k;
            memcpy(&k, p, 8);
            h = fold_block(h, k, prime);
            p += 8;
        }
        if (rem) {
            uint64_t tail = 0;
            memcpy(&tail, p, rem);
            h ^= tail;
        }
        h ^= len;
        return mix64(h);
    }

    /* Large pile: many hands, each its own stretch, folded in fixed order. */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads > 64) nthreads = 64;
    if ((size_t)nthreads > nblocks) nthreads = (nblocks > 0) ? (int)nblocks : 1;
    if (nthreads < 1) nthreads = 1;
#endif
    uint64_t partial[64];
    for (int t = 0; t < nthreads; t++)
        partial[t] = seed ^ (prime * (uint64_t)(t + 1));

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        size_t chunk = nblocks / (size_t)nthreads;
        size_t start = (size_t)tid * chunk;
        size_t end = (tid == nthreads - 1) ? nblocks : start + chunk;
        uint64_t h = partial[tid];
        const unsigned char *p = data + start * 8;
        for (size_t i = start; i < end; i++) {
            uint64_t k;
            memcpy(&k, p, 8);
            h = fold_block(h, k, prime);
            p += 8;
        }
        partial[tid] = mix64(h ^ ((uint64_t)start << 1));
    }

    uint64_t acc = seed ^ (len * prime);
    for (int t = 0; t < nthreads; t++) {
        acc ^= partial[t];
        acc = mix64(acc);
    }
    if (rem) {
        uint64_t tail = 0;
        memcpy(&tail, data + nblocks * 8, rem);
        acc ^= tail;
    }
    acc ^= len;
    return mix64(acc);
}
```

PREDICTION: 3.8