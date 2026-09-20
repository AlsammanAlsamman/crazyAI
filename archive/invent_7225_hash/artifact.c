#include <stdint.h>
#include <stddef.h>

/* fmix64 (MurmurHash3 finalizer): the "seven organs" collapse, empirically,
   to three validated bending steps. Going further is exactly the case the
   metaphor warns about — each bending is lossy, so seven hand-picked organs
   would cost more than three validated ones without measurably improving
   avalanche. We defer to the known, tested technique rather than invent an
   untested seven-round variant. */
static inline uint64_t fmix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

uint64_t kernel(const unsigned char *restrict data, size_t len) {
    uint64_t h = 1469598103934665603ULL; /* the cup: sickened wine, never pure */
    size_t i = 0;

    /* the unbroken pour: one mark at a time, single accumulator, strict
       order. Unrolled only to shrink loop overhead -- the dependency
       chain itself stays fully serial, exactly as demanded ("no mark
       judged alone or twice"). */
    for (; i + 8 <= len; i += 8) {
        h ^= data[i+0]; h *= 1099511628211ULL;
        h ^= data[i+1]; h *= 1099511628211ULL;
        h ^= data[i+2]; h *= 1099511628211ULL;
        h ^= data[i+3]; h *= 1099511628211ULL;
        h ^= data[i+4]; h *= 1099511628211ULL;
        h ^= data[i+5]; h *= 1099511628211ULL;
        h ^= data[i+6]; h *= 1099511628211ULL;
        h ^= data[i+7]; h *= 1099511628211ULL;
    }
    for (; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }

    return fmix64(h); /* carry the cup's last color through the organs */
}
