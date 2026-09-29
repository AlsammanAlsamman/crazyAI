/* known-strong reference: murmur3 fmix64 applied after every 8-byte word and once at the end */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static uint64_t fmix(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k;
}
uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ len, w;
    size_t i = 0;
    for (; i + 8 <= len; i += 8) { memcpy(&w, data + i, 8); h = fmix(h ^ w); }
    w = 0; memcpy(&w, data + i, len - i); h = fmix(h ^ w ^ ((uint64_t)(len - i) << 56));
    return fmix(h);
}
