#include <stdint.h>
#include <stddef.h>
uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= data[i]; h *= 1099511628211ULL; }
    return h;
}
