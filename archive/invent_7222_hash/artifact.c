#include <stdint.h>
#include <stddef.h>

uint64_t kernel(const unsigned char *restrict data, size_t len) {
    /* seat the stone at a starting position that already reflects the pile's length */
    uint64_t h = (uint64_t)len;

    #pragma GCC unroll 8
    for (size_t i = 0; i < len; i++) {
        h += data[i];   /* press this mark's weight into the stone's current seated place */
        h += h << 20;   /* quarter turn through the groove: fold the current position forward */
        h ^= h >> 12;   /* the stone's memory of the last press bends how deep this one sinks */
    }

    /* final seating against the wooden numbered keeps: one settled reading, nothing else leaves the desk */
    h += h << 6;
    h ^= h >> 22;
    h += h << 30;
    return h;
}
