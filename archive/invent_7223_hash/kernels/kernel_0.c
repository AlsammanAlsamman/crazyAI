#include <stdint.h>
#include <stddef.h>

/* Bob Jenkins' one-at-a-time hash, generalized from 32-bit to 64-bit by
 * scaling its shift constants to preserve the original ratios-of-width
 * (10/32, 6/32 in the loop; 3/32, 11/32, 15/32 in the finalizer).
 * Mixing is done purely by add/shift/xor ("tumbles"), no multiplication -
 * a fixed count of tumbles (3) per byte, one running accumulator (the
 * sphere), only the final tumbled value kept (the last hairline crack).
 */
uint64_t kernel(const unsigned char * restrict data, size_t len) {
    uint64_t h = 0;                     /* the sphere: sole running memory */

    for (size_t i = 0; i < len; i++) {
        h += data[i];                   /* press the mark into the sphere */
        h += (h << 20);                 /* tumble 1 */
        h ^= (h >> 12);                 /* tumble 2 (strike -> crack)     */
    }

    /* the last, smallest crack: one final fixed tumble, kept as the token */
    h += (h << 6);
    h ^= (h >> 22);
    h += (h << 30);

    return h;
}
