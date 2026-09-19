#include <omp.h>

void kernel(int n, const double *in, double *out) {
    /* copy the garden: leaves start exactly as the input, in the straight path's order */
    for (int i = 0; i < n; i++) out[i] = in[i];

    /* walk the ribs, fixed order, trunk to lotus petal: h = 1, 2, 4, ..., n/2 */
    for (int h = 1; h < n; h <<= 1) {
        int step = h << 1;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += step) {
            for (int j = i; j < i + h; j++) {
                double a = out[j];
                double b = out[j + h];
                out[j]     = a + b;   /* leaf raised: this rib's stripe, +1 branch */
                out[j + h] = a - b;   /* leaf flat:   this rib's stripe, -1 branch */
            }
        }
    }
}
