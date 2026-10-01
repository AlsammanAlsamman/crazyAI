        }
    }

    /* step 9: the closing.  The tail from any treasure to the far ends of
       both ropes is priced by the very same arithmetic, so the walk simply
       continues to d = 2n and the surviving closings meet at (n,n).      */
    return (int)buf[(2 * n) % 3][n];
}
#endif

/* ------------------------------ the kernel ------------------------------ */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n >= 64 && n <= 14000) {
        char *rawA = (char *)malloc((size_t)n + 128);
        char *rawB = (char *)malloc((size_t)n + 128);
        size_t bsz = (size_t)(n + 129);
        int16_t *raw = (int16_t *)malloc(3 * bsz * sizeof(int16_t));
        if (rawA && rawB && raw) {
            char *A, *BR;
            int16_t *b0, *b1, *b2;
            int i, lb, s, k, need, res;

            /* step 1: lay the first rope along the east wall, doorway end at
               the doorway, and count its knots from the door outward.  Lay
               the second rope from the same doorway across the floor at a
               slant -- stored reversed so the slanted count runs forward.
               Nothing is written on the floor: the counts are the addresses. */
            A = rawA + 64; BR = rawB + 64;
            memset(rawA, 1, (size_t)n + 128);   /* off-rope: dye 1 */
            memset(rawB, 2, (size_t)n + 128);   /* off-rope: dye 2, never 1 */
            memcpy(A, a, (size_t)n);
            for (i = 0; i < n; ++i) BR[i] = b[n - 1 - i];
            b0 = raw + 64; b1 = raw + bsz + 64; b2 = raw + 2 * bsz + 64;

            /* step 8 (global, part 1): a standing tally to drop heads
               against -- the perfectly in-step walk, O(n).               */
            lb = 0;
            for (i = 0; i < n; ++i) lb += (a[i] == b[i]) ? 1 : -1;

            /* one cheap narrow walk sharpens the standing tally */
            k = 32; if (k > n) k = n;
            s = sweep(n, A, BR, k, b0, b1, b2);
            if (s > lb) lb = s;

            /* step 6 arithmetic as a certificate: any walk outside
               |i-j| <= need scores <= n - 5*need <= lb, so it is bare
               floor and never needs naming.                            */
            need = (n - lb + 4) / 5;
            if (need < 1) need = 1;
            if (need > n) need = n;

            /* step 10: of all the closings that survive, keep the smallest
               cost -- i.e. the largest tally.  Announce it, coil the ropes. */
            res = (need <= k) ? s : sweep(n, A, BR, need, b0, b1, b2);

            free(rawA); free(rawB); free(raw);
            return res;
        }
        free(rawA); free(rawB); free(raw);
    }
#endif
    return nw_scalar(n, a, b);
}
