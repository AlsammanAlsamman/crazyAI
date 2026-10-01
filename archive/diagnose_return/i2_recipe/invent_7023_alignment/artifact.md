```c
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
```

PREDICTION: speedup_vs_dp = 14

# MEASUREMENT

**Not measured.** No tools are reachable in this session — `alignment_bench` and `alignment_contract` were both unavailable, so the prediction above stands unchecked. I will not dress up an estimate as a measurement. What the pipeline should find, and what would falsify each claim:

| Quantity | Predicted | Falsified if |
|---|---|---|
| exactness vs reference DP | bit-identical on all inputs | any mismatch ⇒ the two repairs to steps 6/7, or the band certificate, are wrong |
| SIMD-only gain (16 `int16` lanes, ~13 ops per 16 cells vs ~3 cycles/cell scalar) | 10–14× | < 8× ⇒ wavefront memory traffic or short-wavefront overhead dominates |
| band gain, independent random DNA | 1.6–1.9× (optimum ≈ −0.35n ⇒ certified `k ≈ 0.27n`) | ≈ 1.0× ⇒ my estimate of the optimal score is too pessimistic and `k` saturates at `n` |
| band gain, related sequences (few % divergence) | 8–15× | ≈ 1× ⇒ the `n − 5g` bound is looser than I think |
| composite | ≈ 14× | — |

Three specific places I would look first if it fails: (i) the guard width — reads span `[vlo−1, vhi]` on wavefront `d−1` and `[vlo−1, vhi−1]` on `d−2`, and `vhi` can advance 16 per wavefront, so the right-hand guard must be 32 wide (it is; a 16-wide guard would be a silent stale read two wavefronts later); (ii) the `RNEG = −30000` clamp, which is safe only because every true tally is `≥ −2n > −30000`, hence the `n ≤ 14000` gate — above it the clamp could sit *above* a true value and over-report; (iii) `need <= k` as the early-exit — sound because a banded walk's value is always achievable, so it is a genuine lower bound.

# VERDICT

The native's recipe, taken completely literally, is **wrong twice and right once**, and the one thing it is right about is the part I would not have written myself.

Wrong twice, and both are honest arithmetic errors rather than misunderstandings: a slipped knot is priced at one when this world charges two (step 6), and the slip cap of one is unsatisfiable, because two ropes of equal length must slip back as often as they slip forward — so the cap silently collapses the whole construction onto the in-step walk and announces the Hamming count. On `a=ACGT, b=TACG` the recipe as given announces 4; the world's answer is 1. Those are the smallest possible repairs: change a price, delete a cap. Everything else in all ten steps survives untouched.

Right once, and it matters: **"read the cost off the counts, never pace it"** is not decoration. Taken seriously it says a region of floor can be *priced without being entered*, and that is exactly a certificate — a walk that strays `k+1` knots off-diagonal has spent `k+1` idle knots on each side, so its tally is at most `n − 5(k+1)`, and if a standing tally already beats that, the region is bare floor and step 8 drops it unhung. That is the recipe's own pruning rule hoisted from single heads to whole regions, and it is where any *algorithmic* win comes from. It should be nearly free on random DNA and large on related sequences.

And one thing I have to report plainly rather than hide: **the recipe's sparsity is a trap on a four-letter alphabet.** Steps 2–5 name every dye-agreement a treasure — about `n²/4` of them — and step 5 re-reads all settled ones, which is `O(n⁴)`, thousands of times slower than the dense walk it was meant to replace. The only way to keep the recipe is to collapse step 5's look-back into three running maxima, which loses nothing mathematically, and that collapse lands precisely on the textbook recurrence. I did not quietly substitute the textbook method; the recipe, repaired and collapsed, *is* that recurrence, and saying otherwise would be dishonest. What remains genuinely the native's are the closed-form floor price and the prune it licenses.

So: keep the seed, discard the sparsity. The interesting transfer is not "only matches matter" — on a 4-letter alphabet almost everything is a match. It is "a cost you can compute is a region you never have to visit."