# APPROACH

Bouton's theorem still requires two O(n) passes over the pile array: one to XOR-reduce all piles, and one to locate a pile where the winning move applies. The known solution's second pass computes `piles[i] ^ xor_all` and compares it to `piles[i]` for every pile until a hit.

That comparison is unnecessary work. The classical proof of Bouton's theorem shows something stronger: if `xor_all != 0`, let `hb` be its **highest set bit**. Because that bit appears an odd number of times across all piles, at least one pile has `hb` set, and for *any* pile with `hb` set, XOR-ing with `xor_all` is guaranteed to clear that top bit (since `xor_all` also has it set) without being able to set any *higher* bit — so the result is unconditionally smaller. That means the search step degenerates from "XOR and compare" to a plain bitwise AND test (`piles[i] & hb`), which is cheaper per element and trivially vectorizable with a compare-to-zero + movemask instead of a subtraction/compare.

Both passes (reduction and search) are simple, data-parallel, branch-light scans, so I vectorize each with AVX2 (guarded by `#if defined(__AVX2__)`, which `-march=native` will define when the build host supports it) and keep scalar fallbacks for the tail and for non-AVX2 targets, guaranteeing correctness everywhere. I skip OpenMP: pile counts in Nim positions are typically small to moderate, and thread-spawn overhead would dominate an O(n) memory scan at those sizes, likely making things *slower*, not faster.

# ARTIFACT

```c
#include <immintrin.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int xor_all = 0;
    int i = 0;

#if defined(__AVX2__)
    __m256i vxor = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        vxor = _mm256_xor_si256(vxor, v);
    }
    {
        __m128i lo  = _mm256_castsi256_si128(vxor);
        __m128i hi  = _mm256_extracti128_si256(vxor, 1);
        __m128i x128 = _mm_xor_si128(lo, hi);
        __m128i x64  = _mm_xor_si128(x128, _mm_srli_si128(x128, 8));
        __m128i x32  = _mm_xor_si128(x64,  _mm_srli_si128(x64, 4));
        xor_all ^= _mm_cvtsi128_si32(x32);
    }
#endif
    for (; i < n; i++) xor_all ^= piles[i];

    if (xor_all == 0) {
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* highest set bit of xor_all: any pile with this bit set works */
    unsigned int hb = 1u << (31 - __builtin_clz((unsigned int)xor_all));

    i = 0;
#if defined(__AVX2__)
    __m256i vhb = _mm256_set1_epi32((int)hb);
    __m256i vzero = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(piles + i));
        __m256i t = _mm256_and_si256(v, vhb);
        __m256i cmp = _mm256_cmpeq_epi32(t, vzero);       /* 0xFFFFFFFF where bit NOT set */
        unsigned int mask = (unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        unsigned int hit = (~mask) & 0xFFu;               /* bits where bit IS set */
        if (hit) {
            int idx = __builtin_ctz(hit);
            int pos = i + idx;
            *out_pile = pos;
            *out_remove = piles[pos] - (piles[pos] ^ xor_all);
            return;
        }
    }
#endif
    for (; i < n; i++) {
        if (piles[i] & hb) {
            *out_pile = i;
            *out_remove = piles[i] - (piles[i] ^ xor_all);
            return;
        }
    }
}
```

# PREDICTION: 1.4

# MEASUREMENT

No benchmarking tools were available in this session (explicitly noted in the environment: "No tools are available in this session"), so I could not execute `nim_bench` or any timing harness myself. The reasoning behind the 1.4× prediction: the AVX2-reduced XOR pass and the AND+movemask search pass both cut per-element work roughly in half compared to the baseline's per-element XOR+compare-and-branch in the search loop, and both loops process 8 ints/cycle-ish instead of 1; for small n (a handful of piles, the common Nim case) the two implementations should be statistically indistinguishable since both are dominated by fixed overhead, so I expect the win to show mainly at larger n. This is a plain estimate, not a measured result — the actual MEASUREMENT/VERDICT should be filled in by whatever harness compiles and times this code, since that capability is not present here.

# VERDICT

Unmeasured in this session — correctness is verified by hand against Bouton's theorem (the highest-bit argument guarantees any pile with `hb` set is a valid, strictly-reducing move, and the empty/zero-XOR case is preserved verbatim from the reference), but the relative-speed claim above is a prediction only, pending actual compilation and timing by the pipeline.