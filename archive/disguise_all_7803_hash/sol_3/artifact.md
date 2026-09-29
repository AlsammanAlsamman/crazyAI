## APPROACH

**Mapping the disguised solution onto the real problem.**

| Grandma Nell's world | This hash |
|---|---|
| Pebbles arriving one at a time, each forced to wait for a "multiplied twist" before the next | FNV‑1a / xxHash: one accumulator, `h ^= byte; h *= prime;` — a serial multiply‑latency chain, ~3 cycles per byte |
| "Don't insist on one spinner or one strict order" | No single accumulator and no strict left‑to‑right dependence: the combine operator is **XOR**, which is associative *and* commutative, so the bracket shape is irrelevant and every leaf can be computed independently |
| "Each pair of hands presses two colors together — an overlay, nothing multiplied" | Leaf transform is **multiply‑free**: `x ^= key; x ^= x>>29; x += x<<13; x ^= x>>31; x += x<<17`. The two `x += x<<s` steps are shift‑and‑add (i.e. multiplication by a sparse odd constant done with an adder) — invertible, and *nonlinear over GF(2)* thanks to carries, which is what kills the symmetry that a pure XOR/rotate tree would have |
| "Half as many little spinners, pair them again, keep going, like a tournament bracket" | 4 independent 256‑bit accumulators (16 lanes of work in flight) are folded pairwise: `(a0^a1)^(a2^a3)`, then the 4 surviving 64‑bit lanes are folded pairwise again in the finalizer. A literal log‑depth tree materialised in scratch memory would re‑read the buffer; the bracket is instead collapsed **into registers**, which is the same value with one pass over memory |
| "Everybody works on their pairs at the same time" | SIMD (AVX2) gives 4 leaves per instruction; 4 accumulators give instruction‑level parallelism so nothing waits on a latency chain; OpenMP splits the block range across threads for buffers ≥ 4 MB. Because keys are a function of the *absolute* block index and the combine is XOR, the parallel result is **bit‑identical** to the serial one for any thread count |
| "Swapping one pebble changes half the blades, because the change rides through every pairing to the top" | Each lane enters the root through a bijection (`+`, rotate, then a full `fmix64`), so a one‑bit input flip *always* produces a nonzero root delta, and the O(1) strong finalizer converts it to ~32 flipped output bits |
| "The multiplied twist is the only way" (the assumption being attacked) | Multiplication survives **only in the O(1) finalizer** (3 `fmix64`), never per byte. "More rounds always better" is rejected too: one pass, two cheap nonlinear steps per word |

**Two correctness details I had to design around** (both are real traps in this mechanism):

1. *Positional distinctness.* With XOR combining, a rotate‑only tree has at most 64 distinct leaf transforms, so two words always share one and swapping them is an instant collision. Fixed by keying each word position with its own key `KB[l] + j*KS[l]` (maintained incrementally by one vector add per block) **and** making the leaf nonlinear — with a linear leaf, position keys cancel out and word swaps still collide.
2. *High‑bit degeneracy.* `x += x<<s` never propagates from bit 63, so an MSB‑only flip gives a fixed, value‑independent delta — which would let the overlapping tail block cancel the main block, and let two MSB flips 32 bytes apart collide. Fixed by leading with `x ^= x>>29` (so every single‑bit flip acquires a sub‑MSB component and becomes carry/value dependent) and by absorbing the overlapping tail into **permuted lanes with rotations**, so a word's two contributions can never land in the same lane.

Small inputs get short paths (overlapping 16/8/4/1‑byte reads — no padding, no `memcpy`, no OOB read), with `len` mixed into the key so different lengths never alias.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* per-position key schedule: key(block j, lane l) = KB[l] + j*KS[l] */
static const uint64_t KB[4] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0xD6E8FEB86659FD93ULL };
static const uint64_t KS[4] = {
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
    0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL };

static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* leaf: multiply-free, invertible, nonlinear (shift-and-add carries) */
static inline uint64_t leaf64(uint64_t x, uint64_t key) {
    x ^= key;
    x ^= x >> 29;
    x += x << 13;
    x ^= x >> 31;
    x += x << 17;
    return x;
}

/* O(1) strong mixer -- the only place a multiply appears */
static inline uint64_t fmix64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31; return x;
}

#if defined(__AVX2__)
#define VLEAF(x) do {                                        \
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 29));       \
    x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 13));       \
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 31));       \
    x = _mm256_add_epi64(x, _mm256_slli_epi64(x, 17));       \
} while (0)
#endif

/* absorb 32-byte blocks [j0,j1) as independent leaves; XOR-fold into out[4].
   XOR is associative+commutative, so this is exactly a tournament bracket
   over those blocks, in any order, from any number of workers. */
static void absorb_range(const unsigned char *data, size_t j0, size_t j1, uint64_t out[4])
{
    uint64_t k[4];
    int l;
    for (l = 0; l < 4; l++) k[l] = KB[l] + (uint64_t)j0 * KS[l];
    const unsigned char *p = data + (j0 << 5);
    size_t n = j1 - j0, i = 0;

#if defined(__AVX2__)
    const __m256i S1 = _mm256_loadu_si256((const __m256i *)KS);
    const __m256i S2 = _mm256_add_epi64(S1, S1);
    const __m256i S3 = _mm256_add_epi64(S2, S1);
    const __m256i S4 = _mm256_add_epi64(S2, S2);
    __m256i base = _mm256_loadu_si256((const __m256i *)k);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;

    for (; i + 4 <= n; i += 4, p += 128) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p      )), base);
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 32)), _mm256_add_epi64(base, S1));
        __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 64)), _mm256_add_epi64(base, S2));
        __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(p + 96)), _mm256_add_epi64(base, S3));
        VLEAF(x0); VLEAF(x1); VLEAF(x2); VLEAF(x3);
        a0 = _mm256_xor_si256(a0, x0);
        a1 = _mm256_xor_si256(a1, x1);
        a2 = _mm256_xor_si256(a2, x2);
        a3 = _mm256_xor_si256(a3, x3);
        base = _mm256_add_epi64(base, S4);
    }
    for (; i < n; i++, p += 32) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)p), base);
        VLEAF(x0);
        a0 = _mm256_xor_si256(a0, x0);
        base = _mm256_add_epi64(base, S1);
    }
    /* pair, press, pair, press */
    a0 = _mm256_xor_si256(a0, a1);
    a2 = _mm256_xor_si256(a2, a3);
    a0 = _mm256_xor_si256(a0, a2);
    {
        uint64_t t[4];
        _mm256_storeu_si256((__m256i *)t, a0);
        out[0] ^= t[0]; out[1] ^= t[1]; out[2] ^= t[2]; out[3] ^= t[3];
    }
#else
    /* scalar fallback: identical lane/key mapping => identical output */
    uint64_t b0 = k[0], b1 = k[1], b2 = k[2], b3 = k[3];
    uint64_t A0 = 0, A1 = 0, A2 = 0, A3 = 0, B0 = 0, B1 = 0, B2 = 0, B3 = 0;
    for (; i + 2 <= n; i += 2, p += 64) {
        A0 ^= leaf64(ld64(p      ), b0);
        A1 ^= leaf64(ld64(p +  8 ), b1);
        A2 ^= leaf64(ld64(p + 16 ), b2);
        A3 ^= leaf64(ld64(p + 24 ), b3);
        B0 ^= leaf64(ld64(p + 32 ), b0 + KS[0]);
        B1 ^= leaf64(ld64(p + 40 ), b1 + KS[1]);
        B2 ^= leaf64(ld64(p + 48 ), b2 + KS[2]);
        B3 ^= leaf64(ld64(p + 56 ), b3 + KS[3]);
        b0 += 2 * KS[0]; b1 += 2 * KS[1]; b2 += 2 * KS[2]; b3 += 2 * KS[3];
    }
    for (; i < n; i++, p += 32) {
        A0 ^= leaf64(ld64(p      ), b0);
        A1 ^= leaf64(ld64(p +  8 ), b1);
        A2 ^= leaf64(ld64(p + 16 ), b2);
        A3 ^= leaf64(ld64(p + 24 ), b3);
        b0 += KS[0]; b1 += KS[1]; b2 += KS[2]; b3 += KS[3];
    }
    out[0] ^= A0 ^ B0; out[1] ^= A1 ^ B1; out[2] ^= A2 ^ B2; out[3] ^= A3 ^ B3;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t acc[4];

    if (len >= 32) {
        size_t nb = len >> 5;                 /* full 32-byte leaf blocks */
        acc[0] = acc[1] = acc[2] = acc[3] = 0;

#ifdef _OPENMP
        if (len >= ((size_t)4 << 20) && omp_get_max_threads() > 1) {
            uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            #pragma omp parallel reduction(^ : r0, r1, r2, r3)
            {
                int nt = omp_get_num_threads(), id = omp_get_thread_num();
                size_t per = (nb + (size_t)nt - 1) / (size_t)nt;
                per = (per + 3) & ~(size_t)3;        /* keep 4-block groups whole */
                size_t b0 = (size_t)id * per;
                size_t b1 = b0 + per;
                if (b0 > nb) b0 = nb;
                if (b1 > nb) b1 = nb;
                uint64_t loc[4] = { 0, 0, 0, 0 };
                if (b1 > b0) absorb_range(data, b0, b1, loc);
                r0 ^= loc[0]; r1 ^= loc[1]; r2 ^= loc[2]; r3 ^= loc[3];
            }
            acc[0] = r0; acc[1] = r1; acc[2] = r2; acc[3] = r3;
        } else
#endif
        absorb_range(data, 0, nb, acc);

        /* final 32 bytes (may overlap): permuted lanes + rotation so the
           duplicated region can never cancel the main absorption */
        {
            const unsigned char *t = data + len - 32;
            uint64_t tk = 0x2545F4914F6CDD1DULL + (uint64_t)len;
            uint64_t v0 = leaf64(ld64(t      ), KB[0] ^ tk);
            uint64_t v1 = leaf64(ld64(t +  8 ), KB[1] ^ tk);
            uint64_t v2 = leaf64(ld64(t + 16 ), KB[2] ^ tk);
            uint64_t v3 = leaf64(ld64(t + 24 ), KB[3] ^ tk);
            acc[1] ^= ROTL64(v0, 11);
            acc[2] ^= ROTL64(v1, 23);
            acc[3] ^= ROTL64(v2, 37);
            acc[0] ^= ROTL64(v3, 53);
        }
    } else if (len >= 16) {
        uint64_t tk = 0x2545F4914F6CDD1DULL + (uint64_t)len;
        acc[0] = leaf64(ld64(data          ), KB[0] ^ tk);
        acc[1] = leaf64(ld64(data + 8      ), KB[1] ^ tk);
        acc[2] = leaf64(ld64(data + len - 16), KB[2] ^ tk);
        acc[3] = leaf64(ld64(data + len - 8 ), KB[3] ^ tk);
    } else {
        uint64_t a, b, x, y;
        if (len >= 8)      { a = ld64(data); b = ld64(data + len - 8); }
        else if (len >= 4) { a = ld32(data); b = ld32(data + len - 4); }
        else if (len)      { a = ((uint64_t)data[0] << 16) |
                                 ((uint64_t)data[len >> 1] << 8) |
                                  (uint64_t)data[len - 1]; b = 0; }
        else               { a = 0; b = 0; }
        x = a ^ (KB[0] + (uint64_t)len);
        y = b ^ KB[1];
        x += ROTL64(y, 31);
        y ^= ROTL64(x, 19);
        x += y;
        return fmix64(x);
    }

    /* root of the bracket: pair, press, pair, press -- each lane enters
       through a bijection, so any single-bit input delta reaches the root */
    {
        uint64_t t0 = fmix64(acc[0] + ROTL64(acc[1], 29) + (uint64_t)len);
        uint64_t t1 = fmix64(acc[2] + ROTL64(acc[3], 41) + 0x9E3779B97F4A7C15ULL);
        return fmix64(t0 ^ ROTL64(t1, 17));
    }
}
```

PREDICTION: 20

(Read as: ~20× the throughput of the FNV‑1a reference on a multi‑kilobyte buffer. Reasoning: FNV‑1a is latency‑bound at ~3 cycles/byte — one dependent `imul` per byte. This kernel issues ~12 AVX2 instructions per 32 bytes with four independent accumulator chains, ≈0.1 cycles/byte when the buffer is cache‑resident, i.e. ~30×, throttled toward ~10–15× once DRAM bandwidth binds; 20 is my central estimate across typical benchmark sizes. Sub‑32‑byte keys should still win ~3×, since only the O(1) finalizer's two multiplies remain.)

## MEASUREMENT

**Not measured — and I will not dress that up.** This session was invoked with no tools available (`hash_bench`, `hash_contract`, and the rest are absent), so I have zero measurements, zero avalanche statistics, and zero improvement iterations to report. The "improve it at most four times" loop did not happen; what is above is the first and only version.

What I *can* report is what I verified by hand rather than by instrument:

- **Contract**: signature is exactly `uint64_t kernel(const unsigned char *data, size_t len)`; no globals mutated; `len == 0` returns a fixed value; no reads outside `[data, data+len)` on any path (the ≥32 path's tail reads the last 32 bytes, the 16–31 path reads the first and last 16, the <16 path uses overlapping 8/4 reads or single bytes).
- **Determinism across thread counts**: leaf keys depend only on the absolute block index and the fold is XOR, so serial and any‑`nt` parallel runs are bit‑identical. The AVX2 and scalar paths use the same lane/key mapping and also agree bit‑for‑bit.
- **Avalanche argument** (analytic, not measured): each input byte lands in exactly one lane via an invertible leaf, each lane enters the root via an injective chain (`+ const`, rotate, `fmix64`), and the root passes through a third `fmix64`. So a one‑bit flip yields a nonzero root delta with certainty, then splitmix64‑grade diffusion. I expect a bias well under 1% per output bit — but *expect* is the operative word.

Honest residual risks, in order of how much they'd bother me: (1) the whole leaf stage is XOR‑accumulated, so quality rests on the leaf's carry nonlinearity plus the per‑position keys — a differential collision search would attack there, not at avalanche; (2) the 4 MB OpenMP threshold is a guess and could be wrong in either direction for the bench's buffer size; (3) if the bench weights very short keys heavily, the two‑multiply finalizer floor (~10–15 cycles) caps the speedup far below 20.

## VERDICT

The mechanism translates cleanly and, I believe, faithfully: the serial multiply‑per‑byte accumulator is gone, replaced by a commutative XOR tournament over independently computed, position‑keyed, multiply‑free leaves, with strong mixing paid exactly once at the root. That is the disguised solution's actual claim — parallel pairing plus cheap overlay, with avalanche preserved because a change "rides along through every pairing it touches" — and it is also, not coincidentally, why the design should be several‑fold to an order of magnitude faster than FNV‑1a.

But the verdict on *performance* is unearned in this session. I predicted 20× before measuring and then could not measure, so the correct statement is: **prediction recorded, result unknown.** If the pipeline's numbers come back at 3× or the avalanche bias is visibly nonzero, the prediction was simply wrong, and the first two places I'd look are the tail/overlap handling and whether the 12‑instruction leaf can be trimmed to 8 without reintroducing the high‑bit degeneracy I designed around.