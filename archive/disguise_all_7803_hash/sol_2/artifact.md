## APPROACH

**Mapping the disguised solution onto the real problem**

| Grandma Nell | Real hash |
|---|---|
| one spinner, blades take colors | one running 64-bit state (plus a few parallel copies) |
| "special twist" = turns worked out by *multiplying* the pebble number with the current turn count | FNV/xxHash's data-dependent `h *= prime` — a multiply on the critical path for **every byte** |
| **drop the multiplied twist** | no multiply anywhere in the streaming loop |
| "flip it over (swap left half and right half) and give it one plain quarter-turn — always the same, no matter the pebble" | a **fixed, data-independent bit rotation** of the state: swap halves = `rotl 32`, plus one plain step = `rotl 33`. 33 is odd, hence coprime to 64, so the rotation orbit visits all 64 bit positions instead of collapsing into a short cycle |
| "press the color on, *overlaying* what's already there rather than painting over it" | combine, don't replace: `^=` (pure overlay) on two lanes and `+=` (blended overlay, i.e. carries — my one cheap source of GF(2)-nonlinearity) on the other two |
| "no need to wait for pebbles one-by-one" | the update `h = rotl(h,r) ^ word` has a **data-independent** shift, so it is trivially widened: 4 AVX2 accumulators × 4 lanes = 16 independent folds, 128 B per iteration; the loop becomes load-bound, not latency-bound |
| "the pattern still ends up unrecognizable after a swapped pebble" | avalanche is provided **once at the end**, not per byte: the fold is injective enough that a single flipped input bit always changes at least one bit of the pre-final state, and a splitmix64 finalizer (2 multiplies **total**, O(1), not O(n)) turns that one-bit difference into ~32 flipped output bits |

Honest disclosure about faithfulness: the mechanism forbids the *per-pebble* multiplied twist, and this implementation has none — the streaming loop is rotate + xor/add only. It keeps two multiplies in the fixed-cost finalizer, because a purely linear pipeline cannot avalanche at all (each input bit would reach exactly one output bit, i.e. 1/64 instead of 1/2). That is the one place the story's "the pattern is unrecognizable" claim needs non-linearity, and putting it at the end instead of per byte is exactly the trade the disguised solution is arguing for.

Structure: AVX2 path for ≥128 B → 32 B/iteration scalar path with 4 accumulators (also the whole path when AVX2 is absent) → word tail → partial word via `memcpy` (never over-reads) → length mixed in (so trailing-zero-length ambiguity is impossible) → finalizer.

## ARTIFACT

```c
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define KERNEL_AVX2 1
#endif

/* "flip + one plain quarter-turn": fixed, data-independent rotations.
   All odd => coprime to 64 => full 64-position orbit, no short cycles. */
#define ROT_A 33u
#define ROT_B 23u
#define ROT_C 41u
#define ROT_D 13u

#define KC0 0x243F6A8885A308D3ULL
#define KC1 0x13198A2E03707344ULL
#define KC2 0xA4093822299F31D0ULL
#define KC3 0x082EFA98EC4E6C89ULL

static inline uint64_t k_rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

static inline uint64_t k_ld64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

/* splitmix64 finalizer: bijective, ~0.5 avalanche. O(1), not per byte. */
static inline uint64_t k_final(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

#ifdef KERNEL_AVX2
static inline __m256i k_rot256(__m256i v, const int r) {
    return _mm256_or_si256(_mm256_slli_epi64(v, r),
                           _mm256_srli_epi64(v, 64 - r));
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;
    uint64_t h0 = KC0, h1 = KC1, h2 = KC2, h3 = KC3;
    uint64_t h;

#ifdef KERNEL_AVX2
    if (n >= 128) {
        __m256i a0 = _mm256_set1_epi64x((long long)KC0);
        __m256i a1 = _mm256_set1_epi64x((long long)KC1);
        __m256i a2 = _mm256_set1_epi64x((long long)KC2);
        __m256i a3 = _mm256_set1_epi64x((long long)KC3);
        uint64_t t[16];
        int i;

        do {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(p +  0));
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            __m256i v2 = _mm256_loadu_si256((const __m256i *)(p + 64));
            __m256i v3 = _mm256_loadu_si256((const __m256i *)(p + 96));
            /* fixed flip, then overlay: xor (pure) / add (blended, carries) */
            a0 = _mm256_xor_si256(k_rot256(a0, ROT_A), v0);
            a1 = _mm256_add_epi64(k_rot256(a1, ROT_B), v1);
            a2 = _mm256_xor_si256(k_rot256(a2, ROT_C), v2);
            a3 = _mm256_add_epi64(k_rot256(a3, ROT_D), v3);
            p += 128;
            n -= 128;
        } while (n >= 128);

        _mm256_storeu_si256((__m256i *)&t[0],  a0);
        _mm256_storeu_si256((__m256i *)&t[4],  a1);
        _mm256_storeu_si256((__m256i *)&t[8],  a2);
        _mm256_storeu_si256((__m256i *)&t[12], a3);
        for (i = 0; i < 16; i++)
            h0 = k_rotl64(h0, ROT_A) ^ t[i];
    }
#endif

    while (n >= 32) {
        h0 = k_rotl64(h0, ROT_A) ^ k_ld64(p +  0);
        h1 = k_rotl64(h1, ROT_B) + k_ld64(p +  8);
        h2 = k_rotl64(h2, ROT_C) ^ k_ld64(p + 16);
        h3 = k_rotl64(h3, ROT_D) + k_ld64(p + 24);
        p += 32;
        n -= 32;
    }

    h = k_rotl64(h0, 11) ^ k_rotl64(h1, 27)
      ^ k_rotl64(h2, 43) ^ k_rotl64(h3, 59);

    while (n >= 8) {
        h = k_rotl64(h, ROT_A) ^ k_ld64(p);
        p += 8;
        n -= 8;
    }
    if (n) {                       /* 1..7 bytes, no over-read */
        uint64_t w = 0;
        memcpy(&w, p, n);
        h = k_rotl64(h, ROT_C) ^ w;
    }

    h ^= (uint64_t)len;
    return k_final(h);
}
```

## PREDICTION

PREDICTION: 18

(Reasoning stated before any measurement: FNV-1a is a serial `xor`+`imul` chain, ≈4 cycles/byte ≈ 0.25 B/cycle. This kernel's bulk loop has no data-dependent latency chain — 16 independent folds, ~12–14 µops per 128 B — so it saturates load/L2 bandwidth at roughly 8–32 B/cycle, i.e. 30–100× on L1/L2-resident buffers, dropping toward DRAM bandwidth (~5–10×) for multi-MB buffers and toward ~2–4× for buffers under ~32 B where fixed overhead dominates. 18× is my estimate of the aggregate the harness will report over a mixed size sweep.)

## MEASUREMENT

**Not measured — reported as a failure of this session, not a success.** `hash_bench`, `hash_contract`, and the symbolic/unconventional tools were listed but no tool execution was available in this environment (confirmed: no callable bench tool), so the prediction above is unverified. What I can state without measuring:

- Contract: signature is exactly `uint64_t kernel(const unsigned char *data, size_t len)`; no writes to `data`, no allocation, no reads past `data+len` (the only sub-8-byte access is `memcpy(&w, p, n)` with `n < 8`), `len == 0` is well-defined, result is a pure function of `(data, len)` and is identical across the AVX2 and scalar paths' *own* outputs (the two paths give different values from each other — that is permitted; a hash need only be deterministic for a given build).
- Avalanche argument (analytic, not empirical): every input bit reaches at least one bit of the pre-final state and no single-bit difference can cancel (it appears in exactly one fold term), and `k_final` is splitmix64's finalizer, which is bijective with measured avalanche bias well under 1% in the published literature. So one-bit input flip ⇒ one-or-more-bit state flip ⇒ ~32 of 64 output bits flip.
- Known weakness I am not hiding: the streaming fold is GF(2)-linear on the xor lanes, so an adversary can construct collisions (two 8 KB-separated equal words in the same lane cancel). The `add_epi64` lanes break exact cancellation for half the input, and length is mixed in, but this is a *fast non-cryptographic* hash, like the FNV baseline it replaces. If the harness runs adversarial/sparse-key collision tests rather than random-key tests, this is where it would fail, and the fix inside the same mechanism (not a different approach) is to make more lanes additive or insert one `vpshufb` cross-byte permute per block.

## VERDICT

The mechanism translates cleanly and, on structural grounds, should be several times to tens of times faster than byte-at-a-time FNV-1a while meeting the avalanche requirement: removing the per-byte multiply removes the serial dependency that caps FNV at ~0.25 B/cycle, and paying for avalanche once in a fixed-cost finalizer is strictly cheaper than paying for it on every byte. The story's other claim — "if the mixing seems weak, run the bag through two or three times" — is the assumption this replaces: extra passes over the data cost O(n) each and buy nothing that one O(1) finalizer does not. The claim I *cannot* stand behind is the number: PREDICTION 18 is unverified because no benchmark tool was reachable in this session, and it should be treated as an estimate until `hash_bench` is actually run against this artifact.