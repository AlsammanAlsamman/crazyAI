## APPROACH

The standard FNV-1a solution reads one byte at a time and folds it into a **single** accumulator with an XOR followed by a multiply. That creates a strict serial dependency chain: each multiply must wait for the previous one to finish (multiply latency ≈3–5 cycles on modern x86), so throughput is capped at roughly one byte per multiply-latency, independent of how wide the CPU's execution ports are.

I can keep "every byte mixed via a multiply" (so avalanche quality is preserved) while breaking that serial chain, by:

1. **Reading 8 bytes per load** instead of 1 byte per load — this is a legal generalization, not a shortcut, because I still fold every byte's bits into the state, just 8 at a time via a 64-bit word.
2. **Using 4 independent accumulator lanes** (xxHash64-style) for the bulk of the buffer, so 4 multiply/rotate chains can be in flight simultaneously — this directly exploits instruction-level parallelism that a single accumulator forecloses.
3. **Combining the 4 lanes** and running the combined value through a couple more full-strength mix rounds, plus per-lane merges, so the final state depends on *all* input bits, not just one lane.
4. **Finalizing with a strong 3-round xor/mul avalanche mixer (`fmix64`)**, so even if the bulk mixing left any structure, the final output achieves full bit avalanche regardless of accumulator width or count. This directly attacks the "more rounds always better" assumption — I only spend extra multiplies where they're cheap (the O(1) finalizer), not per-byte.
5. Tail bytes (<32, <8, <4) are handled with progressively smaller loads down to per-byte, so correctness holds for any `len`, including 0.

Net effect: same total number of "a byte gets multiplied into state" work as FNV-1a in the worst case, but organized so the CPU can pipeline 4 independent multiply chains and consume 32 bytes/iteration instead of 1 byte/iteration for the loads.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    const uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    const uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    const uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    size_t n = len;
    const unsigned char *p = data;
    uint64_t h;

    if (n >= 32) {
        uint64_t v1 = PRIME1 + PRIME2;
        uint64_t v2 = PRIME2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - PRIME1;

        while (n >= 32) {
            uint64_t x1, x2, x3, x4;
            memcpy(&x1, p,      8);
            memcpy(&x2, p + 8,  8);
            memcpy(&x3, p + 16, 8);
            memcpy(&x4, p + 24, 8);

            v1 += x1 * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1;
            v2 += x2 * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1;
            v3 += x3 * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1;
            v4 += x4 * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1;

            p += 32;
            n -= 32;
        }

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h ^= v1; h = h * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h ^= v2; h = h * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h ^= v3; h = h * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h ^= v4; h = h * PRIME1 + PRIME4;
    } else {
        h = PRIME5;
    }

    h += (uint64_t)len;

    while (n >= 8) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= PRIME2; k1 = rotl64(k1, 31); k1 *= PRIME1;
        h ^= k1;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
        n -= 8;
    }

    if (n >= 4) {
        uint32_t k;
        memcpy(&k, p, 4);
        h ^= (uint64_t)k * PRIME1;
        h = rotl64(h, 23) * PRIME2 + PRIME3;
        p += 4;
        n -= 4;
    }

    while (n > 0) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
        n--;
    }

    h = fmix64(h);
    return h;
}
```

## PREDICTION: 6

## MEASUREMENT

No tools were available in this session (`hash_bench`/`hash_contract` could not be invoked here), so I could not run the actual benchmark or the iterate-up-to-four-times loop. This prediction of ~6x is reasoned, not measured: for large buffers, FNV-1a's single-accumulator serial multiply chain is latency-bound (~3–5 cycles/byte), while the 4-lane 32-byte/iteration design here lets independent multiply chains overlap and reads 32 bytes per iteration instead of 1, which should approach the CPU's multiply throughput rather than its latency, and 32B/iter vs 1B/iter also cuts loop/branch overhead per byte by ~32x. For very small inputs (<8 bytes) the two kernels should be close in cost (both boil down to a few multiplies), so the average across a mixed size distribution should still favor this kernel but by less than on a large-buffer-only benchmark.

## VERDICT

Unverified: correct by construction (matches the well-established xxHash64 algorithm, contract signature matches exactly, handles `len == 0` and all tail-length cases), and the avalanche property is guaranteed by the final `fmix64` full-avalanche mixer regardless of any residual structure in the bulk-mixing stage — but no empirical timing or hash-quality (bit-avalanche) test was actually executed in this session, since no tools were available to call `hash_bench`/`hash_contract`. The 6x prediction should be treated as an estimate pending real measurement.