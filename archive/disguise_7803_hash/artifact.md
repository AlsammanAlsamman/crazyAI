# APPROACH

The three disguised solutions map onto three real hashing strategies:

- **Solution 1** = the literal FNV-1a-style baseline: one accumulator, one multiply per byte, strictly serial. Correct, but the multiply forms a true data-dependency chain (each byte's multiply must finish before the next starts), so throughput is bounded by multiply *latency*, not multiply *throughput* — the CPU's pipelines sit idle.
- **Solution 2** = drop the multiply for a fixed rotate + XOR-overlay. Cheaper per byte, but a pure rotate/XOR mixer without multiplication has measurably worse avalanche behavior (linear diffusion only) — risky for the "flip one bit → flip ~half the output" requirement.
- **Solution 3** = stop insisting on *one* accumulator / *one* strict chain. Split the work into several independent lanes that can be updated in parallel (no cross-lane dependency), then merge the lanes pairwise/tree-style at the end.

The fastest **correct** design combines Solution 3's structural idea (independent lanes → tree merge, which breaks the serial dependency chain and lets the CPU pipeline several multiplies at once) with Solution 1's multiply-based mixing (kept, because it's what actually gives good avalanche — Solution 2's cheaper mixer is not worth the diffusion risk). This is exactly the structural trick xxHash64 uses: 4 independent 64-bit lanes consuming 32 bytes/iteration with multiply+rotate, merged at the end with a final avalanche mix. I did not reach for OpenMP thread-level parallelism because buffer sizes in a hash benchmark are typically too small to amortize thread spawn/join cost; the lane-level ILP parallelism is safe and profitable at every size.

# ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = PRIME1 + PRIME2;
        uint64_t v2 = PRIME2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - PRIME1;

        const unsigned char *limit = end - 32;
        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p, 8);
            memcpy(&k2, p + 8, 8);
            memcpy(&k3, p + 16, 8);
            memcpy(&k4, p + 24, 8);

            v1 += k1 * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1;
            v2 += k2 * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1;
            v3 += k3 * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1;
            v4 += k4 * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1;

            p += 32;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h ^= v1; h = h * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h ^= v2; h = h * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h ^= v3; h = h * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h ^= v4; h = h * PRIME1 + PRIME4;
    } else {
        h = PRIME5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= PRIME2; k1 = rotl64(k1, 31); k1 *= PRIME1;
        h ^= k1;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h ^= (uint64_t)k1 * PRIME1;
        h = rotl64(h, 23) * PRIME2 + PRIME3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
    }

    h ^= h >> 33;
    h *= PRIME2;
    h ^= h >> 29;
    h *= PRIME3;
    h ^= h >> 32;

    return h;
}
```

Correctness notes: handles `len == 0` (falls to `h = PRIME5`, no tail bytes, straight to avalanche), uses `memcpy` for all multi-byte loads (no strict-aliasing/alignment UB), loop guards (`p + 8 <= end`, `p + 4 <= end`) are exact so no over-read past `data + len`, tail is fully consumed byte-by-byte so every input byte participates in the final state, and the final 3-round xor-mul-xor-mul-xor avalanche finisher is the standard, well-tested xxHash64 finalizer (empirically passes avalanche/SMHasher-style tests).

# PREDICTION: 6

(Predicting roughly a 6x wall-clock speedup over the byte-at-a-time FNV-1a baseline on the benchmark's mixed buffer sizes: large gains on large buffers from breaking the serial multiply-latency chain into 4 independent, pipeline-overlapping lanes processing 32 bytes/iteration, smaller but still positive gains on small buffers from the 8-byte/4-byte tail fast paths versus per-byte FNV.)

# MEASUREMENT

No `hash_bench` / `hash_contract` tool execution occurred in this session (tools were reported unavailable here) — per the task instructions, the prediction above was recorded before any measurement, and the compile/benchmark pass is expected to run externally in the pipeline against this artifact. I have not fabricated numbers here; the pipeline's actual `hash_bench` throughput comparison against the FNV-1a baseline, plus `hash_contract`'s avalanche/distribution check, should be treated as the real result, to be compared against the PREDICTION above.

# VERDICT

Structurally this is a well-established, previously-validated design (xxHash64) rather than a speculative one, so I have high confidence it is both **correct** (satisfies the exact `uint64_t kernel(const unsigned char*, size_t)` contract, reads exactly `len` bytes once, no OOB access) and **materially faster** than serial byte-wise FNV-1a, because it eliminates the single-accumulator multiply-latency bottleneck (Solution 1's core weakness) by using four independent lanes merged tree-style (Solution 3's core idea) while keeping multiplication-based mixing for avalanche quality (rejecting Solution 2's cheaper-but-weaker rotate/overlay-only mixer). If the external measurement contradicts the 6x prediction, the most likely cause is a benchmark dominated by very small buffers (<32 bytes), where the lane setup overhead reduces the advantage to something closer to 1.5–3x rather than the large-buffer 6x+ regime.