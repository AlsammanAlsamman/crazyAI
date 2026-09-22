# APPROACH

The three disguised solutions map onto three real hashing strategies:

- **Solution 1** ("one jar, one extra flip") = keep FNV‑1a's single accumulator, just add one more mixing step per byte. This does **not** break the sequential dependency chain — each byte's multiply still has to wait for the previous byte's multiply to finish, so latency-bound throughput is basically unchanged from the baseline.
- **Solution 2** ("four jars, round-robin, merge once at the gate") = exactly xxHash's internal trick: four *independent* accumulator lanes that each get every 4th chunk of input, mixed with their own multiply-rotate-multiply step, combined only once at the end. Because the four lanes have no data dependency on each other, the CPU can run their multiply chains in parallel (out-of-order/ILP), turning a latency-bound loop into a throughput-bound one — same total work, much better instruction-level parallelism, and it processes 32 bytes/iteration with only 2 multiplies per 8 bytes (vs. FNV-1a's 1 multiply per **1** byte).
- **Solution 3** ("no jar while walking, pairwise tree merge with helpers") = a multi-thread/tree parallel reduction. This is real and can win for huge buffers, but it needs thread spawn/join or a chunk-parallel OpenMP region, which adds fixed overhead that will dominate for the small/medium buffer sizes a `kernel(data,len)` micro-hash is normally called with, and it complicates handling of odd/tail lengths for no benefit at those sizes.

Solution 2 is the most direct, contract-preserving translation: it stays single-threaded (no OpenMP launch overhead, correct for every `len` including 0), needs no scratch memory, and gets its speedup purely from ILP instead of thread parallelism — this is literally why xxHash64 is faster than FNV-1a in practice. So I implement xxHash64 (seed = 0) as the kernel: 4-lane 32-byte main loop, standard 8/4/1-byte tail, final avalanche mix. Avalanche quality is inherited from a hash that's been extensively tested (SMHasher-clean), and speed comes from fewer multiplies/byte plus independent lanes.

# ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t read64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t read32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3 1609587929392839161ULL
#define P4 9650029242287828579ULL
#define P5 2870177450012600261ULL

static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = rotl64(acc, 31);
    acc *= P1;
    return acc;
}
static inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;
    const uint64_t seed = 0;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + P1 + P2;
        uint64_t v2 = seed + P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - P1;

        do {
            v1 = xxh_round(v1, read64(p)); p += 8;
            v2 = xxh_round(v2, read64(p)); p += 8;
            v3 = xxh_round(v3, read64(p)); p += 8;
            v4 = xxh_round(v4, read64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh_merge_round(h64, v1);
        h64 = xxh_merge_round(h64, v2);
        h64 = xxh_merge_round(h64, v3);
        h64 = xxh_merge_round(h64, v4);
    } else {
        h64 = seed + P5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        h64 ^= xxh_round(0, read64(p));
        h64 = rotl64(h64, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)read32(p) * P1;
        h64 = rotl64(h64, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * P5;
        h64 = rotl64(h64, 11) * P1;
        p += 1;
    }

    h64 ^= h64 >> 33;
    h64 *= P2;
    h64 ^= h64 >> 29;
    h64 *= P3;
    h64 ^= h64 >> 32;

    return h64;
}
```

Correctness: reads every byte exactly once in order (main loop consumes 32-byte blocks, tail handles the 0–31 remaining bytes via 8/4/1-byte passes), no undefined behavior (unaligned loads via `memcpy`), well-defined for `len == 0` (falls straight to the `seed+P5` branch, tail loops don't execute, only the length and final avalanche affect the output). Every input byte passes through at least one multiply and is xor'd/rotated into the state before the next byte is consumed, and the final avalanche (`xor-shift, multiply` ×3) ensures single-bit flips anywhere spread across all 64 output bits — this is the standard, heavily-tested xxHash64 finalizer.

# PREDICTION: 3

# MEASUREMENT

Not run in this session — no tools (including `hash_bench`) were available to me here; per the session instructions, the pipeline will compile and benchmark this artifact against the FNV-1a baseline separately. My prediction above (≈3× throughput improvement, likely larger for big buffers where the 32-byte/4-lane ILP loop dominates, smaller or possibly near-parity for tiny buffers dominated by the tail path and setup) is recorded before any such run.

# VERDICT

Pending the pipeline's actual `hash_bench` run. Expected outcome: a clear win on medium/large buffers (fewer multiplies per byte: ~0.25/byte in the main loop vs. FNV-1a's 1/byte, plus 4 independent multiply chains giving the out-of-order core real ILP instead of a single latency-bound dependency chain), with avalanche quality at least as good as the baseline since xxHash64's mixing/finalizer is a well-established, SMHasher-vetted design. Main risk to the prediction: for very small inputs (well under 32 bytes) the extra tail-path branching could erase some or all of the advantage over the trivially simple FNV-1a loop — if `hash_bench` weights small-buffer latency heavily, the real speedup could land below 3× or, in the worst case, near 1×.