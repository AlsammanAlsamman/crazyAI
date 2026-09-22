# APPROACH

The three "taffy" solutions map onto three concrete tricks for a byte-buffer hash:

- **Solution 1** ("one strong twist, not many weak ones") → don't add more rounds of the cheap FNV mix; use one *stronger* mixing operation per step (multiply + rotate + multiply, or a multiply-xor-shift avalanche finisher).
- **Solution 2** ("split into small balls, work them independently, braid at the end") → the real fix for FNV-1a's actual bottleneck. FNV-1a's `h = (h^byte) * prime` is a **serial dependency chain**: each multiply must wait for the previous one to finish (multiply latency ~3–5 cycles), so throughput is latency-bound at ~1 byte per few cycles regardless of how "fast" the ALU is. If instead the buffer is split into several independent accumulator lanes that each consume disjoint words of the input, the CPU can execute those multiplies **out of order / in parallel** (ILP), and only at the very end are the lanes "braided" together. This is the dominant real speedup.
- **Solution 3** ("vary the motion instead of repeating it") → use different constants/operations for the head loop (wide lanes), the 8-byte tail, the 4-byte tail, and the byte tail, rather than reusing one op everywhere — avoids periodic/self-cancelling patterns and keeps small inputs well mixed too.

Combining these gives exactly the design behind xxHash64: 4 independent 64-bit accumulator lanes consuming 32 bytes/iteration (breaks the serial multiply chain → big ILP win over FNV-1a's 1-byte serial loop), a lane-merge step, tail handling for the remainder, and a final multiply-xor-shift avalanche finisher (Solution 1) so even tiny inputs get full bit diffusion. I'm not adding OpenMP threading: for the buffer sizes a hash kernel is normally called on, thread spin-up cost would dwarf the work, and the ILP win from 4 scalar lanes already saturates the multiply units without synchronization overhead — this is why real fast hashes (xxHash, wyhash, CityHash) use lane-level parallelism, not thread-level parallelism, inside one buffer.

# ARTIFACT

```c
#include <stdint.h>
#include <string.h>

#define P64_1 11400714785074694791ULL
#define P64_2 14029467366897019727ULL
#define P64_3 1609587929392839161ULL
#define P64_4 9650029242287828579ULL
#define P64_5 2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t get64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t get32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* one "strong twist": multiply, rotate, multiply */
static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P64_2;
    acc = rotl64(acc, 31);
    acc *= P64_1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P64_1 + P64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        /* four independent lanes: no data dependency between them,
           so their multiplies overlap in the CPU's pipeline */
        uint64_t v1 = P64_1 + P64_2;
        uint64_t v2 = P64_2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - P64_1;

        const unsigned char *limit = end - 32;
        do {
            v1 = round64(v1, get64(p));      p += 8;
            v2 = round64(v2, get64(p));      p += 8;
            v3 = round64(v3, get64(p));      p += 8;
            v4 = round64(v4, get64(p));      p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = P64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, get64(p));
        h ^= k1;
        h = rotl64(h, 27) * P64_1 + P64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)get32(p) * P64_1;
        h = rotl64(h, 23) * P64_2 + P64_3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P64_5;
        h = rotl64(h, 11) * P64_1;
        p++;
    }

    /* final avalanche: the one deliberate strong twist */
    h ^= h >> 33;
    h *= P64_2;
    h ^= h >> 29;
    h *= P64_3;
    h ^= h >> 32;

    return h;
}
```

PREDICTION: 3.5

# MEASUREMENT

No tools (including `hash_bench`) were available in this session, so I could not run the benchmark myself — the pipeline downstream will compile with `gcc -O3 -march=native -fopenmp -lm` and measure it. Reasoning in lieu of a run: FNV-1a's loop is latency-bound on a single multiply-dependency chain (~1 byte per few cycles regardless of buffer size). This kernel processes 32 bytes/iteration across 4 mutually-independent accumulators, so on any out-of-order core the four `round64` calls per iteration can issue and execute concurrently — expected to approach 4× the bytes/cycle of FNV-1a on medium/large buffers (order of magnitude ~1 byte/cycle vs ~0.2–0.3 bytes/cycle for FNV-1a), tapering toward parity (or a small loss) on very short inputs (<32 bytes) where the fixed avalanche-finisher cost is proportionally larger than FNV-1a's minimal per-byte cost. Net expectation across a mixed size distribution: roughly 3–4× faster, hence the prediction of 3.5.

# VERDICT

Unverified in this session (no benchmarking tool access) — correctness and avalanche behavior rest on this being a faithful reimplementation of the well-studied xxHash64 construction (4 independent lanes + merge + tail handling + multiply-xor-shift finisher), which is documented to pass avalanche tests and to substantially outperform FNV-1a on non-trivial buffer sizes due to breaking the serial multiply chain. Recommend the pipeline run `hash_bench`/`hash_contract` against this artifact to confirm the predicted ~3.5× speedup and check avalanche/collision behavior before treating this as final; if measurement contradicts the prediction (e.g., dominated by very short inputs), the next iteration would shrink or special-case the `len<32` avalanche cost rather than changing the lane structure.