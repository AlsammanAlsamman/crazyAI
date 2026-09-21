# APPROACH

The reference FNV-1a has a fundamental performance flaw: it forms a single serial dependency chain — each byte's `xor` then `multiply` must complete before the next byte can be processed, and it consumes only **1 byte per multiply**. Multiplies have ~3-5 cycle latency, so FNV-1a is latency-bound at roughly one byte every few cycles regardless of how wide the machine is.

To go faster while keeping (and actually improving) avalanche quality, I challenge the "one accumulator, one byte at a time" assumptions from the prompt:

- **Read 8 bytes per step, not 1** — a single 64-bit load absorbs 8x more input per mixing operation.
- **Use 4 independent accumulator lanes** processed in the bulk loop. Four independent chains let the CPU's multiple multiply/ALU ports overlap the multiply latency (instruction-level parallelism), instead of stalling on one chain — this is exactly the trick xxHash64 uses and is well-proven for both speed and avalanche.
- **Merge the 4 lanes with rotations + multiplies** at the end so entropy from all lanes spreads into every output bit (better avalanche than a single accumulator, since a single-bit input flip now perturbs one lane fully and that lane's rotate/merge step fans it out across the full 64-bit state).
- **Strong final avalanche mixer** (xor-shift/multiply, MurmurHash3-style finalizer) guarantees good bit diffusion even for tiny inputs where the bulk loop never runs.
- Tail bytes (<32, then <8, <4) are handled with progressively smaller reads so no byte is ever hashed one-at-a-time unless truly necessary (last ≤7 bytes).

I skip OpenMP: spawning threads for what's likely a single buffer hash (unknown, possibly modest size) would add fixed overhead per call that a single-threaded 4-lane pipeline of this quality would rarely repay unless buffers are huge (many MB+), and the benchmark harness presumably calls `kernel` many times over varying/smaller sizes.

# ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t rd64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t rd32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

static inline uint64_t mergeround(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = P1 + P2;
        uint64_t v2 = P2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - P1;
        const unsigned char *limit = end - 32;

        do {
            v1 = round64(v1, rd64(p));      p += 8;
            v2 = round64(v2, rd64(p));      p += 8;
            v3 = round64(v3, rd64(p));      p += 8;
            v4 = round64(v4, rd64(p));      p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = mergeround(h, v1);
        h = mergeround(h, v2);
        h = mergeround(h, v3);
        h = mergeround(h, v4);
    } else {
        h = P5;
    }

    h += (uint64_t)len;

    while ((size_t)(end - p) >= 8) {
        h ^= round64(0, rd64(p));
        h = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if ((size_t)(end - p) >= 4) {
        h ^= (uint64_t)rd32(p) * P1;
        h = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P5;
        h = rotl64(h, 11) * P1;
        p++;
    }

    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;

    return h;
}
```

# PREDICTION: 3.5

(predicted speedup factor of this kernel's throughput vs. the byte-at-a-time FNV-1a baseline, largest for buffers of a few hundred bytes and up, where the 4-lane/8-bytes-per-step loop dominates runtime)

# MEASUREMENT

No `hash_bench` / `hash_contract` tool was available in this session (explicitly disabled per the task instructions), so I could not execute the harness myself. I have not fabricated numbers — the prediction above is the only figure I can honestly provide; actual measurement must come from the pipeline that compiles and benchmarks this artifact.

Reasoning behind the predicted magnitude: FNV-1a does 1 mixing multiply per byte with a fully serial dependency chain (latency-bound, ~1 byte per ~3-4 cycles). This kernel does 1 multiply-heavy round per 8 bytes across 4 independent lanes in the bulk path, so the serial critical path per byte drops roughly 8x in load granularity and the 4-way independence lets the CPU overlap multiply latencies instead of stalling — realistically yielding somewhere in the 3-5x throughput range for medium/large buffers, tapering toward parity (or a small loss from extra finalizer overhead) on very short inputs (<16 bytes) where the bulk loop never engages.

# VERDICT

Correctness: the kernel is a faithful implementation of the well-known xxHash64 algorithm (bulk 4-lane 32-byte rounds + tail handling down to per-byte + strong xor-shift/multiply finalizer), which is extensively validated in the wild for both determinism and avalanche behavior — flipping any single input bit propagates through a `round64`/`mergeround` step and the final avalanche mix, giving close to 50% output-bit-flip probability, matching or exceeding FNV-1a's avalanche quality.

Speed: strictly dominates FNV-1a on non-trivial buffers on any measure I can reason about (fewer, wider, more ILP-friendly operations per byte); the contract signature is matched exactly (`uint64_t kernel(const unsigned char *data, size_t len)`), no OpenMP/threading overhead was introduced since it wasn't clearly justified without knowing benchmark buffer sizes, and unaligned reads go through `memcpy` so there is no undefined-behavior risk from strict-aliasing or misaligned casts. I was not able to run `hash_bench` in this session to confirm the numeric prediction — that verification is deferred to the pipeline.