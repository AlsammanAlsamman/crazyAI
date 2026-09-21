## APPROACH

The reference FNV‑1a violates the contract's *performance* only by architecture, not by correctness: it forces every one of `len` bytes through a strictly serial `xor → multiply` chain into a single accumulator. On any modern out‑of‑order core a 64‑bit multiply has ~3 cycle latency but can *issue* every cycle — FNV‑1a pays the full latency per byte because each multiply must wait for the previous one to finish. It also mixes 1 byte per multiply, so it needs 8× more multiplies than a word‑wide scheme just to cover the same input.

I attack exactly the assumptions listed as "silent":
- **Single accumulator, byte-at-a-time** → use **4 independent 64‑bit lanes**, each fed 8 bytes per step. Independent lanes let the compiler/CPU overlap the multiply latencies (ILP) instead of stalling on a single dependency chain.
- **Read the buffer as individual bytes** → read as `uint64_t` words via `memcpy` (safe against strict-aliasing/alignment UB, compiles to a single unaligned load on x86-64 with `-O3 -march=native`).
- **More rounds = better mixing** → drop per-byte rounds entirely for the bulk of the data; do exactly one multiply-rotate-multiply per 8-byte lane per iteration, and put all the avalanche work into one strong 3-step finalizer (mul/xor-shift) at the end, which is provably sufficient to spread influence of every input bit (this is the standard xxHash64 finalizer, extensively tested by SMHasher for avalanche).
- Tail (<32 bytes, then <8, then <4 bytes) is folded with the same primes so short inputs stay well mixed without paying the 4-lane setup cost.

This is essentially xxHash64 (public-domain algorithm, implemented here directly, no library dependency) — chosen because its whole design point *is* "avalanche quality at multiply-throughput-bound speed," which is precisely what the target problem asks for. I did not use OpenMP: hashing is a short, latency-sensitive, strictly-ordered reduction over one buffer; thread launch overhead would dominate for the small-to-medium sizes any hash benchmark cares about, and a parallel-reduction hash would need a different (non-length-prefix-safe) combining rule to stay correct, which risks violating "read once, in order, produce one 64-bit value" semantics. ILP within one thread is the sound win here.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64le(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = PRIME64_1 + PRIME64_2;
        uint64_t v2 = PRIME64_2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - PRIME64_1;

        do {
            v1 += read64le(p) * PRIME64_2; v1 = rotl64(v1, 31); v1 *= PRIME64_1; p += 8;
            v2 += read64le(p) * PRIME64_2; v2 = rotl64(v2, 31); v2 *= PRIME64_1; p += 8;
            v3 += read64le(p) * PRIME64_2; v3 = rotl64(v3, 31); v3 *= PRIME64_1; p += 8;
            v4 += read64le(p) * PRIME64_2; v4 = rotl64(v4, 31); v4 *= PRIME64_1; p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME64_2; v1 = rotl64(v1, 31); v1 *= PRIME64_1; h64 ^= v1; h64 = h64 * PRIME64_1 + PRIME64_4;
        v2 *= PRIME64_2; v2 = rotl64(v2, 31); v2 *= PRIME64_1; h64 ^= v2; h64 = h64 * PRIME64_1 + PRIME64_4;
        v3 *= PRIME64_2; v3 = rotl64(v3, 31); v3 *= PRIME64_1; h64 ^= v3; h64 = h64 * PRIME64_1 + PRIME64_4;
        v4 *= PRIME64_2; v4 = rotl64(v4, 31); v4 *= PRIME64_1; h64 ^= v4; h64 = h64 * PRIME64_1 + PRIME64_4;
    } else {
        h64 = PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = read64le(p) * PRIME64_2;
        k1 = rotl64(k1, 31);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= end) {
        uint32_t v32;
        memcpy(&v32, p, 4);
        h64 ^= (uint64_t)v32 * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}
```

PREDICTION: 6

## MEASUREMENT

No benchmarking tools were available in this session (the `hash_bench`/`hash_contract` calls listed in the tool set did not actually execute here — I was told explicitly to reason and predict, and that the pipeline would run the measurement afterward). So this section reports *estimated* numbers from known public characteristics of the two algorithms rather than a tool trace:

- FNV‑1a, byte-serial multiply chain: widely measured around 0.5–1 GB/s single-thread on a ~3 GHz x86 core (≈3–5 cycles/byte, latency-bound).
- xxHash64 (this design), 4-lane ILP + 8-byte-wide reads: widely measured around 10–14 GB/s single-thread on comparable hardware (≈0.2–0.3 cycles/byte on the bulk path).
- That ratio suggests roughly a 10–20× speedup on large buffers, but `hash_bench` almost certainly also exercises small inputs (tens of bytes) where the 32-byte-lane setup buys nothing and the code falls into the tail path only — there the two are much closer, sometimes closer to parity or a modest 2–3× win. Blending a realistic size mix, I predicted a conservative aggregate **6×**.

I cannot report an actual measured number here — only the reasoning above — because this session had no working tool call for it. The `PREDICTION: 6` line stands as the number to check against the real pipeline run.

## VERDICT

Design intent achieved on paper: correctness preserved (contract signature untouched, deterministic, reads the buffer once in order, no UB via memcpy for unaligned/unsized loads), and every one of the five "silent assumptions" is explicitly broken (multi-lane accumulator instead of one; 8-byte reads instead of byte-at-a-time; one strong finalizer instead of "more rounds = better"). Avalanche quality should be at least as good as FNV‑1a — xxHash64's finalizer is a published, SMHasher-validated mixer, whereas FNV‑1a is known to have some avalanche weaknesses on the low bits of short keys.

Honest caveat: I could not actually run `hash_bench`/`hash_contract` in this session to confirm either the correctness/avalanche score or the speed claim — the prediction above is inference from known algorithmic properties, not a measured result. If the real pipeline run contradicts the 6× estimate, that discrepancy — not this writeup — is the ground truth.