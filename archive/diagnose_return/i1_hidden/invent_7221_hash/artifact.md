## MAPPING

**Seed 1 — "one crease per mark, angle from the mark *and* the previous crease"**

| World object | Computational object |
|---|---|
| sheet (pink/orange, only size matters) | the input buffer; content-type/endianness irrelevant, only `len` matters |
| pile of marks, read in order, never skipping | `data[0..len-1]`, consumed strictly in address order |
| one fold per mark | one state update per byte |
| fold angle = mark + previous crease | `h = (h ^ data[i]) * K` — a serial dependency chain |
| paper carries forward everything told | single running 64-bit state |

*Verdict:* maximally literal for a byte-at-a-time FNV/`dp`-style hash. **Upholds** the target assumption exactly — one serial chain, each byte folded before the next is read. This is the baseline, not an idea.

**Seed 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"**

| World object | Computational object |
|---|---|
| four wire birds | four independent 64-bit accumulator lanes `v0..v3` |
| a different bird's beak per fold | four distinct odd constants `B0..B3` / round-robin word→lane assignment |
| pressed against all four *at once* | four lane updates issued in the same iteration → 4-way ILP, four independent dependency chains |
| "the crease counts as set only when all four agree" | no lane is authoritative; the digest exists only after a merge that requires all four |
| refold tighter | 64-bit×64-bit → 128-bit product, folded `hi ^ lo` (a "tighter" fold than a truncated multiply) |
| spiral + snail-shell markings lined up *wrong on purpose*, scrambled | prime rotations (17, 43) at the merge, so lane-symmetric or shifted inputs cannot cancel |
| "a clean line-up would mean two piles look the same" | the explicit anti-collision requirement |

**Seed 3 — "carry the whole wad to the water's edge, thin it to one coin, throw away every scrap"**

| World object | Computational object |
|---|---|
| water's edge | the finalizer, run exactly once after all input is consumed |
| thinning "the way a body thins going under with one small suitcase" | avalanche compression: `xor-shift / multiply / xor-shift` — width preserved, entropy compacted |
| the one small suitcase carried under | `len`, mixed in at the very end (and at lane init) |
| one hard dense corner no bigger than a coin | the returned `uint64_t` |
| every sheet, scrap, and failed bird-reading thrown into the mud | **no scratch memory, no lookup table, no retained state** — register-only, zero footprint |
| "if one mark differed the whole spiral folds to a different corner" | the avalanche criterion itself |

## CHOSEN SEED

**Seed 2** (four wire birds), with Seed 3's water's-edge thinning as its mandatory tail — the contract returns one 64-bit value, so the "thin to a coin" step is not optional and not a separate idea.

Seed 2 is chosen over Seed 3 because its mapping is the more *literal*: "four birds" is a countable, concrete object that becomes a countable, concrete thing in the machine (four accumulators), whereas Seed 3's mapping is a single generic finalizer. Seed 1 is rejected because it is faithful to the wrong thing — it reproduces the assumption we were asked to break.

## ASSUMPTION BROKEN

> *"each byte must be mixed into the running state before the next byte is read"*

The native never has one running state. There are **four** creases in flight, and a fold is pressed against a different bird while the others are still being read — no bird waits for another. Literally: 64 bytes are loaded per iteration into **four independent dependency chains**, and bytes 32–63 are read and mixed while bytes 0–31 are still in the multiplier pipeline. There is no point at which "the state" exists mid-stream; agreement is deferred to the merge. This converts the latency-bound serial chain (~4–5 cycles/byte) into a throughput-bound one (~1 multiply latency per 64 bytes), which is the entire speed win.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the four wire birds: distinct odd "beaks" ---- */
#define B0 0xa0761d6478bd642fULL
#define B1 0xe7037ed1a0b428dbULL
#define B2 0x8ebc6af09c88c6e3ULL
#define B3 0x589965cc75374cc3ULL
#define B4 0x1d8e4e27c47d124fULL
#define B5 0xeb44accab455d165ULL

/* one fold: two things pressed together into one crease.
   full 128-bit product, folded hi^lo -- the "tighter" refold. */
static inline uint64_t fold2(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t lo = a * b;
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    uint64_t hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return lo ^ hi;
#endif
}

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* unaligned-safe, strict-aliasing-safe reads (gcc -O3 folds these to MOVs) */
static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }
static inline uint64_t rd3(const unsigned char *p, size_t k) {
    return ((uint64_t)p[0] << 16) | ((uint64_t)p[k >> 1] << 8) | (uint64_t)p[k - 1];
}

/* the water's edge: thin the wad to one dense corner, with len as the suitcase */
static inline uint64_t thin(uint64_t h, uint64_t len) {
    h = fold2(h ^ B0, len ^ B1);
    h ^= h >> 32; h *= B3;
    h ^= h >> 29; h *= B4;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;
    uint64_t h;

    if (n >= 64) {
        /* four birds, each holding its own crease; len enters all four up front */
        uint64_t v0 = B0 ^ (uint64_t)len;
        uint64_t v1 = B1 ^ (uint64_t)len;
        uint64_t v2 = B2 ^ (uint64_t)len;
        uint64_t v3 = B3 ^ (uint64_t)len;
        do {
            __builtin_prefetch(p + 384, 0, 0);
            uint64_t w0 = rd8(p),      w1 = rd8(p + 8);
            uint64_t w2 = rd8(p + 16), w3 = rd8(p + 24);
            uint64_t w4 = rd8(p + 32), w5 = rd8(p + 40);
            uint64_t w6 = rd8(p + 48), w7 = rd8(p + 56);
            /* each crease = this mark folded against the crease before it;
               four beaks pressed in the same breath -> four independent chains */
            v0 = fold2(v0 ^ w0, B4 ^ w1) ^ w1;
            v1 = fold2(v1 ^ w2, B5 ^ w3) ^ w3;
            v2 = fold2(v2 ^ w4, B0 ^ w5) ^ w5;
            v3 = fold2(v3 ^ w6, B1 ^ w7) ^ w7;
            p += 64; n -= 64;
        } while (n >= 64);
        /* only when all four beaks agree does the crease count as set;
           prime rotations keep the spirals deliberately misaligned */
        h = fold2(rotl64(v0, 17) ^ v1 ^ B2, rotl64(v2, 43) ^ v3 ^ B3)
            ^ (v0 + v1 + v2 + v3);
    } else {
        /* GUARD: for short pulls the four birds are never set up at all */
        h = B0 ^ (uint64_t)len;
    }

    while (n >= 16) {
        uint64_t a = rd8(p), b = rd8(p + 8);
        h = fold2(h ^ a, B2 ^ b) ^ b;
        p += 16; n -= 16;
    }
    if (n) {
        uint64_t a, b;
        if (n >= 8)      { a = rd8(p);       b = rd8(p + n - 8); }
        else if (n >= 4) { a = rd4(p);       b = rd4(p + n - 4); }
        else             { a = rd3(p, n);    b = (uint64_t)n;    }
        h = fold2(h ^ a ^ B4, b ^ B5) ^ b;
    }

    return thin(h, (uint64_t)len);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 25**

(stated before any measurement; no measurement had been taken at the time of writing, and as it turns out none could be)

Reasoning: the `dp`-style baseline is a Seed-1 serial chain — one multiply latency (3–5 cycles) plus a load per byte, ≈4 cycles/byte ≈ 0.75 GB/s at 3 GHz. This kernel's critical path per 64-byte iteration is one `mulx` (≈4 cycles) plus an XOR, with four such chains overlapped, so the loop is issue- and load-bound at roughly 6–8 cycles per 64 bytes ≈ 0.11 cycles/byte in L1/L2. That is ~35× on paper; I discount to **25** because for any buffer larger than L2 the kernel becomes memory-bandwidth-bound (~15–20 GB/s single-thread) while the baseline never is, and because the finalizer's ~15 cycles are amortized over whatever the harness's median length is.

Avalanche prediction: **≈0.50** (essentially ideal). Every input word enters a full 128-bit product folded `hi^lo`, which diffuses across all 64 bits of its lane in one step; the merge crosses all four lanes; the tail is a 3-stage xor-shift-multiply finalizer with a known-good avalanche profile.

## MEASUREMENT

**Not performed. Reporting this plainly rather than inventing numbers.**

This session was invoked with no tools available — `hash_bench`, `hash_contract`, and the `symbolic_*`/`unconventional_*` tools were all listed as available in the task statement but none are actually callable here (the only MCP servers present are Gmail, Drive, Slack and Docs, and two of those are additionally unauthenticated). I therefore have:

- **throughput: unmeasured** — the 25× figure above is a static cycle-count argument, not an observation.
- **avalanche: unmeasured** — the ≈0.50 figure is a structural argument about the 128-bit fold and finalizer, not an observation.
- **contract conformance: unverified against `hash_contract`** — the signature is transcribed literally from the statement (`uint64_t kernel(const unsigned char *data, size_t len)`), argument order unchanged, but I could not confirm it against the tool.

The pipeline will compile and measure this. If the measured speedup lands below ~10×, the most likely cause is that the harness hashes short buffers (< 64 B), where the four birds never fly and the finalizer dominates — in that regime this kernel should be roughly at parity with, or modestly ahead of, the baseline, and my 25× would simply be wrong for the workload that was actually run. I am naming that in advance rather than after the fact.

## VERDICT

The native's idea survives translation intact and is not a disguised restatement of the textbook method. The textbook byte-at-a-time hash is Seed **1**, and I did not ship it. What is shipped is exactly Seed 2: four gauges held at once, no authoritative mid-stream state, agreement deferred to a merge — and that single structural choice (four dependency chains instead of one) is where all of the predicted speed comes from, not from any generic optimization bolted on afterward.

**Risks I name, and how each is addressed (per rule 4 — nothing risky ships unguarded):**

1. **Four-lane setup is pure overhead for small inputs.** *Guarded:* `if (n >= 64)` — the lanes are never initialized, never merged, and never rotated for anything under 64 bytes. Short inputs take a single-fold path with one 128-bit multiply and the finalizer. The 16-byte and 1–15-byte tails use last-N-bytes overlapping reads, so there is no per-byte loop anywhere in the kernel.
2. **Streaming from RAM starves the loop.** *Guarded:* one `__builtin_prefetch(p + 384, 0, 0)` per 64-byte iteration (6 cache lines ahead), free in the shadow of the multiplies.
3. **Unaligned or aliased buffers.** *Guarded:* every read goes through `memcpy`, which `-O3` lowers to a plain unaligned `mov`. No `uint64_t*` cast, no alignment precondition on the caller.
4. **`__uint128_t` absent.** *Guarded:* `#if defined(__SIZEOF_INT128__)` with a correct 32×32 portable fallback. (On the stated toolchain the fast path is always taken.)
5. **OpenMP: considered and deliberately dropped, not shipped.** Threading a hash makes the digest depend on the thread count unless the partition is fixed, and thread-team startup (~µs) exceeds the entire hashing cost for anything under ~1 MB. Since I cannot measure the harness's buffer sizes here, shipping a threaded path would be shipping a mechanism whose own risk I cannot check. The native carries one wad to the water himself; the four birds are his only parallelism, and they are ILP, not threads. So: single-threaded, no `#pragma omp` anywhere.

**Honest limitation, not a speed risk:** this is an unkeyed, non-cryptographic hash. `fold2` is a `mum`-style folded multiply and has the known degenerate case `fold2(0, x) = 0`; I mitigate the sticky-zero fixed point by re-injecting the message word (`^ w1`) after every fold, so a lane can never lose the input that entered it, and by seeding all four lanes with `len`. That makes accidental collapse ~2⁻⁶⁴, but it is not a defense against an adversary who chooses inputs. Do not use it for hash tables exposed to untrusted keys.