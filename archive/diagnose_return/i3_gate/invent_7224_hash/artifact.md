# MAPPING

## SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."

| World object | Problem object |
|---|---|
| pile of marks, in their given order | `data[0..len)`, the byte buffer |
| one mark | one input byte |
| walking them past the servant | traversing the buffer |
| the inflated house on a stick swatted against the mouth | the enforced *silence* of the serial accumulator: nothing is "spoken" (no read‑modify‑write of one shared value) per mark |
| laying each mark on the flat stone at the riverbank | position‑addressed access: `data + i` — the index *is* the order |
| "the order never gets spoken aloud and lost" | order information is carried by **position** (lane index, block index, chunk index, length), not by the *sequence of accumulator updates* |

**Assumption broken:** *"the whole buffer must be read once, start to end, in order"* — and, as a direct consequence, *"each byte must be mixed into the running state before the next byte is read."* Because the order is already pinned by the stone (position), the marks may be picked up in any order, in parallel, by 16 lanes and by several runners at once.

## SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."

| World object | Problem object |
|---|---|
| heap of even count | a fixed 128‑byte block (16 × `uint64`) |
| musk deer's fixed strides | fixed‑stride block iteration, one cache line pair per step |
| uphill twist | `ROTR(a,8)` |
| downhill fold | `+ b`, `ROTL(b,3) ^ a` |
| doubling‑back that eats its own trail | the second Speck half‑round with the roles of *a* and *b* swapped |
| "no heap keeps the shape it entered with" | the round is a bijection that leaves no lane unchanged |
| "the final stone … carried forward into the next heap's racing" | the chaining value; state persists across blocks |
| "never let it ossify … a pillar that brings the roof down" | no fixed point: additive constants + distinct nonzero lane IVs, so an all‑zero buffer is not an absorbing state |

**Assumption broken:** *"mixing one byte requires a multiplication"* (the absorb path is pure add‑rotate‑xor; zero multiplies per byte) and *"the state is a single accumulator"* (16 lanes).

## SEED 3 — "The owl calls the final folded shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned."

| World object | Problem object |
|---|---|
| searching the result in squares | the 16 lanes viewed as a **4×4 matrix** |
| turning ninety degrees | column round → diagonal round alternation |
| folding corners into its own center | the diagonal quarter‑round `G(v0,v5,v10,v15)` etc. |
| "until one changed mark smears across every square" | full diffusion = the strict‑avalanche target |
| the owl taking the pen | the finalizer (`fmix64`), the *only* place multiplication is spent |
| "the dreamer inside the dreamer" | nested mixing: `owl(fold(permute(state)))` |
| one fixed size regardless of how many marks | 64‑bit output, length folded in |
| heaps burned at the shrine, no walking back | one‑way compression: 1024 bits → 64, intermediates never stored |

**Assumption broken:** *"more mixing rounds always means better mixing"* — heavy mixing is spent **once**, at O(1) cost, not per byte.

---

# CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks the preferred assumption ("the whole buffer must be read once, start to end, in order"), and its mapping is the most literal: the swat that enforces silence is *exactly* the removal of the per‑byte serial dependency that FNV‑1a lives on, and the flat stone is *exactly* random‑access indexing. Seeds 2 and 3 are the machinery this seed makes legal, so I kept them intact as the race and the shrine.

# ASSUMPTION BROKEN

**"The whole buffer must be read once, start to end, in order."**
FNV‑1a's hash value is defined by a chain: `h ← f(h, byte)` repeated `len` times, a dependency chain of ~4 cycles per byte (xor 1 + imul 3). The native's claim is that the ordering information lives in the *stone* (position), not in the *chain*. So I define the hash as a function of **(byte value, byte position)** only: lane index `i mod 16`, block index, chunk index and total length all enter explicitly. Once that is true, the buffer may be read 128 bytes at a time by 16 independent lanes, and (for very large piles) by several runners at once, with the result unchanged.

**Where the mechanism lands on validated ground (step 4).** Taken literally, the native's own objects *are* three published, validated constructions, so I let them arrive there rather than inventing anything:

* "heaps raced with uphill twist / downhill fold / doubling‑back" → the **Speck128 ARX round** (α=8, β=3), lane‑parallel — the xxHash3/Blake3 *parallel‑lane absorb* pattern.
* "searching the result in squares, turning ninety degrees, folding corners into the center" → the **BLAKE2b permutation**: a 4×4 matrix of 64‑bit words, column round then diagonal round, rotations 32/24/16/63 (RFC 7693). This is literally what the words describe.
* "the owl takes the pen" → **murmur3 `fmix64`**, the standard validated 64‑bit finalizer.
* "a pile too small to make a heap" → the **xxh3 short‑input path** (overlapping head/tail reads).

**Two regimes, recognised in‑world (step 5).** The native gathers marks *into heaps of even count*: a pile too small to fill a heap never reaches the mountain (`len < 32` → riverbank short path, ~25 cycles, so the fixed shrine cost can never make us slower than FNV on tiny inputs — this is the guard for the one condition where my mechanism could lose). A pile too vast for one runner is split among runners, each carrying its own stone to its own shrine, and the stones are folded at the end (`len ≥ 4 MiB` → OpenMP over deterministic 1 MiB chunks; below that, one runner). Partitioning depends only on `len`, so the hash is identical with or without threads and with or without `-fopenmp`.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define RL(x,r) (((x) << (r)) | ((x) >> (64 - (r))))
#define RR(x,r) (((x) >> (r)) | ((x) << (64 - (r))))

/* --- the flat stone at the riverbank: position-addressed, never "spoken" --- */
static inline uint64_t ld64(const unsigned char *p) { uint64_t x; memcpy(&x, p, 8); return x; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t x; memcpy(&x, p, 4); return (uint64_t)x; }

/* --- the owl takes the pen: murmur3 fmix64, the ONLY multiplications spent --- */
static inline uint64_t owl(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

/* distinct, nonzero lane stones (digits of pi) - nothing may ossify to zero */
static const uint64_t STONE_IV[16] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL, 0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
    0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL, 0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL, 0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
};

/* --- the musk deer's stride: uphill twist, downhill fold, doubling back ---
   8 Speck128-style ARX steps over 8 independent lane-pairs. No multiplies.  */
static inline void heap_round(uint64_t *v) {
    int j;
    for (j = 0; j < 4; j++) { v[j]    = RR(v[j], 8)     + v[j+4];  v[j+8]  = RR(v[j+8], 8)   + v[j+12]; }
    for (j = 0; j < 4; j++) { v[j+4]  = RL(v[j+4], 3)   ^ v[j];    v[j+12] = RL(v[j+12], 3)  ^ v[j+8];  }
    for (j = 0; j < 4; j++) { v[j+4]  = RR(v[j+4], 8)   + v[j];    v[j+12] = RR(v[j+12], 8)  + v[j+8];  }
    for (j = 0; j < 4; j++) { v[j]    = RL(v[j], 3)     ^ v[j+4];  v[j+8]  = RL(v[j+8], 3)   ^ v[j+12]; }
}

/* --- the shrine: search the result in squares. BLAKE2b's G on a 4x4 grid,
       turned ninety degrees (columns) then folded corner-to-centre (diagonals) */
#define G(a,b,c,d) do {                        \
    (a) = (a) + (b); (d) = RR((d) ^ (a), 32);  \
    (c) = (c) + (d); (b) = RR((b) ^ (c), 24);  \
    (a) = (a) + (b); (d) = RR((d) ^ (a), 16);  \
    (c) = (c) + (d); (b) = RR((b) ^ (c), 63);  \
} while (0)

#if defined(__AVX2__)
#define VRL(x,r) _mm256_or_si256(_mm256_slli_epi64((x),(r)), _mm256_srli_epi64((x),64-(r)))
#define VRR(x,r) _mm256_or_si256(_mm256_srli_epi64((x),(r)), _mm256_slli_epi64((x),64-(r)))
#endif

/* one runner: race every heap of this stretch, then carry the stone to the shrine */
static uint64_t heap_race(const unsigned char * restrict data, size_t len, uint64_t id) {
    uint64_t v[16];
    uint64_t t = id * 0x9E3779B97F4A7C15ULL;
    const unsigned char * restrict p = data;
    size_t nb = len >> 7, r = len & 127, i;
    int j, rnd;

    for (j = 0; j < 16; j++) v[j] = STONE_IV[j] + t;

#if defined(__AVX2__)
    {
        __m256i A = _mm256_loadu_si256((const __m256i *)(v + 0));
        __m256i B = _mm256_loadu_si256((const __m256i *)(v + 4));
        __m256i C = _mm256_loadu_si256((const __m256i *)(v + 8));
        __m256i D = _mm256_loadu_si256((const __m256i *)(v + 12));
        for (i = 0; i < nb; i++, p += 128) {
            A = _mm256_xor_si256(A, _mm256_loadu_si256((const __m256i *)(p +  0)));
            B = _mm256_xor_si256(B, _mm256_loadu_si256((const __m256i *)(p + 32)));
            C = _mm256_xor_si256(C, _mm256_loadu_si256((const __m256i *)(p + 64)));
            D = _mm256_xor_si256(D, _mm256_loadu_si256((const __m256i *)(p + 96)));
            A = _mm256_add_epi64(VRR(A, 8), B);
            C = _mm256_add_epi64(VRR(C, 8), D);
            B = _mm256_xor_si256(VRL(B, 3), A);
            D = _mm256_xor_si256(VRL(D, 3), C);
            B = _mm256_add_epi64(VRR(B, 8), A);
            D = _mm256_add_epi64(VRR(D, 8), C);
            A = _mm256_xor_si256(VRL(A, 3), B);
            C = _mm256_xor_si256(VRL(C, 3), D);
        }
        _mm256_storeu_si256((__m256i *)(v +  0), A);
        _mm256_storeu_si256((__m256i *)(v +  4), B);
        _mm256_storeu_si256((__m256i *)(v +  8), C);
        _mm256_storeu_si256((__m256i *)(v + 12), D);
    }
#else
    for (i = 0; i < nb; i++, p += 128) {
        for (j = 0; j < 16; j++) v[j] ^= ld64(p + 8 * j);
        heap_round(v);
    }
#endif

    if (r) {                              /* the short heap at the end of the pile */
        unsigned char tail[128];
        memset(tail, 0, sizeof tail);
        memcpy(tail, p, r);
        for (j = 0; j < 16; j++) v[j] ^= ld64(tail + 8 * j);
        heap_round(v);
    }

    for (rnd = 0; rnd < 2; rnd++) {       /* turn ninety degrees, fold the corners in */
        G(v[0], v[4], v[8],  v[12]);
        G(v[1], v[5], v[9],  v[13]);
        G(v[2], v[6], v[10], v[14]);
        G(v[3], v[7], v[11], v[15]);
        G(v[0], v[5], v[10], v[15]);
        G(v[1], v[6], v[11], v[12]);
        G(v[2], v[7], v[8],  v[13]);
        G(v[3], v[4], v[9],  v[14]);
    }

    {                                     /* burn the heaps: 1024 bits -> 64, one way */
        uint64_t x0 = (v[0] ^ v[8])  + RL(v[4]  ^ v[12], 17);
        uint64_t x1 = (v[1] ^ v[9])  + RL(v[5]  ^ v[13], 29);
        uint64_t x2 = (v[2] ^ v[10]) + RL(v[6]  ^ v[14], 41);
        uint64_t x3 = (v[3] ^ v[11]) + RL(v[7]  ^ v[15], 53);
        uint64_t y0 = x0 ^ RL(x1, 23);
        uint64_t y1 = x2 ^ RL(x3, 47);
        return y0 + RL(y1, 31) + (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    }
}

/* a pile too small to make a heap: straight to the shrine (xxh3-style short path) */
static uint64_t riverbank_short(const unsigned char *data, size_t len) {
    uint64_t a, b;
    if (len >= 8) {
        a = ld64(data);
        b = ld64(data + len - 8);
        if (len > 16) { a ^= ld64(data + 8); b ^= ld64(data + len - 16); }
    } else if (len >= 4) {
        a = ld32(data);
        b = ld32(data + len - 4);
    } else if (len > 0) {
        a = (uint64_t)data[0];
        b = ((uint64_t)data[len >> 1] << 8) | (uint64_t)data[len - 1];
    } else {
        a = 0; b = 0;
    }
    a ^= STONE_IV[0] + (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    b ^= STONE_IV[5];
    a += RL(b, 31);
    b ^= RL(a, 17);
    return owl(a ^ owl(b));
}

uint64_t kernel(const unsigned char * restrict data, size_t len) {
    if (len < 32)                        /* regime 1: no heap can be formed */
        return riverbank_short(data, len);

    if (len < ((size_t)4 << 20))         /* regime 2: one runner, SIMD lanes */
        return owl(heap_race(data, len, 0));

    {                                    /* regime 3: the pile needs many runners */
        size_t nch = len >> 20;          /* ~1 MiB of marks per runner, 4..64 runners */
        if (nch > 64) nch = 64;
        {
            size_t cs = (len / nch) & ~(size_t)127;   /* whole heaps only */
            uint64_t stones[64];
            uint64_t acc;
            size_t k;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long ii = 0; ii < (long)nch; ii++) {
                size_t off = (size_t)ii * cs;
                size_t cl  = ((size_t)ii == nch - 1) ? (len - off) : cs;
                stones[ii] = heap_race(data + off, cl, (uint64_t)ii + 1);
            }
            acc = 0x243F6A8885A308D3ULL ^ ((uint64_t)len * 0x9E3779B97F4A7C15ULL);
            for (k = 0; k < nch; k++)    /* the stones folded at the last shrine */
                acc = RL(acc, 27) ^ owl(stones[k] + (uint64_t)k * 0xC2B2AE3D27D4EB4FULL);
            return owl(acc);
        }
    }
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 15.0**

Stated before any measurement (no tools were available in this session, so *nothing* below is measured — see MEASUREMENT).

Reasoning behind the number: FNV‑1a's inner loop is a hard dependency chain of `xor`(1 cy) + `imul`(3 cy) ≈ **4 cycles/byte** ≈ 0.75 GB/s at 3 GHz, and no compiler can break it. My absorb is ~40 µops per 128 bytes with two independent chains ≈ **0.08–0.10 cycles/byte** ≈ 30 GB/s in L1/L2, falling to ~10–12 GB/s single‑threaded when the buffer exceeds L3 (memory bound), rising again on the ≥4 MiB threaded path. Fixed costs: ~25 cycles for `len < 32`, ~110 cycles of shrine+seal for `len ≥ 32`. So the per‑size ratio I expect is roughly: 8 B ≈ 1.2×, 32 B ≈ 1.3×, 128 B ≈ 2.5×, 1 KiB ≈ 15×, 64 KiB ≈ 30×, 1 MiB ≈ 30×, 64 MiB ≈ 20–35× (threaded). 15.0 is my estimate of whatever aggregate the harness reports; if it benchmarks only a single large in‑cache buffer, I expect it to come out nearer 30 and my prediction to be **too low**.

Avalanche prediction: ≈ 0.500 mean output‑bit flip probability per input‑bit flip (score ≥ 0.99 on any normalised scale), with no bias at any size, because (a) every absorb round is a bijection so a single‑bit difference can never die, and (b) the last absorbed bit still passes through two full BLAKE2b double‑rounds, a rotation‑staggered 16→1 fold, and `fmix64`.

# MEASUREMENT

**Not measured. `hash_bench` and `hash_contract` were not available in this session — no tool calls were possible.** I am reporting this plainly rather than inventing numbers: the prediction above is unvalidated, and the pipeline's measurement supersedes it.

What I *did* do, in place of measurement, was four design revisions, each driven by an analytic failure I could name (this is the "at most four improvements" budget, spent on paper):

1. **Single 64‑byte / 8‑lane state → 128‑byte / 16‑lane state.** With 8 lanes the whole state is one serial ARX chain per block (~12 cycles latency per 64 B, latency‑bound). Splitting into two independent pairs (A,B) and (C,D) doubles ILP and makes the loop throughput‑bound.
2. **Per‑lane rotation constants → uniform constants per step.** Per‑lane distinct rotates need `vprolvq` (AVX‑512) or variable shifts and would have blocked clean AVX2 codegen. Lane asymmetry now comes from the distinct IVs and the data, exactly as in ChaCha/BLAKE2, where all four columns share rotation constants.
3. **Cross‑lane shuffles inside the hot loop → cross‑lane mixing only at the shrine.** Re‑reading the native: *"I search **the result** in squares"* — the 90° turning is applied to the result, not to every heap. Removing the per‑block `vpermq` cost ~20% of the loop and is what the text actually says.
4. **Added the `len < 32` riverbank path.** My own analysis said the ~110‑cycle fixed shrine cost would make the kernel *slower than FNV‑1a below ~24 bytes*. That is a named condition where the mechanism loses, so per the rules it is guarded, not shipped: a size check with a validated xxh3‑style short path as the fallback. Likewise the thread path is gated at 4 MiB (where OpenMP fork overhead is <2% of the work) with the single‑runner path as fallback, and produces bit‑identical results whether or not `-fopenmp` is used.

# VERDICT

The native's description is not a metaphor for a novel hash; read literally it *is* three validated constructions bolted end to end, and the translation gets better the less I improvise. "The order fixed on the stone rather than spoken aloud" is the single load‑bearing idea: it says the ordering information belongs in the *addressing*, not in the *dependency chain*, and that is precisely the assumption FNV‑1a spends 4 cycles per byte to uphold. Everything else follows — 16 lanes, 128‑byte heaps, and finally several runners, all legal only because position already encodes order.

Honest statement of residual risk, and how each is handled rather than hidden:

* **Small inputs.** The 4×4 shrine is O(1) but not free (~110 cycles). Guarded: `len < 32` takes the 25‑cycle riverbank path. Between 32 and ~128 bytes I expect only ~1.2–2.5× over FNV — better, but not dramatically so, and I would not claim otherwise.
* **Threads.** Only above 4 MiB, where the units of work (1 MiB per runner) dwarf fork overhead; below that, and on a 1‑core machine, the code is identical to the serial path. If the harness never benchmarks ≥4 MiB, this branch is dead code and costs one compare.
* **Cryptographic strength.** None claimed. The lanes are independent during absorb, so crafted multi‑block collisions are reachable; single‑bit avalanche — the thing actually being graded — is not affected, and non‑cryptographic hashes (xxHash, wyhash) make the same trade.
* **The prediction may be wrong in either direction.** 15.0 is a point estimate over an unknown size mix. If the harness measures one large in‑cache buffer I expect ~30 and will have under‑predicted; if it weights sizes under 64 bytes heavily I expect ~3 and will have over‑predicted. I would rather name that spread than quote a number that cannot be falsified.

One thing I deliberately did **not** do: the native says "a calculation grown too satisfied with itself is a pillar that will bring the roof down," and the corresponding concrete failure — an ARX state with no additive asymmetry is a fixed point on an all‑zero buffer, so `hash(0^n)` would be constant across `n` — is real. It is closed by the distinct nonzero pi‑digit IVs, the additions inside Speck and G, and folding `len` in twice. That warning earned its place in the code.