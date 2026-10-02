## MAPPING

**SEED 1 — the servant, the inflated house, the flat stone**

| world object | computational object |
|---|---|
| the pile of marks, in their given order | the input buffer `data[0..len)` |
| walking marks past the servant | the read phase: pure loads, *no* state update while reading |
| inflated house swatted against my mouth → enforced silence | the accumulator is forbidden to speak (update) during the read; mixing is deferred |
| the flat stone at the riverbank | a 64-byte staging block — an AVX2 register pair for full blocks, a zeroed stack block for the tail |
| "laid on the stone" | each byte's identity is its *position* (block index × 8 lanes × byte-in-word), not its turn in a sequence |
| "the order never spoken aloud and lost" | order carried by placement + `len` committed at the seal, never by the order of ALU operations |

*Breaks:* **"each byte must be mixed into the running state before the next byte is read"** and, as a direct consequence, **"the whole buffer must be read once, start to end, in order"** — inside a laid-down block there is *no* order of operations at all; 8 lanes are read and mixed simultaneously.

**SEED 2 — the musk deer's stride**

| world object | computational object |
|---|---|
| heaps of even count, deer's *fixed strides* | fixed 64-byte blocks, constant stride, no data-dependent addressing |
| the mountain path, one per heap | 8 independent 64-bit lanes, each with its own rotation and its own constant |
| uphill twist | `ROTL64` |
| downhill fold | `^` |
| doubling-back that eats its own trail | `+` — carry propagation makes the step non-linear over GF(2); the word enters twice, once xored, once added |
| "nothing stays still except the final stone" | every one of the 8 lanes is rewritten on every block; there is no passive accumulator |
| "never let it ossify — a satisfied calculation is a pillar that brings the roof down" | **no fixed points**: each lane adds a nonzero constant, so an all-zero block still advances the state; `x=0` is not a fixed point |

*Breaks:* **"the state is a single accumulator updated in place, one value"** and **"mixing one byte requires a multiplication"** (zero multiplies anywhere in the per-byte path).

**SEED 3 — the owl, the dreamer inside, the burning**

| world object | computational object |
|---|---|
| searching the result in squares, turning ninety degrees | the 8 lanes read as a 4×2 square: column pairs `(i,i+4)`, then a ninety-degree turn to diagonal pairs `(i,((i+1)&3)+4)` — a ChaCha/BLAKE3-shaped double round |
| folding corners into its own center | the 8-cycle lane graph `0-4-3-7-2-6-1-5-0` (diameter 4); three double rounds carry any lane into every lane |
| the owl takes the pen | the finalizer (the one place a multiply is allowed, because it is O(1), not per byte) |
| "inside the dreamer inside" | nested fold 8 → 4 → 2 → 1 lanes |
| sealed small, the size of any other token | 64-bit output, independent of `len` |
| intermediate heaps burned the moment the next swallows them | O(1) state, streaming, nothing buffered; 512→64-bit one-way compression |

*Breaks:* **"more mixing rounds always means better mixing"** — there is exactly *one* ARX step per lane per block, and all the heavy diffusion is concentrated in three rounds that run **once**, not per byte.

## CHOSEN SEED

**SEED 1** (the servant's enforced silence / the flat stone). It is the only one of the three that breaks the preferred assumption — *the whole buffer must be read once, start to end, in order* — and it breaks it literally: the servant's swat is exactly the prohibition on touching the accumulator between reads, and the flat stone is exactly a positional staging block. Seeds 2 and 3 are its necessary continuation (what the laid-down heap is mixed with, and how the shape is sealed), so all three ship as one mechanism; Seed 1 is the one that chooses the architecture.

**Arrival at validated technique (step 4).** Taken literally, Seed 1 forces: positional striped lanes, no cross-lane traffic in the hot loop, merge only at the end. That is not an invention — it is the architecture of **xxHash3** (striped parallel accumulators, merged by `mergeAccs`). Seed 2's multiply-free uphill-twist/downhill-fold/doubling-back is **SpookyHash**'s ARX bulk mixing. Seed 3's ninety-degree turn into diagonals is **ChaCha/BLAKE3**'s column-then-diagonal round, and the owl's pen is the **Murmur3/splitmix64** 64-bit finalizer. Four validated components, assembled by the metaphor; nothing novel was invented where a known-good primitive existed.

## ASSUMPTION BROKEN

Primary: *the whole buffer must be read once, start to end, in order* — and with it *each byte must be mixed into the running state before the next byte is read*. Within a 64-byte heap, the 8 words are read and mixed with no ordering between them; order survives only as position. Secondary: *single accumulator* (8 lanes), *multiplication per byte* (pure ARX in the bulk path), *more rounds is better* (one step per block; diffusion concentrated in an O(1) seal).

**Regime recognition (step 5).** The known_way describes one regime (byte-at-a-time through one accumulator) but the problem has two in practice — short keys and long buffers — and the metaphor itself encodes the test: *"I gather the marks into small heaps of even count."* A pile too small to form a heap cannot be raced, so it goes straight to the stone and the owl. Three runtime paths, selected by size, each with the simpler one as fallback:

- `len < 16` — too small for a heap: positional overlapping reads + owl's seal only (~15 ops; no 64-byte stone, no square rounds).
- `16 ≤ len < 64` — one heap, zero-padded on the stone; bulk loop never entered.
- `len ≥ 64` — SIMD lane-striped racing, then stone for the tail, then the squares and the owl.

**Thread parallelism declined, deliberately.** The metaphor has one walker and one servant, and its unit of work is a 64-byte heap — six orders of magnitude below an OpenMP fork/join's worth of work. Adding threads would also make the hash value depend on scheduling unless I built a fixed-chunk tree, which is a different (BLAKE3) mechanism the native did not describe. SIMD is where the metaphor's parallelism actually lives, so SIMD is where it stops.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

/* The eight stones at the shrine: distinct, nonzero.  Nonzero is the
   anti-ossification guard -- x=0 is not a fixed point of a lane step, so a
   buffer of zeros still moves the state every heap. (pi fraction digits) */
static const uint64_t HK[8] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t x; memcpy(&x,p,8); return x; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t x; memcpy(&x,p,4); return (uint64_t)x; }

/* the owl takes the pen: 64 -> 64 avalanche (Murmur3/splitmix finalizer) */
static inline uint64_t owl(uint64_t h){
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

/* one heap raced in the deer's pattern: uphill twist (rotate), downhill fold
   (xor), doubling-back that eats its own trail (add -- carries, so the step is
   not GF(2)-linear).  Eight lanes, eight strides, no multiply, no cross-lane
   traffic: eight independent dependency chains for the out-of-order engine. */
static inline void deer(uint64_t *restrict v, const unsigned char *restrict p){
    uint64_t w0=ld64(p),    w1=ld64(p+8),  w2=ld64(p+16), w3=ld64(p+24);
    uint64_t w4=ld64(p+32), w5=ld64(p+40), w6=ld64(p+48), w7=ld64(p+56);
    v[0] = ROTL64(v[0]^w0,13) + (w0^HK[0]);
    v[1] = ROTL64(v[1]^w1,23) + (w1^HK[1]);
    v[2] = ROTL64(v[2]^w2,29) + (w2^HK[2]);
    v[3] = ROTL64(v[3]^w3,31) + (w3^HK[3]);
    v[4] = ROTL64(v[4]^w4,37) + (w4^HK[4]);
    v[5] = ROTL64(v[5]^w5,41) + (w5^HK[5]);
    v[6] = ROTL64(v[6]^w6,47) + (w6^HK[6]);
    v[7] = ROTL64(v[7]^w7,53) + (w7^HK[7]);
}

/* searching the result in squares: the eight lanes as a 4x2 square -- columns
   (i,i+4), then a ninety-degree turn into the diagonals (i,((i+1)&3)+4).
   Lane graph 0-4-3-7-2-6-1-5-0, diameter 4; three of these reach everywhere. */
static inline void square(uint64_t *restrict v){
    int i;
    for (i = 0; i < 4; i++){
        v[i]   += v[i+4];
        v[i+4]  = ROTL64(v[i+4],17) ^ v[i];
        v[i]    = ROTL64(v[i],43);
    }
    for (i = 0; i < 4; i++){
        int j = ((i+1)&3)+4;
        v[i] += v[j];
        v[j]  = ROTL64(v[j],31) ^ v[i];
        v[i]  = ROTL64(v[i],23);
    }
}

/* the shrine: squares, then the dreamer inside the dreamer (8->4->2->1),
   then the owl's seal.  O(1): all the expensive mixing lives here, none of
   it per byte. */
static inline uint64_t shrine(uint64_t *restrict v, size_t len){
    uint64_t a,b,c,d,x,y;
    v[0] ^= (uint64_t)len;                  /* the order, never lost */
    v[5] += (uint64_t)len + 0x9E3779B97F4A7C15ULL;
    square(v); square(v); square(v);
    a = v[0] ^ ROTL64(v[4],11);
    b = v[1] + ROTL64(v[5],29);
    c = v[2] ^ ROTL64(v[6],47);
    d = v[3] + ROTL64(v[7],19);
    x = a + b;  y = c ^ d;
    x = ROTL64(x,31) + y;
    x ^= ROTL64(y,17);
    return owl(x + (uint64_t)len);
}

/* regime A: a pile too small to form a heap.  Marks laid by position with
   overlapping reads (never outside the buffer), straight to the owl. */
static uint64_t tiny_pile(const unsigned char *p, size_t len){
    uint64_t a,b;
    if (len >= 8){ a = ld64(p) ^ HK[0];        b = ld64(p+len-8) ^ HK[1]; }
    else if (len >= 4){ a = ld32(p) ^ HK[2];   b = ld32(p+len-4) ^ HK[3]; }
    else if (len){ a = (((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8)
                        | (uint64_t)p[len-1]) ^ HK[4];
                   b = HK[5]; }
    else { a = HK[6]; b = HK[7]; }
    a += (uint64_t)len * 0x9E3779B97F4A7C15ULL;
    b ^= ROTL64(a,37);
    a += ROTL64(b,23);
    return owl(a ^ ROTL64(b,41));
}

uint64_t kernel(const unsigned char *data, size_t len){
    uint64_t v[8];
    const unsigned char *p = data;
    size_t n = len;
    int i;

    if (len < 16) return tiny_pile(data, len);      /* regime check + fallback */

    for (i = 0; i < 8; i++) v[i] = HK[i];

#if defined(__AVX2__)
    if (n >= 64){                                   /* regime C: racing heaps */
        __m256i V0 = _mm256_loadu_si256((const __m256i *)(const void *)&v[0]);
        __m256i V1 = _mm256_loadu_si256((const __m256i *)(const void *)&v[4]);
        const __m256i K0 = _mm256_loadu_si256((const __m256i *)(const void *)&HK[0]);
        const __m256i K1 = _mm256_loadu_si256((const __m256i *)(const void *)&HK[4]);
        const __m256i S0 = _mm256_setr_epi64x(13,23,29,31);
        const __m256i S1 = _mm256_setr_epi64x(37,41,47,53);
        const __m256i T0 = _mm256_setr_epi64x(64-13,64-23,64-29,64-31);
        const __m256i T1 = _mm256_setr_epi64x(64-37,64-41,64-47,64-53);
        do {
            /* the flat stone: 64 marks laid down by position, in silence */
            __m256i W0 = _mm256_loadu_si256((const __m256i *)(const void *)p);
            __m256i W1 = _mm256_loadu_si256((const __m256i *)(const void *)(p+32));
            __m256i X0 = _mm256_xor_si256(V0, W0);
            __m256i X1 = _mm256_xor_si256(V1, W1);
            X0 = _mm256_or_si256(_mm256_sllv_epi64(X0,S0), _mm256_srlv_epi64(X0,T0));
            X1 = _mm256_or_si256(_mm256_sllv_epi64(X1,S1), _mm256_srlv_epi64(X1,T1));
            V0 = _mm256_add_epi64(X0, _mm256_xor_si256(W0,K0));
            V1 = _mm256_add_epi64(X1, _mm256_xor_si256(W1,K1));
            p += 64; n -= 64;
        } while (n >= 64);
        _mm256_storeu_si256((__m256i *)(void *)&v[0], V0);
        _mm256_storeu_si256((__m256i *)(void *)&v[4], V1);
    }
#else
    while (n >= 64){ deer(v, p); p += 64; n -= 64; }   /* scalar fallback,
                       bit-identical to the AVX2 path; SLP-vectorizable */
#endif

    {   /* regime B / the tail: the last marks laid on a zeroed stone */
        unsigned char stone[64];
        memset(stone, 0, sizeof stone);
        if (n) memcpy(stone, p, n);
        deer(v, stone);
    }
    return shrine(v, len);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12

Written before any measurement (and no measurement tool is reachable in this session — see MEASUREMENT). Reasoning: FNV-1a is latency-bound on a 64-bit `imul` dependency chain, ≈3 cycles/byte ≈ 1.2 GB/s. This kernel spends ~12 vector ALU µops + 2 loads per 64 bytes with 8 independent chains, so its ceiling is ~14 B/cycle (~50 GB/s) and in practice it will hit a cache/DRAM bandwidth wall first: ~8–25× for L2/L3-resident buffers, ~8× if the benchmark buffer is DRAM-resident. 12 is my single-number bet. Avalanche prediction: mean bit-flip fraction 0.49–0.51 (score ≈ ideal), because every lane feeds the fold and the fold passes through an invertible Murmur3 finalizer, so a one-bit input change is a one-lane change that is then fully avalanched by construction. For `len < 16` I expect roughly 1–2× FNV (small absolute win) with avalanche unchanged, which is why that path is deliberately 15 ops and not the 64-byte machinery.

## MEASUREMENT

Not measured. This session was invoked with no tools available — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were listed but are not callable here, so I have no throughput number and no avalanche score to report. I am stating that plainly rather than reporting plausible-looking figures: **the prediction above is untested.** (Separately: the `claude.ai PubMed` MCP server is unauthorized and cannot be authorized from a non-interactive session — authorize it in claude.ai connector settings if it is ever needed; it is irrelevant to this task.)

Falsification conditions I would accept, when the pipeline measures it:
- avalanche mean flip fraction outside 0.48–0.52, or any output bit with flip probability outside ~0.45–0.55 → the three `square` rounds plus the fold are insufficient and the native's own test ("if even a corner survives untouched I throw the whole method away") says start over at the riverbank: raise to four rounds or move one `square` into the per-block path.
- speedup < 4× on a large buffer → the lane-striped absorb is not the bottleneck I think it is; the likely culprit is `vpsllvq` throughput, fixed by switching to a single rotation amount per vector with `_mm256_slli_epi64` and distinguishing lanes only via `HK`.
- speedup < 1× for `len < 64` → the `memset`+`memcpy` stone is too expensive at small sizes; replace regime B with overlapping positional reads (extend `tiny_pile` to 63 bytes).

## VERDICT

The mechanism is the native's, taken literally, and it lands on validated ground rather than novelty: the servant's enforced silence gives positional striped lanes (xxHash3's architecture), the deer's twist/fold/doubling-back gives multiply-free ARX bulk mixing (SpookyHash), the ninety-degree turn into diagonals gives a ChaCha/BLAKE3 double round, and the owl's pen is the Murmur3 finalizer. Zero multiplications touch a data byte; the single multiply pair runs once per call.

Stated risks, all addressed in the shipped code rather than left as caveats:
- *Small inputs could be slower than the simple accumulator.* Guarded by `len < 16` → `tiny_pile`, and the 64-byte block machinery is never entered below `len = 64`.
- *A lane could ossify on zero data.* Every `HK[i]` is nonzero, so a zero block still advances every lane; `len` is injected at the seal so zero-runs of different length cannot collide.
- *Thread parallelism could cost more than it earns and make the result nondeterministic.* Dropped entirely, with the metaphor's own reason: one walker, one servant, 64-byte heaps.
- *Out-of-bounds reads.* All loads are either inside a full 64-byte block, inside the zeroed 64-byte stone, or (in `tiny_pile`) overlapping reads provably within `[p, p+len)`. All loads go through `memcpy`, so no strict-aliasing violation.

What I am least sure of: the absorb step is lane-local, so this hash is eight independent ARX chains merged at the end. That is enough for avalanche (which is what is measured) and is what xxHash3 does, but it means I have *not* demonstrated collision resistance against an adversary choosing inputs — the avalanche score cannot see that, and I will not claim it.