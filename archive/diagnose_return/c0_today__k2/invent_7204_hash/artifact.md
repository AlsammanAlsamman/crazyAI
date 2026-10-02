No tools are exposed in this session, so I reasoned, predicted, and wrote the artifact without measuring — and I say so plainly in MEASUREMENT rather than inventing numbers.

---

## MAPPING

### SEED 1 — "counting-bars + eight violin strings bound under one bridge; one pluck shivers into every string" (the mixing engine)

| World object | Problem object |
|---|---|
| counting-bars that "alone hold true count" (the bars the tensed speak and death lean against) | integer registers / exact unsigned 64-bit machine arithmetic — no float, count must be exact |
| eight violin strings strung over the landscape | `uint64_t acc[8]` — a 512-bit state, eight lanes laid over the buffer |
| "pluck the string nearest the mark's own weight" | lane index = byte offset within the 64-byte stripe (`j = off/8`); each 8-byte lane goes to its own string |
| "never softly, always the same force" | branchless, constant-cost per byte — no data-dependent work |
| strings "bound under one bridge", a shudder in one bends all the rest | cross-lane coupling: `acc[j^1] += v` each stripe, plus a full-lane bridge (scramble + lane rotation) each block |
| "catch them mid-tremor and carry it forward" | state is never reset between stripes; carried across the whole buffer |
| **Assumption broken** | **"the state is a single accumulator updated in place, one value"** |

### SEED 2 — "the feather's ink-weight, drawn fresh, never repeated" (strict order into the running count)

| World object | Problem object |
|---|---|
| the inkwell that never runs dry | a fixed 192-byte secret `SEC[24]` — reusable, constant, never consumed |
| a fresh child-feather per mark, no two marks share one | each input lane is XORed with a *distinct* secret word before mixing; the secret window slides one word per stripe and the lane permutation rotates per block, so no (position, weight) pair recurs |
| the feather records not the mark's *shape* but its *weight*, "pigment sinks differently for every stroke" | the byte's value is not stored; only its position-weighted contribution `(lo32·hi32)` of `v ^ secret[j]` |
| **Assumption broken** | **"mixing one byte requires a multiplication"** (per *byte*: 8 bytes share one widening 32×32→64 multiply) and partly **"must be read start to end, in order"** (order lives in the position weights, not in a serial dependency) |

### SEED 3 — "the storm's single pass; the pickers destroy every feather and drop"

| World object | Problem object |
|---|---|
| the storm channelling the sky's dissolving pigment | the finalizer — a merge + avalanche pass |
| it passes **once** over the strings, "no more" | exactly one finalization: four 64×64→128 folds of the eight lanes, then one `xor-shift / multiply / xor-shift` avalanche. Zero strong-diffusion rounds during absorption |
| "dissolves it down into eight small notes" | 8 bytes = the 64-bit return value |
| the pickers take apart every used feather and drop so nothing of the original order survives above | scratch state is local and dropped; the fold is non-invertible — the buffer is not recoverable from the token |
| "change one mark, even the last, and the whole chord comes out unrecognizable" | avalanche requirement, including the final stripe (which is why the last stripe is read from the buffer's end and the storm folds *all* eight lanes) |
| **Assumption broken** | **"more mixing rounds always means better mixing"** |

---

## CHOSEN SEED

**SEED 3** — the storm's single pass. It is the only one of the three that breaks the preferred assumption ("more mixing rounds always means better mixing"), and it is the most different from the known way: FNV-1a spends a full strong mixing round (xor + 64-bit multiply, a 4–5 cycle latency chain) on *every single byte*; SEED 3 says all strong diffusion is deferred to **one** pass at the very end, and absorption is deliberately weak and wide.

SEED 1 and SEED 2 are not discarded — the native described one machine, and the storm is only safe to use once *because* the strings are eight and coupled and the feathers are position-unique. They supply the absorber; SEED 3 governs the design.

## ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** Replaced by: *weak, wide, cheap absorption + exactly one strong collapse*. Diffusion is a property of the whole pipeline, not of per-byte rounds. Per-byte work drops to ~1 widening multiply per 8 bytes with no serial dependency; all avalanche is bought once, at the end, where it costs O(1) instead of O(n).

Secondary consequence (SEED 1): the single accumulator becomes eight coupled lanes, which is what removes the latency chain.

**Step 4 — arriving at the validated technique rather than inventing.** Once the mapping is taken literally, the machine it describes *is* XXH3's long-input structure: an eight-lane 512-bit accumulator (eight strings), each lane XORed with a position-indexed word of a fixed 192-byte secret (fresh feather from the inkwell that never runs dry), a `lo32 × hi32` widening multiply-accumulate (the pluck, same force every time), a swap-add into the paired lane (`acc[j^1] += v` — the neighbour shivers), a scramble at each 1024-byte block boundary (the bridge), and `mergeAccs` + a single `avalanche` (the storm, one pass, eight notes). For short inputs the shape it describes is wyhash's: one string, overlapping end-reads, one fold, same storm. So I let the mechanism land on those two validated designs rather than invent a new mixer. The only places I did not simply copy XXH3 are where the native is *more* specific than XXH3:

- "**every** string shivering a little" — XXH3's inter-stripe coupling is only pairwise (`j^1`), permanently pairing lanes {0,1},{2,3},{4,5},{6,7}. I add a 3-lane rotation after each block scramble, which re-pairs them as {3,4},{5,6},{7,0},{1,2}, so the coupling graph becomes a connected 8-cycle and every pluck reaches every string "before it settles." It is a permutation, hence bijective, so it cannot lose state entropy; it costs 8 moves per 1024 bytes.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- counting-bars: exact unsigned integer arithmetic only ---- */
#define XP64_1 0x9E3779B185EBCA87ULL
#define XP64_2 0xC2B2AE3D27D4EB4FULL
#define XP64_3 0x165667B19E3779F9ULL
#define XP64_4 0x85EBCA77C2B2AE63ULL
#define XP64_5 0x27D4EB2F165667C5ULL
#define XP32_1 0x9E3779B1ULL
#define XP32_2 0x85EBCA77ULL
#define XP32_3 0xC2B2AE3DULL

/* ---- the inkwell that never runs dry: 24 fixed drops (192 bytes).
       High-entropy fixed secret; any such secret works, no bit-compat claimed. ---- */
static const uint64_t SEC[24] = {
    0xbe4ba423396cfeb8ULL, 0x1cad21f72c81017cULL, 0xdb979083e96dd4deULL, 0x1f67b3b7a4a44072ULL,
    0x78e5c0cc4ee679cbULL, 0x217cffcc7dd05a82ULL, 0x8e2443f7744608b8ULL, 0x4c263a81e69035e0ULL,
    0xcb00c391bb52283cULL, 0xa32e531b8b65d088ULL, 0x4ef90da297486471ULL, 0xd8acdea946ef1938ULL,
    0x3f349ce33f76faa8ULL, 0x1d4f0bc7c7bbdcf9ULL, 0x3159b4cd4be0518aULL, 0x647378d9c97e9fc8ULL,
    0xc3ebd33483acc5eaULL, 0xeb6313faffa081c5ULL, 0x49daf0b751dd0d17ULL, 0x9e68d429265516d3ULL,
    0xfca1477d58be162bULL, 0xce31d07ad1b8f88fULL, 0x280416958f3acb45ULL, 0x7e404bbbcafbd7afULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

/* 64x64 -> 128, folded: the one place strong diffusion is spent */
static inline uint64_t fold128(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t const r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t const al=a&0xFFFFFFFFULL, ah=a>>32, bl=b&0xFFFFFFFFULL, bh=b>>32;
    uint64_t const ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t const cross=(ll>>32)+(lh&0xFFFFFFFFULL)+hl;
    uint64_t const upper=(lh>>32)+(cross>>32)+hh;
    uint64_t const lower=(cross<<32)|(ll&0xFFFFFFFFULL);
    return lower ^ upper;
#endif
}

/* ---- the storm: ONE pass, eight notes ---- */
static inline uint64_t storm(uint64_t h){ h ^= h>>37; h *= XP64_3; h ^= h>>32; return h; }

/* ---- the pluck: nstripes x 64 bytes onto the eight strings.
       stripe s uses secret bytes [secoff+8s, secoff+8s+64). ---- */
static inline void absorb(uint64_t *restrict acc, const unsigned char *restrict p,
                          size_t nstripes, size_t secoff)
{
    const unsigned char *const sb = (const unsigned char *)SEC + secoff;
#if defined(__AVX2__)
    __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)acc);
    __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(acc + 4));
    for (size_t s = 0; s < nstripes; s++) {
        const unsigned char *const in = p  + 64*s;
        const unsigned char *const sk = sb +  8*s;
        __m256i const d0 = _mm256_loadu_si256((const __m256i *)(const void *)in);
        __m256i const d1 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 32));
        __m256i const k0 = _mm256_loadu_si256((const __m256i *)(const void *)sk);
        __m256i const k1 = _mm256_loadu_si256((const __m256i *)(const void *)(sk + 32));
        __m256i const x0 = _mm256_xor_si256(d0, k0);          /* fresh feather */
        __m256i const x1 = _mm256_xor_si256(d1, k1);
        __m256i const p0 = _mm256_mul_epu32(x0, _mm256_srli_epi64(x0, 32)); /* ink-weight */
        __m256i const p1 = _mm256_mul_epu32(x1, _mm256_srli_epi64(x1, 32));
        __m256i const w0 = _mm256_shuffle_epi32(d0, _MM_SHUFFLE(1,0,3,2));  /* neighbour shivers */
        __m256i const w1 = _mm256_shuffle_epi32(d1, _MM_SHUFFLE(1,0,3,2));
        a0 = _mm256_add_epi64(_mm256_add_epi64(a0, w0), p0);
        a1 = _mm256_add_epi64(_mm256_add_epi64(a1, w1), p1);
    }
    _mm256_storeu_si256((__m256i *)(void *)acc,       a0);
    _mm256_storeu_si256((__m256i *)(void *)(acc + 4), a1);
#else
    for (size_t s = 0; s < nstripes; s++) {
        const unsigned char *const in = p  + 64*s;
        const unsigned char *const sk = sb +  8*s;
        for (int j = 0; j < 8; j++) {                 /* 8 independent strings */
            uint64_t const v = ld64(in + 8*j);
            uint64_t const k = v ^ ld64(sk + 8*j);
            acc[j ^ 1] += v;                          /* the neighbour shivers */
            acc[j]     += (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);
        }
    }
#endif
}

/* ---- the bridge: bijective, so no state entropy is lost.
       scramble each string, then rotate the strings so the pairing graph
       becomes a connected 8-cycle: every pluck reaches every string. ---- */
static inline void bridge(uint64_t *restrict acc)
{
    uint64_t t[8];
    for (int j = 0; j < 8; j++) {
        uint64_t a = acc[j];
        a ^= a >> 47;
        a ^= SEC[16 + j];
        a *= XP32_1;
        t[j] = a;
    }
    for (int j = 0; j < 8; j++) acc[j] = t[(j + 3) & 7];
}

/* ---- regime 2: too few marks to reach across all eight strings.
       One string, overlapping end-reads, then the same single storm. ---- */
static uint64_t short_path(const unsigned char *data, size_t len)
{
    uint64_t a, b;
    uint64_t seed = SEC[0] ^ ((uint64_t)len * XP64_2);
    if (len >= 16) {
        size_t i = 0, n = len;
        while (n > 16) {                              /* 16 marks per pluck */
            seed = fold128(ld64(data + i) ^ SEC[1] ^ seed, ld64(data + i + 8) ^ SEC[2]);
            i += 16; n -= 16;
        }
        a = ld64(data + len - 16);
        b = ld64(data + len -  8);
    } else if (len >= 8) {
        a = ld64(data);          b = ld64(data + len - 8);
    } else if (len >= 4) {
        a = ld32(data);          b = ld32(data + len - 4);
    } else if (len > 0) {
        a = ((uint64_t)data[0] << 16) | ((uint64_t)data[len >> 1] << 8) | (uint64_t)data[len - 1];
        b = 0;
    } else { a = 0; b = 0; }
    uint64_t h = fold128(a ^ SEC[3] ^ seed, b ^ SEC[4]);
    h += (uint64_t)len * XP64_1;
    return storm(h);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the native counts the marks first and picks his regime */
    if (len < 64) return short_path(data, len);

    uint64_t acc[8] = { XP32_3, XP64_1, XP64_2, XP64_3, XP64_4, XP32_2, XP64_5, XP32_1 };

    size_t const nbStripes = (len - 1) / 64;     /* every stripe but the last */
    size_t const nbBlocks  = nbStripes / 16;     /* 1024-byte blocks */

    for (size_t b = 0; b < nbBlocks; b++) {
        absorb(acc, data + b * 1024, 16, 0);
        bridge(acc);                             /* once per 1024 bytes */
    }
    size_t const rem = nbStripes - nbBlocks * 16;
    if (rem) absorb(acc, data + nbBlocks * 1024, rem, 0);

    /* the last mark is never skipped: final stripe read from the very end,
       with its own window of the inkwell */
    absorb(acc, data + len - 64, 1, 120);

    /* ---- the storm passes once, and only once ---- */
    uint64_t h = (uint64_t)len * XP64_1;
    h += fold128(acc[0] ^ SEC[1], acc[1] ^ SEC[2]);
    h += fold128(acc[2] ^ SEC[3], acc[3] ^ SEC[4]);
    h += fold128(acc[4] ^ SEC[5], acc[5] ^ SEC[6]);
    h += fold128(acc[6] ^ SEC[7], acc[7] ^ SEC[8]);
    return storm(h);
}
```

Bounds audit (every read, since there is no correctness check to catch me): `nbBlocks*16 ≤ nbStripes = (len-1)/64`, so the block loop reads at most byte `nbBlocks*1024 - 1 ≤ 64*nbStripes - 1 ≤ len - 2`; the `rem` run likewise ends at `≤ len-2`; the final stripe reads exactly `[len-64, len)` with `len ≥ 64`. Secret reads: stripe `s ≤ 15` touches bytes `[8s, 8s+64) ⊆ [0,184)`, and the last stripe touches `[120,184)` — both inside the 192-byte table. In `short_path`, every branch reads only within `[0,len)`, and the `while (n > 16)` guard leaves the final 16 bytes to the overlapping end-reads, so every byte of every length 1..63 is absorbed.

Two regimes, both recognized at runtime from the metaphor itself: **`len < 64`** (the marks don't reach across all eight strings → one string, wyhash-shaped, same single storm) and **`len ≥ 1024`** (enough sweeps for the bridge to engage at all). No OpenMP: the native's own account forbids it ("never doubling back… I catch them mid-tremor and carry that tremor forward"), the carried state is a true serial dependency across blocks, and at plausible benchmark sizes thread spin-up would dominate. Vectorization only, per the instruction's default.

## PREDICTION

PREDICTION: speedup_vs_dp = 18

Reasoning behind the number, stated before measuring: FNV-1a is a serial `xor`+`imul` dependency chain, ~4–5 cycles per byte (≈0.8–1.1 GB/s at 3.5 GHz) and cannot vectorize. This kernel does one `vpmuludq` per 8 bytes across 8 independent lanes, ~0.15–0.25 cycles/byte for cache-resident data (≈15–25 GB/s), so 15–30× on large buffers, pulled down by any small-size mix in the harness. Avalanche score: I predict it lands in the "good" band (≈0.49–0.51 bit-flip fraction, no biased bit), because all eight lanes pass through 128-bit folds and a final avalanche, so even a flip in the last byte reaches every output bit.

Falsifiers I would accept: speedup below ~5× on large buffers (means the absorb didn't vectorize — check for `vpmuludq` in the disassembly), or avalanche materially off 0.5 (means one storm pass is genuinely not enough and the native is wrong about "no more").

## MEASUREMENT

**Not measured — no tools are available in this session.** `hash_bench` and `hash_contract` were not callable, so I ran zero of the four permitted improvement iterations. I will not report numbers I did not obtain. The prediction above stands as written and un-tuned; the harness that compiles this will produce the first real measurement, and if it falls outside the falsifier bands above, the design is wrong, not the measurement.

What I substituted for measurement, and what it does *not* cover: a by-hand bounds audit of every load (above), a bijectivity argument for the bridge, and the fact that the absorb/merge/finalize structure is the published XXH3 long-input shape and the short path the published wyhash shape — both of which pass SMHasher in their original form. That is evidence about the *shape*, not about my transcription of it. The specific untested deltas are (a) the 3-lane rotation in `bridge`, (b) the secret constants, reproduced from memory and therefore probably not bit-identical to XXH3's `kSecret` — harmless for quality since any fixed high-entropy secret serves, but it means this kernel is **not** XXH3-compatible and should not be described as such, and (c) whether GCC's auto-vectorizer handles the non-AVX2 scalar fallback well.

## VERDICT

The native's claim is a real, falsifiable engineering claim and it is correct in the literature: mixing strength per byte and mixing strength of the hash are not the same quantity. FNV-1a's one strong round per byte buys avalanche at the cost of an O(n) serial multiply chain; the native buys the same avalanche once, in O(1), at the end, and spends the per-byte budget on *width* instead. Eight coupled strings under one bridge is how the serial chain disappears. "More rounds is better" is false specifically because the final fold is where diffusion actually gets paid for.

Risks I named, and how each is addressed rather than left standing:

- *"Overhead if small."* Guarded: `len < 64` takes `short_path`, a single-lane validated path, so the 8-lane setup and 192-byte secret never load for small inputs. Residual I will not hide: for `len ≤ 4` the fixed finalization cost (~10 cycles for one fold plus the storm) exceeds FNV-1a's ~15 cycles only marginally and may measure at parity or slightly behind. That cost is constant, not scaling, and is the unavoidable price of any strong finalizer; the only way to remove it would be to weaken the storm, which would break the native's own avalanche requirement. So I kept it and state it, rather than shipping an unguarded claim.
- *"Only helps if the problem is large."* Addressed by the same two-regime check — there is no size at which the kernel takes the wide path without enough bytes to fill it.
- *Thread parallelism.* Dropped entirely rather than guarded, because the metaphor's unit of work is one carried tremor and splitting it would change the hash, and because I cannot measure the crossover here. Shipping an unmeasured OpenMP path would be exactly the "mechanism whose own stated risk I didn't address" failure.

Honest bottom line: the mechanism is sound and the mapping is literal, but with no measurement in this session the speedup figure is a prediction and nothing more. If it misses, the most likely cause is vectorization of the absorb failing, not the native being wrong.

---

One housekeeping note: the `claude.ai PubMed` MCP server is listed as needing authorization and cannot be authorized from this non-interactive session — if you need it, authorize it via your claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session); until then its tools are unavailable. Nothing in this task required it.