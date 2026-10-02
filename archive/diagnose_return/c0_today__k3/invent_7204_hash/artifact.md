No tools were available in this session, so everything below is reasoning + the artifact. The prediction line is stated before (and without) any measurement, and the MEASUREMENT section reports honestly that nothing was measured here.

---

# MAPPING

## SEED 1 — "counting-bars and violin strings bound under one bridge; one pluck shivers into every other string"

| World object | Problem object |
|---|---|
| counting-bars (the only things that hold true count) | the machine's 64-bit integer registers / SIMD lanes — the only place exact arithmetic happens |
| the pile of marks, in their given order | the byte buffer `data[0..len-1]`, read in order |
| eight violin strings strung over the landscape | an 8 × 64-bit state `s[0..7]` = 512 bits of state (two `__m256i`) |
| "I pluck the string nearest the mark's own weight" | word *w* of the stream is absorbed into lane `w mod 8` — i.e. a 64-byte block maps one 8-byte word onto each string |
| the one bridge all eight strings are bound under | a cross-lane coupling step: every absorb step mixes A(lanes 0-3) ↔ B(lanes 4-7) **and** rotates lanes inside each half, so one word's difference reaches all eight strings |
| "the whole instrument answers, every string shivering a little" | the update is a butterfly: `A' = perm₁(A+M+F) ^ rot(B+M+F)`, `B' = perm₃(B+M+F) + rot(A+M+F)` — both halves are functions of both halves |
| "never softly, always the same force; the force must never vary" | fixed, data-independent op count per block; no branch on data values |
| "I catch them mid-tremor and carry that tremor forward" | the state is never reset between blocks; strict serial carry |

**Silent assumption broken:** *"the state is a single accumulator updated in place, one value."*

## SEED 2 — "the feather's ink-weight per mark, drawn fresh and never repeated"

| World object | Problem object |
|---|---|
| the inkwell that never runs dry | a fixed table of 8 odd constants `KSTR[8]` (an inexhaustible pigment source: a *stride*, not a finite keystream) |
| a fresh child-feather per mark, never shared | a position-unique addend: `F[lane] = (block+1)·KSTR[lane]`, maintained by `F += KSTR` per block — unique per (lane, block); the byte's offset inside its little-endian word makes it unique per *byte* |
| the feather records only the weight, not the shape — "pigment sinks differently for every stroke" | the byte is reduced to an **additive weight**, not transformed: `state += word + F`. Pigment is *added*, never multiplied |
| "no two marks ever share a feather" → strict order is fixed into the count | position-dependence makes the absorb order-sensitive without needing a multiply to break symmetry |

**Silent assumption broken:** *"mixing one byte requires a multiplication."* (Per-byte mixing here is pure add/xor/rotate/permute — ARX.)

## SEED 3 — "the storm's single pass collapses the final tremor into a small fixed token; then the pickers destroy every discarded feather and drop"

| World object | Problem object |
|---|---|
| "Only when the last mark has been plucked do I wait — truly wait, hands off" | **no strong mixing during the loop at all**; strength is deferred entirely to finalization |
| the storm passing **once** over the strings | exactly one finalization pass: a fixed, O(1), message-free diffusion of the 512-bit state — not repeated per byte, not repeated per block |
| the sky's dissolving pigment (the only dissolving agent in the world) | the only `imul` in the whole kernel: the final avalanche multiplier |
| "dissolves it down into eight small notes, no more" | 8 bytes = exactly 64 output bits |
| the pickers taking apart every feather and drop "so nothing of the original order survives" | the irreversible 512 → 64 fold; all per-position feathers and intermediate lane values are destroyed (stack scratch, non-invertible compression) |
| "change one mark, even the last, and the whole chord comes out a different token" | avalanche must hold for the *last* byte too → the single storm pass must be strong enough alone |

**Silent assumption broken:** *"more mixing rounds always means better mixing."* The native explicitly refuses to add rounds per mark — he plucks once, with constant force, and spends his entire mixing budget in **one** terminal pass.

---

# CHOSEN SEED

**SEED 3** — the storm's single pass.

It is the one of the three that breaks the preferred assumption ("more mixing rounds always means better mixing"), and it is the farthest from FNV-1a/xxHash, which do a mixing round *per byte or per lane-chunk* and then a comparatively light finalizer. SEED 3 inverts the budget: the per-byte work is the weakest thing in the kernel (one add, no multiply, no round), and essentially all diffusion strength lives in one O(1) terminal storm. SEEDs 1 and 2 are the machinery SEED 3 needs (a wide coupled state to park weak per-byte work in, and position-unique pigment so weak work still fixes order), so they are implemented as its support, not discarded.

# ASSUMPTION BROKEN

> **"more mixing rounds always means better mixing."**

Dropped. Mixing rounds are *not* spread over the input — they are concentrated into a single final pass. Corollaries dropped alongside (from SEEDs 1-2): the state is 512 bits across 8 coupled lanes, not one accumulator; and no multiplication touches any input byte.

**This is not an invention.** Taken seriously, the native describes the **sponge / wide-state ARX construction**: cheap keyed absorb into a wide permuted state, then one strong finalization — the structure of ChaCha/BLAKE2-BLAKE3's compression and of the Keccak-style absorb-then-squeeze, and the storm itself is literally **MurmurHash3's `fmix64`**, a validated finalizer. I let the metaphor land on these rather than hand-rolling a novel finalizer.

**Thread parallelism: declined, deliberately.** The native says "never skipping, never doubling back… I catch them mid-tremor and carry that tremor forward." The metaphor's unit of work is a strictly serial 64-byte pluck; there are no independent large units to hand to threads. Vectorization (the eight simultaneous strings) is where the metaphor's own parallelism is, so that is all I took. No OpenMP.

**Regime recognition (the native's own check).** "I lay the pile of marks upon the counting-bars… for the bars alone hold true count" — he counts the pile *before* plucking. A pile too small to span all eight strings never reaches the bridge at all: `len < 64` takes a two-string short path with a single `fmix64` and no storm. This also discharges the risk my own verdict names (a fixed 6-step storm would dominate tiny inputs): it is guarded by a size check with a cheaper fallback, not left as a caveat. The tail of a long buffer (`len % 64`) is absorbed from a zero-padded 64-byte scratch block rather than by re-reading the last 64 bytes, because re-reading is "doubling back," which the native forbids.

---

# ARTIFACT

```c
/* "Eight strings under one bridge": wide-state ARX sponge.
 *   per byte  : one add of a position-unique constant  (no multiply, no round)
 *   per block : one butterfly pluck, 8 x 64-bit state  (constant force)
 *   once      : one storm pass + irreversible 512->64 fold (the only multiplies)
 * Portable scalar path and AVX2 path compute the identical function.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x,n) (((x) << (n)) | ((x) >> (64 - (n))))

/* the eight strings' resting pitches */
static const uint64_t KIV[8] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL
};
/* the inkwell that never runs dry: strides giving a fresh, never-repeated
   ink-weight F[lane] = (block+1)*KSTR[lane] for every mark */
static const uint64_t KSTR[8] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
    0x85EBCA77C2B2AE63ULL, 0x2545F4914F6CDD1DULL,
    0xD6E8FEB86659FD93ULL, 0xA24BAED4963EE407ULL
};

/* the storm: the sky's dissolving pigment. One pass. The only multiplies in
   the kernel. (MurmurHash3 fmix64 - validated, not invented.) */
static inline uint64_t storm64(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}

static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* REGIME 2: a pile too small to span all eight strings never reaches the
   bridge. Two strings, no storm pass, one fmix64. */
static uint64_t short_pile(const unsigned char *d, size_t len) {
    uint64_t a = KIV[0] ^ (uint64_t)len;
    uint64_t b = KIV[5] + ROTL64((uint64_t)len, 29);
    if (len >= 16) {
        size_t i = 0;
        do {
            a += ld64(d + i)     + KSTR[0]; a = ROTL64(a, 29); a ^= b;
            b += ld64(d + i + 8) + KSTR[5]; b = ROTL64(b, 41); b ^= a;
            i += 16;
        } while (i + 16 <= len);
        if (i < len) {                      /* last 16 bytes, fresh feathers */
            a += ld64(d + len - 16) + KSTR[2]; a = ROTL64(a, 31); a ^= b;
            b += ld64(d + len - 8)  + KSTR[6]; b = ROTL64(b, 43); b ^= a;
        }
    } else if (len >= 8) {
        a += ld64(d)           + KSTR[0]; a = ROTL64(a, 29); a ^= b;
        b += ld64(d + len - 8) + KSTR[5]; b = ROTL64(b, 41); b ^= a;
    } else if (len >= 4) {
        a += (uint64_t)ld32(d)           + KSTR[1]; a = ROTL64(a, 27); a ^= b;
        b += (uint64_t)ld32(d + len - 4) + KSTR[6]; b = ROTL64(b, 43); b ^= a;
    } else if (len > 0) {
        uint64_t t = (uint64_t)d[0];
        t |= (uint64_t)d[len >> 1] << 8;
        t |= (uint64_t)d[len - 1] << 16;
        a += t + KSTR[3]; a = ROTL64(a, 25); a ^= b;
    }
    return storm64(a ^ ROTL64(b, 27));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * __restrict p = data;
    uint64_t s[8];

    /* the count goes onto the counting-bars first: regime check */
    if (len < 64) return short_pile(p, len);

#if defined(__AVX2__)
    {
    #define VROTL(x,n) _mm256_or_si256(_mm256_slli_epi64((x),(n)), _mm256_srli_epi64((x),64-(n)))
    #define PERM1(x)   _mm256_permute4x64_epi64((x), 0x39)   /* lane i <- i+1 */
    #define PERM3(x)   _mm256_permute4x64_epi64((x), 0x93)   /* lane i <- i+3 */
    /* one pluck: constant force, whole instrument answers, feathers advance */
    #define PLUCK(MA,MB,R1,R2) do {                                            \
            __m256i tA = _mm256_add_epi64(_mm256_add_epi64(A,(MA)), FA);        \
            __m256i tB = _mm256_add_epi64(_mm256_add_epi64(B,(MB)), FB);        \
            A = _mm256_xor_si256(PERM1(tA), VROTL(tB,(R1)));                   \
            B = _mm256_add_epi64(PERM3(tB), VROTL(tA,(R2)));                   \
            FA = _mm256_add_epi64(FA, SA); FB = _mm256_add_epi64(FB, SB);      \
        } while (0)

        const __m256i L = _mm256_set1_epi64x((long long)len);
        __m256i A  = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)KIV), L);
        __m256i B  = _mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(KIV+4)), L);
        const __m256i SA = _mm256_loadu_si256((const __m256i*)KSTR);
        const __m256i SB = _mm256_loadu_si256((const __m256i*)(KSTR+4));
        __m256i FA = SA, FB = SB;
        size_t i = 0;

        for (; i + 64 <= len; i += 64) {
            __m256i MA = _mm256_loadu_si256((const __m256i*)(p + i));
            __m256i MB = _mm256_loadu_si256((const __m256i*)(p + i + 32));
            PLUCK(MA, MB, 23, 40);
        }
        if (i < len) {                       /* tail: padded, never doubling back */
            unsigned char pad[64];
            memset(pad, 0, 64);
            memcpy(pad, p + i, len - i);
            PLUCK(_mm256_loadu_si256((const __m256i*)pad),
                  _mm256_loadu_si256((const __m256i*)(pad + 32)), 29, 47);
        }
        {   /* THE STORM: one pass, hands off the marks, no message at all */
            const __m256i Z = _mm256_setzero_si256();
            PLUCK(Z, Z, 17, 31); PLUCK(Z, Z, 23, 40); PLUCK(Z, Z, 29, 47);
            PLUCK(Z, Z, 37, 53); PLUCK(Z, Z, 11, 19); PLUCK(Z, Z, 43, 59);
        }
        _mm256_storeu_si256((__m256i*)s,       A);
        _mm256_storeu_si256((__m256i*)(s + 4), B);
    #undef PLUCK
    #undef PERM3
    #undef PERM1
    #undef VROTL
    }
#else
    {
        uint64_t f[8], m[8];
        int k; size_t i = 0;
        for (k = 0; k < 8; k++) { s[k] = KIV[k]; f[k] = KSTR[k]; }
        for (k = 0; k < 4; k++) s[k]     ^= (uint64_t)len;
        for (k = 4; k < 8; k++) s[k]     += (uint64_t)len;

    #define PLUCK(M,R1,R2) do {                                                \
            uint64_t tA[4], tB[4]; int j;                                      \
            for (j = 0; j < 4; j++) tA[j] = s[j]   + (M)[j]   + f[j];          \
            for (j = 0; j < 4; j++) tB[j] = s[j+4] + (M)[j+4] + f[j+4];        \
            for (j = 0; j < 4; j++) s[j]   = tA[(j+1)&3] ^ ROTL64(tB[j],(R1)); \
            for (j = 0; j < 4; j++) s[j+4] = tB[(j+3)&3] + ROTL64(tA[j],(R2)); \
            for (j = 0; j < 8; j++) f[j] += KSTR[j];                           \
        } while (0)

        for (; i + 64 <= len; i += 64) {
            for (k = 0; k < 8; k++) m[k] = ld64(p + i + 8*(size_t)k);
            PLUCK(m, 23, 40);
        }
        if (i < len) {
            unsigned char pad[64];
            memset(pad, 0, 64);
            memcpy(pad, p + i, len - i);
            for (k = 0; k < 8; k++) m[k] = ld64(pad + 8*(size_t)k);
            PLUCK(m, 29, 47);
        }
        {   /* THE STORM */
            uint64_t z[8]; for (k = 0; k < 8; k++) z[k] = 0;
            PLUCK(z, 17, 31); PLUCK(z, 23, 40); PLUCK(z, 29, 47);
            PLUCK(z, 37, 53); PLUCK(z, 11, 19); PLUCK(z, 43, 59);
        }
    #undef PLUCK
    }
#endif

    /* the pickers: irreversible 512 -> 64, nothing of the order survives */
    {
        uint64_t h = (s[0] + ROTL64(s[1], 11)) ^ (s[2] + ROTL64(s[3], 23));
        h += (s[4] + ROTL64(s[5], 37)) ^ (s[6] + ROTL64(s[7], 53));
        h ^= (uint64_t)len;
        return storm64(h);          /* eight small notes, no more */
    }
}
```

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 15**

Reasoning stated before any measurement, so it can be falsified cleanly:

- **Baseline (FNV-1a):** the loop is a serial `xor`→`imul` dependency chain, ≈4 cycles/byte latency-bound ⇒ ≈0.7 GB/s at ~3 GHz. One byte per 4 cycles.
- **This kernel, long regime:** 64 bytes per pluck; the pluck's critical path is `add → add → vpermq → vxor` ≈ 6 cycles, with both halves independent and the feather advance off the critical path. ⇒ ≈6-7 cycles per 64 bytes ≈ 9-10 bytes/cycle ≈ 28-30 GB/s in-cache, dropping toward DRAM bandwidth (~10-15 GB/s) for buffers that miss L2. Instruction count ≈18 per 64 bytes = 0.28 instr/byte.
- So the honest per-size expectation is **≈35-40× in cache, ≈15-20× DRAM-bound, ≈1-2× at 16 bytes, ≈0.5-1× at 1-4 bytes** (where `fmix64`'s two multiplies are an irreducible floor that FNV-1a avoids by simply not avalanching). I predict **15** as the harness's aggregate, and I expect the single biggest source of error to be whether the benchmark's mean is dominated by small keys (would push it under 5) or by large buffers (would push it over 25).
- **Avalanche:** I predict **0.49-0.50**. The riskiest case is a flip in the *final* 64-byte block, which sees only one weak pluck before finalization — it must survive on the storm alone. Lane-diffusion trace of the butterfly: a difference in `s0` reaches `{s0,s3,s4,s5}` after one storm step and all eight within three; six steps plus `fmix64` leaves ≈2× margin. If avalanche comes back below ~0.45, the diagnosis is "six storm steps are not enough," and the fix is more storm steps (free for large inputs) — not more per-byte work.

# MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were not available in this session, so I have measured nothing and I am not going to report numbers as if I had. The kernel is submitted unmeasured and un-iterated; the four allowed improvement rounds are unspent. Concretely unverified: (a) that `speedup_vs_dp` is anywhere near 15, (b) that avalanche reaches ~0.5, (c) that the AVX2 and scalar paths agree (they are written to, and I traced the permute immediates `0x39`/`0x93` against `result[i] = src[(i+1)&3]` / `src[(i+3)&3]`, but I did not run them), (d) that it compiles clean under `gcc -O3 -march=native` on the target. Treat every number in PREDICTION as a falsifiable claim, not a result.

# VERDICT

The native's design is a real construction, not a metaphor that collapses under translation: **cheap keyed absorb into a wide coupled state, with the entire mixing budget spent in one terminal pass** is the sponge/ARX pattern behind ChaCha, BLAKE3's compression, and Keccak's absorb-squeeze, and the storm is literally MurmurHash3's `fmix64`. Breaking "more rounds means better mixing" is exactly what buys the speed: FNV-1a pays a 3-cycle multiply latency *per byte* to get diffusion it could have deferred; the native pays nothing per byte and settles the whole debt once, off the hot loop, where its cost amortizes to zero.

What I will not paper over:

1. **The absorb is weakly nonlinear.** Add/xor/rotate/permute with position-unique addends is enough to guarantee a single-bit flip never cancels (hence good avalanche, which is what the bench measures), but it is *not* enough to rule out crafted multi-word differentials cancelling in the state. This hash is non-cryptographic and should not be used where an adversary picks the input. A single-bit avalanche score near 0.5 would **not** be evidence against this concern — it does not probe it.
2. **Small inputs are the honest weak spot,** and I addressed rather than caveated it: the `len < 64` short path drops to two strings and skips the storm entirely, so the fixed 6-step cost can never dominate a tiny buffer. Below ~4 bytes the two `fmix64` multiplies may still make this *slower* than FNV-1a. I accept that rather than weaken the finalizer, because FNV-1a's speed there comes from not avalanching at all.
3. **I declined thread parallelism on the metaphor's own authority** ("never skipping, never doubling back"), and one core already sits near memory bandwidth on large buffers — OpenMP here would buy little and would require abandoning the carried tremor, which is the construction.
4. **If it loses,** the likely cause is that the benchmark weights small keys heavily, where no amount of wide-state cleverness beats "do almost nothing." The mechanism's claim is about bytes, not calls.

---

*Note on tooling: the `claude.ai` and `PubMed` MCP servers listed in this environment require authorization and cannot be authorized from a non-interactive session — for claude.ai connectors, via your claude.ai connector settings; for others, via `claude mcp` or `/mcp` in an interactive session. None of them were needed here; the missing pieces were `hash_bench` / `hash_contract`.*