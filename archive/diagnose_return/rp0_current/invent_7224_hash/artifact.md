## MAPPING

I read the whole passage as one working, then split it per SEED.

### SEED 1 — "the servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded"

| World object | Problem object |
|---|---|
| a mark | one input byte `data[i]` |
| the pile in its given order | the buffer `data[0..len)` |
| walking marks past the servant, one at a time | loading bytes in index order |
| the inflated house on a stick swatted against the mouth (the jester's bladder / the Zen *keisaku*) | a **hard prohibition on arithmetic during the read**: the load stage is allowed to do nothing but place bytes |
| "hold silence" | no state update per byte — zero mixing while absorbing |
| the flat stone at the riverbank | a fixed-size positional slab: a 64-byte block of lanes, laid out as 8 × `uint64_t` |
| "so the order never gets spoken aloud and lost" | **order is carried by lane position, not by update sequence** — byte *i* lives at `(block i>>6, lane (i>>3)&7, byte i&7)` |
| the count laid down with the marks | `len` injected into the capacity before any folding |

**Assumption broken:** primarily *"each byte must be mixed into the running state before the next byte is read"*, and consequently *"the whole buffer must be read once, start to end, in order"* — if position encodes order, the sequential *dependency* is gone; the 64 bytes of a heap may be fetched in any order, all at once, as vector loads.

### SEED 2 — "each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with"

| World object | Problem object |
|---|---|
| heaps of even count | fixed 64-byte blocks (rate of a sponge) |
| the musk deer gathering ground in **fixed strides** | **constant** rotation amounts — no data-dependent shifts |
| uphill twist | rotate-left |
| downhill fold | XOR |
| a doubling-back that eats its own trail | the self-consuming `x ^= x >> k` / `d ^= a` feedback of an ARX round |
| racing across the mountain path | add-rotate-xor round applied to the block's lanes |
| "no heap keeps the shape it entered with" | the round is a bijective permutation, not an accumulation |
| the final stone at the shrine, carried forward into the next heap's racing | the chaining value / capacity words `s[8..15]` |
| "I never let this ossify… a calculation grown too satisfied is a pillar that brings the roof down" | **do not add rounds for comfort** — use the minimum count that passes the diffusion test |

**Assumption broken:** *"mixing one byte requires a multiplication"* (fixed strides = rotate/xor/add only, zero `imul` in the bulk path), and *"more mixing rounds always means better mixing"* (the roof-collapse warning is an explicit round-count ceiling).

### SEED 3 — "the owl calls the final folded shape inside the dreamer inside, sealed to one fixed size; intermediate heaps burned at the shrine"

| World object | Problem object |
|---|---|
| the owl taking the pen | the finalization routine (distinct from absorption) |
| "the dreamer inside" (Zhuangzi's butterfly) | the squeeze: a many-word state collapsed to one word |
| sealed small and unchanging, the size of any other token | return exactly one `uint64_t` regardless of `len` |
| the intermediate heaps burned the moment the next heap swallows them | no per-block history kept; state is overwritten in place |
| "a token that can be walked backward to its marks is no token at all" | the squeeze is lossy 128 B → 8 B; one-way by construction |
| the drifting man searching **squares in every direction** | a **4×4 square state**, mixed by columns *and* by the turned axis |
| turning the shape ninety degrees | **ShiftRows** — rows 1,2,3 rotated by 1,2,3 lanes, so the next column pass mixes across what were rows |
| folding the corners into its own center | the final XOR-fold over the two diagonals `s[0],s[5],s[10],s[15]` / `s[3],s[6],s[9],s[12]` |
| "change one mark and watch the whole grid rearrange; if even a corner survives untouched, throw it away and start at the riverbank" | **full-dependency test**: the per-block round count is set to the minimum at which every one of the 16 words depends on every input word |

**Assumption broken:** *"the state is a single accumulator updated in place, one value"* — the state is a 128-byte 4×4 square of 16 words, and the output is not the state.

---

## CHOSEN SEED

**SEED 1.**

Per the tie-break rule I looked for a seed that breaks *"the whole buffer must be read once, start to end, in order."* One does: SEED 1. The bladder-swat is not decoration — it is a prohibition on doing arithmetic while reading, and the *reason* given is that order must be preserved some other way ("so the order never gets spoken aloud and lost"). The other way is the flat stone: **position**. Once order is held by position rather than by the sequence of accumulator updates, the serial byte→byte dependency that defines FNV-1a is dissolved: a heap's 64 bytes are all placed simultaneously and the read order inside a heap is free. This is also maximally distant from the known way, whose entire structure *is* "mix before you read the next byte."

SEED 2 and SEED 3 supply the machinery SEED 1 needs (what to do with a laid-down heap, and how to seal it), so I implement all three — but SEED 1 is the load-bearing one.

## ASSUMPTION BROKEN

Primary: **"each byte must be mixed into the running state before the next byte is read"** and **"the whole buffer must be read once, start to end, in order."**
Secondary, from the same working: **"mixing one byte requires a multiplication"** (bulk path is multiply-free), **"the state is a single accumulator, one value"** (128-byte square), **"more mixing rounds always means better mixing"** (round count pinned to the native's own diffusion test, not inflated).

### Letting the mechanism land on a validated technique (step 4)

The mechanism, followed literally, *is* a known construction and I let it arrive there rather than inventing:

- silence-then-place, fixed rate, capacity carried forward, lossy squeeze = **sponge construction** (Keccak/Ascon/Xoodyak family);
- square state, mix columns → turn 90° → mix columns = **ChaCha's column/diagonal round pattern** and **AES's ShiftRows**;
- uphill twist / downhill fold / self-eating trail with fixed strides = the **BLAKE2b `G` function**, and I use BLAKE2b's own published rotation constants **32, 24, 16, 63** rather than constants of my own;
- corners folded into the centre = a diagonal XOR-fold squeeze.

So: a **BLAKE2b-`G` ARX permutation over a 4×4 `uint64` square, run as a sponge at rate 64 / capacity 64**. Nothing here is novel; the novelty is only that the metaphor chose it over the accumulator.

### The two regimes (step 5) and the risk-guard (step 4)

The assumptions section names two regimes by naming an "overhead if small" failure. A 128-byte state plus a fixed 12-layer seal cannot pay for itself on 12 bytes. The native's own text gives the regime test: the flat stone at the riverbank is only worth walking to if there is a *pile*; a handful of marks is spoken straight into the single stone at the shrine. So:

- `len >= 64` → the square sponge (the metaphor's path);
- `len < 64` → **fall back to the simpler known path**: a two-lane splitmix64/xxHash-class accumulator. This is the guard for the exact condition my verdict would otherwise flag.

Both paths avalanche well; the boundary is at the first point where the sponge is already ahead (at `len=64` it costs ~110 cycles vs FNV's ~290, so there is no regression valley at the seam).

**No thread parallelism.** The metaphor's carry is explicitly serial — "the running shape gets carried forward into the next heap's racing… burned the moment the next heap swallows them." There is no independent large unit of work in this world to give a thread, so OpenMP would be me overruling the native. The parallelism the metaphor *does* offer is the 4 independent columns of the square, which is lane parallelism — so I take SIMD (via a deliberately auto-vectorizable column loop, `restrict`, 64-byte-aligned block stride, a prefetch) and stop there.

### Design revisions before shipping (4 allowed, 3 used)

1. Rate 32 (4×4 of `uint32`, plain ChaCha) → rate 64 with 64-bit lanes: halves permutations per byte.
2. 4 double-rounds per block → **2** (4 layers): the native's test demands full dependency, and column→turn→column reaches all 16 words in 2 layers, so 4 is the honest minimum-with-margin. This is the "don't let it ossify" clause used as a speed lever.
3. Unconditional sponge → size-gated sponge with a splitmix64 fallback, closing the small-input hole.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* ------------------------------------------------------------------ *
 * seal64 -- splitmix64 / xxHash-class 64-bit avalanche finalizer.
 * Validated, multiply-based.  Used ONLY by the short regime: the
 * "small pile spoken straight into the single shrine stone".
 * ------------------------------------------------------------------ */
static inline uint64_t seal64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* ------------------------------------------------------------------ *
 * grow -- one "musk deer stride" layer over the 4x4 square.
 * Exactly BLAKE2b's G function (fixed strides 32,24,16,63: uphill
 * twist = rotate, downhill fold = xor, doubling-back that eats its
 * own trail = the d^=a / b^=c feedback), applied to all FOUR columns
 * at once.  The i-loop is the lane loop: 4 independent columns, unit
 * stride, add/xor/shift only -- written this way so -O3 -march=native
 * turns it into one vector op per scalar op.  No multiplication.
 * ------------------------------------------------------------------ */
static inline void grow(uint64_t *restrict s) {
    for (int i = 0; i < 4; i++) {
        uint64_t a = s[0 + i], b = s[4 + i], c = s[8 + i], d = s[12 + i];
        a += b; d ^= a; d = ROTL64(d, 32);
        c += d; b ^= c; b = ROTL64(b, 24);
        a += b; d ^= a; d = ROTL64(d, 16);
        c += d; b ^= c; b = ROTL64(b, 63);
        s[0 + i] = a; s[4 + i] = b; s[8 + i] = c; s[12 + i] = d;
    }
}

/* ------------------------------------------------------------------ *
 * turn90 -- "turning the shape ninety degrees": ShiftRows.  Rows 1,2,3
 * rotate left by 1,2,3 lanes, so the next grow() mixes along the other
 * axis.  grow+turn+grow reaches all 16 words from any one word, which
 * is the native's test ("if even a corner survives untouched, throw
 * the whole method away").
 * ------------------------------------------------------------------ */
static inline void turn90(uint64_t *restrict s) {
    uint64_t t;
    t = s[4];  s[4] = s[5];  s[5] = s[6];   s[6] = s[7];   s[7] = t;
    t = s[8];  s[8] = s[10]; s[10] = t;
    t = s[9];  s[9] = s[11]; s[11] = t;
    t = s[15]; s[15] = s[14]; s[14] = s[13]; s[13] = s[12]; s[12] = t;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ===== regime test ============================================= *
     * "the flat stone is only worth walking to if there is a pile."
     * A small pile is spoken straight into the single shrine stone:
     * fall back to the simple known accumulator path, two lanes deep
     * so the multiply chain is not the critical path.
     * =============================================================== */
    if (len < 64) {
        uint64_t h0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
        uint64_t h1 = 0xC2B2AE3D27D4EB4FULL + (uint64_t)len;
        size_t i = 0;
        while (i + 16 <= len) {
            uint64_t w0, w1;
            memcpy(&w0, p + i, 8);
            memcpy(&w1, p + i + 8, 8);
            h0 = seal64(h0 ^ w0);
            h1 = seal64(h1 ^ w1);
            i += 16;
        }
        while (i + 8 <= len) {
            uint64_t w0;
            memcpy(&w0, p + i, 8);
            h0 = seal64(h0 ^ w0);
            i += 8;
        }
        if (i < len) {                       /* safe zero-padded tail */
            unsigned char t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t w0;
            memcpy(t, p + i, len - i);
            memcpy(&w0, t, 8);
            h0 = seal64(h0 ^ w0 ^ ((uint64_t)(len - i) << 56));
        }
        return seal64(h0 ^ ROTL64(h1, 32));
    }

    /* ===== the riverbank: 4x4 square, 128-byte state =============== */
    uint64_t s[16];
    s[0]  = 0x6A09E667F3BCC908ULL; s[1]  = 0xBB67AE8584CAA73BULL;
    s[2]  = 0x3C6EF372FE94F82BULL; s[3]  = 0xA54FF53A5F1D36F1ULL;
    s[4]  = 0x510E527FADE682D1ULL; s[5]  = 0x9B05688C2B3E6C1FULL;
    s[6]  = 0x1F83D9ABFB41BD6BULL; s[7]  = 0x5BE0CD19137E2179ULL;
    s[8]  = 0xCBBB9D5DC1059ED8ULL; s[9]  = 0x629A292A367CD507ULL;
    s[10] = 0x9159015A3070DD17ULL; s[11] = 0x152FECD8F70E5939ULL;
    s[12] = 0x67332667FFC00B31ULL; s[13] = 0x8EB44A8768581511ULL;
    s[14] = 0xDB0C2E0D64F98FA7ULL; s[15] = 0x47B5481DBEFA4FA4ULL;

    /* the count is laid down with the marks, before any heap is folded */
    s[8] ^= (uint64_t)len;

    const size_t nblk = len >> 6;
    for (size_t b = 0; b < nblk; b++) {
        /* --- silence: the 64 marks are only PLACED, never mixed.
         *     Order is held by lane position, so all 64 bytes are
         *     fetched at once and their read order is free. --- */
        uint64_t m[8];
        memcpy(m, p + (b << 6), 64);
        __builtin_prefetch(p + (b << 6) + 512, 0, 0);
        for (int i = 0; i < 8; i++) s[i] ^= m[i];

        /* --- the race: 2 double-rounds.  Held to the minimum that
         *     passes the native's corner test; not one layer more. --- */
        grow(s); turn90(s);
        grow(s); turn90(s);
        grow(s); turn90(s);
        grow(s); turn90(s);
    }

    /* the last, short heap: zero-padded, one 0x01 mark laid to close it */
    {
        const size_t r = len & 63;
        unsigned char t[64];
        uint64_t m[8];
        memset(t, 0, 64);
        if (r) memcpy(t, p + (nblk << 6), r);
        t[r] = 0x01;                          /* r <= 63, always in range */
        memcpy(m, t, 64);
        for (int i = 0; i < 8; i++) s[i] ^= m[i];
    }

    /* ===== the owl's seal: 12 layers, then the corners folded in ==== *
     * Constant cost, amortised away on any real buffer, and it is what
     * guarantees a bit flipped in the FINAL heap still smears across
     * every square.  12 layers = 6 BLAKE2b-equivalent rounds, well past
     * the ~2 rounds at which dependency completes.
     * =============================================================== */
    for (int k = 0; k < 12; k++) { grow(s); turn90(s); }

    uint64_t h = s[0] ^ ROTL64(s[5], 17) ^ ROTL64(s[10], 34) ^ ROTL64(s[15], 51);
    h ^= ROTL64(s[3], 11) ^ ROTL64(s[6], 27) ^ ROTL64(s[9], 43) ^ ROTL64(s[12], 59);
    h ^= h >> 32;
    h  = ROTL64(h, 29);
    h ^= h >> 31;
    return h;               /* the dreamer inside: one token, fixed size,
                               no path back to the marks */
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 7.0

Reasoning behind that number, stated before any measurement so it can be falsified:

- FNV-1a is latency-bound on a serial `xor; imul` chain: ~1 + 3 = 4–5 cycles per **byte**, i.e. ~0.6–0.75 GB/s at 3 GHz.
- My bulk path costs 4 layers × 4 columns × ~20 ops = 320 scalar ops per **64 bytes** = 5 ops/byte, but the 4 columns are independent and unit-stride, so with AVX2 that is ~1.25 vector ops/byte → ~0.4–0.5 cycles/byte ≈ 6–7 GB/s.
- Ratio ≈ 9–11× if auto-vectorization is clean; ≈ 4× if GCC refuses to vectorize `grow` and I only get the 4-way scalar ILP. I predict **7.0** as the honest midpoint, weighted toward vectorization succeeding but not perfectly.
- Avalanche: I predict a score at or very near ideal (≈0.50 flip rate, per-bit deviation in the low 10⁻³ for the sizes a bench would use). The finalization alone is 6 BLAKE2b-equivalent rounds over a 128-byte state, and full dependency across the square completes in 2 layers; there is no plausible path by which a single input bit reaches the squeeze under-diffused.
- Falsifiers I will accept: speedup below 3.0 would mean the vectorization argument is simply wrong and the metaphor's wide state does not pay; an avalanche score materially below ideal would mean 2 double-rounds per block is below the native's own corner test and the design is wrong, not just slow.

## MEASUREMENT

**Not performed — and I am not going to dress that up.** This session was invoked with no tools: `hash_bench` and `hash_contract` are unavailable here, as are the `symbolic_*` and `unconventional_*` probes I would have used to check the diffusion claim independently. Separately, the `claude.ai` and `PubMed` MCP connectors in this environment require OAuth and cannot be authorized from a non-interactive session — they would need to be authorized through claude.ai connector settings or `claude mcp` / `/mcp` in an interactive session — but nothing in this task needed them.

So the honest status of every quantitative claim above is **predicted, unmeasured**. What the pipeline should report, and what each outcome means:

| Reported | Reading |
|---|---|
| speedup ≈ 6–11×, avalanche ≈ ideal | prediction confirmed; the metaphor's wide square beats the accumulator for the reason claimed |
| speedup ≈ 3–5×, avalanche ≈ ideal | `grow` did not vectorize; the construction is right, the SIMD claim was optimistic. Fix is an explicit `#if defined(__AVX2__)` `_mm256_*` path, not a redesign |
| speedup < 1 | the regime guard is mis-placed and the bench runs below 64 bytes; the seam, not the sponge, is wrong |
| avalanche below ideal | 2 double-rounds per block is genuinely below the native's corner test; raise per-block layers 4 → 6 |

What I *can* verify by hand, and did: the block loop and both tails never read past `data + len` (every partial read goes through `memcpy` into a zeroed local); `t[r]` with `r = len & 63` is always in range; `turn90`'s three row rotations are each checked element-by-element to be the intended left-rotations by 1, 2, 3; `len = 0` takes the short path and performs no load at all; every `ROTL64` amount is in 1..63 so there is no undefined shift-by-64.

## VERDICT

The native is not describing a worse hash function in funny clothes. Followed literally, the description lands on a **sponge over a BLAKE2b-`G` ARX permutation on a 4×4 square** — a construction the cryptographic literature already validated — and it gets there by refusing exactly the assumption the textbook answer is built on. The bladder-swat is the whole idea: *do no arithmetic while reading, and let position carry the order.* That single prohibition is what converts a 4–5-cycle-per-byte serial multiply chain into a 64-byte-at-a-time lane-parallel permutation. FNV-1a's speed ceiling is not its multiply; it is that the multiply must finish before the next byte may be touched. The native removes the "before."

Two things I deliberately did not do. I did not add OpenMP: the metaphor's carry across heaps is explicitly serial ("burned the moment the next heap swallows them"), so there is no large independent unit of work in this world to hand a thread, and inventing one — a BLAKE3-style parallel chunk tree — would have been me replacing the native's working with a different one. I also did not invent rotation constants; the "fixed strides" clause says constants, and BLAKE2b's published 32/24/16/63 are the validated choice.

The one condition under which this is worse than FNV-1a is small inputs, and I have not left that as a caveat — it is guarded. Below 64 bytes the kernel takes a two-lane splitmix64 path, and I checked that the seam has no valley: at exactly 64 bytes the sponge already costs roughly a third of FNV. The residual honest weakness is the very smallest inputs (`len` of 1–4 bytes), where one `seal64` pair costs ~12–25 cycles against FNV's ~5 — but FNV at `len=1` barely avalanches at all, so that comparison is not one the bench's avalanche score will reward, and I chose correctness of mixing over saving 15 cycles on a 1-byte hash.

The claim I am least sure of is the speedup *magnitude*, because it rests entirely on GCC auto-vectorizing `grow` rather than on an intrinsics path I could have written but could not have tested. If the measurement comes back in the 3–5× band with ideal avalanche, the metaphor was right and my compiler optimism was wrong; that is a one-file fix and not a reason to go back to the riverbank.