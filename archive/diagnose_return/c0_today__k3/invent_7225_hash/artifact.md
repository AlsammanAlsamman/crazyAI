## MAPPING

**SEED 1 — the unbroken pour into the impure cup**

| world object | problem object |
|---|---|
| a mark in the pile | one input byte `data[i]` |
| the pile | the buffer, length `len` |
| the cup that is never pure / "sickened" (cut) wine | a vessel whose content is pre-mixed garbage with memory: a CRC-32C register + its 256-entry polynomial table (software) or the fused `crc32` instruction (hardware). "Pure wine holds no memory" = an identity/clean table mixes nothing; a *cut* table carries every previous drop |
| "a pull on the cup" | one `crc32` step consuming the mark(s) |
| "taste the cup's color" | read the 32-bit CRC register |
| "that color becomes the color waiting for the next pull" | the CRC value is the seed operand of the next CRC step — a true serial data dependency |
| "a single unbroken pour, no mark judged alone or twice" | exactly one pass, each byte touched once, no lanes, no re-read |
| **assumption broken** | **A3: "mixing one byte requires a multiplication."** A pull is a table lookup / carry-less polynomial division. Zero multiplies in the per-byte loop. |

**SEED 2 — the seven organs**

| world object | problem object |
|---|---|
| "the cup's last color" | the CRC + body left at end of the pour |
| "I don't stop at the cup" | the loop state is *not* the hash; it is a pre-hash |
| "carry it through my own organs" | a finalizer applied **once per call**, outside the loop |
| "bent seven times, once per organ" | 7 bending steps, each with its *own* character (its own shift, its own prime) |
| "each bending throws away half of what came before" | `x ^= x >> 32`-class xorshift: half the register is discarded |
| "keeping only what refuses to sit still" | XOR keeps only bit positions where the two halves disagree |
| "the body carries a sickness until every organ compensates" | the diffusion of one changed bit is completed *after* the data is gone |
| "the garden door / the nightingale's last note / trade beauty for beauty exactly" | the strict-avalanche criterion: exactly half the output bits must flip. The organ count is decided by this test, not by appetite |
| **assumption broken** | **A5: "more mixing rounds always means better mixing."** Two things break it. (a) The *per-byte* rounds are reduced to almost nothing and the avalanche is bought once, at the end — fewer total rounds, better avalanche. (b) Taken literally, "throw away half" repeated identically is an **involution**: `x ^= x>>32` applied twice is the identity. Seven identical bendings = one bending (or none). The door, not the count, is what makes mixing good. |

**SEED 3 — change one mark, pour again, resemblance condemns**

| world object | problem object |
|---|---|
| "change one mark, anywhere, even the quietest one" | flip one input bit — including a leading `0x00`, the classic blind spot |
| "pour the whole thing through again from the first cup" | recompute the whole hash, no incremental shortcut |
| "if the new token still resembles the old one's shape" | output Hamming distance far from 32/64 |
| "throw the whole method away" | the acceptance gate is a measured differential test |
| "a door someone forgot to turn" | a hash that is really a permutation of a too-small state |
| **assumption broken** | A5 again, but as *epistemology, not mechanism*: rounds are not the metric, one-bit-flip dissimilarity is. It is a test harness, not a kernel. |

## CHOSEN SEED

**SEED 2 (the seven organs).** It is the only seed that breaks the preferred assumption *as a mechanism* rather than as a test, and it is maximally different from the known way: FNV/xxHash spend their mixing strength **per byte** (one multiply per byte, inside the dependency chain); the organs spend it **once per buffer**, outside the chain. SEED 1 is retained as its necessary premise — "I don't stop at the cup" cannot be implemented without a cup — and SEED 3 is retained as the acceptance gate. SEED 3 alone is not a kernel.

## ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** The kernel does the *weakest useful* thing per byte (one carry-less `crc32` pull + one add/rotate, no multiply) and then exactly **seven** bendings once at the end. Total multiplies for an N-byte buffer: **3**, versus FNV's **N**. The per-byte work was cut to the bone and the avalanche got *better*, because avalanche is a property of the last 20 cycles, not of the first N.

Secondary break (SEED 1): **"mixing one byte requires a multiplication"** — the loop contains no multiply at all.

**Arriving at the validated technique, not inventing one (step 4).** The literalisation lands exactly on two real, validated constructions rather than novelties: the cup = **CRC-32C via SSE4.2 `_mm_crc32_u64`** (the standard hardware-CRC fast path), and the seven organs = **Murmur3's `fmix64` / `splitmix64` finaliser family**, extended by one organ. I did not invent a mixer; the metaphor walked into the known-good one. I did *not* take the one place the metaphor could have pushed me past validated ground: three parallel cups + `pclmulqdq` CRC-combine would be faster, but it breaks "a single unbroken pour" and reintroduces the very multiplication the cup exists to avoid.

**Regime recognition inside the metaphor (step 5).** The known-way section describes two regimes (bulk stripes vs. short/tail inputs), so the native must recognise both at runtime:
- *How many marks remain in the pile?* A **draught** (8 marks in one pull) can only be poured while eight marks remain; below that you pour **drops** (one mark per pull). → `len>=32` unrolled, `>=8` single-word, `<8` byte-wise. Same cup, same organs, both regimes.
- *Which tavern am I in?* Where the cup is fused into the wall (hardware `crc32`), taste once per drop. Where it is a clay cup you must stir yourself (no SSE4.2/CRC32 extension), taste **once per draught** instead of once per drop and let the cup's body carry the mass — otherwise the clay cup is slower than the known way, which is exactly the self-named risk the rules forbid shipping unguarded.
- **No thread parallelism.** The metaphor's unit of work is one pour through one cup; it is inherently serial, and splitting it needs carry-less multiplication to recombine. At benchmark sizes the serial pour already runs at memory-ish speed. I state this as a refusal, not an oversight.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- which tavern am I in? (regime 2: cup fused into the wall, or clay) ---- */
#if defined(__SSE4_2__)
  #include <nmmintrin.h>
  #define CUP_HW_X86 1
#elif defined(__ARM_FEATURE_CRC32)
  #include <arm_acle.h>
  #define CUP_HW_ARM 1
#endif

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

#if !defined(CUP_HW_X86) && !defined(CUP_HW_ARM)
/* the clay cup: 256 pre-mixed colors of sickened wine (CRC-32C, reflected) */
static uint32_t cup_tbl[256];
static int cup_ready = 0;
static void cup_fill(void) {
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? ((c >> 1) ^ 0x82F63B78u) : (c >> 1);
        cup_tbl[i] = c;
    }
    cup_ready = 1;              /* idempotent: a benign race recomputes the same table */
}
#endif

/* one pull, one drop */
static inline uint32_t pull_drop(uint32_t c, unsigned char b) {
#if defined(CUP_HW_X86)
    return _mm_crc32_u8(c, b);
#elif defined(CUP_HW_ARM)
    return __crc32cb(c, b);
#else
    return (c >> 8) ^ cup_tbl[(c ^ b) & 0xFFu];
#endif
}

/* one pull, a draught of eight marks */
static inline uint32_t pull_draught(uint32_t c, uint64_t w) {
#if defined(CUP_HW_X86)
    return (uint32_t)_mm_crc32_u64((uint64_t)c, w);
#elif defined(CUP_HW_ARM)
    return __crc32cd(c, w);
#else
    /* clay cup: taste once per draught, not once per drop, so the clay tavern is
       never slower than the known way. The cup's body still holds every mark. */
    uint32_t f = (uint32_t)w ^ (uint32_t)(w >> 32);
    c = (c >> 8) ^ cup_tbl[(c ^ f) & 0xFFu];
    c = (c >> 8) ^ cup_tbl[(c ^ (f >> 11)) & 0xFFu];
    return c;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    size_t n = len;

    /* the cup is never pure: it starts already cut, and the pile's size is itself
       a mark -- so a pile of leading silences (0x00...) cannot pass unnoticed. */
    uint32_t color = 0x9E3779B9u ^ (uint32_t)len;            /* what the tongue tastes */
    uint64_t body  = 0x6A09E667F3BCC909ULL ^ (uint64_t)len;  /* what the cup holds    */

#if !defined(CUP_HW_X86) && !defined(CUP_HW_ARM)
    if (!cup_ready) cup_fill();
#endif

    /* regime 1a: four draughts while thirty-two marks remain */
    while (n >= 32) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, p,      8);
        memcpy(&w1, p +  8,  8);
        memcpy(&w2, p + 16, 8);
        memcpy(&w3, p + 24, 8);
        body = rotl64(body + w0, 29); color = pull_draught(color, w0 ^ body);
        body = rotl64(body + w1, 29); color = pull_draught(color, w1 ^ body);
        body = rotl64(body + w2, 29); color = pull_draught(color, w2 ^ body);
        body = rotl64(body + w3, 29); color = pull_draught(color, w3 ^ body);
        p += 32; n -= 32;
    }
    /* regime 1b: one draught while eight marks remain */
    while (n >= 8) {
        uint64_t w; memcpy(&w, p, 8);
        body = rotl64(body + w, 29); color = pull_draught(color, w ^ body);
        p += 8; n -= 8;
    }
    /* regime 1c: the dregs, drop by drop */
    while (n) {
        unsigned char b = *p++; n--;
        body  = rotl64(body ^ (uint64_t)b, 7) + 0x9E3779B97F4A7C15ULL;
        color = pull_drop(color, b);
    }

    /* the cup's last color, carried out of the tavern */
    uint64_t x = body ^ (((uint64_t)color << 32) | (uint64_t)(color ^ 0x5BF03635u));

    /* --- the seven organs: each throws away half, keeps what refuses to sit still,
           and each bends with its own character (identical organs would undo
           each other -- an involution -- which is the whole point) --- */
    x ^= x >> 33;  x *= 0xFF51AFD7ED558CCDULL;   /* 1, 2 */
    x ^= x >> 29;  x *= 0xC4CEB9FE1A85EC53ULL;   /* 3, 4 */
    x ^= x >> 32;  x *= 0x9E3779B97F4A7C15ULL;   /* 5, 6 */
    x ^= x >> 31;                                /* 7: the garden door */
    return x;
}
```

Four design-stage revisions, all folded in before any measurement and all honestly un-measured: (1) a single CRC chain only — no 3-lane + `pclmulqdq` recombine, because it needs the forbidden multiply; (2) the **body** added as the cup's second aspect (volume as well as colour), because a bare CRC chain carries only 32 bits of state — "a door someone forgot to turn," exactly SEED 3's condemnation — and the body's add/rotate chain (2 cycles) hides free inside the CRC's 3-cycle latency shadow; (3) nonzero cup init + `len` folded into both aspects, to kill the leading-zero blind spot that SEED 3's "even the quietest mark" names directly; (4) the clay-cup path retuned to one taste per draught so the no-hardware regime cannot regress below the known way.

## PREDICTION

Stated before any measurement (no bench tool was available in this session, so this prediction is pure cycle accounting, not a post-hoc fit):

**PREDICTION: speedup_vs_dp = 10.0**

Reasoning: FNV-1a's critical path is `xor` (1) + `imul` (3) per byte ≈ 4 cycles/byte = 0.25 B/cycle. The pour's critical path is one `crc32` (latency 3, throughput 1) per 8 bytes = 2.67 B/cycle; the body chain is 2 cycles and hides underneath. Ratio ≈ 10.7×, call it 10 after loop overhead. The seven organs cost ~20 cycles once, so at `len = 8` the advantage shrinks to roughly 1.4× and at `len ≥ 256` it should exceed 10×. Avalanche predicted **0.49–0.51**: a single input bit flip always changes the CRC (CRC detects single-bit errors for any offset) *and* always changes the body (add/rotate is injective), and `fmix64`-family organs are validated to ~0.5 bias-free.

## MEASUREMENT

**Not measured.** No tools were available in this session, so `hash_bench` and `hash_contract` were never run; I will not report numbers I did not take. What is above is a prediction and nothing more.

What would falsify it, stated so the pipeline's run is a real test:
- **avalanche < 0.47** → the seven organs are not reaching the garden door, and by SEED 3's own rule the method is condemned, not patched. The first thing to check is whether a 32-bit colour plus a linear-ish body is leaking structure past `fmix64`.
- **speedup < 3×** → either `-march=native` did not enable SSE4.2 (check for the clay path) or `crc32` throughput, not latency, is the limit.
- **speedup < 1× at small lengths** → the seven-organ tax dominates; the fix is fewer organs (the door, not the count, decides), not more.

## VERDICT

The metaphor is a precise description of **hardware-CRC pour + `fmix64` finaliser**, and it got there by its own logic: the impure cup is a polynomial table, the pull is `crc32`, the unbroken pour is the serial seed dependency, the seven organs are a one-shot xorshift-multiply finaliser, the garden door is the strict-avalanche criterion. Two real assumptions fall: no multiply per byte, and *fewer* total mixing rounds producing *better* avalanche — N multiplies become 3.

Honest limits, each addressed rather than left as a caveat:
- *"Only helps if the pile is big"* — guarded: three length regimes (32/8/1) in one pass, no path that reads the buffer twice, and the byte path is still cheaper than FNV's per-byte multiply.
- *"Could be slower where the cup is clay"* — guarded: the no-hardware path tastes once per draught instead of once per drop, so it stays ahead of the known way instead of falling behind it.
- *"A 32-bit cup is a door someone forgot to turn"* — guarded: the cup's body carries 64 bits and every mark, and its add-carries make the pour nonlinear, so the state entering the organs is ≥64 bits of data-dependent entropy, not 32.
- *Unfixed and stated plainly:* this is **not** cryptographic and not adversary-resistant. The pour is close to linear; someone who wants collisions can solve for them. That is also true of FNV and xxHash, and it is not what was asked, but I am not going to pretend the knot would hold against someone deliberately pulling threads rather than one thread by accident.
- *Refused:* thread parallelism. One cup, one pour. The metaphor forbids it and the cycle count does not need it.