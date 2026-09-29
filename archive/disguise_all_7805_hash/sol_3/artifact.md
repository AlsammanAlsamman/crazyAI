## APPROACH

**Mapping the taffy mechanism onto the real problem**

| Boardwalk | Hash kernel |
|---|---|
| The lump of taffy | the hash state |
| One drop poured at a time | one input word consumed per mixing step |
| "always the identical twist after every drop" | FNV‑1a's single `xor; imul` step, repeated identically for every byte |
| "more folds always help" (the mentor's bad advice) | more multiply‑xor‑shift rounds per byte → in fact pure latency cost, and repeated identical rounds converge toward a fixed statistical shape ("muddy brown") without adding distinguishing power |
| **the fix: a small set of *different* hand motions, cycled by position in the card** | **a small set of four *different*, structurally distinct mixing primitives, cycled by word position in the buffer** |
| "each motion disturbs the taffy in a different way" | each primitive uses a different algebraic family: xor‑rotate‑multiply, add‑xorshift‑rotate (no multiply at all), rotate‑both‑then‑multiply, rotate‑add‑xorshift (no multiply) |
| "so she never needs to do any motion more than once per drop" | exactly **one** primitive application per 8‑byte word — single pass, no extra rounds |
| "variety does the work extra repetition was supposed to do" | diffusion comes from primitive heterogeneity + one strong terminal avalanche, not from round count |

The four motions are:

```
TWIST  (a,v): a = rotl(a ^ v, 31) * KA
STRETCH(a,v): t = a + v; t ^= t>>29; a = rotl(t,21) + KB      (no multiply)
BRAID  (a,v): a = (rotl(a,27) ^ rotl(v,17)) * KC
SHEAR  (a,v): t = rotl(a,41) + (v ^ KD); a = t ^ (t>>37)      (no multiply)
```

Two design consequences that are *what makes this fast*, and both fall straight out of the mechanism rather than being bolted on:

1. **Because the motion depends on position, different positions touch the state differently and independently** — so the state is carried as several lanes (8 for the 64‑byte main block, 4 for medium inputs), motions cycled `TWIST, STRETCH, BRAID, SHEAR, TWIST, STRETCH, BRAID, SHEAR` across the block. This breaks the listed assumptions "single accumulator" and "each byte mixed before the next is read" without changing the mechanism. It replaces FNV's 4‑cycle serial `xor→imul` dependency per *byte* with ~11 cycles of fully overlapped work per *64 bytes*.
2. **Two of the four motions use no multiplication at all**, which directly kills the assumption "mixing one byte requires a multiplication" and halves pressure on the single `imul` port — variety buying speed, not just quality.

**Avalanche is argued, not hoped for.** Every motion is a bijection in `a` for fixed `v` *and* injective in `v` for fixed `a` (each is a composition of xor‑with‑constant, add, rotate, `x ^ (x>>k)`, and multiply by an odd constant — all bijections). Therefore a single flipped input bit, which lands in exactly one word and hence exactly one lane, *provably cannot be cancelled*: that lane's accumulator differs at the end of the pass. Lanes are merged with xor / rotate / 128‑bit mulfold plus additive rotated terms (so a lone lane difference survives merging), `len` is folded in, and the result passes through the well‑tested `fmix64` finalizer, which avalanches any nonzero difference — including a lone bit‑63 difference — across all 64 output bits. Tail bytes are covered by a single overlapping last‑8‑byte read, so **no read ever leaves `[data, data+len)`**.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define KA 0x9E3779B185EBCA87ULL
#define KB 0xC2B2AE3D27D4EB4FULL
#define KC 0x165667B19E3779F9ULL
#define KD 0x27D4EB2F165667C5ULL
#define KE 0x85EBCA77C2B2AE63ULL
#define KF 0x9FB21C651E98DF25ULL
#define KG 0xD6E8FEB86659FD93ULL
#define KH 0xA0761D6478BD642FULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rd4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* terminal avalanche (fmix64) */
static inline uint64_t fmix(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}

#if defined(__SIZEOF_INT128__)
static inline uint64_t mulfold(uint64_t x, uint64_t y) {
    __uint128_t p = (__uint128_t)x * (__uint128_t)y;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
}
#else
static inline uint64_t mulfold(uint64_t x, uint64_t y) {
    uint64_t xl = (uint32_t)x, xh = x >> 32, yl = (uint32_t)y, yh = y >> 32;
    uint64_t ll = xl * yl, lh = xl * yh, hl = xh * yl, hh = xh * yh;
    uint64_t cross = (ll >> 32) + (uint32_t)lh + hl;
    uint64_t hi = hh + (cross >> 32) + (lh >> 32);
    uint64_t lo = (cross << 32) | (uint32_t)ll;
    return lo ^ hi;
}
#endif

/* --- the four different "hand motions": one application per word, cycled by position --- */
#define TWIST(a, v)   do { (a) = rotl64((a) ^ (v), 31) * KA; } while (0)
#define STRETCH(a, v) do { uint64_t t_ = (a) + (v); t_ ^= t_ >> 29; \
                           (a) = rotl64(t_, 21) + KB; } while (0)
#define BRAID(a, v)   do { (a) = (rotl64((a), 27) ^ rotl64((v), 17)) * KC; } while (0)
#define SHEAR(a, v)   do { uint64_t t_ = rotl64((a), 41) + ((v) ^ KD); \
                           (a) = t_ ^ (t_ >> 37); } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (len <= 16) {                                  /* short cards: one shaped fold */
        if (len == 0) return fmix(KA);
        if (len < 4) {
            uint64_t c = ((uint64_t)data[0] << 16)
                       | ((uint64_t)data[len >> 1] << 8)
                       | ((uint64_t)data[len - 1])
                       | ((uint64_t)len << 24);
            return fmix((c ^ KB) * KE);
        }
        if (len < 8) {
            uint64_t w = (rd4(data) << 32) | rd4(data + len - 4);
            return fmix((w ^ KB ^ ((uint64_t)len * KD)) * KE);
        }
        if (len == 8) return fmix((rd8(data) ^ KB) * KE);
        {
            uint64_t lo = rd8(data) ^ KB;
            uint64_t hi = rd8(data + len - 8) ^ KC;
            uint64_t m  = mulfold(lo, hi) + rotl64(lo, 31) + rotl64(hi, 47)
                        + (uint64_t)len * KD;
            return fmix(m);
        }
    }

    {
        const unsigned char *p = data, *end = data + len;
        uint64_t a0 = KA ^ (uint64_t)len, a1 = KB, a2 = KC, a3 = KD;

        if (len > 64) {                               /* main pass: 64 B, 8 positions */
            uint64_t a4 = KE, a5 = KF, a6 = KG, a7 = KH;
            do {
                TWIST  (a0, rd8(p     ));
                STRETCH(a1, rd8(p +  8));
                BRAID  (a2, rd8(p + 16));
                SHEAR  (a3, rd8(p + 24));
                TWIST  (a4, rd8(p + 32));
                STRETCH(a5, rd8(p + 40));
                BRAID  (a6, rd8(p + 48));
                SHEAR  (a7, rd8(p + 56));
                p += 64;
            } while ((size_t)(end - p) >= 64);
            a0 ^= rotl64(a4, 29);
            a1 ^= rotl64(a5, 31);
            a2 ^= rotl64(a6, 33);
            a3 ^= rotl64(a7, 35);
        }

        while ((size_t)(end - p) >= 32) {              /* 4 positions */
            TWIST  (a0, rd8(p     ));
            STRETCH(a1, rd8(p +  8));
            BRAID  (a2, rd8(p + 16));
            SHEAR  (a3, rd8(p + 24));
            p += 32;
        }
        {
            unsigned i = 0;
            while ((size_t)(end - p) >= 8) {            /* keep cycling the motions */
                uint64_t v = rd8(p);
                switch (i & 3u) {
                    case 0:  TWIST  (a0, v); break;
                    case 1:  STRETCH(a1, v); break;
                    case 2:  BRAID  (a2, v); break;
                    default: SHEAR  (a3, v); break;
                }
                p += 8; ++i;
            }
        }
        if (p != end) {                                 /* len >= 17 here: in bounds */
            uint64_t v = rd8(end - 8);
            BRAID(a1, v);
        }

        {
            uint64_t m = mulfold(a0 ^ KE, a1 ^ KF) + mulfold(a2 ^ KG, a3 ^ KH)
                       + rotl64(a0, 13) + rotl64(a1, 27)
                       + rotl64(a2, 41) + rotl64(a3, 55)
                       + (uint64_t)len * KD;
            return fmix(m);
        }
    }
}
```

## PREDICTION

PREDICTION: 12

(Interpretation: speedup factor versus the FNV‑1a baseline on the benchmark's size mix. Reasoning behind the number: FNV‑1a is latency‑bound at ≈4 cycles/byte (`xor`+`imul` serial chain) ≈ 0.25 B/cycle. The main pass is ≈43 uops and only 4 multiplies per 64 bytes across 8 independent lanes → ≈11 cycles/64 B ≈ 5.8 B/cycle, i.e. ~23× on L1/L2‑resident data, falling toward ~10× if DRAM bandwidth caps it, and only ~3–4× if the harness is dominated by very short keys. 12 is the blend I'd bet on.)

## MEASUREMENT

**Not measured — reported plainly as a gap, not glossed over.** This session was invoked with no tool access (`hash_bench` / `hash_contract` were listed but are not callable here; the MCP connectors present are unrelated and unauthorized). I therefore made **zero** of the four allowed improve‑and‑measure iterations, and the prediction above stands unverified by me. The pipeline's own compile‑and‑measure step is the first real measurement this artifact will receive.

What I *did* verify by hand, since I could not verify it by running:

- **Bounds:** every read is inside `[data, data+len)`. `len<4` uses only `data[0]`, `data[len>>1]`, `data[len-1]`; `4..7` uses two 4‑byte reads at `0` and `len-4`; `≥8` uses 8‑byte reads at `0` and `len-8`; the tail overlap read `end-8` is only reached when `len ≥ 17`.
- **Coverage:** every byte enters the state in every branch (full blocks, then 32‑byte groups, then 8‑byte words, then one overlapping final word).
- **Rotate legality:** all rotate amounts are in 1..63 (31, 21, 27, 17, 41, 29, 31, 33, 35, 13, 27, 41, 55, 47); 37 and 29 and 33 appear only as shifts.
- **No‑cancellation:** all four motions are bijective in the accumulator and injective in the input word, every 8‑byte word feeds exactly one lane, and lane merging keeps a lone lane difference alive into `fmix64`. `len` is mixed in, so `"abc"` and `"abc\0"` cannot collide by construction of the length term.
- **Determinism / portability:** no OpenMP, no UB, unaligned reads via `memcpy`, `__int128` used only behind `__SIZEOF_INT128__` with a scalar fallback.

If the measurement comes back below prediction, the honest next moves *within this mechanism* (not replacing it) are, in order: widen the main pass to 128 bytes with 16 position‑cycled lanes; drop the two remaining multiplies in the merge for short inputs; and if large buffers turn out to be DRAM‑bound, add an OpenMP split whose chunk boundaries are a pure function of `len` so determinism survives.

## VERDICT

The mechanism translates cleanly and, I believe, favorably: **heterogeneous position‑cycled mixing primitives, applied exactly once per word, replace FNV's identical‑round‑per‑byte serial chain** — and the "variety instead of repetition" idea is what licenses the single pass and the multiply‑free motions that make it fast. Avalanche is supported by a structural no‑cancellation argument plus a standard strong finalizer, which is stronger ground than "it looked random."

But the claim is untested. **I cannot report a verdict on speed, only on design.** Predicted 12× versus FNV‑1a; measured: nothing, because no benchmark tool was reachable in this session. Treat the artifact as correct‑by‑construction and fast‑by‑analysis, and the 12 as a bet still open.