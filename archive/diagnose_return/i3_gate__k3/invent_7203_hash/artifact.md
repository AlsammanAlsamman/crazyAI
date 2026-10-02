## MAPPING

### SEED 1 — the chained line sharpening the blank pillar

| world object | problem object |
|---|---|
| a mark in the pile | one byte of `data`, in its given index order |
| "leads them one by one past the knowing walls" | sequential load of the buffer; the walls = the fixed taps that know where to XOR |
| a chained prisoner | one stage of a shift register holding part of the angle |
| "does not look at the mark, only feels how it changes the angle" | the byte is never examined, compared, or multiplied — it is XORed in and the register is shifted |
| "passes that angle, not the mark, to the next" | only register state crosses stage boundaries; the data word is dead after one stroke |
| "the angle carries backward through the line" | the feedback polynomial: the top stage feeds back into low taps every stroke |
| "a mark dropped at the very start still trembles in the hand of the last" | full forward dependency — every earlier byte is still in the register |
| the blank pillar that "keeps sharpening onward as if nothing had passed" | the register retains no readable image of the data; it is not the hash |
| **concretely** | `crc32c` — a 32-stage LFSR with backward feedback, i.e. `_mm_crc32_u64`, **one instruction, no multiply** |

**Assumption broken:** *"mixing one byte requires a multiplication."* A shift register with backward feedback mixes 8 bytes per instruction with zero multiplies.

### SEED 2 — the flood that never recedes, bending anchored wires

| world object | problem object |
|---|---|
| the horizonless pit | the hash's internal state space, not externally visible |
| the river that floods every solid argument | the absorb loop; "every solid argument" = it is purely GF(2)-linear, so no algebraic invariant survives it readable |
| "rises on each pass" | one loop iteration = one pass |
| "does not recede until every mark has gone through" | the state is *never* read until `i == len` |
| "a token pulled while the water is still up is worthless, the shapes still swimming" | literally true: a mid-loop CRC lane has ~0 avalanche; it is not a usable hash |
| wires anchored "past where any traveler fused its source" | 8 fixed, non-derived seed constants (digits of π, e, √2) |
| "they do not move themselves; the flood moves them" | accumulators are updated only by data, never self-scrambled |
| "bending each wire by exactly the angle ground on that pass" | on pass *p*, wire *w* absorbs word *w* of that pass through the chain |
| eight wires, not one pillar-reading | 8 independent 32-bit lanes = 256 bits of state |

**Assumption broken:** *"the state is a single accumulator updated in place, one value"* — and secondarily *"each byte must be mixed into the running state before the next byte is read"*: eight bytes-streams are in flight at once and the state is deliberately unreadable until the flood drains.

### SEED 3 — the silhouette against the sun

| world object | problem object |
|---|---|
| the flood draining | loop exit |
| "the shadows stop their chitter / the walls are still settling" | the in-flight dependency chains retire; the lanes are tied off |
| climbing to the rim and reading once | exactly one finalize, applied once |
| "which lean, which stand straight, **which cross another**" | the projection is made of *crossings* = pairwise 64×64→128 products of wires, the only multiplications in the whole kernel |
| the silhouette: small, fixed, drawn only once | one 64-bit value from 256 bits of wire |
| "I keep nothing but that shadow-shape" | lanes, marks, floodwater all discarded |
| "the marks the prisoners forget as soon as sharpened past" | **per-mark work is deliberately weak and thrown away** |
| "I never ask the river where its source lies" | not invertible, not a PRF; one changed mark bends every wire after it and the silhouette returns a stranger's shape |

**Assumption broken:** ***"more mixing rounds always means better mixing."*** The native spends essentially *no* strength per byte — one shift-register stroke, linear, zero multiplies — and buys the entire avalanche with a single read-once projection at the end. Strength is relocated, not repeated.

## CHOSEN SEED

**SEED 3**, with SEEDs 1 and 2 as its obligatory body (the silhouette can only be read off wires that a flood bent, and the flood can only carry angles a chain ground). SEED 3 is the only one of the three that breaks the preferred assumption, and it is maximally opposite to FNV-1a/xxHash, which do the reverse: a strong multiply round *per byte*, then return the accumulator almost as-is.

## ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** Rounds are not what buys avalanche here — *one* strong projection does. Per-byte work is driven to the floor (one `crc32q`, 1 cycle throughput, no multiply) precisely because it does not need to be strong; it only needs to be order-sensitive and lossless. Also broken as a consequence: *mixing requires a multiplication* (SEED 1) and *the state is one accumulator* (SEED 2).

Where this lands, honestly: the mechanism converges on a **validated real-world construction — CityHash's `CityHashCrc128` / FarmHash's CRC variants**, which fold data with `_mm_crc32_u64` into several fixed-seeded accumulators and finish with CityHash's `Hash128to64`-style multiply fold. Per step 4 I let the metaphor *arrive* there rather than inventing a novel finisher: the finalizer is Murmur3's `fmix64` plus 128-bit multiply folds, both long-validated. This is deliberately *not* the xxHash3 shape my previous attempt collapsed into: no 256-byte secret, no per-stripe 32×32 multiplies, no periodic accumulator scramble, no overlapping final stripe.

**Two regimes, recognized in-world.** The native's pit tells him which regime he is in by *how high the water rises*: a deep flood reaches all eight anchors and bends them in wide passes; a shallow pile never floods the riverbed, so the marks are led down the line **one by one** to the nearest prisoner instead. That is the runtime check `len >= 256` / `len >= 64` / else, with the one-by-one path as the fallback — which is also the guard demanded by step 4 for the only condition under which the wide path could lose (short buffers, where 64-byte passes have no work to amortize). Second guard: no `crc32` hardware ⇒ a portable xorshift shift-register chain, still multiply-free. **No OpenMP**: there is one river and one pit in this world; splitting the flood into independent floods is not in the metaphor, and a single core already runs the chain at ~8 B/cycle, past DRAM bandwidth, so threads would buy nothing but a risk I'd then have to guard.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64))
#  include <immintrin.h>
#  define CHAIN_HW 1
#else
#  define CHAIN_HW 0
#endif

/* =====================================================================
   THE PIT.  Three objects, nothing else.

   grind()   one chained prisoner's stroke.  He never looks at the mark;
             it only changes the angle of what he was already sharpening,
             and he passes the ANGLE, not the mark, onward.  A 32-stage
             shift register whose top stage carries backward into the low
             taps on every stroke, so a mark dropped at the very start
             still trembles in the hand of the last one sharpening.
             NO MULTIPLICATION.  NO BRANCH.  NO TABLE.

   ANCHOR[]  where the eight wire shapes are fused into the riverbed,
             past where any traveler has ever fused its source.

   silhouette()  the single shadow the wires throw against the sun once
             the flood has drained: which lean, which stand straight,
             WHICH CROSS ANOTHER.  Read once.  Kept alone.
   ===================================================================== */

static const uint32_t ANCHOR[8] = {
    0x243F6A88u, 0x85A308D3u, 0xB7E15162u, 0x8AED2A6Au,
    0x9E3779B9u, 0x7F4A7C15u, 0x6A09E667u, 0xBB67AE85u
};

#if CHAIN_HW
/* the chain, in hardware: one stroke, one cycle of throughput */
#  define GRIND64(a, m)  ((uint32_t)_mm_crc32_u64((uint64_t)(uint32_t)(a), (uint64_t)(m)))
#  define GRIND8(a, b)   ((uint32_t)_mm_crc32_u8 ((uint32_t)(a), (unsigned char)(b)))
#else
/* the same chain, by hand: shifts and xors only, still no multiply */
static inline uint32_t sr_grind(uint32_t a, uint64_t m) {
    uint64_t s = ((uint64_t)a ^ m) ^ 0x9E3779B97F4A7C15ULL;
    s ^= s << 13;          /* the angle passes forward down the line  */
    s ^= s >> 7;           /* and carries backward through it         */
    s ^= s << 17;
    return (uint32_t)s ^ (uint32_t)(s >> 32);
}
#  define GRIND64(a, m)  sr_grind((uint32_t)(a), (uint64_t)(m))
#  define GRIND8(a, b)   sr_grind((uint32_t)(a), (uint64_t)(unsigned char)(b) | 0x100ULL)
#endif

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* two wires crossing: the only place in the kernel a multiply is allowed */
static inline uint64_t cross(uint64_t u, uint64_t v) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)u * (__uint128_t)v;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t ul = (uint32_t)u, uh = u >> 32, vl = (uint32_t)v, vh = v >> 32;
    uint64_t ll = ul * vl, lh = ul * vh, hl = uh * vl, hh = uh * vh;
    uint64_t mid = lh + hl + (ll >> 32);
    uint64_t lo  = (ll & 0xFFFFFFFFu) | (mid << 32);
    uint64_t hi  = hh + (mid >> 32);
    return lo ^ hi;
#endif
}

/* the token: one projection of the eight wires, drawn once, kept alone */
static uint64_t silhouette(const uint32_t *w, uint64_t len) {
    uint64_t a = (((uint64_t)w[0] << 32) ^ (uint64_t)w[1]) ^ 0x452821E638D01377ULL;
    uint64_t b = (((uint64_t)w[2] << 32) ^ (uint64_t)w[3]) ^ 0xBE5466CF34E90C6CULL;
    uint64_t c = (((uint64_t)w[4] << 32) ^ (uint64_t)w[5]) ^ 0xC0AC29B7C97C50DDULL;
    uint64_t d = (((uint64_t)w[6] << 32) ^ (uint64_t)w[7]) ^ 0x3F84D5B5B5470917ULL;

    uint64_t lean  = cross(a, b);                 /* which lean          */
    uint64_t stand = cross(c, d);                 /* which stand straight*/
    uint64_t z     = cross(lean ^ len, stand ^ 0x9E3779B97F4A7C15ULL);

    z ^= z >> 33; z *= 0xFF51AFD7ED558CCDULL;     /* the shadow settles  */
    z ^= z >> 29; z *= 0xC4CEB9FE1A85EC53ULL;
    z ^= z >> 32;
    return z;
}

/* one pass of the flood: eight wires bent, each by the angle its own
   chain ground on this pass.  Nothing is read back out. */
#define PASS(q)                                                        \
    do {                                                               \
        w0 = GRIND64(w0, ld64((q) +  0)); w1 = GRIND64(w1, ld64((q) +  8)); \
        w2 = GRIND64(w2, ld64((q) + 16)); w3 = GRIND64(w3, ld64((q) + 24)); \
        w4 = GRIND64(w4, ld64((q) + 32)); w5 = GRIND64(w5, ld64((q) + 40)); \
        w6 = GRIND64(w6, ld64((q) + 48)); w7 = GRIND64(w7, ld64((q) + 56)); \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t n = len;

    uint32_t w0 = ANCHOR[0], w1 = ANCHOR[1], w2 = ANCHOR[2], w3 = ANCHOR[3];
    uint32_t w4 = ANCHOR[4], w5 = ANCHOR[5], w6 = ANCHOR[6], w7 = ANCHOR[7];

    /* ---- REGIME ONE: the deep flood.  The water reaches every anchor,
       so it rises in wide passes and bends all eight wires at once.     */
    if (n >= 256) {
        do {
            __builtin_prefetch(p + 512, 0, 0);
            PASS(p);  PASS(p + 64);  PASS(p + 128);  PASS(p + 192);
            p += 256; n -= 256;
        } while (n >= 256);
    }
    while (n >= 64) { PASS(p); p += 64; n -= 64; }

    /* ---- REGIME TWO: the shallow pile.  The water never floods the
       riverbed, so the marks are led down the line one by one to the
       nearest prisoner.  This is also the tail of regime one.           */
    {
        uint32_t w[8];
        unsigned k = 0;
        w[0]=w0; w[1]=w1; w[2]=w2; w[3]=w3; w[4]=w4; w[5]=w5; w[6]=w6; w[7]=w7;

        while (n >= 8) { w[k]      = GRIND64(w[k],      ld64(p)); p += 8; n -= 8; k++; }
        while (n)      { w[k & 7u] = GRIND8 (w[k & 7u], *p++);           n--;    k++; }

        /* the water goes down; the angle carries backward through the
           line, so the last one sharpening holds every mark.            */
        {
            uint32_t carry = w[7];
            int i;
            for (i = 0; i < 8; i++) { carry = GRIND64(w[i], carry); w[i] = carry; }
        }

        /* climb to the rim.  Read the shadow once.  Keep nothing else.   */
        return silhouette(w, (uint64_t)len);
    }
}
```

## PREDICTION

Stated before any measurement (no measurement tool was reachable in this session — see MEASUREMENT):

**PREDICTION: speedup_vs_dp = 18**

Reasoning, so the number is falsifiable rather than decorative:
- FNV-1a's loop is a *latency* chain: `imul` is 3 cycles, one byte per link ⇒ ≈0.33 B/cycle ≈ 1.1 GB/s at 3.3 GHz.
- The flood is a *throughput* machine: `crc32q` is 1/cycle (single port) with 3-cycle latency, and 8 independent wires more than cover that latency ⇒ 8 crc per 64 bytes = **8 B/cycle ≈ 26 GB/s** L1/L2-resident; DRAM-bound (~10–15 GB/s) for buffers past L3.
- 8.0 / 0.33 ≈ 24× in cache; blended with DRAM-bound large sizes and short-buffer overhead, **~18×**.
- Avalanche: predicted **0.49–0.50** (near ideal). A 1-bit input flip changes exactly one wire by a fixed nonzero CRC pattern; the backward sweep spreads it to the later wires; then four dependent multiplies (two crossings, one crossing, `fmix64`) carry it to all 64 output bits.
- Where the prediction could be wrong: if the harness weights very short keys heavily, the shallow path's ~50-cycle fixed cost (chain + sweep + 4-multiply read) compresses the win toward **2–4×**, since FNV only costs ~3 cycles/byte there.

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were declared available for this task but no tool channel exists in this session, so I ran nothing. I will not dress up an estimate as a measurement. What I can state without a harness, and what the pipeline should check against the prediction above:

| quantity | status |
|---|---|
| throughput vs FNV-1a baseline | **unmeasured**; predicted ≈18× (24× in-cache, 2–4× worst case on short keys) |
| avalanche score | **unmeasured**; predicted 0.49–0.50 |
| `len == 0` | reasoned: returns a fixed nonzero constant (anchors swept and projected) |
| length sensitivity | reasoned: `len` is folded into the third crossing, so `"abc"` ≠ `"abc\0"` |
| contract | signature is byte-identical to `uint64_t kernel(const unsigned char *data, size_t len)`; `restrict` is applied to a local copy only |
| portability | compiles without SSE4.2 and without `__int128` via the two fallback paths |

If the measured avalanche comes back low (say <0.45), the honest diagnosis is pre-stated: the absorb phase is entirely GF(2)-linear, so *all* nonlinearity lives in `silhouette()`, and the fix would be inside that one read — not by adding rounds to the loop, which is exactly what SEED 3 forbids.

## VERDICT

The core of this kernel **is** the native's machine, not a textbook hash wearing its vocabulary:

- **Per-mark work is a shift register, not a multiplier.** One `crc32q` per 8 bytes. Zero multiplications anywhere in the absorb loop — this is the literal "he does not look at it, only feels how it changes the angle." My previous attempt's 32×32 multiplies per stripe are gone.
- **The state is eight wires anchored at fixed points, bent only by the flood, never read mid-flood.** No self-scramble, no secret table, no mid-loop folding. The lanes genuinely are worthless as a hash until the end, exactly as the native warns.
- **Exactly one projection.** Four dependent multiplies total, once per call, reading all eight wires through crossings. All avalanche comes from there.
- **Both regimes are in-world and runtime-selected** by how high the water rises, with the one-by-one path as fallback.

Honest statement of what this mechanism cannot do, and what I did about each:

1. **It is not adversary-resistant.** The whole absorb phase is linear over GF(2) with 256 bits of state, so a motivated attacker can solve for colliding inputs by linear algebra regardless of how strong the silhouette is. This is inherent to "per-mark work deliberately weak," is equally true of `CityHashCrc128`, and is *not* fixable by adding sweeps (they are linear too). I did **not** paper over it by smuggling a multiply back into the loop, because that would delete the seed. Not a cryptographic or HashDoS-hardened hash.
2. **Short-buffer overhead** — the one condition where the wide path loses. Guarded: `len >= 256` → `len >= 64` → one-by-one chain, with the short path never touching a 64-byte pass.
3. **No `crc32` instruction** → portable xorshift shift-register chain, same mechanism, ~4–5 B/cycle instead of 8.
4. **Thread parallelism deliberately omitted**, so there is no unguarded risk to address: one river, one pit, and a single core already outruns DRAM.
5. **Throughput ceiling.** `crc32q` is one port, 8 B/cycle. `pclmulqdq`-based folding would reach 16 B/cycle — but a carry-less multiply *is* a multiply, which SEED 1 forbids. I accepted being ~2× slower than the fastest possible x86 fold rather than breaking the mechanism to win a benchmark. That is the trade, stated plainly rather than buried.

The one place the metaphor and the hardware disagree: "the angle carries backward through the line" happens continuously in the world. In the kernel it happens at two grains — inside every stroke (the CRC feedback taps, which is literally a backward carry on every bit of every mark) and once across the eight wires at recession. I rejected a per-block cross-wire sweep after costing it: 8 serial strokes per 512 bytes is a 27% throughput loss, and since the absorb phase is linear either way it buys no structural strength, only spreading that the silhouette's crossings already provide.