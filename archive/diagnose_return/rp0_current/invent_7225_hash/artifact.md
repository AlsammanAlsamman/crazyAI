## MAPPING

**SEED 1 — the sickened cup, one unbroken pour**

| World object | Problem object |
|---|---|
| the pile of marks | the byte buffer `data[0..len)` |
| one mark | one input byte (or one handful = one 8-byte word) |
| the cup that is never pure / the tavern's sickened wine | a fixed **impure lookup structure** — a CRC-32C polynomial / its 256-entry table; the hardware `crc32` unit is the tavern's own cup |
| "pure wine holds no memory" | a pure permutation of the byte carries nothing forward — an identity-like mix |
| "wine that is cut remembers every drop added to it" | the drop is **added to** the cup (raw word XORed into the 64-bit state) *and* the cup is drawn from |
| a *pull* on the cup | one `crc32` step — indexing the impurity. **Not a multiply.** |
| tasting the cup's color, that color waiting for the next pull | the 64-bit accumulator is the seed of the next pull: strict loop-carried dependency |
| single unbroken pour, "no mark judged alone or twice" | one chain, one pass, no overlapping re-reads, no parallel accumulators |

Breaks: **"mixing one byte requires a multiplication."** (Affirms in-order single-pass, affirms single accumulator.)

**SEED 2 — seven organs**

| World object | Problem object |
|---|---|
| the cup's last color | the accumulator at end of buffer |
| my own organs (seven) | a 7-step finalizer applied once, O(1), off the hot loop |
| a bending | one mixing operation |
| "throws away half of what came before" | either `x >> k` (the shifted-out half is destroyed) or `x *= C` (the **upper 64 bits of the 128-bit product are destroyed**) |
| "keeping only what refuses to sit still" | `x ^= …` — keep only the bits that differ |
| the garden door / the nightingale's last note / trade beauty for beauty exactly | the avalanche criterion: exactly ½ of output bits flip |
| "if not exact, I bend again" | round count is decided by measurement, not by doctrine |

Breaks: **"more mixing rounds always means better mixing"** — but only weakly, and in the *opposite* direction from the assumption's spirit: the native makes rounds *destructive* and then makes an external door, not the round count, the arbiter. It does not assert that fewer rounds can be better.

**SEED 3 — one mark changed, condemn the method**

| World object | Problem object |
|---|---|
| changing one mark, even the quietest | flip one bit, including in the last/first byte |
| pouring the whole thing again from the first cup | recompute `kernel` end to end |
| "resembles the old token's shape" | output Hamming distance far from 32/64 |
| "a knot that remembers its old shape is just a door someone forgot to turn" | a hash with poor avalanche is an identity function in disguise |

Breaks: none of the five kernel assumptions. This is a **test protocol**, not a mechanism — it is exactly what `hash_bench`'s avalanche score does.

## CHOSEN SEED

**SEED 1**, with SEED 2 as its mandatory finalizer and SEED 3 as the acceptance gate.

Plainly: **no seed cleanly breaks "more mixing rounds always means better mixing."** SEED 2 only bends toward it (destructive rounds, empirical stopping) and SEED 3 is a test, not a mechanism. So I fall back to the most literal seed, which is also the one most different from FNV-1a/xxHash: SEED 1 removes the multiply from the per-byte path entirely and replaces it with a draw from an impure cup.

Crucially, step 4 applies here in my favour: "a pull on an impure cup, chained" is not something I have to invent. It is **CRC-32C** — and on the target compile line (`-march=native`) the tavern's cup is a *hardware instruction*, `_mm_crc32_u64`, 3-cycle latency, 1/cycle throughput. Hardware-CRC accumulation plus a strong nonlinear finalizer is a validated production technique (FarmHash/CityHashCrc use exactly this shape). The metaphor arrives at it; I did not smuggle it in.

## ASSUMPTION BROKEN

**"mixing one byte requires a multiplication."** There is not a single multiply in the pour. All 64-bit multiplies live in the seven organs — three of them, executed **once per call**, not once per byte.

Two secondary consequences I keep faithful rather than optimizing away:

- The native's "single unbroken pour" **forbids** the standard xxHash trick of 4 independent lanes. I therefore did **not** use multiple accumulators and did **not** use OpenMP. A 3-lane split cup would be ~2.5× faster on multi-MB buffers. The native's pour is one pour; I report the cost rather than hide it.
- "No mark judged alone or twice" **forbids** the usual overlapping-tail read (`load8(p+len-8)`), so the tail is handled by a real 4-byte pull plus ≤3 single drops.

**Regime recognition (step 5).** The known-way section spans two regimes (per-byte in-order streaming vs. bulk), so the native weighs the pile in his palm first:

- **palm-sized pile (`len ≤ 8`)** — never touches the cup at all; the marks are held all at once and passed straight through the organs. This is the *guard* for my own stated risk (below).
- **heavy pile (`len ≥ 32`)** — handfuls of thirty-two: four chained pulls per fold, which hides the fold's latency behind the CRC chain (13 cycles / 32 B instead of 4 cycles / 8 B).
- **middling pile** — handfuls of eight, then a handful of four, then stragglers drop by drop.

**The risk I must guard, named up front:** the seven organs cost ~17 cycles unconditionally. For `len ≤ 4`, FNV-1a costs ~16 cycles total, so an unguarded version of my kernel would be *slower than the known way*. The `len ≤ 8` palm path is that guard: it skips the cup, the loop setup and all branching, leaving only the irreducible finalizer. (FNV-1a at `len ≤ 4` is faster only because it has no finalizer at all and therefore fails SEED 3's garden door outright — it is faster and broken.)

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(__amd64__))
  #include <immintrin.h>
  #define CUP_HW 1
#else
  #define CUP_HW 0
#endif

/* ---- the cup that is never pure -------------------------------------- */
/* A pull is a CRC-32C step: an index into a fixed impurity.  No multiply. */

#if CUP_HW
  /* the tavern's own cup: hardware, 3-cycle latency, 1/cycle throughput */
  #define CUP8(c,b)  ((uint64_t)_mm_crc32_u8 ((unsigned int)(uint32_t)(c),(unsigned char)(b)))
  #define CUP32(c,v) ((uint64_t)_mm_crc32_u32((unsigned int)(uint32_t)(c),(uint32_t)(v)))
  #define CUP64(c,v) ((uint64_t)_mm_crc32_u64((uint64_t)(c),(uint64_t)(v)))
#else
  /* the cup I carry: 256 pre-poisoned colors, same polynomial */
  static uint32_t cup_tab[256];
  static int cup_ready = 0;
  static void cup_fill(void) {
      unsigned i; int k;
      for (i = 0; i < 256; ++i) {
          uint32_t c = (uint32_t)i;
          for (k = 0; k < 8; ++k)
              c = (c >> 1) ^ (0x82F63B78u & (uint32_t)(0u - (c & 1u)));
          cup_tab[i] = c;                      /* idempotent: benign race */
      }
      cup_ready = 1;
  }
  static inline uint64_t cup8_(uint64_t c, unsigned char b) {
      uint32_t x = (uint32_t)c;
      return (uint64_t)(cup_tab[(x ^ b) & 0xFFu] ^ (x >> 8));
  }
  static inline uint64_t cup32_(uint64_t c, uint32_t v) {
      int i; for (i = 0; i < 4; ++i) { c = cup8_(c, (unsigned char)v); v >>= 8; } return c;
  }
  static inline uint64_t cup64_(uint64_t c, uint64_t v) {
      int i; for (i = 0; i < 8; ++i) { c = cup8_(c, (unsigned char)v); v >>= 8; } return c;
  }
  #define CUP8(c,b)  cup8_ ((c),(unsigned char)(b))
  #define CUP32(c,v) cup32_((c),(uint32_t)(v))
  #define CUP64(c,v) cup64_((c),(uint64_t)(v))
#endif

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* ---- the seven organs -------------------------------------------------
   Each bending throws away half of what came before:
     a shift destroys the shifted-out half;
     a 64-bit multiply destroys the upper half of the 128-bit product.
   Four shifts + three multiplies = seven.  Constants are the validated
   splitmix64 pair plus one degski64 constant; none are invented here.   */
static inline uint64_t seven_organs(uint64_t x) {
    x ^= x >> 30;                      /* 1 */
    x *= 0xBF58476D1CE4E5B9ULL;        /* 2 */
    x ^= x >> 27;                      /* 3 */
    x *= 0x94D049BB133111EBULL;        /* 4 */
    x ^= x >> 31;                      /* 5 */
    x *= 0xD6E8FEB86659FD93ULL;        /* 6 */
    x ^= x >> 32;                      /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    size_t n = len;
    /* the cup is never empty: it starts stained with the weight of the pile */
    uint64_t color = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;

#if !CUP_HW
    if (!cup_ready) cup_fill();
#endif

    /* --- palm-sized pile: never touches the cup, straight to the organs --- */
    if (n <= 8) {
        if (n >= 4) {
            uint32_t a, b;
            memcpy(&a, p,         4);
            memcpy(&b, p + n - 4, 4);
            color ^= ((uint64_t)a << 32) ^ (uint64_t)b;
        } else if (n > 0) {
            color ^=  (uint64_t)p[0]
                   ^ ((uint64_t)p[n >> 1] << 19)
                   ^ ((uint64_t)p[n - 1]  << 41);
        }
        return seven_organs(color);
    }

    /* --- heavy pile: handfuls of thirty-two ---
       Four pulls chain through the cup (3 cycles each); the drops
       themselves are added to the cup off the critical path.          */
    while (n >= 32) {
        uint64_t w0, w1, w2, w3, c;
        memcpy(&w0, p +  0, 8);
        memcpy(&w1, p +  8, 8);
        memcpy(&w2, p + 16, 8);
        memcpy(&w3, p + 24, 8);
        c = CUP64(color, w0);
        c = CUP64(c,     w1);
        c = CUP64(c,     w2);
        c = CUP64(c,     w3);
        color = (rotl64(color, 37) ^ w0 ^ rotl64(w1, 13)
                                   ^ rotl64(w2, 29) ^ rotl64(w3, 47)) + c;
        p += 32; n -= 32;
    }

    /* --- handfuls of eight --- */
    while (n >= 8) {
        uint64_t w; memcpy(&w, p, 8);
        color = (rotl64(color, 37) ^ w) + CUP64(color, w);
        p += 8; n -= 8;
    }

    /* --- one handful of four --- */
    if (n >= 4) {
        uint32_t v; memcpy(&v, p, 4);
        color = (rotl64(color, 37) ^ (uint64_t)v) + CUP32(color, v);
        p += 4; n -= 4;
    }

    /* --- stragglers, drop by drop; no mark read twice --- */
    while (n) {
        uint64_t b = (uint64_t)(*p);
        color = (rotl64(color, 37) ^ b) + CUP8(color, b);
        ++p; --n;
    }

    return seven_organs(color);
}
```

Three design notes, all forced by the metaphor rather than by the textbook:

1. **The drop is added to the cup as well as pulled through it.** "Wine that is cut remembers every drop *added to it*" is literal: the raw 64-bit word is XORed into the state in addition to being CRC'd. This is not decoration — `crc32_u64` funnels 64 input bits into 32 state bits, which would make 8-byte keys collide at 2⁻³² instead of 2⁻⁶⁴. The raw add keeps the state genuinely 64-bit. It is free: it sits off the loop-carried CRC chain.
2. **`+` rather than `^` for the fold.** CRC, rotate and XOR are all GF(2)-linear; a wholly linear accumulator has, in principle, blind spots that no finalizer can repair. Integer addition's carries are nonlinear over GF(2) and cost the same one cycle.
3. **`rotl64(color, 37)`**: gcd(37,64)=1, so state bits traverse every position, and 37 is not a multiple of 8, so byte-shifted inputs cannot cancel.

## PREDICTION

PREDICTION: speedup_vs_dp = 7.5

Stated before any measurement, with the reasoning so it can be scored honestly:

- Loop-carried latency of the 32-byte handful = 4 × crc32 (3 cyc) + 1 add = **13 cycles / 32 B = 2.46 B/cycle**. FNV-1a's chain is imul (3) + xor (1) = **0.25 B/cycle**. Large-buffer ratio ≈ **9.8×**.
- Small/medium buffers pay the fixed ~17-cycle finalizer, dragging the mix down; at `len` = 16 I expect ~1.8–2.5×, at `len` = 64 ~4×.
- Blended over a typical size sweep: **7.5×**, and I will call the prediction wrong if it lands outside 5–11×.
- Avalanche: **≈ 0.50** (I expect 0.498–0.502). If it comes back below 0.47 the mechanism has failed SEED 3 and must be thrown away, not patched.
- Absolute throughput on large buffers: ~8 GB/s at ~3.3 GHz.

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were unavailable in this session — no tools were exposed. I am reporting that plainly rather than inventing numbers; the pipeline will compile and measure, and the prediction above is on the record before it does.

What I can assert without the tool, and what I cannot:

- *Can assert:* the instruction-level latency arithmetic above (crc32 r64: latency 3, throughput 1 on Skylake-and-later and Zen 1–5; imul r64: latency 3). The throughput prediction rests on published latencies, not on hope.
- *Cannot assert:* the avalanche score. `seven_organs` is splitmix64's validated core plus one extra bending, so its own avalanche is known-good; what is untested here is whether the accumulator ever produces a **zero** state delta for some single input bit flip, which the finalizer could not repair. The raw-word add and the nonlinear `+` fold are specifically there to make that unlikely, but "unlikely" is not "measured." This is the one place the artifact could genuinely fail SEED 3's gate.

## VERDICT

The native's method translates literally and lands, without being steered, on a validated real-world technique: hardware-CRC accumulation plus a strong nonlinear finalizer. The cup is the CRC polynomial; a pull is one `crc32`; the unbroken pour is the loop-carried dependency; the seven organs are a splitmix-class finalizer whose "throw away half" is, in both of its forms, the literal truth about what a shift and a 64-bit multiply each discard. The assumption "mixing one byte requires a multiplication" is broken cleanly and for real: zero multiplies per byte, three per call.

Honest accounting of where this loses:

- **Against xxHash/wyhash, not against FNV, this is probably slower on large buffers.** The native's "single unbroken pour" is a hard constraint that costs roughly 2.5×: four independent CRC lanes would hit ~8 B/cycle instead of 2.46. I did not take that speedup, because taking it would mean discarding the seed rather than testing it. If the benchmark is dominated by multi-megabyte buffers and reports a mediocre speedup, the seed is what's wrong, and that is the finding.
- **No thread parallelism, deliberately.** The metaphor's unit of work is one pour, and a pour cannot be split without ceasing to be one pour. At the buffer sizes a hash benchmark uses, OpenMP would be pure overhead anyway.
- **The one condition where my mechanism could be worse than the known way — tiny inputs — is guarded**, by the `len ≤ 8` palm path that bypasses the cup and the loops entirely, leaving only the finalizer that avalanche requires.
- **I expect assumption E to survive untouched, and I expect organs six and seven to be worth nothing.** Five bendings (plain splitmix64) almost certainly already satisfies the garden door; the sixth and seventh buy ~5 cycles of nothing. I kept them because the native said seven and because the ablation is the interesting experiment, not because I believe they help. They cannot make the kernel slower than FNV-1a, so this is a note, not an unaddressed risk. If a later run shows 5 and 7 bendings score identically on avalanche, that is direct evidence against "more mixing rounds always means better mixing" — the one assumption none of the three seeds set out to break.