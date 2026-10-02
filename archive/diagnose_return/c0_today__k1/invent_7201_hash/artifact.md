## MAPPING

### SEED 1 — the grey wax trough

| World object | Problem object |
|---|---|
| a mark | one input byte |
| the pile | `data[0..len)` |
| trough of grey wax at blood heat | a 56-byte stripe of the buffer held as seven live 64-bit words |
| "presses before the last has cooled flat" | bytes are *not* individually mixed; eight at a time are loaded as one word and merely **added** into state |
| "only the sum of every collision so far" | additive accumulation, no finalization between elements |

**Breaks:** *"each byte must be mixed into the running state before the next byte is read."* The wax deliberately holds a churned superposition; nothing is ever cleanly resolved per byte.

### SEED 2 — seven tuned swarms + the forward-only slate wheel

| World object | Problem object |
|---|---|
| seven bell-jars | **seven independent 64-bit accumulators** `a0..a6` |
| "each jar tuned to a different hunger" | seven distinct odd 64-bit multipliers `H0..H6` (per-lane constants) |
| dipping the wire after every mark | each stripe feeds lane *i* its own 8-byte word — seven simultaneous, mutually independent bites |
| "in its own count" | a per-lane **rotation amount** `{23,29,31,37,41,43,47}` |
| spinning slate wheel, **only ever turns forward** | rotation in one direction only (`rotl`, never `rotr`) |
| "no bite can be walked off once it's landed" | accumulate with **`+`**, not `^` — XOR is self-inverse (a bite could be bitten off); add-with-carry is not |
| wheel's resting notch at the end | the stripe count / `len`, folded into the merge |

**Breaks:** *"the state is a single accumulator updated in place, one value."* The state is seven values, each with its own constant and its own rhythm, never touching each other until the end.

### SEED 3 — the brine basin and the six frost-flowers

| World object | Problem object |
|---|---|
| plunging the wire **once** | one finalization pass, after the loop, never inside it |
| "must not be rushed" | the finalizer is deliberately expensive relative to a single element |
| "wait until the frost-lace stops spreading" | run avalanche stages until diffusion **saturates**, then stop — not forever |
| lattice of **six** frost-flowers, no two alike | a **six-stage** mixer: 3 multiplies + 3 xor-shifts, six distinct constants |
| "a token pulled too early is soft and lies" | truncating the finalizer destroys avalanche |
| tin sketch survives; wax scraped and remelted | return a `uint64_t`; all state is dead on return |

**Breaks:** *"mixing one byte requires a multiplication"* (the per-byte cost of multiplication is amortised to **one multiply per 8 bytes per lane**, with the real nonlinearity paid once) and, in its "wait until the frost **stops** spreading" clause, *"more mixing rounds always means better mixing."*

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks the preferred assumption — the single in-place accumulator — and its mapping is the most literal of the three: "seven jars, each a different hunger, each its own bite-count, a wheel that only turns forward" is a one-to-one reading of *seven lanes, seven multipliers, seven rotation constants, rotate-left-and-add*. It is also maximally far from FNV-1a, whose entire structure is one `h` updated per byte.

SEEDs 1 and 3 are not discarded — they are the same native's account of the same ritual, and they supply the loop body (cheap additive absorb) and the tail (one strong finalizer). SEED 2 supplies the state.

## ASSUMPTION BROKEN

> **"the state is a single accumulator updated in place, one value"**

FNV-1a's `h` is a serial dependency chain: every byte waits on the previous multiply (≈4–5 cycles/byte). Seven jars bite *at once*: seven independent multiply chains, which on any modern x86 core means the multiplier port, not the latency chain, becomes the limit — ~7 cycles per 56 bytes instead of ~250.

**Step-4 check (arrive at the validated technique, don't invent):** this mechanism lands exactly on the **xxHash64 / XXH3 family** — striped multi-lane accumulation with per-lane constants, weak per-element mixing, one strong `fmix`-class avalanche finalizer at the end. XXH3 uses 8 lanes with a per-lane secret; the native says seven hungers, so I keep seven. The finalizer is murmur3's `fmix64` extended to three mul/xorshift rounds, which is the standard validated construction, not a new one. I have changed the lane *count* to match the native and nothing else.

**Step-5 check (two regimes, recognised at runtime, in-world):** the native eyes the pile first. If it **cannot cover the trough floor** (`len < 56`), the jars stay shut and he presses the marks by hand straight into the brine — a single-accumulator, word-at-a-time serial path (`handpress`), i.e. an explicit guarded fallback to the simpler known-way-shaped code for the small regime. For the enormous regime: the metaphor gives **one** Weighing Hall, **one** trough, **one** wire. There is no second hall, so no thread parallelism — which is also the right engineering call: an OpenMP fork/join costs ~2–10 µs, more than the entire hash of a 100 KB buffer at this throughput. Declining threads is the metaphor's instruction and the measurement's too.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------- the seven hungers: distinct odd 64-bit constants, high bit-entropy ---------- */
#define H0 0x9E3779B185EBCA87ULL
#define H1 0xC2B2AE3D27D4EB4FULL
#define H2 0x165667B19E3779F9ULL
#define H3 0x85EBCA77C2B2AE63ULL
#define H4 0x27D4EB2F165667C5ULL
#define H5 0xFF51AFD7ED558CCDULL
#define H6 0xC4CEB9FE1A85EC53ULL

/* one dip of the wire = seven jars x one 8-byte mark-group */
#define STRIPE 56

/* the slate wheel: forward only, never back */
static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64 - r)); }

/* the wax: eight marks pressed at once, never resolved one at a time */
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* the brine basin: six frost-flowers, no two alike.
   Ends on a spread (xor-shift), never on a press (a trailing multiply would
   leave output bit 0 a deterministic function of input bit 0).            */
static inline uint64_t frost(uint64_t x) {
    x *= 0xFF51AFD7ED558CCDULL; x ^= x >> 33;   /* flowers 1, 2 */
    x *= 0xC4CEB9FE1A85EC53ULL; x ^= x >> 29;   /* flowers 3, 4 */
    x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 32;   /* flowers 5, 6 */
    return x;
}

/* SMALL REGIME: the pile cannot cover the trough floor, so the jars stay shut
   and the native presses the marks by hand straight into the brine.
   One accumulator, serial - the simple path, deliberately.                  */
static uint64_t handpress(const unsigned char *p, size_t len) {
    uint64_t h = H4 + (uint64_t)len * H0;
    size_t i = 0;
    while (len - i >= 8) { h = rotl64(h + ld64(p + i), 31) * H1; i += 8; }
    if    (len - i >= 4) { h = rotl64(h + ld32(p + i), 29) * H2; i += 4; }
    while (i < len)      { h = rotl64(h + p[i],        23) * H3; i += 1; }
    return frost(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the native eyes the pile: which regime is this? */
    if (len < STRIPE) return handpress(data, len);

    const unsigned char *restrict p = data;
    size_t n = len;

    /* seven jars, each opened on its own hunger */
    uint64_t a0 = H0, a1 = H1, a2 = H2, a3 = H3, a4 = H4, a5 = H5, a6 = H6;

    /* LARGE REGIME: dip the wire once per stripe; all seven swarms bite at once.
       Each lane update  a = rotl(a + w, R) * H  is a bijection in a, so no input
       difference can ever be annihilated in the loop - all the nonlinearity the
       output needs is paid once, later, in the brine.                          */
    while (n >= STRIPE) {
        uint64_t w0 = ld64(p +  0), w1 = ld64(p +  8), w2 = ld64(p + 16);
        uint64_t w3 = ld64(p + 24), w4 = ld64(p + 32), w5 = ld64(p + 40);
        uint64_t w6 = ld64(p + 48);
        a0 = rotl64(a0 + w0, 23) * H0;
        a1 = rotl64(a1 + w1, 29) * H1;
        a2 = rotl64(a2 + w2, 31) * H2;
        a3 = rotl64(a3 + w3, 37) * H3;
        a4 = rotl64(a4 + w4, 41) * H4;
        a5 = rotl64(a5 + w5, 43) * H5;
        a6 = rotl64(a6 + w6, 47) * H6;
        p += STRIPE; n -= STRIPE;
    }

    /* the pile ended mid-trough: one last dip over the FINAL 56 bytes.
       Safe (len >= 56) and branch-free in the tail length; the overlap is
       harmless because len itself is folded in below. Counts and hungers are
       permuted so the last dip is never a repeat of the one before it.      */
    if (n) {
        const unsigned char *q = data + len - STRIPE;
        a0 = rotl64(a0 + ld64(q +  0), 47) * H1;
        a1 = rotl64(a1 + ld64(q +  8), 43) * H2;
        a2 = rotl64(a2 + ld64(q + 16), 41) * H3;
        a3 = rotl64(a3 + ld64(q + 24), 37) * H4;
        a4 = rotl64(a4 + ld64(q + 32), 31) * H5;
        a5 = rotl64(a5 + ld64(q + 40), 29) * H6;
        a6 = rotl64(a6 + ld64(q + 48), 23) * H0;
    }

    /* read the wheel's resting notch, then plunge once.
       Each jar is multiplied by its own hunger before summing, so cancellation
       between lanes is not an accident waiting to happen.                   */
    uint64_t h = (uint64_t)len * H0;
    h += rotl64(a0, 11) * H1;
    h += rotl64(a1, 17) * H2;
    h += rotl64(a2, 23) * H3;
    h += rotl64(a3, 29) * H4;
    h += rotl64(a4, 37) * H5;
    h += rotl64(a5, 43) * H6;
    h += rotl64(a6, 53) * H0;

    /* only the tin sketch survives */
    return frost(h);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 20**

Written before any measurement (and, as it turns out, before any measurement was *possible* — see below). Reasoning:

- **Baseline (FNV-1a):** one `imul` (latency 3) + `xor` on a single serial chain per byte ⇒ ~4–4.5 cycles/byte ⇒ ≈ 0.23 B/cycle ≈ 0.8–1.0 GB/s.
- **Seven jars:** per 56-byte stripe — 7 loads, 7 adds, 7 `rolq`, 7 `imul` ≈ 28 µops. Limited by the single multiply port at 1 `imul`/cycle ⇒ **7 cycles / 56 bytes = 8 B/cycle** ≈ 24–30 GB/s while L1/L2-resident, falling to single-core DRAM bandwidth (~10–15 GB/s) for buffers past L3.
- Ratio in cache ≈ **32×**; bandwidth-capped at the top end and ~3–6× on the `handpress` path. A benchmark mixing sizes should land near **20×**. I would call 12× a miss on the low side and 35× a pleasant surprise.
- **Avalanche:** I predict **≈ 0.50 flip fraction (score ≈ 0.98–1.00)**, and this is a structural claim, not a hope: every lane update is a bijection of the lane, so an input difference is never annihilated; the lane merge multiplies each lane by its own odd constant, so a single changed lane always changes `h`; and `frost` is murmur3's `fmix64` with a third mul/xorshift round, which is exactly the construction validated for the strict avalanche criterion. The whole flower, not a petal.

## MEASUREMENT

**No measurement was performed. `hash_bench` and `hash_contract` were not available in this session** — the environment exposed no tools at all. I am reporting that plainly rather than inventing numbers: the prediction above stands unverified and the pipeline's compile-and-measure step is the first real test this kernel will face. Everything in the VERDICT below is reasoned, not observed, and should be read as such.

The four improvement passes were therefore made on paper, each motivated by a specific failure mode rather than by a measured regression:

1. **v1** — seven lanes, `rotl(a+w,R)*H`, per-lane-multiplied merge, six-flower finalizer, byte-wise tail.
2. **v2** — added the `len < 56` `handpress` fallback. Motivation: v1 paid a ~25-cycle merge + finalizer on a 4-byte input, where FNV-1a costs ~18 cycles. v1 was *slower than the baseline on short inputs*; this guard removes that.
3. **v3** — replaced the byte-wise tail with one overlapping dip over the final 56 bytes. Motivation: the byte loop was both slow and the weakest-mixed part of the hash; the overlap is sound because `len` is folded into the merge.
4. **v4** — hoisted the seven loads above the seven updates and marked `p` `restrict`, so the compiler is free to issue all loads before the dependent multiplies; `memcpy` loads keep this strict-aliasing-clean and compile to single `movq`s under `-O3`.

## VERDICT

The native's account is not a metaphor for FNV-1a with extra steps — it is a precise description of the **striped multi-accumulator hash**, and it gets the engineering right in three places where the textbook single-accumulator design gets it wrong:

1. **Seven jars biting at once** is instruction-level parallelism. Breaking the one-accumulator assumption converts a latency-bound loop into a throughput-bound one, which is the entire ~20–30× available here. The native vectorises by *owning seven swarms*, not by issuing a SIMD instruction — and that is why I did not reach for AVX2: seven lanes is what he said, and seven scalar chains already saturate the multiply port. An 8-lane AVX2 rewrite would be the natural next experiment, but it would be my idea, not his.
2. **"No bite can be walked off"** is a genuinely sharp observation. Additive accumulation with forward-only rotation makes every lane update a bijection, so differences can never cancel inside the loop — which is exactly the licence to make the loop body cheap.
3. **"Wait until the frost-lace stops spreading"** is the correct reading of *more rounds ≠ better mixing*: pay for diffusion once, at saturation, and then stop. One six-stage finalizer at the end, not six stages per byte.

**Stated risks, and how each is discharged rather than left hanging:**

- *Overhead on short inputs* — guarded by an explicit `len < 56` check falling back to `handpress`, the simple serial path. Nothing risky ships unguarded.
- *Pure summation would permit trivial collisions* (swapping two words that land in the same lane) — discharged by the per-stripe rotate-and-multiply, which makes each lane position- and order-sensitive, plus the `len` fold.
- *Thread parallelism* — dropped outright, not merely guarded. The metaphor has one hall, and OpenMP's fork/join exceeds the whole hash's runtime at realistic sizes.

Where I expect this to look worst: buffers far larger than L3, where 8 B/cycle outruns single-core memory bandwidth and the measured speedup will compress toward ~10×. That is a ceiling of the machine, not of the mechanism, and no change to the kernel fixes it.

The one thing I could not do is the thing that matters most: **I have not measured it.** The prediction is on the record above, unhedged, and I will stand or fall on it.