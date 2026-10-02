## 1. MAPPING

### SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."

| World object | Problem object |
|---|---|
| sailor-chain, one knot added per mark | multi-slot state; the row is *lengthened*, never reset |
| "never breaking an old knot" | no lane is ever overwritten from scratch — every lane is a running accumulator |
| the chain *hangs* between fire and wall | the chain's current sag = one scalar `hang`, recomputed from **all** lanes each row |
| the shadow is of the whole chain, "the whole line hangs together or not at all" | each lane's next update takes `hang` as an addend → total cross-lane coupling |
| "the row grows" / chain length | a byte counter `ctr` folded into `hang` → position/length dependence |

**Breaks:** *"the state is a single accumulator updated in place, one value."* The state is a row of six, and the coupling term is a function of the entire row.

### SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."

| World object | Problem object |
|---|---|
| a mark | one input word (its "strokes" are its set bits) |
| counting the mark's strokes as pebbles into a small cup | `popcount(w)` |
| the cup's fill decides how far I swing the incense | rotation amount `r = (popcount(w)<<1) + fixed_arc` |
| short chain → bounded arc | rotation is mod 64 — a closed arc |
| the smoke *bends the fire's throw* before it lands | the rotate is applied to `state + word + hang`, i.e. it bends the projection, it does not add to it |
| the bend is set by the mark itself, not by me | **data-dependent** rotation — the mixing operator is chosen by the data |

**Breaks:** *"mixing one byte requires a multiplication."* The entire mixer is add + rotate + xor. No `*` appears anywhere in the kernel.

### SEED 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."

| World object | Problem object |
|---|---|
| wait for the smoke to thin to one straight thread; read only when the shadow is still | absorb cheaply; do *no* heavy mixing per byte — mix once, at the end |
| the wall's six fixed cracks, the same six every time | six fixed lanes / six fixed arc constants |
| wet clay pressed to the wall | the linear collapse of the six lanes into one 64-bit word |
| water running over the stub until only hardened ridges remain | the finalizer: shift-add-xor erosion (Thomas Wang's multiply-free 64-bit mix) |
| throw away the smoke, the pebbles, the old print | no retained intermediates; one-way |

**Breaks:** *"more mixing rounds always means better mixing"* (and partly *"each byte must be mixed into the running state before the next byte is read"*): the per-byte round is deliberately weak, and all the avalanche budget is spent once, on the final print.

---

## 2. CHOSEN SEED

**SEED 2.** It is the one that breaks "mixing one byte requires a multiplication," which the instructions tell me to prefer, and its mapping is the most literal of the three — "cup's fill → swing distance" is exactly "popcount → rotate amount," with nothing left over. Seeds 1 and 3 are not discarded; the native uses all three in one procedure, so they supply the state shape (six lanes + whole-row `hang`) and the finalizer. Seed 2 supplies the *mixing primitive*, which is the thing the known way gets from `imul`.

**Arriving at a validated technique rather than inventing one (step 4).** Multiplication-free mixing driven by the data is not something I had to invent: the mechanism lands squarely on two validated families, and I let it land there rather than steering around them.
- *add–rotate–xor absorb with a multi-lane state* = SipHash / BLAKE2 / ChaCha construction.
- *rotation amount chosen by the data* = RC5/RC6 data-dependent rotations.
- *shift-add-xor finalizer* = Thomas Wang's 64-bit integer hash, used verbatim, with an xor-rotate rinse on each end (the xxh3 `rrmxmx`-style spread, also multiply-free here).

**Why the avalanche should be structurally good, not just finalizer-good.** Flipping *any single input bit* changes that word's popcount by exactly ±1, hence the swing by exactly 2. So the lane becomes `rotl(x', r±2)` instead of `rotl(x, r)` — the same bit-string displaced by two positions, which differs from the original in ≈32 of 64 positions *before* the finalizer ever runs. The native's own claim — "moving even one mark throws every shadow after it into a different bend" — is literally the avalanche argument.

---

## 3. ASSUMPTION BROKEN

> **"mixing one byte requires a multiplication"**

Replaced by: *the data picks the rotation*. Secondary casualties: the single accumulator (six lanes), and "more rounds is better" (one weak absorb round, one strong final erosion).

**Two regimes (step 5).** The known-way section names both a short-input path and a bulk folding path, so one strategy is not acceptable. The native encodes the test himself: a rope *hangs* between fire and wall, and a short rope is **taut** — a taut row reads whole in one shadow and needs no incense swing across the piazza. So the runtime regime check is literally "does the row sag?": `len < 48` (one full row of six knots) → taut path, no six-lane setup, no per-row `hang` recurrence, plus a sub-ladder for `len < 8` where there is no whole mark to read. `len >= 48` → sagging path. This also discharges the risk my own verdict names (fixed finalization cost would dominate tiny inputs): the small regime is guarded and falls back to the cheap path.

**No thread parallelism.** The metaphor's unit of work is one knot (8 bytes) and one row (48 bytes) — far below any OpenMP threshold at realistic buffer sizes. I used vectorization-adjacent hints only (`restrict`, `memcpy` loads, six independent ILP chains). AVX2 has no per-lane variable rotate and no 64-bit popcount, so the literal six-lane *scalar* form is also the fastest form available; this is one of the rare cases where fidelity and speed agree.

---

## 4. ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= the wall's six fixed cracks: the same six every time ===== */
#define CR0 0x9E3779B97F4A7C15ULL
#define CR1 0xBB67AE8584CAA73BULL
#define CR2 0x3C6EF372FE94F82BULL
#define CR3 0xA54FF53A5F1D36F1ULL
#define CR4 0x510E527FADE682D1ULL
#define CR5 0x1F83D9ABFB41BD6BULL

/* ============ the six resting angles of the brick-incense chain ============ */
#define A0  7u
#define A1 19u
#define A2 31u
#define A3 43u
#define A4 11u
#define A5 53u

static inline uint64_t rotl64(uint64_t v, unsigned r) {
    unsigned s = r & 63u;                       /* the swing is a closed arc */
    return (v << s) | (v >> ((64u - s) & 63u));
}

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint32_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}

/* ===== the stream: water over the clay stub until only the ridges remain.
   Thomas Wang's multiplication-free 64-bit finalizer, with an xor-rotate
   rinse at each end.  There is no '*' operator anywhere in this file.   ===== */
static inline uint64_t ridges(uint64_t k) {
    k ^= rotl64(k, 49) ^ rotl64(k, 24);
    k  = (~k) + (k << 21);
    k ^= k >> 24;
    k  = k + (k << 3) + (k << 8);
    k ^= k >> 14;
    k  = k + (k << 2) + (k << 4);
    k ^= k >> 28;
    k  = k + (k << 31);
    k ^= k >> 30;
    return k;
}

/* one full row of six knots: six marks walk past the fire, each one's
   pebble-count setting its own swing, every lane taking the hang of the
   whole chain as it stood a moment before.                               */
#define ROW(q)                                                            \
    do {                                                                  \
        uint64_t w0 = ld64((q)),       w1 = ld64((q) +  8),               \
                 w2 = ld64((q) + 16),  w3 = ld64((q) + 24),               \
                 w4 = ld64((q) + 32),  w5 = ld64((q) + 40);               \
        unsigned c0 = (unsigned)__builtin_popcountll(w0);                 \
        unsigned c1 = (unsigned)__builtin_popcountll(w1);                 \
        unsigned c2 = (unsigned)__builtin_popcountll(w2);                 \
        unsigned c3 = (unsigned)__builtin_popcountll(w3);                 \
        unsigned c4 = (unsigned)__builtin_popcountll(w4);                 \
        unsigned c5 = (unsigned)__builtin_popcountll(w5);                 \
        s0 = rotl64(s0 + w0 + hang, (c0 << 1) + A0);                      \
        s1 = rotl64(s1 + w1 + hang, (c1 << 1) + A1);                      \
        s2 = rotl64(s2 + w2 + hang, (c2 << 1) + A2);                      \
        s3 = rotl64(s3 + w3 + hang, (c3 << 1) + A3);                      \
        s4 = rotl64(s4 + w4 + hang, (c4 << 1) + A4);                      \
        s5 = rotl64(s5 + w5 + hang, (c5 << 1) + A5);                      \
        ctr += 48u;                       /* the row only ever lengthens */\
        hang = ((s0 ^ s2) ^ s4) + rotl64((s1 ^ s3) ^ s5, 29) + ctr;       \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* ============ regime 1: a short rope hangs TAUT — read it whole ======= */
    if (len < 48) {
        if (len == 0) return ridges(CR0);

        if (len < 8) {                    /* not even one whole mark to walk */
            uint64_t a, b;
            if (len >= 4) {
                a = ((uint64_t)ld32(p + len - 4) << 32) | (uint64_t)ld32(p);
                b = rotl64(a, 27) + (uint64_t)len;
            } else {
                a = ((uint64_t)p[0] << 16)
                  | ((uint64_t)p[len >> 1] << 8)
                  |  (uint64_t)p[len - 1];
                b = a + ((uint64_t)len << 5);
            }
            unsigned c = (unsigned)__builtin_popcountll(a);
            uint64_t x = rotl64(a + CR0, (c << 1) + 11u) + (b ^ CR1) + (uint64_t)len;
            return ridges(x);
        }

        uint64_t x = CR0 ^ (uint64_t)len;
        uint64_t y = CR3 + (uint64_t)len;
        size_t n = len;
        while (n >= 8) {
            uint64_t w = ld64(p);
            unsigned c = (unsigned)__builtin_popcountll(w);
            x = rotl64(x + w + y, (c << 1) + A0);
            y = (y ^ w) + rotl64(x, 29);
            p += 8; n -= 8;
        }
        if (n) {                               /* last mark, read overlapping */
            uint64_t w = ld64(data + len - 8);
            unsigned c = (unsigned)__builtin_popcountll(w);
            x = rotl64(x + w + y, (c << 1) + A2);
            y = (y ^ w) + rotl64(x, 37);
        }
        return ridges(x + rotl64(y, 23) + (uint64_t)len);
    }

    /* ===== regime 2: the rope SAGS across the piazza — six knots a swing === */
    uint64_t s0 = CR0 ^ (uint64_t)len, s1 = CR1, s2 = CR2,
             s3 = CR3, s4 = CR4,       s5 = CR5 + (uint64_t)len;
    uint64_t hang = CR2 + (uint64_t)len;
    uint64_t ctr  = 0;
    size_t   n    = len;

    while (n >= 48) {
        ROW(p);
        p += 48; n -= 48;
    }
    if (n) {                     /* the last row, re-hung over the chain's end */
        const unsigned char *q = data + len - 48;
        hang ^= (uint64_t)n;
        ROW(q);
    }

    /* the wet clay takes the print of the whole settled row, then the stream */
    uint64_t fold = (s0 + rotl64(s1, 11)) ^ (s2 + rotl64(s3, 29));
    fold += (s4 + rotl64(s5, 47)) ^ hang;
    return ridges(fold + (uint64_t)len);
}
```

---

## 5. PREDICTION

**PREDICTION: speedup_vs_dp = 9.0**

Stated before any measurement, with its arithmetic exposed so it can be scored honestly:

- Baseline FNV-1a is one loop-carried `imul` (3 cyc) + `xor` (1 cyc) per byte → **≈4–5 cycles/byte**, ~0.22 B/cycle.
- The sagging path consumes 48 bytes per row. Per row: 6 loads, 6 `popcnt`, 12 adds, 6 variable `rol` (1 µop on Zen/Ice Lake, ~3 on Skylake with flag merging), 6 `lea`+mask, ~8 µops of `hang`, ~3 loop overhead → **≈65–85 µops / 48 B ≈ 1.4–1.8 µops/byte ≈ 0.35–0.45 cycles/byte**. The loop-carried critical path (add → rot → xor-tree → rot → add ≈ 8–9 cycles) is well under the ~17–22 cycles of issue work per row, so it is throughput-bound, not latency-bound.
- Ratio on large buffers: **≈10–13×**. The taut path runs ~1.5–2.5 cycles/byte → **≈2–4×**. I assume the harness mixes sizes, so I shade the large-buffer figure down to **9.0** rather than quoting the best case.
- Avalanche prediction: **0.49–0.50**, on the ±1-popcount → ±2-rotation argument above (expected Hamming distance ≈32/64 from the absorb alone, before `ridges`). If it comes back below ~0.45 the structural argument is wrong and I will say so.

---

## 6. MEASUREMENT

**Not measured. `hash_bench`, `hash_contract` and the symbolic/unconventional tools were listed but are not available in this session — no tool call was possible, so I ran zero experiments.** I am reporting that plainly rather than dressing up the prediction as a result. The numbers in PREDICTION are a static µop/latency model, not an observation, and the four allotted improvement rounds were therefore not spent — the artifact above is a first draft that has never been compiled or scored by me.

What the pipeline should check, in order of how likely each is to falsify me:
1. **Avalanche < 0.45** would kill the central claim. The most likely culprit is the single `ridges` call collapsing six lanes through one linear fold; the fix is a second `ridges` on an independent half-fold (cheap, once per call).
2. **Speedup < 4× on large buffers** would mean the variable `rol r64, cl` is costing more than modelled (likely on pre-Ice-Lake Intel, 3 µops each, 18 µops/row). The in-metaphor fix that keeps the seed intact is to let one swing serve a pair of knots (3 variable rotates per row instead of 6) — the incense swings once for two sailors.
3. **Short-input regression vs FNV** at `len` 8–32 would mean `ridges` is too long for the taut path; trim the leading xor-rotate rinse there.

---

## 7. VERDICT

The native's procedure is not a metaphor for FNV-1a with extra steps; it is a genuinely different machine, and every piece of it corresponds to something real:

- The **pebble-cup → swing** is data-dependent rotation, and it does the job multiplication does in the known way — it makes the mixing operator depend on the data — at 1 µop instead of 3 cycles of latency, and it makes avalanche *structural*: one flipped bit always moves the popcount by one, always moves the swing by two, always displaces the whole lane. That is the strongest part of the design and the reason I expect the avalanche score to come in near 0.5.
- The **six-lane sailor-chain** is the thing that actually buys the speed, not the rotation: six independent dependency chains let the machine retire 48 bytes per ~17 cycles where FNV retires one byte per ~4.5. The native's insistence that the chain never breaks and that the shadow is of the *whole* row is what keeps six lanes from being six independent weak hashes.
- The **clay print and the stream** are the admission that the per-byte round is deliberately weak — which is the right trade and the opposite of "more rounds is always better."

Stated risks and how each is discharged, as required: the fixed finalizer cost would hurt tiny inputs → guarded by the `len < 48` taut path and the `len < 8` ladder, both of which skip the six-lane setup and the `hang` recurrence entirely. Thread parallelism would be pure overhead at these work-unit sizes → not used at all, rather than used behind a guess. The one risk I have **not** discharged is the honest one: *none of this is measured.* Everything above is a prediction with its reasoning shown, and the first `hash_bench` run may well contradict it — in particular point (2), the variable-rotate throughput, which is the assumption I hold least confidently and which varies by microarchitecture in a way my model cannot see from here.