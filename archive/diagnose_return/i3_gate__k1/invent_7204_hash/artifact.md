No tools are available in this session, so everything below is reasoning + the artifact. The prediction is stated before any measurement, and the MEASUREMENT section honestly reports that I could not measure.

---

# MAPPING

## SEED 1 — "counting-bars and violin strings bound under one bridge; a pluck shivers into every string"

| World object | Problem object |
|---|---|
| counting-bars ("the bars alone hold true count") | 64-bit integer registers — exact wraparound arithmetic, no floats, no rounding |
| eight violin strings strung over the landscape | `uint64_t a[4], b[4]` — a 512-bit multi-lane state |
| a pluck on one string | XOR-absorb of one input word into one lane |
| "the whole instrument answers… strings bound under one bridge" | a cross-lane diffusion step: `a[j] += b[j]; b[j] = rotl(b[j],23) ^ a[j]; a[j] = rotl(a[j],41)` plus a lane re-stringing (lower half turns by 1, upper by 2) |
| "never softly, always the same force" | branch-free, constant work per step — no data-dependent control flow |
| "a shudder bends the pitch of all the rest **before it settles**" | diffusion is started but not completed per step |

**Breaks:** *"the state is a single accumulator updated in place, one value."*

## SEED 2 — "the feather's ink-weight per mark, drawn fresh and never repeated"

| World object | Problem object |
|---|---|
| a mark | an input byte (grouped 8 to a bar) |
| the inkwell that never runs dry | an unbounded stream of odd constants: `ctr += 0x9E3779B97F4A7C15` |
| a fresh child-feather, never reused | per-position ink: `ctr ^ INK[j]` — unique for every (chord, lane) |
| "the feather records not the mark's shape, only its weight" | the byte-word is never absorbed raw; it is absorbed as `w + (ctr ^ INK[j])` — an add, which is nonlinear w.r.t. the XOR absorb |
| "the way pigment sinks differently for every stroke" | the position-dependent constant makes a transposed pair of marks produce a different weight → order is fixed into the state |

**Breaks:** *"mixing one byte requires a multiplication"* — order and non-linearity come from a counter-add and a carry, not an `imul`. (It also softens *"each byte must be mixed before the next byte is read"*: eight marks are weighed in parallel.)

## SEED 3 — "the storm's single pass collapses the final tremor into a small fixed token; the pickers destroy every discarded feather and drop"

| World object | Problem object |
|---|---|
| "I catch them mid-tremor and carry that tremor forward" | exactly **one** reduced round of the bridge per 64-byte chord — deliberately incomplete mixing |
| "only when the last mark has been plucked do I wait — hands off" | the heavy work is deferred entirely to finalization |
| the storm's **single pass** over the strings | one fixed finalization: 16 rounds of the same bridge, run once, not per byte |
| "dissolves it down into eight small notes, no more" | squeeze: one byte per string, 8 strings → 64-bit token |
| the pickers taking apart feathers, used drops, mid-tremor counts | the 512-bit state is discarded; 448 bits never leave — capacity truncation, nothing of the running order survives in the output |
| "change one mark, even the last… unrecognizable" | the final chord still passes through 1 + 16 = 17 bridge rounds → full avalanche |

**Breaks:** *"more mixing rounds always means better mixing."* The native explicitly refuses to let the strings settle between marks. Mixing effort is moved out of the O(n) loop into an O(1) tail.

---

# CHOSEN SEED

**SEED 3.** It is the only one of the three that attacks the preferred assumption head-on, and its mapping is maximally literal: "one pass, hands off, eight notes, no more" is a finalization permutation with a fixed round count and a truncating squeeze. SEED 1 supplies the object the storm reads (the coupled 8-lane state), so SEED 1 is implemented as the machinery SEED 3 operates on; SEED 2 supplies the per-position ink. All three are one instrument — I did not discard the other two, I subordinated them.

**Where this lands, deliberately, on a validated technique:** a multi-lane state + cheap per-block permutation + one strong final permutation + truncated squeeze *is* the **sponge/duplex construction** (Keccak, Gimli, Xoodyak, Ascon), with an ARX round in the style of ChaCha/SipHash. Per step 4, I let the metaphor arrive there rather than inventing a new primitive. The native's "pickers destroy everything" is precisely the sponge's capacity; the "eight small notes" is the squeeze; "caught mid-tremor" is the reduced-round absorb that Ascon/Xoodyak use for exactly the reason the native gives.

---

# ASSUMPTION BROKEN

> **"more mixing rounds always means better mixing"**

Rejected. Mixing rounds spent *inside* the per-byte loop cost O(n) and buy almost nothing, because any difference injected early is going to be hammered by everything that follows anyway. Only the **last** chord is at risk of under-mixing, and it costs O(1) to fix that — one storm at the end. So the correct schedule is: **1 cheap round per 64 bytes + 16 rounds once**, not 1 strong round per byte. FNV-1a does the opposite: a full-strength `imul` per byte (n multiplies, 3-cycle serial latency each) and *zero* finalization.

Two further assumptions fall out as collateral: the state is eight coupled lanes, not one accumulator; and no multiplication appears anywhere in the bulk loop.

---

# ARTIFACT

Two regimes, recognized at runtime from the pile's size (the native's own test: *does the pile cover the bars?*):

- **`len < 64` — the instrument stays in its case.** A 512-bit state plus a 16-round storm is pure overhead for a handful of marks. One bar, one multiply-chain per 8-byte mark with fresh ink, short finalizer. This is the validated xxh3/wyhash short-input shape, and it is the mandated guard on my own stated risk ("the storm is an O(1) tax that dominates tiny inputs").
- **`len >= 64` — the full instrument.** Chords of 64 bytes, one reduced bridge round each, one 16-round storm, eight-note squeeze.

**No OpenMP.** The metaphor forbids it outright — *"never skipping, never doubling back… I catch them mid-tremor and carry that tremor forward"* is a strictly serial duplex. Thread-level parallelism would require independent sub-streams, which the native explicitly refuses. Vectorization only: fixed-trip 4-wide loops that GCC's SLP vectorizer turns into AVX2 lanes, `restrict`, `memcpy` for aliasing-safe wide loads.

Four revisions made during design (before any measurement, since none was possible): (1) dropped data-dependent lane selection — an indexed scatter kills SLP vectorization, and the bridge's re-stringing already rotates which string a mark lands on; (2) replaced `h ^= rotl(h,k)` with `h ^= h >> k` — `x ^ rotl(x,k)` is singular over GF(2)⁶⁴ and silently loses a bit; (3) raised the storm from 12 to 16 rounds for diffusion margin, affordable because it is O(1); (4) replaced the three small-size brackets with one 8-byte-at-a-time loop with per-position ink — simpler and faithful to "no two marks share a feather."

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* The inkwell that never runs dry: eight odd drops. */
static const uint64_t INK[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL,
    0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL
};
#define DROP   0x9E3779B97F4A7C15ULL
#define STORM  16   /* the storm passes once -- O(1), not O(n) */

static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* THE BRIDGE: one reduced round. A pluck on any string shivers into the rest.
   Each (a[j],b[j]) step is an invertible Feistel-like ARX pair, so a difference
   can never die; the re-stringing (lower half turns by 1, upper by 2) changes
   the pairing every round, so all eight strings couple within ~4 rounds. */
static inline void bridge(uint64_t a[4], uint64_t b[4], uint64_t rc)
{
    int j;
    uint64_t t0, t1, t2, t3;
    for (j = 0; j < 4; j++) {
        a[j] += b[j];
        b[j]  = ROTL64(b[j], 23) ^ a[j];
        a[j]  = ROTL64(a[j], 41);
    }
    t0 = a[1]; t1 = a[2]; t2 = a[3]; t3 = a[0];
    a[0] = t0 ^ rc; a[1] = t1; a[2] = t2; a[3] = t3;
    t0 = b[2]; t1 = b[3]; t2 = b[0]; t3 = b[1];
    b[0] = t0; b[1] = t1; b[2] = t2; b[3] = t3;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t a[4], b[4], w[8], ctr, h;
    size_t i, n, rem;
    int j, r;

    /* ---- REGIME 1: the pile does not cover the bars (len < 64).
       Raising the whole instrument would cost a fixed 16-round storm for a
       handful of marks. One bar, one mark at a time, each with a fresh drop. */
    if (len < 64) {
        uint64_t k = 0;
        h = INK[0] ^ ((uint64_t)len * DROP);
        for (i = 0; i + 8 <= len; i += 8) {
            k += DROP;                    /* a feather never used twice */
            h ^= rd8(p + i) + k;
            h *= INK[1];
            h ^= h >> 29;
        }
        if (i < len) {                    /* the last 1..7 marks */
            uint64_t t;
            size_t rr = len - i;
            if (len >= 8)     t = rd8(p + len - 8);                        /* overlapping read */
            else if (rr >= 4) t = rd4(p + i) | (rd4(p + len - 4) << 32);
            else              t = ((uint64_t)p[0] << 16)
                                | ((uint64_t)p[rr >> 1] << 8)
                                | ((uint64_t)p[len - 1]);
            k += DROP;
            h ^= t + k;
            h *= INK[2];
            h ^= h >> 32;
        }
        h *= INK[3]; h ^= h >> 29;        /* a brief gust, not the full storm */
        h *= INK[4]; h ^= h >> 32;
        return h;
    }

    /* ---- REGIME 2: the full instrument. Eight strings over the counting-bars. */
    for (j = 0; j < 4; j++) { a[j] = INK[j]; b[j] = INK[j + 4]; }
    a[0] ^= (uint64_t)len * DROP;
    b[3] ^= ROTL64((uint64_t)len + 1, 32);

    ctr = 0;
    n = len >> 6;
    for (i = 0; i < n; i++) {
        ctr += DROP;                                  /* fresh drop from the inkwell */
        memcpy(w, p, 64);                             /* a chord of eight marks */
        for (j = 0; j < 4; j++) a[j] ^= w[j]     + (ctr ^ INK[j]);     /* weight, not shape */
        for (j = 0; j < 4; j++) b[j] ^= w[j + 4] + (ctr ^ INK[j + 4]);
        bridge(a, b, ctr);                            /* caught mid-tremor: ONE round */
        p += 64;
    }
    rem = len & 63;
    if (rem) {
        unsigned char tail[64];
        memset(tail, 0, 64);
        memcpy(tail, p, rem);
        memcpy(w, tail, 64);
        ctr += DROP;
        for (j = 0; j < 4; j++) a[j] ^= w[j]     + (ctr ^ INK[j]);
        for (j = 0; j < 4; j++) b[j] ^= w[j + 4] + (ctr ^ INK[j + 4]);
        bridge(a, b, ctr ^ (uint64_t)rem);
    }

    /* ---- THE STORM: hands off, one pass over the strings. O(1), never per byte. */
    for (r = 0; r < STORM; r++)
        bridge(a, b, INK[r & 7] + (uint64_t)(r * 0x9E3779B1u));

    /* ---- Eight small notes, no more: each string collapses to one byte,
       every bit of it folded in; the other 448 bits go to the pickers. */
    h = 0;
    for (j = 0; j < 4; j++) {
        uint64_t v = a[j], u = b[j];
        v ^= v >> 32; v ^= v >> 16; v ^= v >> 8;
        u ^= u >> 32; u ^= u >> 16; u ^= u >> 8;
        h |= (v & 0xFFULL) << (8 * j);
        h |= (u & 0xFFULL) << (8 * (j + 4));
    }
    return h;
}
```

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 10.0**

Stated before any measurement. Reasoning: FNV-1a's loop-carried `xor;imul` chain is ~4 cycles/byte (≈0.8 GB/s at 3 GHz) and cannot be broken by the compiler. The chord loop's critical path is one add plus one rotate per 64 bytes (~5–6 cycles, ≈11 B/cycle), so the bulk path should be memory/load-bound at roughly 10–25 GB/s, i.e. 15–25× on large buffers. The O(1) 16-round storm pulls medium sizes down (≈3× at 64 B, ≈4× at 63 B via the short path), so a mixed-size harness should land near 10×.

Avalanche: I predict **≥ 0.49 mean bit-flip probability / score ≥ 0.95**. Any single input bit changes exactly one lane before 1 + 16 = 17 bridge rounds of an invertible ARX permutation with changing lane pairings; full cross-lane coupling needs ~4 rounds and full intra-lane bit diffusion ~6–8, so 17 is roughly 2× margin. The short path is a 3-multiply chained moremur, which is known-good.

---

# MEASUREMENT

**Not performed. No tools were available in this session** — `hash_bench` and `hash_contract` could not be called, so I have no measured throughput and no measured avalanche score. I am reporting that plainly rather than inventing numbers.

What should be checked when the pipeline runs it, and what would falsify my claims:

1. **Avalanche < ~0.45** ⇒ `STORM = 16` is insufficient or the eight-note squeeze (8 bits per lane) is throwing away too much. First fix: replace the squeeze with `h = (a[0]^b[2]) + ROTL64(a[1]^b[3],32) ^ ...` full-width fold; second fix: `STORM = 24`.
2. **speedup < 3×** ⇒ GCC did not SLP-vectorize the 4-wide loops, or the `memcpy(w,p,64)` is not folding into two vector loads. Fix: explicit `__m256i` intrinsics for the bridge.
3. **Any size where speedup < 1.0** ⇒ the 64-byte regime threshold is misplaced; raise it (the small path is already ~4× at 63 B, so the threshold can move up to 128 or 256 safely).

---

# VERDICT

The native is describing a **reduced-round duplex sponge**, and taken literally it is a good design, not a curiosity. The one claim that carries all the weight is the refusal to let the strings settle between marks: *mixing effort belongs at the end, not in the loop*. FNV-1a pays a 3-cycle multiply latency per byte and then performs **no finalization at all** — exactly inverted. Moving all the strength into one O(1) storm is what makes a 10×-class speedup possible while *improving* avalanche rather than trading it away.

Honest statements of risk, and how each is addressed rather than waved at:

- **"The storm is a fixed tax that dominates small inputs."** This is my own mechanism's stated weakness, so it is guarded, not acknowledged: `len < 64` takes a separate single-bar path with no storm. Without that guard the kernel would be *slower than the baseline* somewhere around 17–40 bytes. With it, I believe there is no size at which this loses to FNV-1a — that is the single most important thing for the measurement to confirm or refute.
- **One reduced round per 64 bytes is a real security weakening**, not just a speed choice. For an adversarial, keyed, DoS-resistant hash you would want 2+ rounds per block or a true cryptographic permutation. The target problem asks for avalanche and speed, not unforgeability, so this is the right trade here — but I will not claim more than that.
- **The squeeze keeps 64 of 512 bits.** That is the native's "pickers" and it is correct sponge practice, but it means the output quality rests entirely on the storm being long enough. If the avalanche score disappoints, that is the first place to look.
- **No thread parallelism**, because the metaphor's unit of work is a strictly serial tremor carried forward. Adding OpenMP would have meant abandoning the native's mechanism and substituting independent-chunk parallel hashing — the textbook move the brief forbids.

What I would not claim: that this beats xxHash3 or wyhash. It is in the same structural family and the same performance neighbourhood; it is a faithful, measured-shaped implementation of the native's instrument, not a new record.