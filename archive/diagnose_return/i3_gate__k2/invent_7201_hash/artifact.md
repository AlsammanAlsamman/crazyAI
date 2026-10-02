## 1. MAPPING — all three seeds

### SEED 1 — the grey wax trough

| World object | Problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| one mark | one 8-byte word read whole (`memcpy`-load) |
| grey wax at blood heat | `uint64_t W` — one shared running accumulator, never reset |
| "presses its shape in before the last has cooled flat" | each word is injected with 1 cheap op; no diffusion is completed before the next word arrives |
| "only the sum of every collision so far" | `W += w` — literally a running sum, carries colliding across bit positions |
| trough scraped clean and remelted | `W` lives in a register, discarded on return |

**Assumption broken:** *"each byte must be mixed into the running state before the next byte is read."* The wax accepts the next press mid-cool: injection is decoupled from diffusion, which is deferred wholesale to the end.

### SEED 2 — seven tuned swarms, forward-only wheel

| World object | Problem object |
|---|---|
| the dipped wire | the streaming state as a whole |
| seven bell-jars, each a different hunger | `v0..v6`: seven simultaneously-live 64-bit accumulators, seven distinct seed constants |
| "bite in its own count and rhythm" | per-lane fixed rotation amount `{13,29,41,7,53,19,37}` — counts/rhythms, **no multiplication** |
| dipping the wire after every mark | lane *i* reads the trough's prefix sum `W` *after* mark *i* |
| spinning slate wheel, forward only, no bite walked off | `rotl` only (never `rotr`, never subtract); rotation amounts are compile-time immediates fixed forever |
| one dip serves seven jars | one loop iteration = 7 marks = a 56-byte stripe |

**Assumption broken:** *"the state is a single accumulator updated in place, one value"* — and, as a bonus, *"mixing one byte requires a multiplication"* (the hot loop has zero multiplies) and *"the whole buffer must be read once, start to end, in order"* (seven offsets are consumed per dip).

### SEED 3 — the brine basin and the frost-flower lattice

| World object | Problem object |
|---|---|
| plunging the wire *once* into brine | a single finalizer applied after the stream ends, never per byte |
| "must not be rushed… a token pulled too early is soft and lies" | a weak finalizer = poor avalanche; the stream alone is not a hash |
| waiting until the frost-lace **stops spreading** | run diffusion to its fixpoint — and stop there |
| fixed lattice of **six frost-flowers, no two alike** | six pairwise-distinct constants: `len·φ`, `>>30`, `×BF58…`, `>>27`, `×94D0…`, `>>31` |
| sketched onto tin, wax remelted | only the `uint64_t` return survives; all state is scratch |
| "the whole flower, not a petal, comes out changed" | the avalanche requirement itself |

**Assumption broken:** *"more mixing rounds always means better mixing."* The native freezes to the fixpoint and not one round further; past the fixpoint the lattice is already fixed and extra rounds buy nothing but time.

## 2. CHOSEN SEED

**SEED 2.** It is the one of the three that breaks the preferred assumption (*single accumulator updated in place*), and its mapping is the most literal: "seven jars tuned to different hungers" is seven live accumulators with seven distinct constants, "count and rhythm" is add-and-rotate, "a wheel that only turns forward" is `rotl` with fixed immediates. It is also maximally distant from FNV-1a, whose entire identity is *one* value and *one* multiply per byte.

SEEDs 1 and 3 are not discarded — they are the same native's other apparatus in the same ritual, so the trough (SEED 1) and the brine (SEED 3) appear as the shared wax `W` and the single six-flower finalizer. Nothing in the description is dropped.

## 3. ASSUMPTION BROKEN

Primary: **the state is a single accumulator updated in place, one value** → eight live state words (one wax trough + seven bite-patterns).
Secondary, falling out of the same metaphor: **mixing requires a multiplication** (hot loop is xor/add/rotl only), **each byte fully mixed before the next is read** (diffusion deferred to the brine), **more rounds is better** (freeze to fixpoint, stop).

**Regime recognition (required, and in-world):** the native weighs the pile in his hand first. A pile too thin to feed all seven jars never gets the seven-fold dipping — it goes in "one trickle" into a single accumulator, in order, and is brined at once. That is the `len < 56` branch, and the same trickle also swallows the end-of-pile remainder, so there is exactly one tail code path. This is also the guard for the risk my own verdict names (stripe setup + fold overhead on short inputs).

**Why no threads:** the Hall has one trough, one wire, one wheel — the metaphor's own parallelism is seven swarms biting *the same wire in one dip*, i.e. instruction-level parallelism, not seven halls. Concretely: an OpenMP fork costs ~10⁴ cycles ≈ the time to hash ~100 KB here, so threading would lose at every plausible bench size. `restrict`, whole-word `memcpy` loads, constant-immediate rotations and a 56-byte fully-unrolled body are the vectorization-side hints instead.

**Known-technique check (step 4):** multi-lane striped accumulation folded by a strong terminal avalanche *is* the validated real-world answer (xxHash64/XXH3), so I let the metaphor land on it rather than inventing around it — and the brine is not a new finalizer but Stafford's `mix13` (the splitmix64 finalizer), measured bias <0.1%. The novel parts are only the ones the native actually insists on: seven lanes instead of four, a shared prefix-sum wax instead of independent lanes, and zero multiplies in the stream.

## 4. ARTIFACT

Design iterations (4, all pre-measurement reasoning — no tools in this session): (1) seven independent lanes, no trough — fast but silently deleted the native's central object; (2) reinstated the shared wax `W` with per-mark accumulation and lanes biting the wire *after* each mark — costs ~25% throughput, buys literalness **and** cross-lane diffusion (a flipped bit in mark 3 reaches lanes 3–6 and every later lane); (3) replaced my invented six-stage finalizer with the validated `mix13` plus a length flower; (4) added the thin-pile regime check and reused it as the tail.

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===================== THE WEIGHING HALL =====================
   W        : the grey wax trough - one shared running impression of
              every mark so far ("only the sum of every collision")
   v0..v6   : seven bite-patterns on the dipped wire, one per jar,
              each tuned to its own hunger and its own rhythm
   rotl     : the spinning slate wheel - forward only, fixed notches
   brine()  : one plunge, six frost-flowers, frozen to the fixpoint
   return   : the tin sketch; wax, wire and swarms are all remelted
   ============================================================= */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));            /* never turns back */
}

static inline uint64_t load64(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;  /* one mark, read whole */
}

/* the brine: six frost-flowers, no two alike. Stafford mix13 (the
   validated splitmix64 finalizer) is the fixpoint where the frost-lace
   stops spreading - stop there, not one round further. */
static inline uint64_t brine(uint64_t x, uint64_t len) {
    x ^= len * 0x9E3779B97F4A7C15ULL;              /* flower 1: pile depth */
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;      /* flowers 2, 3 */
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;      /* flowers 4, 5 */
    x ^= x >> 31;                                  /* flower 6 */
    return x;
}

/* one trickle: the thin-pile regime and the end-of-pile remainder.
   A single accumulator, strictly in order - the old way, kept exactly
   where the native keeps it: for piles too thin to feed seven jars. */
static inline uint64_t trickle(uint64_t h, const unsigned char *p, size_t r) {
    while (r >= 8) {
        uint64_t w = load64(p);
        h = rotl64(h ^ w, 31) + w;
        p += 8; r -= 8;
    }
    if (r) {
        uint64_t k = (uint64_t)r * 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < r; i++) k ^= (uint64_t)p[i] << (8u * (unsigned)i);
        h = rotl64(h ^ k, 23) + k;
    }
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t r = len;

    /* weigh the pile in the hand: too thin to reach all seven jars? */
    if (r < 56)
        return brine(trickle(0x9E3779B97F4A7C15ULL, p, r), (uint64_t)len);

    uint64_t W  = 0x6A09E667F3BCC908ULL;           /* the wax */
    uint64_t v0 = 0x9E3779B185EBCA87ULL, v1 = 0xC2B2AE3D27D4EB4FULL,
             v2 = 0x165667B19E3779F9ULL, v3 = 0x85EBCA77C2B2AE63ULL,
             v4 = 0x27D4EB2F165667C5ULL, v5 = 0xD6E8FEB86659FD93ULL,
             v6 = 0xA0761D6478BD642FULL;           /* seven hungers */

    do {   /* one dip: seven marks pressed, seven swarms bite the wire */
        W += load64(p +  0); v0 = rotl64(v0 ^ W, 13) + W;
        W += load64(p +  8); v1 = rotl64(v1 ^ W, 29) + W;
        W += load64(p + 16); v2 = rotl64(v2 ^ W, 41) + W;
        W += load64(p + 24); v3 = rotl64(v3 ^ W,  7) + W;
        W += load64(p + 32); v4 = rotl64(v4 ^ W, 53) + W;
        W += load64(p + 40); v5 = rotl64(v5 ^ W, 19) + W;
        W += load64(p + 48); v6 = rotl64(v6 ^ W, 37) + W;
        W = rotl64(W, 23);                         /* the wheel notches on */
        p += 56; r -= 56;
    } while (r >= 56);

    /* lift the wire: all seven bite-patterns and the wax onto one strand */
    uint64_t h = rotl64(v0,  1) + rotl64(v1,  9) + rotl64(v2, 17)
               + rotl64(v3, 25) + rotl64(v4, 33) + rotl64(v5, 41)
               + rotl64(v6, 49);
    h ^= W * 0x9FB21C651E98DF25ULL;                /* the wax's own witness */
    h  = trickle(h, p, r);                         /* the pile's last marks */
    return brine(h, (uint64_t)len);                /* plunge once, sketch to tin */
}
```

Properties I am claiming, to be checked: every lane step `v = rotl(v ^ W, R) + W` is invertible in `v` for fixed `W`, so no mark can be swallowed; the lanes read *prefix sums* of the wax, so mark *i* of a stripe reaches lanes *i..6* of that stripe and every lane of every later stripe; the 7-lane fold uses seven pairwise-distinct rotations plus a multiplied wax term, so a change confined to one lane cannot cancel; the critical path is the wax chain, 7 adds + 1 rotate = 8 cycles per 56 bytes (≈7 B/cycle ceiling), while throughput is ~39 instructions per 56 bytes (≈5.6 B/cycle on a 4-wide core) — so the shared trough rides *just under* the issue-width bound rather than becoming the bottleneck.

## 5. PREDICTION

**PREDICTION: speedup_vs_dp = 16**

Stated before any measurement, with the model that produced it: FNV-1a is latency-bound at imul(3)+xor(1) ≈ 4 cycles/byte ≈ 0.25 B/cycle. This kernel should run ~5.5–7 B/cycle on the deep path (≈20–25 GB/s at 3.5 GHz, in-cache), giving ~22–28× on large piles; if the harness blends in short inputs (16–64 B) where the fold+brine fixed cost (~25 cycles) dominates, those cases give only ~8–10×. Blended central estimate: **16**, plausible range 8–28.

Avalanche prediction: **~0.500 mean output-bit flip fraction** (score ≥0.99 if normalized to 1.0), with no input-bit position below 0.47 or above 0.53 — because any nonzero difference surviving into the brine is avalanched by `mix13`, whose measured bias is <0.1%, and injectivity of every lane step guarantees the difference is nonzero.

Falsifiers I will accept: avalanche < 0.47 would mean the stream is cancelling differences before the brine (most likely suspect: the additive 7-lane fold); throughput < 4× would mean the shared wax chain, not issue width, is the binding constraint and SEED 1's trough must be coarsened to one press per stripe.

## 6. MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were unavailable in this session — no tools were callable, so there is no measured throughput and no measured avalanche score to report. I will not dress up the model above as data: the four revisions listed in ARTIFACT are reasoned design iterations, not measured ones, and the prediction is untested. The pipeline's compile-and-measure step is the first real contact with evidence.

The one thing I can assert without the bench is compile-and-run safety: no out-of-bounds read (all loads are bounds-guarded; sub-8-byte tails are read byte-wise), no UB rotation (every rotation amount is a nonzero compile-time constant), no strict-aliasing violation (`memcpy` loads), `len == 0` returns `brine(seed, 0)` without dereferencing `data`, and the signature is the contract's verbatim.

## 7. VERDICT

The native's apparatus, translated literally, lands on a construction that is recognizably the validated state of the art — striped multi-lane accumulation plus a strong terminal avalanche — which is the right outcome, not a disappointment: it means the metaphor is sound rather than merely exotic. Where it genuinely departs from xxHash it departs in the direction the native insisted on, and each departure is a claim the bench can refute:

- **Seven lanes, not one accumulator** — the assumption actually broken. Should be the whole speedup.
- **Zero multiplies in the hot loop** — "counts and rhythms" only. Multiplies are confined to the brine, where they cost O(1).
- **A shared wax trough feeding lanes prefix sums** — the costliest fidelity choice (~25% throughput vs. independent lanes). It pays for itself in cross-lane diffusion, but it is the first thing I would coarsen (one press per 56-byte stripe instead of per mark) if measurement shows the wax chain binding.
- **Freeze to the fixpoint, not past it** — six flowers, a known-tested finalizer, no extra rounds.

Named risk, and how it is addressed rather than merely confessed: the seven-fold dipping has fixed setup and fold cost, so it would *lose* to byte-at-a-time FNV-1a on very short buffers. That is guarded by the `len < 56` weighing-of-the-pile check, which falls back to a single in-order accumulator — the simpler path — and the same path absorbs the tail. No unguarded risky mechanism ships. Threading is deliberately absent, justified by both the metaphor (one hall, one trough, one wire) and arithmetic (fork cost ≈ 100 KB of hashing).

If the measured avalanche comes back below 0.47, my first suspect is named in advance: the purely additive part of the 7-lane fold. The fix would be to multiply each rotated lane by its own odd constant before summing — still O(1), still inside the metaphor (each swarm's bites weighed by its own hunger as the wire is lifted).