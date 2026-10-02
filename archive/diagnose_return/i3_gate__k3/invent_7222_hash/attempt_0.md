## 1. SEED → PROBLEM MAPPINGS

### SEED 1 — "the stone never resets between marks; every fold carries the callus of all folds before it"

| World object | Problem object |
|---|---|
| die-stone | the running hash state |
| never resets | state is initialized once (from a fixed inherited basis), never re-zeroed inside the buffer |
| callus of all previous folds | serial data dependence: state after mark *i* is a function of marks 0..*i* |
| one fold per mark, no more | exactly one mixing step per unit of input — no extra rounds |

**Assumption broken:** *"more mixing rounds always means better mixing."* The native folds **once per mark, never twice** — the pile's own length supplies the rounds. But the "never resets, carry it forward" part is *exactly* what FNV-1a already does, so this seed is the least different from the known way.

### SEED 2 — "each mark's weight is pressed into the stone's already-turned position, not onto a clean face"

| World object | Problem object |
|---|---|
| die-stone with faces | the state is an **orientation**, i.e. several values (4 lanes), not one number |
| "turn it a quarter through the chalk-bank groove" | a quarter revolution = advance the **seat index** by one of four → lane cycle `q = (q+1) & 3` |
| "the same seated place" | the byte always lands in *the seat currently facing the desk* — same code, different lane |
| "already-turned position, not a clean face" | mark *i* and mark *i+1* land in **different** lanes → they are not in the same dependence chain |
| "early marks bend how later marks land" | within one lane, successive presses are serially dependent (multiply by the accumulated value's rotation) |
| "no two folding-paths that started differently walk the same last step" | each fold must be a **bijection** on the state (rot, add, xor, multiply-by-odd) — distinct prefixes can never merge |

**Assumptions broken:** **"the state is a single accumulator updated in place, one value"** (the state is a four-seat orientation), *and* "each byte must be mixed into the running state before the next byte is read" (four presses are in flight at once, because they land on four different faces).

### SEED 3 — "only the final seated number leaves the desk; groove-dust and intermediate turns are swept away"

| World object | Problem object |
|---|---|
| groove-dust, chalk residue, intermediate turns | internal state wider than the output; nothing of it is returned |
| "lift the stone, read it against the wooden numbered keeps" | a **finalizer**: fixed external constants applied once at the end (xorshift–multiply avalanche chain) |
| "only that reading is the token" | the 256-bit lane state is collapsed to 64 bits exactly once, at the end |

**Assumption broken:** also **"the state is a single accumulator updated in place, one value"** — the state may be wider than the result precisely *because* the residue is discarded. (Secondarily it breaks "the whole buffer must be read … and that's the answer": the answer is a *reading of*, not the state itself.)

---

## 2. CHOSEN SEED

**SEED 2.** Two of the three seeds (2 and 3) break the preferred assumption "the state is a single accumulator updated in place, one value." Seed 2 is chosen because its mapping is the most literal — a quarter turn is a *concrete, countable* four-position cycle, which forces a concrete number of lanes rather than a vague "wider state" — and it is the most different from the known way: FNV-1a/xxHash-the-baseline fold the buffer through **one** chain; Seed 2 forbids two consecutive marks from touching the same face. Seed 3 is kept as a subordinate license (it is what permits a 256-bit state behind a 64-bit output and supplies the "wooden keeps" finalizer). Seed 1 is rejected as least different — it describes the baseline.

## 3. ASSUMPTION BROKEN

> **the state is a single accumulator updated in place, one value**

Replaced by: the state is a **die's seated orientation** — four 64-bit faces plus a seat index that advances one quarter per mark. Mark *i* presses only face *i* mod 4. Consecutive marks therefore sit on **independent dependence chains** and the multiply latency of one press overlaps the next three presses. The baseline's single chain is latency-bound (≈4–5 cycles per *byte*); four seats make it throughput-bound instead.

Per step 4, I let the mechanism land on a **validated** technique rather than invent one: four-lane striping with a bijective rot/add/multiply round and an xorshift–multiply finalizer is the xxHash64/SMHasher-validated family. The metaphor independently produced the lane count (quarter turn = 4 seats), the bijectivity requirement ("no two folding-paths … walk the same last step"), the one-round-per-mark rule, and the final-read-only output. I did not invent a novel mixer.

### Full literal object map

| World | Computation |
|---|---|
| a mark, read "as a weight, a flat thing invested into three" | 2³ = 8 raw bytes read as one little-endian `uint64_t` weight |
| the desk's numbered groove | the byte buffer, read strictly in order, sequentially |
| cycle-light, same starlight colour at any hour | fixed compile-time constants; data-independent control flow in the hot loop |
| bone-colored die-stone, inherited | the fixed initial basis `STONE`, never derived from the data |
| four seats of the chalk-bank groove | `l0..l3`, four 64-bit lanes |
| quarter turn per mark | `q = (q+1) & 3` (unrolled to a fixed 4-lane pattern in the bulk loop) |
| "drop the weight into the seated place" | `lane += w` |
| "the stone's memory bends how deep this one goes" | `rotl64(·, 29) * K1` — the press depth is the *accumulated* value's own rotation, multiplied |
| "nothing washes clean" | no lane is ever reset mid-buffer |
| one fold per mark, centuries like a handful | exactly one round per 8 bytes; one `do`-free `while` loop, O(n), no extra passes |
| wooden numbered keeps at the desk's edge | the 4 merge rounds + 5-step avalanche finalizer, fixed constants, run exactly once |
| sweep the residue away | 256 bits of lane state discarded; only the 64-bit reading returns |
| "two different piles never seat the same way" | the pile's length is folded in before the reading |
| **"never lifting it fully off the desk"** | **no threads.** The metaphor forbids a second stone; OpenMP is rejected on the metaphor's own authority, which also removes the thread-spawn risk at benchmark sizes |

### Regime recognition (step 5)

The known_way section describes two regimes ("centuries of marks … a handful"; byte-at-a-time vs. whole-buffer). The native recognizes which he is in **by whether the pile is long enough for the stone to complete a revolution of the groove**:

* `len >= 64` — **centuries**: the full four-seat groove, 64 bytes (two revolutions) per iteration, one light press per mark.
* `len < 64` — **a handful**: there is no groove to turn, so he keeps the stone in one seated place (single accumulator, zero lane set-up / lane-merge cost) and, because there are few presses, **presses each one deeper** (a pre-weighting multiply on the off-critical-path side, free in latency terms). This is the guarded fallback to the simpler path demanded by step 4 — the four-seat mechanism's only stated risk is lane set-up + 4-way merge overhead on short piles, and it is paid only when `len >= 64`.

Tail marks (`len % 64`) are still pressed one per quarter-turn, and leftover single bytes one per quarter-turn, so no input byte is ever skipped or double-counted.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the wooden numbered keeps: fixed, "that same starlight colour no matter the hour" ---- */
#define K1    0x9E3779B185EBCA87ULL
#define K2    0xC2B2AE3D27D4EB4FULL
#define K3    0x165667B19E3779F9ULL
#define K4    0x27D4EB2F165667C5ULL
#define K5    0x85EBCA77C2B2AE63ULL
#define STONE 0x9E3779B97F4A7C15ULL   /* the inherited bone-coloured die-stone */

static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }

/* a mark: "a flat thing about to be invested into three" -> 2^3 bytes read as one weight */
static inline uint64_t mark64(const unsigned char *p) { uint64_t w; memcpy(&w, p, 8); return w; }

/* light press: exactly one fold per mark; a long pile supplies its own folds.
   bijective in both arguments -> two differing folding-paths can never re-merge. */
static inline uint64_t press_light(uint64_t s, uint64_t w) { return rotl64(s + w, 29) * K1; }

/* deep press: few marks means few presses, so each press carries a pre-weighting
   (the extra multiply depends only on the mark, so it is off the state's critical path). */
static inline uint64_t press_deep(uint64_t s, uint64_t w)  { return rotl64(s + w * K2, 31) * K1; }

/* one leftover mark, pressed alone */
static inline uint64_t press_byte(uint64_t s, unsigned b)  { return rotl64(s ^ (uint64_t)b, 11) * K3; }

/* lifting the stone and reading it against the keeps -- the only thing that leaves the desk */
static inline uint64_t read_keeps(uint64_t h) {
    h ^= h >> 33; h *= K4;
    h ^= h >> 29; h *= K5;
    h ^= h >> 32;
    return h;
}
static inline uint64_t seat_lane(uint64_t acc, uint64_t lane) {
    return (acc ^ (rotl64(lane * K2, 31) * K1)) * K1 + K4;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    const unsigned char *const end = data + len;

    /* ---- regime check: is this a handful, or centuries? ---- */
    if (len < 64) {
        /* a handful: no groove to turn, one seated place, each mark pressed deep */
        uint64_t h = STONE + (uint64_t)len * K3;
        while ((size_t)(end - p) >= 8) { h = press_deep(h, mark64(p)); p += 8; }
        while (p < end)                { h = press_byte(h, *p++); }
        return read_keeps(h + (uint64_t)len);
    }

    /* centuries: the four seats of the chalk-bank groove, a quarter turn per mark,
       two full revolutions (64 bytes) per sweep of the groove */
    {
        uint64_t l0 = STONE + K1 + K2;
        uint64_t l1 = STONE + K2;
        uint64_t l2 = STONE;
        uint64_t l3 = STONE - K1;

        while ((size_t)(end - p) >= 64) {
            l0 = press_light(l0, mark64(p +  0));
            l1 = press_light(l1, mark64(p +  8));
            l2 = press_light(l2, mark64(p + 16));
            l3 = press_light(l3, mark64(p + 24));
            l0 = press_light(l0, mark64(p + 32));
            l1 = press_light(l1, mark64(p + 40));
            l2 = press_light(l2, mark64(p + 48));
            l3 = press_light(l3, mark64(p + 56));
            p += 64;
        }

        /* the tail of the pile: still one press and one quarter turn per mark */
        {
            uint64_t lane[4];
            unsigned q = 0;
            lane[0] = l0; lane[1] = l1; lane[2] = l2; lane[3] = l3;
            while ((size_t)(end - p) >= 8) {
                lane[q] = press_deep(lane[q], mark64(p));
                p += 8; q = (q + 1) & 3u;
            }
            while (p < end) {
                lane[q] = press_byte(lane[q], *p++);
                q = (q + 1) & 3u;
            }
            l0 = lane[0]; l1 = lane[1]; l2 = lane[2]; l3 = lane[3];
        }

        /* read the final seated position; sweep the rest off the desk */
        {
            uint64_t acc = rotl64(l0, 1) + rotl64(l1, 7) + rotl64(l2, 12) + rotl64(l3, 18);
            acc = seat_lane(acc, l0);
            acc = seat_lane(acc, l1);
            acc = seat_lane(acc, l2);
            acc = seat_lane(acc, l3);
            acc += (uint64_t)len;
            return read_keeps(acc);
        }
    }
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 14.0**

Reasoning, stated before any measurement: the FNV-1a baseline is latency-bound on a single chain — `xor` + `imul` ≈ 4–5 cycles **per byte** (≈0.8–1.0 GB/s). The four-seat loop's critical path per lane is `add` + `rot` + `mul` ≈ 5 cycles, and each lane consumes 16 bytes per 64-byte sweep, so 64 bytes cost ≈10 cycles of latency and only 8 `imul` issues → ≈6.4 bytes/cycle ≈ 25 GB/s on L1/L2-resident data, falling to whatever single-core DRAM bandwidth allows (~10–20 GB/s) on buffers past L3. Hence ~25× best case, ~12× memory-bound, ~2–3× on 16-byte keys; 14× is my honest point estimate for a mixed or large-buffer benchmark.

Secondary prediction: **avalanche ≈ 0.50 (within 0.49–0.51)**. Every press is a bijection, so a one-bit input flip guarantees a non-zero lane difference; the merge rounds are bijective in each lane, so a single changed lane cannot cancel in the sum; and the final five-step xorshift–multiply chain avalanches any non-zero 64-bit difference, including a high-bit-only one (that is what the leading `h ^= h >> 33` is for).

## MEASUREMENT

**Not measured — and I will not pretend otherwise.** This session was invoked with no tools available (`hash_bench`, `hash_contract` and the symbolic/unconventional tools are all absent), so there are exactly zero measurements behind this artifact and zero of the permitted four improvement rounds were spent. The prediction above is unvalidated arithmetic about ports and latencies, not data. The pipeline that compiles this will produce the first real numbers.

Falsifiable thresholds I commit to in advance, so the measurement can cleanly refute me:

* `speedup_vs_dp < 6` on a ≥1 MB buffer → the four-seat mechanism failed; the likely cause is that `memcpy`-based loads did not fold into single `mov`s, or the benchmark is DRAM-bound at a level where the baseline is also near bandwidth.
* `speedup_vs_dp < 1.0` at any tested size → the regime guard is mis-tuned and the `len < 64` threshold must rise.
* `avalanche < 0.49` → the one-press-per-mark rule (SEED 1's "centuries fold the same as a handful") is too thin and `press_light` must become `press_deep` in the bulk loop, costing ≈1.6× throughput. This is the single most likely failure point, and it is the first thing I would change.

## VERDICT

The metaphor is not decoration here: it produced a specific, defensible design and then *constrained* it in three places where I would otherwise have over-engineered.

* It gave the lane count. "A quarter turn" is 4, not "some number of accumulators I'll tune" — and 4 lanes × 16 bytes is, by coincidence or not, close to the ILP sweet spot for a 3-cycle 64-bit multiply.
* It forbade threads. "Never lifting it fully off the desk" rules out a second stone, so there is no OpenMP, and therefore no thread-spawn overhead risk to guard against. I did not need a size check for a mechanism the metaphor refused to let me build.
* It demanded bijectivity. "No two folding-paths that started differently ever end up walking the same last step" is literally injectivity of the round function, which is why every operation is rot/xor/add/multiply-by-odd and nothing is a lossy shift.

Stated risks, and how each is addressed rather than left hanging:

1. *Lane set-up and the 4-way merge are pure overhead on short inputs.* Guarded: `len < 64` takes a single-accumulator path with no lanes and no merge, which is the simpler known-way-shaped path. This is the one condition where the mechanism could lose, and it is fenced off by a runtime size check, not by hope.
2. *One light press per mark may under-mix.* Partly addressed (deep presses on the short path and on all tail marks, plus a validated finalizer), partly a live risk — it is the explicit refutation threshold above. If avalanche comes in below 0.49 the honest fix is to accept the ~1.6× throughput loss, not to argue with the number.
3. *The whole thing is unmeasured.* That is the real verdict. The design is principled and the arithmetic is checkable, but a hash kernel that has never been run through a bench is a hypothesis, not a result. Treat 14.0 as a claim to be shot at.