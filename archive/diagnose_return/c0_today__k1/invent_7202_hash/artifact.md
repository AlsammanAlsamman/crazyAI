No tools are available in this session, so every number below that is labelled a prediction is a prediction, and the MEASUREMENT section honestly reports that nothing was measured here.

---

## MAPPING

### SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."

| World object | Problem object |
|---|---|
| sailor-chain | the hash state, a **row of lanes**, not one number |
| one knot added per mark | each byte lands on exactly one lane; the row is the chain |
| "never breaking an old knot, only ever lengthening" | lanes are only ever updated, never reset or re-read from memory |
| the wall's **six fixed cracks**, "the six places my token will ever hold" | the row is exactly **6 lanes** wide, fixed for every input length — the growing chain is read through a fixed six-slot aperture, so byte *i* lands on lane *i mod 6* |
| "the whole line hangs together or not at all" | the lanes are **coupled**: lane *j* absorbs lane *j+1* each step, and lane 5 closes the loop onto the already-updated lane 0 |
| fire → chain → wall | load → lane update → final read |

**Silent assumption broken:** *"the state is a single accumulator updated in place, one value."*

### SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."

| World object | Problem object |
|---|---|
| a mark | one input byte `b` |
| the mark's "own strokes" counted as pebbles | `popcount(b)` — the set bits of that byte, 0..8 |
| the small cup's fill | the integer 0..8 |
| the **short chain** of the brick-incense | a short swing: shift = `6 * popcount(b)`, a small multiple, six paces per pebble (six, the wall's number) |
| smoke bends the fire's throw | the byte is injected at a **data-dependent bit offset** instead of a fixed one |
| where the shadow lands on the wall | `(b \| 0x100) << (6 * popcount(b))` — the 9th marker bit is the knot's own body, guaranteeing the landing pattern is injective in `b` (values 256..511 span exactly one octave, so `v<<s == v'<<s'` forces `v=v', s=s'`) |
| absorbing the landed shadow | `lane = rotl(lane, R_j) ^ lane_next ^ landing` — **xor, rotate, and a data-dependent shift. No multiply.** |

**Silent assumption broken:** *"mixing one byte requires a multiplication."*

### SEED 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."

| World object | Problem object |
|---|---|
| waiting for the smoke to thin to one straight thread | all six lanes are folded to one 64-bit value before anything is read |
| the settled, still shadow | the folded value `h` |
| pressing the wet clay | the fold: `L0..L5` combined with six fixed crack-rotations, mixing `+` and `^` so the fold is not GF(2)-linear |
| the stream, the water running over it | **one** finalizer pass — `z^=z>>30; z*=K1; z^=z>>27; z*=K2; z^=z>>31` (SplitMix64 / Murmur3 `fmix64`) |
| "only the hardened ridges remain" | the avalanche-complete output; the two multiplies live **here only**, once per call, not once per byte |
| throwing away the smoke, the cup, the old print | no history, no scratch buffer, no per-byte retained state |
| "a hooded reader working backward from print to pile" | one-wayness: the finalizer destroys the lane structure |
| *"only the new one is kept"* | one wash, not many — **more rounds is not the goal** |

**Silent assumption broken:** *"more mixing rounds always means better mixing."* (And secondarily, the per-byte multiplication, since the only multiplies in the whole kernel are the two in this single wash.)

---

## CHOSEN SEED

**SEED 2** — the pebble-cup swinging the incense.

It is the one that breaks the preferred assumption (*"mixing one byte requires a multiplication"*) and its mapping is completely mechanical: pebbles → `popcount`, cup fill → an integer 0..8, short chain → a small multiple, swing → a variable left-shift, landing spot → the injected value. It is also maximally far from FNV-1a/xxHash, whose entire per-byte step *is* the multiply.

The three seeds are one machine, so SEED 1 supplies the state shape (six coupled lanes) and SEED 3 supplies the single terminal wash. SEED 2 governs the inner loop, which is where the known way spends all its time.

**Letting the mechanism arrive at a validated technique (step 4):** three places where the metaphor pointed at something that already exists, so I used the existing thing rather than inventing:

- *six lanes read at six fixed cracks* → striped multi-accumulator absorption, exactly xxHash's reason for 4 lanes (break the serial dependency). I did not invent a new topology, I used six because the native said six.
- *the stream and the ridges* → `fmix64`/SplitMix64 finalizer verbatim, a published, well-tested 64-bit avalanche stage. I did not design a finalizer.
- *the bent landing spot of a mark is a property of the mark alone* → the whole `(b|0x100) << 6·popcount(b)` map is memoised into a 2 KB L1-resident table, which is simply a precomputed lookup — the oldest trick there is. It drops the inner step from ~9 µops/byte to 5.

**What the metaphor explicitly denied me:** there is one fire and one piazza. No second fire ⇒ **no thread parallelism**. This also happens to be right — hashing a few KB is memory-and-µop bound, and OpenMP fork/join would cost more than the entire hash at these sizes. I default to ILP (six independent dependency chains) + `restrict`, per the instruction to prefer that over threads.

---

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."**

The inner loop contains **zero multiplications**. Per byte: one zero-extending load, one table load, one rotate-by-constant, two xors. The multiply is exiled to the single terminal wash, where it is paid once per call instead of once per byte — which is exactly what the native does by throwing away every intermediate print and keeping only the one that goes into the stream.

Also broken, as a consequence: *"the state is a single accumulator updated in place"* (six coupled lanes), and *"more mixing rounds always means better mixing"* (one wash, not three).

**Two regimes, recognised in-metaphor (step 5).** The known way has a short-input regime and a long-input striped regime. The native weighs the pile in his hand before the first mark passes the fire: a pile too short to carry the chain once around all six cracks never gets the chain at all — it walks on **a single knot**. That is the `len < 48` guard (48 = 8 full six-knot passes, comfortably past the 6 passes needed for the coupling to reach every lane), falling back to a one-lane serial walk that pays no fold and no lane setup. Both paths use the same bent-shadow landing table and the same single wash, so the two regimes are one machine, not two hashes bolted together.

**Self-stated risk, addressed:** my own verdict below says the six-lane chain only pays for itself once the pile is long enough to amortise the six-lane fold. That risk is guarded by the `len < 48` branch with the single-knot fallback, not left as a caveat.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- *
 *  SAILOR-CHAIN HASH
 *  six knots on a closed chain, hung between the fire and the wall;
 *  each mark's pebble-count swings the incense, bending where its
 *  shadow lands; one wash in the stream at the end, nothing kept.
 *  No multiplication anywhere inside the walk.
 * ---------------------------------------------------------------- */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* the stream: wash the clay stub until only the hardened ridges remain.
   SplitMix64 / Murmur3-style fmix64 - a published, validated avalanche
   stage. ONE wash, never more: the old print is thrown away, not stacked. */
static inline uint64_t ridges(uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

/* SHADOW[b] = (b | 0x100) << (6 * popcount(b))
   the cup's fill (popcount) swings the incense six paces per pebble;
   the 0x100 marker is the knot's own body. Because every entry lies in
   [256,511] before the swing - exactly one octave - v<<s == v'<<s'
   forces v==v' and s==s', so distinct marks NEVER land identically.
   Max shift 48, max bit touched 56: nothing is ever shifted off the wall. */
static uint64_t SHADOW[256];
static int      SHADOW_LIT = 0;

static void light_the_fire(void) {
    for (unsigned b = 0; b < 256u; ++b)
        SHADOW[b] = (uint64_t)(b | 0x100u)
                  << (6u * (unsigned)__builtin_popcount(b));
    SHADOW_LIT = 1;            /* idempotent: a race rewrites equal values */
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (!SHADOW_LIT) light_the_fire();

    const unsigned char * __restrict p = data;
    const uint64_t      * __restrict T = SHADOW;

    /* ---- weigh the pile in the hand: which regime is this? ----
       a pile too short to carry the chain once around all six cracks
       never gets the chain; it walks on a single knot.            */
    if (len < 48) {
        uint64_t h = 0x9E3779B97F4A7C15ULL
                   ^ ((uint64_t)len << 32) ^ (uint64_t)len;
        for (size_t i = 0; i < len; ++i) {
            uint64_t s = T[p[i]];
            h = rotl64(h ^ s, 23) + s;     /* xor, rotate, add - no multiply */
        }
        return ridges(h);
    }

    /* ---- the six fixed cracks: the only six places the token holds ---- */
    uint64_t L0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    uint64_t L1 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t L2 = 0x165667B19E3779F9ULL;
    uint64_t L3 = 0x27D4EB2F165667C5ULL;
    uint64_t L4 = 0x85EBCA77C2B2AE63ULL;
    uint64_t L5 = 0xD6E8FEB86659FD93ULL;

    /* Each knot takes its own mark AND its neighbour's hang: the whole line
       hangs together. Lanes 0..4 read the neighbour's PRE-pass value (free,
       no extra move); lane 5 closes the loop onto the already-updated lane 0,
       so information runs round the ring and every lane feels every mark.
       The update matrix is invertible over GF(2): a difference, once
       introduced, can never die out before the fold. */
    size_t i = 0;
    for (; i + 12 <= len; i += 12) {
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i +  0]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i +  1]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i +  2]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i +  3]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i +  4]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i +  5]];
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i +  6]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i +  7]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i +  8]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i +  9]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 10]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i + 11]];
    }
    for (; i + 6 <= len; i += 6) {
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i + 0]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i + 1]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i + 2]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i + 3]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 4]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i + 5]];
    }

    /* the last few marks, each still on its own knot */
    switch (len - i) {
        case 5: L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 4]];  /* fall through */
        case 4: L3 = rotl64(L3, 31) ^ L4 ^ T[p[i + 3]];  /* fall through */
        case 3: L2 = rotl64(L2, 19) ^ L3 ^ T[p[i + 2]];  /* fall through */
        case 2: L1 = rotl64(L1, 11) ^ L2 ^ T[p[i + 1]];  /* fall through */
        case 1: L0 = rotl64(L0,  7) ^ L1 ^ T[p[i + 0]];  /* fall through */
        default: break;
    }

    /* ---- the smoke thins to one thread; press the clay to the wall ----
       read the settled shadow against the six fixed cracks. '+' and '^'
       alternate so the fold is not linear over GF(2); each lane enters
       injectively, so a difference in any one lane survives the press. */
    uint64_t h = L0 ^ rotl64(L1, 13);
    h = (h + rotl64(L2, 29)) ^ rotl64(L3, 43);
    h = (h ^ rotl64(L4, 17)) + rotl64(L5, 53);
    h ^= (uint64_t)len;

    /* ---- the stream. nothing else is kept. ---- */
    return ridges(h);
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 3.2**

Stated before any measurement, with the reasoning so it can be scored as a prediction rather than a guess:

- **Reference FNV-1a** is latency-bound on a single serial chain: `xor` (1 cycle) + `imul` (3 cycles) = **~4.0 cycles/byte**. `-O3` cannot vectorise it; the multiply chain is irreducible.
- **This kernel**, inner loop, per byte: `movzx` load, table load, `rol imm`, `xor`, `xor` = **5 µops, 2 loads**. Six independent chains means it is throughput-bound, not latency-bound (per-lane latency is rot+xor+xor = 3 cycles per six bytes = 0.5 cycles/byte, far under the throughput floor). On a 4-wide / 2-load-port core: max(5/4, 2/2) ≈ **1.25 cycles/byte**, call it 1.3 with loop overhead. On Zen 4/5 (6-wide, 3 load ports) it should reach ~0.9.
- 4.0 / 1.3 ≈ **3.1**; with wider cores, up to ~4.4. I predict **3.2** and would not be surprised by anything in **2.6–4.0**. Below 2.2 I would call the prediction wrong.
- **Avalanche**: I predict **0.49–0.51** (essentially ideal). The argument is not hand-waving: the lane-update map is invertible over GF(2), so a single flipped input bit yields a provably nonzero lane difference at the fold; each lane enters the fold injectively; and `fmix64` avalanches any nonzero 64-bit difference. The data-dependent swing (`6·popcount`) supplies the nonlinearity that pure rotate-xor would lack.
- The 2 KB `SHADOW` table is the one real quality-of-implementation risk: it is L1-resident and input-indexed, so it is fast but it is a cache-timing side channel. This hash is not a MAC and makes no secrecy claim, so I accept it — but I am stating it rather than hiding it.

---

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were not available in this session, so I have no throughput number and no avalanche score to report. Reporting the prediction as if it were a result would be the one thing I will not do.

What the harness should check, in order:

1. **Throughput vs the FNV-1a reference** across sizes — specifically at `len = 8, 16, 32` (short-knot path), `len = 64, 256` (chain path, fold still visible), and `len ≥ 4096` (chain path amortised). The claim is that the crossover is handled, not that one size wins.
2. **Avalanche**. If it comes back below ~0.45 the most likely culprit is cancellation in the six-lane fold, and the first repair I would try is replacing the fold's `^`/`+` ladder with a `+ rotl` tree, not adding a second `ridges` pass — a second wash would be exactly the "more rounds is better" assumption the native rejects, and I would rather fail honestly on his terms than quietly become xxHash.
3. **First-call cost.** If the bench times a single cold call on a tiny buffer, the 256-iteration table init will dominate and the number will look terrible. That is a measurement artefact, not a kernel property, and the fix is to emit the 256 constants as a literal `static const` array.

---

## VERDICT

The native's machine is a real hash, and the translation is literal end to end: pebbles are `popcount`, the short chain is a small shift multiple, the bent landing spot is the injected value, the six cracks are six lanes, the chain hanging together is the ring coupling, the stream is `fmix64`, and throwing away the smoke and the old print is the absence of any retained state. Nothing in the kernel is there because a textbook said so.

The substantive claim is that **the multiply is not load-bearing per byte**. FNV-1a uses the multiply to do two jobs at once — diffuse bits upward and destroy structure. The native splits those jobs: a data-dependent *shift* places each byte somewhere structure-dependent (cheap, 1 µop), a constant rotate carries old content away from the landing zone (1 µop), and all the structure-destroying is deferred to one wash at the end. That decomposition is what buys the ~3×: the expensive nonlinearity is paid O(1) times instead of O(n) times.

**Where this is honestly weaker than the known way.** Per-byte entropy injection is thinner than FNV's — nine bits per byte landing in one of nine bit-windows, versus a full 64-bit multiply-diffusion. On a long buffer of highly structured input (say, a megabyte of `0x00` with one byte changed) the lane difference reaching the fold is a rotated sparse pattern, and I am relying entirely on `fmix64` to blow it up. A bit-flip test will not catch that; a collision-torture test on adversarial near-identical inputs might. I would not ship this as a MAC or anywhere an adversary chooses the inputs, and the input-indexed table reinforces that. For the stated job — avalanche on a byte buffer, fast — I expect it to hold.

**Risks I named and then actually closed, rather than leaving as caveats:**
- *"The six-lane chain only pays off once the pile is long enough to amortise the fold"* → guarded by `len < 48` with the single-knot fallback, which the metaphor supplies on its own (weighing the pile in the hand).
- *"Thread parallelism might help on huge buffers"* → rejected, because the metaphor has exactly one fire and because OpenMP fork/join exceeds the whole hash at benchmark sizes. ILP and `restrict` only.
- *"A second finalizer pass would be safer"* → rejected on purpose. That is the assumption the native breaks, and hedging it would mean shipping his idea with the interesting part removed.

If the measurement comes back under 2.2×, the decomposition claim is wrong at the µop level and I will say so plainly rather than retune constants until a number appears.