## MAPPING

### SEED 1 — "A grey wax trough accumulates every mark's impression atop all prior impressions before any can cool clean."

| World object | Problem object |
|---|---|
| the pile | the input buffer `data[0..len)` |
| a mark | one byte (in practice one 8-byte word, the smallest press the trough takes) |
| grey wax at blood heat | state that is deliberately *unfinished* — never a clean digest, only a running sum of collisions |
| "presses before the last has cooled flat" | each byte's contribution is **not** avalanched before the next arrives; per-byte mixing is deferred |
| "never a clean picture of any single mark" | the running state has no avalanche property at all mid-stream, by design |
| scraped clean and remelted | the state is scratch, discarded, not returned |

**Silent assumption broken:** *"each byte must be mixed into the running state before the next byte is read."*

### SEED 2 — "Seven tuned insect-swarms bite a dipped wire in counts fixed forever by a forward-only spinning slate wheel."

| World object | Problem object |
|---|---|
| seven bell-jars | **seven independent 64-bit accumulators** `a0..a6` — not one |
| each jar "tuned to a different hunger" | a distinct odd 64-bit initialisation constant per lane |
| the wire dipped after every mark | the 56-byte block read: lane *i* takes bytes `[8i, 8i+8)` |
| a *bite* | an integer **addition** (`acc + word`) — a bite, not a grinding |
| "in its own count and rhythm" | a distinct rotation amount `R_i ∈ {7,11,17,23,29,37,43}` per lane |
| spinning slate wheel, **forward only, never back** | `rotl` only, never `rotr`; and `+` not `^` — a second identical bite cannot *walk off* the first the way XOR self-cancels |
| no multiplication anywhere in the streaming loop | the per-byte cost is add+rotate, 2 µops per 8 bytes |

**Silent assumptions broken:** *"the state is a single accumulator updated in place, one value"* **and** *"mixing one byte requires a multiplication."*

### SEED 3 — "A brine basin freezes the wire's bite-pattern into a unique frost-flower lattice, sketched to tin as the final token."

| World object | Problem object |
|---|---|
| plunging the wire **once**, at the end | a single finalisation, after the whole buffer, not per byte |
| "must not be rushed … wait until the frost-lace stops spreading" | use a **full-strength, validated** avalanche mixer, not a truncated one |
| "a token pulled too early is soft and lies" | a short finaliser leaves correlated output bits |
| **six** frost-flowers from **seven** jars, no two alike | a binary fold tree over 7 leaves has exactly **6 internal nodes** — six combine steps, each with a distinct rotation |
| the brine itself (salt) | `len` injected at the finaliser — the count of marks salts the basin |
| sketched onto **tin**, never the wax | the returned 64-bit token is a *derived* value; the lanes are thrown back in the trough |
| "the whole flower, not a petal, comes out changed" | avalanche ≈ 0.5 |

**Silent assumption broken:** *"more mixing rounds always means better mixing"* — one unhurried terminal mix beats `len` rushed ones.

---

## CHOSEN SEED

**SEED 2.** It is the only seed that breaks the preferred assumption (*single accumulator updated in place*) — seven bell-jars are literally seven accumulators — and it is the furthest from the known way: FNV-1a/xxHash fold the buffer through multiply-xor chains, while the swarms use only *bites* (adds) and a *forward-only wheel* (rotates), with no multiplication at all until the brine. Seeds 1 and 3 are the same native's same ritual and are kept as the deferral discipline and the fold/finalise discipline respectively, but SEED 2 is the mechanism.

## ASSUMPTION BROKEN

1. **"The state is a single accumulator updated in place, one value."** → seven lanes, seven independent dependency chains. The FNV baseline is *latency*-bound (xor→imul, ~4 cycles per byte); seven lanes are *throughput*-bound instead, which is the entire speedup.
2. **"Mixing one byte requires a multiplication."** → add + rotate only in the hot loop. The three multiplications in the whole kernel live in the brine.
3. **"Each byte must be mixed before the next is read"** and **"more rounds is better"** → all avalanche is deferred to one terminal mix.

**Where the mechanism lands on a validated known technique (step 4, deliberately):** multi-lane interleaved accumulation + one strong terminal avalanche *is* the xxHash64/XXH3 family architecture, and the brine is Pelle Evensen's published **moremur** mixer (a measured-lower-bias drop-in for MurmurHash3's `fmix64`). I did not invent a finaliser; the metaphor's "wait until the frost stops spreading" is satisfied by taking the best validated mixer off the shelf. What is genuinely non-standard is the multiply-free rotate-add lane step (xxHash multiplies per lane) and the 7-lane/6-node fold tree.

**Why avalanche must hold (argued before measuring):** every lane step `a ← rotl(a + w, R)` is a bijection in `a` *and* in `w`, so flipping any input bit necessarily changes exactly the lane that owns that byte. Every one of the six fold nodes `x ← rotl(x,r) + y` is a bijection in each argument, so that difference cannot be cancelled on its way out. The brine therefore receives two *distinct* 64-bit words and is a validated strong mixer → ≈32 output bits flip. Avalanche comes from the bijection chain plus one good mixer, not from per-byte work.

**Regime recognition (step 5), encoded in-world:** the known way describes two regimes — byte-at-a-time FNV (fine for short piles) vs. block folding (long piles). The native's own check: *does the pile wet all seven wires?* If the pile cannot cover the trough floor (`len < 56`, one full dip), he dips **one jar only** and goes straight to the brine — no seven-lane setup, no 6-node fold, no 42-cycle epilogue to amortise over 30 bytes. That is the guarded fallback for the small regime, and it is the explicit answer to the risk my own verdict names.

**Thread parallelism: declined, honestly.** The native has one wire and one wheel. At benchmark sizes the 7-lane loop issues ~56 bytes per ~4 cycles (~15 B/cycle, ~50 GB/s), which is already past single-socket DRAM bandwidth — above L2 the trough is bandwidth-bound, so OpenMP would buy nothing while adding fork overhead on small inputs. Vectorisation hints only (`restrict`, `memcpy` loads, compile-time rotate immediates); seven is prime and resists SIMD packing, which is also why the scalar form is the literal one.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The forward-only slate wheel: turns one way, never back. r is always 1..63. */
#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* A mark pressed into the wax: one aligned-agnostic 8-byte press, no UB. */
static inline uint64_t wax_press(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* The brine basin, salted with the count of marks, left until the frost stops
   spreading: Evensen's "moremur" mixer (validated low-bias fmix64 successor). */
static inline uint64_t brine(uint64_t w, uint64_t marks) {
    w ^= marks * 0x9E3779B97F4A7C15ULL;
    w ^= w >> 27; w *= 0x3C79AC492BA7B653ULL;
    w ^= w >> 33; w *= 0x1C69B3F74AC4AE35ULL;
    w ^= w >> 27;
    return w;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ---- regime check: does the pile wet all seven wires? ---------------- */
    if (len < 56) {
        /* Too small a pile: one jar only, then straight to the brine. */
        uint64_t a = 0x9E3779B97F4A7C15ULL;
        size_t n = len;
        while (n >= 8) { a = ROTL64(a + wax_press(p), 23); p += 8; n -= 8; }
        if (n >= 4) { uint32_t t; memcpy(&t, p, 4);
                      a = ROTL64(a + (uint64_t)t, 29); p += 4; n -= 4; }
        while (n--)  { a = ROTL64(a + (uint64_t)*p, 31); p++; }
        return brine(a, (uint64_t)len);
    }

    /* ---- seven bell-jars, each tuned to a different hunger -------------- */
    uint64_t a0 = 0x9E3779B97F4A7C15ULL;
    uint64_t a1 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t a2 = 0x165667B19E3779F9ULL;
    uint64_t a3 = 0x85EBCA77C2B2AE63ULL;
    uint64_t a4 = 0x27D4EB2F165667C5ULL;
    uint64_t a5 = 0xD6E8FEB86659FD93ULL;
    uint64_t a6 = 0xA0761D6478BD642FULL;

    /* Marks press into the trough; every jar bites the wire in its own
       count and rhythm. No multiplication, no backward turn, no XOR that
       could let a bite be walked off. Seven independent chains. */
    size_t blocks = len / 56;
    for (size_t i = 0; i < blocks; i++) {
        a0 = ROTL64(a0 + wax_press(p +  0),  7);
        a1 = ROTL64(a1 + wax_press(p +  8), 11);
        a2 = ROTL64(a2 + wax_press(p + 16), 17);
        a3 = ROTL64(a3 + wax_press(p + 24), 23);
        a4 = ROTL64(a4 + wax_press(p + 32), 29);
        a5 = ROTL64(a5 + wax_press(p + 40), 37);
        a6 = ROTL64(a6 + wax_press(p + 48), 43);
        p += 56;
    }

    /* The last marks of the pile, dealt round the jars in order (0..55 left). */
    {
        size_t n = len - blocks * 56;
        if (n >= 8) { a0 = ROTL64(a0 + wax_press(p),  7); p += 8; n -= 8; }
        if (n >= 8) { a1 = ROTL64(a1 + wax_press(p), 11); p += 8; n -= 8; }
        if (n >= 8) { a2 = ROTL64(a2 + wax_press(p), 17); p += 8; n -= 8; }
        if (n >= 8) { a3 = ROTL64(a3 + wax_press(p), 23); p += 8; n -= 8; }
        if (n >= 8) { a4 = ROTL64(a4 + wax_press(p), 29); p += 8; n -= 8; }
        if (n >= 8) { a5 = ROTL64(a5 + wax_press(p), 37); p += 8; n -= 8; }
        while (n--) { a6 = ROTL64(a6 + (uint64_t)*p, 53); p++; }
    }

    /* ---- the lattice: six frost-flowers, no two alike -------------------
       Seven leaves fold through exactly six nodes; each node is a bijection
       in both arguments, so no lane's difference can cancel another's.    */
    uint64_t b0 = ROTL64(a0,  5) + a1;
    uint64_t b1 = ROTL64(a2, 13) + a3;
    uint64_t b2 = ROTL64(a4, 19) + a5;
    uint64_t c0 = ROTL64(b0, 31) + b1;
    uint64_t c1 = ROTL64(b2, 41) + a6;
    uint64_t w  = ROTL64(c0, 47) + c1;

    /* The wax is scraped and remelted; only the tin sketch survives. */
    return brine(w, (uint64_t)len);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 14**

(stated before any measurement; secondary predictions: avalanche score ≈ 0.50 ± 0.01, i.e. ~32 of 64 output bits flip per input-bit flip, for both the seven-jar and the one-jar regime.)

Reasoning behind the number: the FNV baseline is latency-bound at ~4 cycles/byte (`xor` 1 + `imul` 3 on a single carried accumulator) ≈ 0.9 GB/s. The seven-jar loop has a 2-cycle dependency chain *per lane* and seven lanes, so it is limited by issue ports — 7 loads + 14 ALU µops per 56 bytes ≈ 3.5–4 cycles/block ≈ 14–16 B/cycle ≈ 50 GB/s in L1/L2. If the benchmark buffer is cache-resident the honest ceiling is 40–55×; if it streams from DRAM the kernel saturates bandwidth at ~12–15 GB/s and the ratio collapses to ~13–16×. I predict the bandwidth-bound case, hence 14, and I will call the prediction wrong rather than reinterpret it if the measurement lands outside 10–20×.

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were listed as available but no tools are callable in this session, so I ran zero measurements and performed zero measured improvement rounds (budget was four; used none). I am reporting that plainly rather than quoting plausible numbers: every figure in the PREDICTION section is a static reasoning estimate, not an observation.

Two revisions were made *pre-measurement*, by reasoning only, and are already in the artifact above:
1. **Removed an overlapping tail read.** The xxHash-style trick of re-reading the final 8 bytes would have let one byte land in two lanes, admitting a ~2⁻⁶⁴ cancellation between two deltas in the fold tree. Replaced with a ≤7-iteration byte loop into `a6` — cost is ~14 cycles, bought a clean one-byte-one-lane argument.
2. **Added the `len < 56` one-jar guard**, so the ~50-cycle seven-lane prologue/fold/brine epilogue is never paid on a pile too small to amortise it.

What would falsify the design, concretely: avalanche below ~0.45 would mean the bijection-plus-mixer argument is wrong somewhere (most likely suspect: the fold tree letting a tail-lane difference land in a near-dead region of the brine's input). Throughput below ~5× would mean the compiler spilled the seven lanes or failed to emit `rol` immediates, and the fix is to cut to four or five jars so everything stays in registers.

## VERDICT

The native's ritual is a *real* and correct hash architecture, and it is not the textbook one. Taken literally, it says: stop finalising per byte (the wax never cools), stop using one accumulator (seven jars), stop multiplying per byte (bites and a forward-only wheel), and spend all your mixing budget once, at the end, unhurried (the brine). That is precisely how the fastest production hashes are built — which is the honest finding here: the metaphor *converges on* the validated multi-lane + strong-finaliser design (xxHash/XXH3 architecture, moremur finaliser) rather than requiring me to invent anything, and I let it, per the rule that a validated technique beats a novel untested one. The one genuinely unorthodox element it contributes is the **multiply-free lane step**: rotate-add only, where xxHash multiplies. That should be strictly cheaper per byte and loses nothing, because the bijection argument shows per-lane strength is irrelevant when a strong terminal mixer exists.

Named risks and how each is addressed rather than left dangling:
- *"Overhead if the pile is small"* — the seven-jar epilogue (6 folds + 3 multiplies) is dead weight below a few hundred bytes. **Guarded**: `len < 56` takes a one-jar path, and the brine is shared so there is no second code-quality cliff.
- *"Only helps if the problem is large"* — the multi-lane ILP win needs enough blocks to amortise. Same guard covers it; between 56 B and ~1 KB the kernel is still at worst comparable to FNV because the inner work per byte is strictly cheaper.
- *"Threads might help on huge buffers"* — **dropped rather than shipped unguarded**, because the loop's own throughput estimate already exceeds DRAM bandwidth, so I cannot claim a thread win I have no measurement for.

Residual weakness I will not paper over: rotate-add lanes are weak as a *cryptographic* primitive and this kernel offers no collision resistance against an adversary who can choose inputs — the lane chains are ARX-without-X and the fold is additive. For the stated target (avalanche + speed, non-adversarial) that is the right trade, the same one xxHash makes. If the benchmark's avalanche score comes back below 0.45, the mapping's central claim — *defer all mixing to one terminal plunge* — is wrong at these sizes, and I would say so rather than quietly bolting a multiply back into the hot loop.