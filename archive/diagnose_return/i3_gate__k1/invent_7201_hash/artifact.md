## MAPPING

**SEED 1 — the grey wax trough**

| World object | Problem object |
|---|---|
| the pile | `data[0..len)` |
| a mark | one input byte |
| trough of grey wax at blood heat | an 8-byte window of the buffer, read as one 64-bit word |
| "presses its shape before the last cooled flat" | **overlapping unaligned 64-bit loads** — the jars read at offsets 0,7,14,…,42, so every word shares 1 byte with its neighbour and no load is ever one clean mark |
| "never a clean picture of any single mark, only the sum of every collision" | each byte is absorbed by *two* lanes, never in isolation |
| scraped clean and remelted | no persistent scratch memory; the window is just a moving pointer |

*Breaks:* "each byte must be mixed into the running state before the next byte is read" — bytes are absorbed in smeared, overlapping groups, not one-at-a-time-then-commit.

**SEED 2 — the seven tuned swarms and the forward-only wheel**

| World object | Problem object |
|---|---|
| seven bell-jars | **seven independent 64-bit accumulators** `a0..a6` |
| each jar tuned to a different hunger | seven distinct odd constants `H_j` added every bite |
| its own bite *count* | seven distinct rotation amounts `R_j` = 7,11,19,23,29,37,43 |
| its own *rhythm* | seven distinct byte offsets into the wax (stride 7) |
| the dipped wire | the 64-bit word the jar bites this round |
| spinning slate wheel, **forward only, never back** | `rotl` only (never `rotr`) and `+` only (never `-`) |
| "no bite can be walked off once it's landed" | carry-propagating **addition**, which cannot cancel, instead of self-inverse XOR alone |
| no multiplication anywhere in the biting | ARX-only absorb loop |

*Breaks:* **"the state is a single accumulator updated in place, one value."** Also breaks "mixing one byte requires a multiplication."

**SEED 3 — the brine basin and the six frost-flowers**

| World object | Problem object |
|---|---|
| plunging the wire **once** | one finalization pass, after the loop, not per byte |
| "must not be rushed … wait until the frost-lace stops spreading" | run the finalizer to full diffusion; never truncate it |
| "a token pulled too early is soft and lies" | weak avalanche from an unfinalized accumulator |
| six frost-flowers, **no two ever alike** | six distinct stages, no repeated shift and no repeated constant |
| sketched onto tin, wax discarded | return a single `uint64_t`; all lanes/temps die |
| "the whole flower, not a petal, comes out changed" | avalanche ≈ 0.5 over all 64 output bits |

*Breaks:* "more mixing rounds always means better mixing" — the rounds are *relocated*, not multiplied: a cheap multiply-free absorb plus exactly one strong finalize, stopped the moment the frost stops spreading.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption, and it breaks it in the most literal way available: seven jars → seven accumulators, seven hungers → seven constants, seven bite-counts → seven rotations, forward-only wheel → add-and-rotate-left only. SEED 1 and SEED 3 are not discarded — they are the load pattern and the finalizer of the same machine.

## ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."** The state is seven accumulators with no data dependency between them, so the 56-byte-deep serial `xor;imul` chain of FNV-1a becomes seven independent 4-op chains that the machine's integer ports run concurrently. Secondary breakage: no multiplication in the absorb loop at all (SEED 2's wheel and hungers), and no in-order byte-at-a-time commit (SEED 1's overlapping smear).

**Letting the metaphor land on validated techniques rather than inventions** (step 4): the seven jars arrive at *striped multi-lane accumulation*, which is exactly the xxHash64 / HighwayHash lane structure; the forward-only wheel arrives at *ARX* (add-rotate-xor) mixing, which is SipHash's and Speck's primitive; the six frost-flowers arrive at the *splitmix64 / Stafford Mix13* finalizer verbatim — a measured, published finalizer with ~0.5 avalanche — rather than a hand-rolled one. Nothing in the hot path is novel; the metaphor chose which validated pieces to assemble.

**Regime recognition** (step 5): the native's own test is whether the pile can feed every jar. `len < 56` (= 7 jars × 8 bytes) takes the one-jar single-trough path; `len >= 56` takes the seven-jar striped path. Both paths end in the same brine.

**Why no SIMD and no threads:** the metaphor's lane count is prime and its rhythm is stride-7 overlapping, so a 4- or 8-wide vector load cannot express it without forging an eighth jar and dropping the smear; 7-way scalar ILP already saturates the ALU ports. The Hall is one Hall — there is no metaphor-unit large enough to justify OpenMP, and a thread-split would make the digest depend on thread count.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the slate wheel: it only ever turns forward, never back ---- */
static inline uint64_t wheel(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* ---- the wax: a mark is never read clean, only as an 8-byte smear ---- */
static inline uint64_t wax(const unsigned char *p) {
    uint64_t w;
    memcpy(&w, p, 8);          /* -O3 emits one unaligned mov */
    return w;
}

/* ---- the brine basin: frost-flowers 2..6, no two alike.
       splitmix64 / Stafford Mix13 finalizer, used verbatim, not invented.  ---- */
static inline uint64_t frost(uint64_t h) {
    h ^= h >> 30;                       /* flower 2 */
    h *= 0xBF58476D1CE4E5B9ULL;         /* flower 3 */
    h ^= h >> 27;                       /* flower 4 */
    h *= 0x94D049BB133111EBULL;         /* flower 5 */
    h ^= h >> 31;                       /* flower 6 */
    return h;
}

/* one swarm biting the wire: xor the smear in, turn the wheel forward,
   add the smear back (a bite cannot be walked off), add the jar's hunger. */
#define BITE(a, word, R, H) \
    do { uint64_t _w = (word); (a) = wheel((a) ^ _w, (R)) + _w + (uint64_t)(H); } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ---- regime test: is the pile thick enough to give all seven jars a bite? ---- */
    if (len < 56) {                      /* too thin: open one jar, bite the marks directly */
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        size_t i = 0;
        for (; i + 8 <= len; i += 8) BITE(h, wax(p + i), 11u, 0x27220A95u);
        uint64_t t = 0;
        for (; i < len; i++) t = (t << 8) | (uint64_t)p[i];
        BITE(h, t, 23u, 0x165667B1u);
        h ^= (uint64_t)len;
        return frost(h);                 /* same brine, never skipped */
    }

    /* ---- seven jars, seven hungers, seven bite-counts, seven rhythms ---- */
    uint64_t a0 = 0x9E3779B97F4A7C15ULL, a1 = 0xBF58476D1CE4E5B9ULL,
             a2 = 0x94D049BB133111EBULL, a3 = 0xD6E8FEB86659FD93ULL,
             a4 = 0xA0761D6478BD642FULL, a5 = 0xE7037ED1A0B428DBULL,
             a6 = 0x8EBC6AF09C88C6E3ULL;

    size_t rem = len;
    while (rem >= 56) {                  /* reads at most p[0..49]; 56 keeps a safe margin */
        BITE(a0, wax(p +  0),  7u, 0x27220A95u);
        BITE(a1, wax(p +  7), 11u, 0x165667B1u);
        BITE(a2, wax(p + 14), 19u, 0x05EBCA77u);
        BITE(a3, wax(p + 21), 23u, 0x42B2AE3Du);
        BITE(a4, wax(p + 28), 29u, 0x27D4EB2Fu);
        BITE(a5, wax(p + 35), 37u, 0x165667C5u);
        BITE(a6, wax(p + 42), 43u, 0x1E3779B1u);
        p   += 49;                       /* stride 7 x 7 jars: the wax never cools clean */
        rem -= 49;
    }

    /* ---- the last marks, round-robin so no jar is starved ---- */
    uint64_t lane[7] = { a0, a1, a2, a3, a4, a5, a6 };
    static const unsigned char RT[7] = { 7u, 11u, 19u, 23u, 29u, 37u, 43u };
    static const uint32_t      HT[7] = { 0x27220A95u, 0x165667B1u, 0x05EBCA77u,
                                         0x42B2AE3Du, 0x27D4EB2Fu, 0x165667C5u,
                                         0x1E3779B1u };
    unsigned j = 0;
    size_t i = 0;
    for (; i + 8 <= rem; i += 8) {
        BITE(lane[j], wax(p + i), RT[j], HT[j]);
        if (++j == 7) j = 0;
    }
    uint64_t t = 0;
    for (; i < rem; i++) t = (t << 8) | (uint64_t)p[i];
    BITE(lane[j], t, RT[j], HT[j]);

    /* ---- flower 1: the wire comes out of the wax; every jar thrown back in ---- */
    uint64_t h = lane[0]
               + wheel(lane[1],  9) + wheel(lane[2], 18) + wheel(lane[3], 27)
               + wheel(lane[4], 36) + wheel(lane[5], 45) + wheel(lane[6], 54);
    h ^= (uint64_t)len;

    /* ---- plunge once, wait for the frost to stop spreading, sketch to tin ---- */
    return frost(h);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 18

Stated before any measurement, with the reasoning that produced it:

- FNV-1a is latency-bound, not throughput-bound: one `xor` + one `imul` per byte on a single carried dependency, ≈ 4 cycles/byte ≈ **0.25 B/cycle**. `-O3 -march=native` cannot vectorize it; the chain is irreducible.
- This kernel consumes 49 bytes per iteration with 7 loads + 7 `xor` + 7 `rol` + 14 `add` = 35 uops, i.e. **0.71 uops/byte**, on seven independent 4-op chains. At ~4.5 uops/cycle that is **≈ 6 B/cycle**, so ≈ 24× on an L1/L2-resident buffer, degrading toward ≈ 8–10× if the harness buffer is large enough to be DRAM-bandwidth-bound. 18 is my midpoint guess over a plausible harness size mix.
- Avalanche: I predict **0.49–0.51**. This is a prediction about splitmix64, not about my own cleverness — a one-bit input flip perturbs 1–2 lanes by 2–3 bits, that difference survives the additive fold (only one or two lanes move, so nothing cancels), and Mix13 is published as spreading any nonzero 64-bit difference to ~half the output bits.
- Sub-16-byte inputs: I predict **slightly below parity** (≈ 0.6–0.9×), because `frost` costs a fixed ~10 cycles that FNV does not pay.

## MEASUREMENT

**Not measured in this session.** `hash_bench`, `hash_contract`, and the symbolic/unconventional tools were listed but no tools are actually callable here, so I ran zero of the four allowed improvement rounds against real numbers. I am reporting that plainly rather than inventing a throughput figure: the numbers above are an instruction-count and dependency-chain estimate, and the pipeline's measurement supersedes them wherever they disagree.

Three refinements are already folded into the artifact *a priori* (each one a design decision I could not validate, and I flag them as unvalidated):

1. **Stride-7 overlapping loads** instead of 7 disjoint words. Costs 12.5% of raw throughput (49 bytes/iteration instead of 56) and buys SEED 1's smear plus a second lane touched by every byte. If measurement shows avalanche is already 0.50 with disjoint words, this 12.5% is recoverable by changing the offsets to 0,8,…,48 and the stride to 56 — the only honest reason to keep it then would be fidelity, not quality.
2. **The hunger addend `+H`.** Costs ~20% throughput (one extra `add` per lane per word). Without it, an all-zero input only rotates the lanes, so all-zero piles of different lengths collapse onto 64 states per lane before the length fold. The addend is an imm32, so it costs a uop but **zero register pressure** — which matters, since 7 lanes + temps + pointer + counter is already near the 16-GPR budget and 64-bit constants in registers would have spilled.
3. **The `len < 56` fallback.** Guards the striping overhead, per step 4.

What I would test first with a working `hash_bench`: whether a 7th frost-flower changes the avalanche score at all. My expectation is that it does not — which would be the metaphor's own claim ("the frost-lace stops spreading") cashing out as the measurable refutation of "more mixing rounds always means better."

## VERDICT

The seven jars are a faithful and useful translation: they dissolve FNV-1a's single carried `imul` chain into seven concurrent multiply-free chains, which is where essentially all of the predicted speedup comes from, and they do so by landing on the already-validated xxHash lane structure + SipHash-class ARX absorb + splitmix64 finalizer rather than on anything I made up. The brine basin is load-bearing in a way worth naming: without it, seven cheap ARX lanes have *poor* avalanche, and the native's warning that "a token pulled too early is soft and lies" is precisely the failure mode of shipping a multi-lane hash with a weak fold.

Named risk conditions and how each is addressed, with nothing left dangling:

- **Striping overhead on small inputs.** This is the mechanism's own stated weak regime, and it is guarded: `len < 56` falls back to the single-trough one-jar path, skipping seven lane initializations and a seven-way fold.
- **Fixed finalizer cost below ~16 bytes.** I predict slight sub-parity in cycles there and I am deliberately *not* guarding it away, because the only available "fallback" is to omit `frost`, and omitting `frost` destroys the one property the target problem actually asks for. This is not an unaddressed optimization risk; it is the cost of the requirement. The optional mechanism (striping) is guarded; the non-optional one (one strong finalize) has no fallback by construction.
- **DRAM-bandwidth saturation on very large buffers.** Shrinks the speedup but cannot invert it: FNV-1a is compute-bound at 0.25 B/cycle and never reaches the bandwidth ceiling, so the floor here is ~1×, not <1×. No guard needed.
- **Unvalidated claim I am most willing to be wrong about:** that the additive seven-lane fold never cancels in a way the avalanche test can see. If the measured score comes back near 0.5 but biased, the fix is to alternate `+` and `^` across the fold — at the cost of the native's forward-only wheel, which I would then report as the metaphor failing on a specific point rather than as a tweak.