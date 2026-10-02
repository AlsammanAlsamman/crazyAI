No tools are available in this session, so everything below is reasoning + artifact only; the pipeline will have to supply the numbers. (Also, for the record: the `claude.ai PubMed` MCP server is unauthenticated here — it would need authorizing via your claude.ai connector settings before it could be used, and nothing in this task needed it. The `brainny` CLI check in my global instructions is skipped silently because no shell is available.)

---

## MAPPING

### SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."

| World object | Problem object |
|---|---|
| the pile of marks, in their given order | the byte buffer `data[0..len)`, read front to back |
| the sphere | the one state object carried across the loop |
| "the sphere's **face**" (one of several on one body) | one 64-bit lane of that state |
| the sphere *rolling* (not swelling) | the state is **permuted**, not accumulated into |
| the footprint left at a stalk, never looked back at | per-byte intermediate / history — never stored anywhere |
| "the old face I let shrink and go" | the lane the mark was pressed into is re-xored and released |

**Assumption broken:** essentially none of the five — it *affirms* "each byte mixed before the next is read" and "one carried state". Its only quarrel with the known way is that the carried thing is a *rolled body with several faces*, not a scalar updated by one arithmetic op, so it mildly dents "the state is a single accumulator updated in place, one value."

### SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."

| World object | Problem object |
|---|---|
| the **coiled** limestone trail | bit positions wrapped modulo 64 — a rotation ring (`ROTL64`) |
| one **turn / tumble** of the sphere | one full round of a fixed permutation of the state |
| the **mason** trail: stalks at fixed, pre-laid stations | the fixed rotation constants 13, 16, 21, 17, 32, 32 |
| striking a stalk, the sphere "sobs once" | one coupling of two faces: **addition** mod 2⁶⁴ (the only nonlinear act) |
| the hairline **crack**, and **the angle of that crack** | the XOR taken after a rotation; the angle *is* the rotate distance |
| the **eggshells the trail sheds**, swept into the plumbing | the carry bits falling out of the top of 64-bit addition, discarded |
| "a **fixed** count of turns, no more, no fewer" | a constant round count per mark — not tuned, not grown with `len` |
| pressing the mark into the sphere's face, then letting that face go | `v3 ^= word;` tumble; `v0 ^= word;` |
| "one wrong mark and every stalk downstream sobs differently" | avalanche: a single flipped input bit must re-route the entire remaining trail |

**Assumption broken:** **"mixing one byte requires a multiplication"** — there is no multiply anywhere in the world: a tumble is a rotation, a sob is an addition, a crack is an XOR-after-rotation. Secondarily it breaks **"more mixing rounds always means better mixing"**: the count is fixed and deliberately *small* ("no more, no fewer"), with the bulk of the work done per 8-mark bale rather than per mark.

### SEED 3 — "Only the final hairline crack in the last stalk is kept as the token; all intermediate cracks and dust swept away."

| World object | Problem object |
|---|---|
| the sphere at the end, explicitly *not* kept | the wide internal state, never returned |
| the last, smallest crack in the final stalk | the finalizer: a twist (`v2 ^= 0xff`) + closing tumbles |
| the token handed over, "small as a token" | the folded 64-bit return value `v0^v1^v2^v3` |
| dust, discarded footprints, eggshells → the cenote | all intermediates, carries, tail padding — dropped on purpose |

**Assumption broken:** "the state is a single accumulator updated in place, one value" — it forces a *separation* of a wide internal state from a narrow output token, which FNV-1a does not have (FNV returns its accumulator raw). It also re-points "more rounds is better" toward spending extra rounds **once, at the end**, instead of per byte.

---

## CHOSEN SEED

**SEED 2.** It is the only seed that breaks the preferred assumption ("mixing one byte requires a multiplication"), and its mapping is the most literal of the three: every noun in it is already a machine instruction. A coil is a modular rotation. A fixed count of turns is a round count. A stalk is a rotation constant. A sob is an `add`. A crack at an angle is `x ^= rotl(y, k)`. An eggshell shed by the trail is the carry bit that falls off the top of a 64-bit add. Nothing in it had to be stretched.

It is also maximally far from the known way: FNV-1a/xxHash are *multiply*-driven (one `imul` is the whole mixer, and it is the whole critical path); this world has no multiplier at all, and it eats the pile in bales of eight rather than one mark at a time.

---

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** Broken outright — the kernel contains zero multiplications. Mixing is add-rotate-xor only.

Also broken, as a consequence: **"more mixing rounds always means better mixing."** The native insists on a *fixed, small* count ("no more, no fewer") per bale, with the heavy rounds reserved for the single closing crack. FNV-1a spends one full mixing round per *byte*; this spends one per *eight bytes* and three at the very end — fewer rounds total, better avalanche, and ~4× the throughput.

### Where the mechanism lands (step 4: arrive at the validated technique, don't invent)

Followed literally, SEED 2 **is SipHash** — specifically **SipHash-1-3**:

- four faces of one sphere → the four 64-bit lanes `v0..v3`;
- the pre-laid mason stations → SipHash's rotation constants 13/16/21/17/32/32;
- sobs → the four `+` of a `SIPROUND`, eggshells → its `mod 2⁶⁴` carries;
- "press into the face, one tumble, let the old face go" → `v3 ^= m; SIPROUND; v0 ^= m;` **exactly**, which is SipHash's compression step with `c = 1`;
- "I take only the last, smallest crack in the final stalk" → `v2 ^= 0xff;` + `d = 3` finalization rounds + the fold `v0^v1^v2^v3`;
- "the eggshells the trail sheds along the way… swept off" → the length pressed into the top byte of the final bale (`len << 56`), so trailing structure can't be forged away.

I did **not** invent a new ARX mixer. SipHash-1-3 is a validated, deployed primitive (the SipHash family is Python's `str` hashing and the Linux kernel's `siphash`; **SipHash-1-3 specifically is Rust's default `HashMap` hasher**, i.e. it is attacked and benchmarked in production daily). Given the choice between a hand-rolled rotate-xor-add chain of my own and the thing the metaphor converges onto anyway, the metaphor's own landing point is the validated one. That is the whole argument of step 4, and here it costs nothing.

### Regime recognition, in-world (step 5)

The native weighs the pile before starting — this is a genuine runtime branch with two paths, not one strategy:

- **"a pile I cannot lift"** (`len >= 8`, with a 4-bale stride once `len >= 32`): bale the marks by eights, one tumble per bale, four bales per pass down the trail to amortise the walk itself.
- **"it fits in one hand"** (`len < 8`): no bales, no loop, no stride setup — a straight-line gather of 1–7 marks into the single final bale, then straight to the closing cracks. Strictly fewer instructions and zero loop/unroll branches on exactly the sizes where fixed overhead is all there is.

What I **refused** to add: a second sphere, or threads. SEED 1 states the sphere is the *sole* carried memory, and the native never looks back at a footprint — so there is no in-world second lane to run in parallel, and OpenMP would be me overruling the native rather than translating him. Vectorization hints only (`restrict`, single-`mov` unaligned bale loads via `__builtin_memcpy`, 4× unrolled stride); no `#pragma omp` anywhere.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

#define ROTL64(x, b) (((x) << (b)) | ((x) >> (64 - (b))))

/* ONE TUMBLE down the mason trail: four stalk-strikes.  Each strike is a
 * sob (add mod 2^64 -- its carry is the eggshell the trail sheds, swept
 * away on purpose), a turn of the coil (rotation by that stalk's fixed
 * angle), and the hairline crack read off at that angle (xor).
 * There is no multiplication anywhere on this trail. */
#define TUMBLE                                                            \
    do {                                                                  \
        v0 += v1; v1 = ROTL64(v1, 13); v1 ^= v0; v0 = ROTL64(v0, 32);     \
        v2 += v3; v3 = ROTL64(v3, 16); v3 ^= v2;                          \
        v0 += v3; v3 = ROTL64(v3, 21); v3 ^= v0;                          \
        v2 += v1; v1 = ROTL64(v1, 17); v1 ^= v2; v2 = ROTL64(v2, 32);     \
    } while (0)

/* Press the bale into the sphere's face, let it tumble a FIXED one turn,
 * then let the old face shrink and go.  No footprint is kept. */
#define PRESS(mexpr)                                                      \
    do {                                                                  \
        const uint64_t mm = (mexpr);                                      \
        v3 ^= mm; TUMBLE; v0 ^= mm;                                       \
    } while (0)

/* Gather 1..7 loose marks into the final bale, low mark first. */
#define GATHER(q, n)                                                      \
    do {                                                                  \
        switch (n) {                                                      \
        case 7: b |= (uint64_t)(q)[6] << 48; /* fall through */            \
        case 6: b |= (uint64_t)(q)[5] << 40; /* fall through */            \
        case 5: b |= (uint64_t)(q)[4] << 32; /* fall through */            \
        case 4: b |= (uint64_t)(q)[3] << 24; /* fall through */            \
        case 3: b |= (uint64_t)(q)[2] << 16; /* fall through */            \
        case 2: b |= (uint64_t)(q)[1] <<  8; /* fall through */            \
        case 1: b |= (uint64_t)(q)[0];       /* fall through */            \
        default: break;                                                   \
        }                                                                 \
    } while (0)

static inline uint64_t load_bale(const unsigned char *p)
{
    uint64_t m;
    __builtin_memcpy(&m, p, sizeof m);        /* one unaligned mov on x86-64 */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&            \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    m = __builtin_bswap64(m);                 /* the trail runs low-mark-first */
#endif
    return m;
}

uint64_t kernel(const unsigned char *restrict data, size_t len)
{
    /* The sphere at the trail's high mouth: one body, four faces. */
    uint64_t v0 = 0x736f6d6570736575ULL ^ 0x0706050403020100ULL;
    uint64_t v1 = 0x646f72616e646f6dULL ^ 0x0f0e0d0c0b0a0908ULL;
    uint64_t v2 = 0x6c7967656e657261ULL ^ 0x0706050403020100ULL;
    uint64_t v3 = 0x7465646279746573ULL ^ 0x0f0e0d0c0b0a0908ULL;

    /* The size of the pile is pressed into the top of the last bale, so a
     * pile's length cannot be swept away with its dust. */
    uint64_t b = (uint64_t)len << 56;

    if (len >= 8) {
        /* ---- REGIME A: a pile I cannot lift.  Bale the marks by eights. */
        const unsigned char *p = data;
        size_t bales = len >> 3;

        if (bales >= 4) {                 /* four bales per walk of the trail */
            size_t walks = bales >> 2;
            bales &= 3;
            do {
                PRESS(load_bale(p));
                PRESS(load_bale(p + 8));
                PRESS(load_bale(p + 16));
                PRESS(load_bale(p + 24));
                p += 32;
            } while (--walks);
        }
        while (bales--) { PRESS(load_bale(p)); p += 8; }

        GATHER(p, len & 7);
    } else {
        /* ---- REGIME B: it fits in one hand.  No bales, no loop, no stride:
         * straight to the final bale and the closing cracks. */
        GATHER(data, len);
    }

    /* the last bale takes its one fixed tumble, like every other */
    v3 ^= b; TUMBLE; v0 ^= b;

    /* I do not keep the sphere.  I take the last, smallest crack it leaves
     * in the final stalk -- three closing tumbles, then the four faces
     * folded into one token.  Everything else goes into the plumbing. */
    v2 ^= 0xffULL;
    TUMBLE; TUMBLE; TUMBLE;
    return v0 ^ v1 ^ v2 ^ v3;
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 3.5**

Stated before any measurement (and, in this session, before any measurement was *possible* — no tools were available). Reasoning, so it can be scored as a real prediction and not a hedge:

- **Baseline (FNV-1a).** The loop-carried chain is `h = (h ^ byte) * P`: one `xor` (1 cycle) plus one `imul` (3 cycles latency) per **byte**, and the multiply latency cannot be hidden because the next byte depends on it. ≈ **4 cycles/byte**, i.e. ~1 GB/s at 3–4 GHz regardless of how wide the machine is.
- **This kernel.** One `SIPROUND` per **8 bytes**. The round's critical path is roughly `v0+=v1 → rotl → v0+=v3 → rotl → xor`, call it 6–8 cycles, and the loads and the 4× unroll keep the front end out of the way. ≈ **0.8–1.1 cycles/byte**, i.e. ~3–4 GB/s — still well under DRAM bandwidth, so it should not flatten out at large sizes.
- 4.0 / 1.0 ≈ 4×; I shade it to **3.5** for tail handling, the fixed 4-round floor, and the chance the harness averages in small sizes.
- **Avalanche: I predict ≈ 0.49–0.50 (essentially ideal).** This is the claim I'm most confident of, because it is not mine: SipHash-1-3's diffusion is the property that got it adopted as Rust's default hasher.
- **Falsification conditions, stated up front.** If measured throughput speedup is < 1.5× I was wrong about the critical path (most likely cause: the harness benchmarks only tiny keys, where the fixed 4 tumbles dominate). If avalanche comes back < 0.45, `c = 1` is too few tumbles per bale for this harness's test, and the honest repair is `c = 2` (plain SipHash-2-4), trading ~40% of the throughput for the metaphor's own "fixed count, no more, no fewer" at a larger fixed count.

---

## MEASUREMENT

**Not measured — and I will not dress that up.** `hash_bench` and `hash_contract` were listed as available but no tools were actually callable in this session, so I have **zero** measured numbers: not throughput, not avalanche, not even a compile. Of the four permitted improvement rounds I used **none**, because iterating without a measurement is not improvement, it is decoration.

What the pipeline should report, and how to read it against the prediction above:

| Quantity | Predicted | Verdict if it comes back otherwise |
|---|---|---|
| `speedup_vs_dp` | 3.5 (range I'd accept as "prediction held": 2.5–4.5) | < 1.5× ⇒ prediction wrong, see falsification note |
| avalanche score | 0.49–0.50 | < 0.45 ⇒ raise to `c = 2` (SipHash-2-4) |
| compile under `gcc -O3 -march=native -fopenmp -lm` | clean (no OpenMP used, no `immintrin.h`, no libm) | — |

The one portability note: bale loads are little-endian, with a `__builtin_bswap64` guard for big-endian targets, so the token is the same on both.

---

## VERDICT

The translation is unusually clean, and that is the interesting result rather than the speed. Taken completely literally — coil = rotation ring, tumble = round, mason stalk = fixed rotation constant, sob = add, crack-angle = rotate distance, eggshell = discarded carry, "press into the face then let the old face go" = `v3 ^= m; round; v0 ^= m`, "only the last hairline crack" = finalizer-and-fold — the native is not describing a vague alternative to FNV. He is describing **SipHash-1-3, step for step**, including the detail that the length rides in the top byte of the last bale. A metaphor that lands exactly on a deployed, attacked, benchmarked primitive is the best possible outcome of step 4: the assumption "mixing one byte requires a multiplication" is broken, and broken by something that has already survived the real world, not by something I made up this afternoon.

The assumption breaks are real, not cosmetic. No multiplication exists in the kernel. And the second break is the counterintuitive one worth flagging: this does **less** mixing work than FNV-1a (one round per 8 bytes instead of one per byte, plus three at the end) and should come out both **faster and better-avalanching** — which is precisely the native's "a fixed count, no more, no fewer," and a direct contradiction of "more mixing rounds always means better mixing."

**The condition where the known way wins, and how it is addressed.** At `len ≤ 4` FNV-1a has genuinely higher throughput: it pays ~4 cycles per byte and nothing else, while this kernel pays a fixed 4 tumbles (~30 cycles) no matter how small the pile. Per the rule that I must not ship a risk I merely narrate, that is guarded: a runtime size check (`len >= 8`) routes short piles down **Regime B**, a straight-line path with no bale loop, no unroll setup and no stride arithmetic — the cheapest form the structure admits. I deliberately did **not** fall back to FNV-1a itself on short inputs, and that choice should be judged openly: the task asks for avalanche *and* speed, FNV-1a's short-input avalanche is poor (a 1-byte input through a single multiply leaves low output bits nearly untouched), so "fall back to the simpler path" cannot mean falling back to a mixer that fails the primary quality criterion on exactly the inputs being handed to it. The guard reduces the overhead; it does not pretend to erase it. If the harness benchmarks only 1–4 byte keys, FNV-1a will win on throughput and I expect to lose that row.

**What I refused.** No threads, no second accumulator lane, no SIMD-across-lanes — not because they wouldn't be faster on large buffers (a 4-lane xxHash-style bulk path would roughly triple this again), but because the native's own seed says the sphere is the *sole* carried memory and that he never looks back at a footprint. Adding lanes would be substituting the textbook method for his idea while keeping his vocabulary, which is the one failure mode this exercise exists to avoid. The honest cost of that fidelity: on multi-megabyte buffers a parallel-lane hash will beat this kernel, and the metaphor is the reason.