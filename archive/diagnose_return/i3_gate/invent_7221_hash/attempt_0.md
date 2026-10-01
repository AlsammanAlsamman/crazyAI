## MAPPING

**SEED 1 — "one crease per mark, angle set by both the mark and the crease before it."**

| World object | Problem object |
|---|---|
| pile of marks, read in order | `data[0..len-1]`, read once, ascending |
| one sheet, laid flat | one 64-bit accumulator |
| a fold | one mix step |
| "angle from the mark **and** the crease before it" | `acc = f(acc, byte)` — strict serial dependency |
| paper "carries forward everything it has been told" | state is a function of the whole prefix |

*Assumption broken: **none.*** This seed is a verbatim restatement of FNV‑1a. Stated plainly so it isn't smuggled in as novelty.

**SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree."**

| World object | Problem object |
|---|---|
| **four** wire birds, each a *different* bird | four 64-bit accumulator lanes, each with a *different* initial value (no lane symmetry ⇒ a permutation of stripes cannot collide) |
| a beak = a gauge you press a corner into | one lane's fold function `acc += m·P2; rotl 31; ·P1` |
| "**at once**" | the four folds are mutually independent ⇒ issued in the same cycles; lane *i+1*'s fold starts before lane *i*'s multiply retires |
| "only when all four beaks agree is the crease set" | the crease is not *readable* until all four lanes are merged — the merge is the agreement |
| spiral markings ∥ snail-shell markings "scrambled on purpose" | golden-ratio / odd-prime 64-bit multipliers with maximally non-aligned bit patterns; a "clean line-up" = an aliasing multiplier = two buffers hashing alike |
| "pink or orange, doesn't matter, only that it's **large enough**" | runtime regime test: the four-bird sheet is only usable when `len ≥ 32`; otherwise a different, smaller sheet |

*Assumption broken:* **"each byte must be mixed into the running state before the next byte is read"** — and, jointly, **"the state is a single accumulator updated in place, one value."**

**SEED 3 — "thin the whole wad at the water's edge to one dense corner; throw away every scrap."**

| World object | Problem object |
|---|---|
| the bulky folded wad | the wide (4×64 = 256-bit) internal state |
| the water's edge / tide-boundary | end of buffer: the finalization point |
| "hold it down until it is small, small, small" | avalanche finalizer: `h^=h>>33; h*=P2; h^=h>>29; h*=P3; h^=h>>32` |
| one hard corner "no bigger than a coin" | the returned `uint64_t` |
| every scrap and failed reading thrown in the mud | no intermediate lane value escapes; only the 64-bit digest |
| "if even one mark had been different… a different unrecognizable corner" | avalanche requirement itself |

*Assumption broken:* **"more mixing rounds always means better mixing"** — one strong terminal thinning replaces many weak per-byte rounds.

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks the preferred assumption, and its mapping is the most literal object-for-object (four birds → four lanes, different birds → different lane seeds, "at once" → independence). SEED 1 *is* the known way, so choosing it would be a no-op; SEED 3 is real but subordinate — in the native's own account the thinning happens *after* the folding, so I keep it as the finalizer rather than as the load-bearing idea.

Per step 4, I did **not** invent a new lane scheme. Four differently-seeded 64-bit lanes over 32-byte stripes, merged by rotate-sum plus per-lane merge rounds, then a terminal xor-shift-multiply avalanche, **is xxHash64's validated structure**. The metaphor lands on it exactly, so I ship that rather than a novel untested variant. Two honest notes on literalism:

- "all four **at once**" tempts an AVX2 vector of 4×64. I rejected it: AVX2 has no 64-bit multiply (`vpmullq` is AVX‑512DQ), so each vector fold costs ~3 `vpmuludq` + shifts + adds, i.e. slower than four scalar `imul`s that the superscalar core already retires concurrently. The "at once" is honored by instruction-level parallelism, which is the machine's real four-beaks-at-once.
- **No threads.** The metaphor has one sheet and one reader working in order; its unit of work is a 32-byte stripe, far too small to justify a team. Adding threads would also make the digest depend on the schedule. Stated rather than silently skipped.

## ASSUMPTION BROKEN

**"Each byte must be mixed into the running state before the next byte is read."** Broken literally: at any instant four folds are in flight against four different birds, so byte *k+1* enters its gauge while byte *k*'s multiply is still resolving. Secondary break: **the state is not one value** — it is four, collapsed to one only at the water's edge (SEED 3).

Regime handling (step 5): `len ≥ 32` → four-bird path; `len < 32` → single small sheet seeded with `P5`, no lane setup, no merge. This is the guard for the risk my own verdict names (four-lane setup + 4 merge rounds is pure overhead on tiny inputs).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the spiral and snail-shell markings: odd, maximally un-aligned multipliers */
#define B1C 0x9E3779B185EBCA87ULL
#define B2C 0xC2B2AE3D27D4EB4FULL
#define B3C 0x165667B19E3779F9ULL
#define B4C 0x85EBCA77C2B2AE63ULL
#define B5C 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;          /* one mov at -O3 */
}
static inline uint32_t rd4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold: the angle is set by the mark AND the crease before it (SEED 1) */
static inline uint64_t fold(uint64_t crease, uint64_t mark) {
    crease += mark * B2C;
    crease  = rotl64(crease, 31);
    crease *= B1C;
    return crease;
}

/* the beaks agreeing: fold a bird's final crease into the coin */
static inline uint64_t press(uint64_t coin, uint64_t bird) {
    coin ^= fold(0, bird);
    coin  = coin * B1C + B4C;
    return coin;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    /* "pink or orange, doesn't matter, only that it's large enough":
       runtime regime test, with a fallback sheet for the small regime. */
    if (len >= 32) {
        const unsigned char *limit = end - 32;
        /* four DIFFERENT wire birds - distinct seeds kill lane symmetry */
        uint64_t b1 = B1C + B2C;
        uint64_t b2 = B2C;
        uint64_t b3 = 0;
        uint64_t b4 = (uint64_t)0 - B1C;

        /* marks read in order, never skipping, never looking ahead;
           four folds in flight at once against four beaks (SEED 2) */
        do {
            b1 = fold(b1, rd8(p +  0));
            b2 = fold(b2, rd8(p +  8));
            b3 = fold(b3, rd8(p + 16));
            b4 = fold(b4, rd8(p + 24));
            p += 32;
        } while (p <= limit);

        /* the crease only counts as set when all four beaks agree */
        h = rotl64(b1, 1) + rotl64(b2, 7) + rotl64(b3, 12) + rotl64(b4, 18);
        h = press(h, b1);
        h = press(h, b2);
        h = press(h, b3);
        h = press(h, b4);
    } else {
        h = B5C;                    /* one small sheet, no birds needed */
    }

    h += (uint64_t)len;             /* the tide-boundary is part of the record */

    /* scraps trimmed off the edge */
    while (p + 8 <= end) {
        h ^= fold(0, rd8(p));
        h  = rotl64(h, 27) * B1C + B4C;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)rd4(p) * B1C;
        h  = rotl64(h, 23) * B2C + B3C;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * B5C;
        h  = rotl64(h, 11) * B1C;
        p++;
    }

    /* the water's edge: thin it down to one dense corner (SEED 3) */
    h ^= h >> 33;
    h *= B2C;
    h ^= h >> 29;
    h *= B3C;
    h ^= h >> 32;
    return h;                       /* the coin; every scrap discarded */
}
```

## PREDICTION

Reasoning, stated before any measurement. FNV‑1a's critical path is `xor`(1) + `imul`(3) ≈ 4 cycles **per byte** ⇒ ~0.25 B/cycle. Each bird's chain is `add`(1) + `rotl`(1) + `imul`(3) ≈ 5 cycles per 8 bytes, and four birds run concurrently ⇒ 32 bytes / ~5 cycles ≈ 6.4 B/cycle ceiling, call it 4.5–6 B/cycle after loads and loop overhead. That is a factor of ~18–25 on large buffers, dragged down by whatever share of the benchmark is short inputs and harness overhead.

**PREDICTION: speedup_vs_dp = 14**

(honest interval 8–25; I expect the avalanche score to be essentially ideal, single-bit flip → ~0.50 output-bit flip probability, since the terminal finalizer is a full xor-shift-multiply cascade and `len` is folded in.)

## MEASUREMENT

**Not measured in this session — no tools were available to me here** (`hash_bench` / `hash_contract` were listed but the session was invoked with none callable). I will not dress up an estimate as a measurement. The numbers above are predictions only; the pipeline's compile-and-measure pass is the test.

Falsification criteria I commit to in advance:
- **speedup < 8×** on large buffers ⇒ my ILP model is wrong (most likely cause: the four `imul`s are not co-issuing, or the benchmark is memory-bandwidth-bound, in which case *no* mixing kernel can win and the metaphor's advantage is invisible).
- **avalanche score materially below the FNV baseline** ⇒ the "scrambled markings" mapping failed and the multipliers alias; remedy would be a second finalizer round, not more per-byte rounds.
- **speedup ≤ 1× at len < 32** ⇒ the `len >= 32` guard is mis-tuned and the threshold should rise.

## VERDICT

The translation is unusually clean: four wire birds are four accumulator lanes, "different birds" is the distinct-seed requirement, "at once" is instruction-level parallelism, "only when all four agree" is the merge, "large enough sheet" is the runtime regime test, and the water's edge is the finalizer. Taking that literally does not produce an exotic new hash — it produces xxHash64's structure, which is the correct outcome under step 4: a validated design beats a novel one, and the native's account is best read as an independent derivation of it rather than a rival to it.

What the mechanism genuinely buys is the broken assumption: a byte no longer waits for the previous byte's multiply, because it goes to a different beak. That is the entire source of the expected speedup, and it is the one thing FNV‑1a structurally cannot do.

Risks I named and then addressed rather than shipped: (a) four-lane setup plus four merge rounds is dead weight on tiny inputs — guarded by `len >= 32` with a single-sheet fallback; (b) AVX2 "all four at once" would be slower than scalar four-at-once on this ISA — dropped, with the reason stated; (c) thread parallelism — dropped, because the metaphor's unit of work is a 32-byte stripe and one reader, not because it was inconvenient. Remaining unaddressed exposure, honestly: if the benchmark is dominated by buffers so large that DRAM bandwidth caps throughput, the speedup will compress toward the bandwidth ratio and my 14× will be wrong on the low side of my own interval — there is no kernel-side fix for that, only an admission.