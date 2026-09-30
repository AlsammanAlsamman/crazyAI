## MAPPING

**SEED 1 — "one crease per mark, angle from the mark *and* the crease before it"**

| World object | Problem object |
|---|---|
| pile of marks, read in order | `data[0..len-1]`, streamed forward |
| one fold per mark | one state update per input word |
| angle = f(mark, previous crease) | `state = f(state, word)` — serial dependency |
| the flat sheet | one accumulator |

*Breaks: nothing.* This seed **is** FNV‑1a/xxHash's inner loop stated in paper. It affirms every silent assumption (serial, in‑place, in order).

**SEED 2 — "test every crease against all four wire birds' beaks at once; refold until they agree"**

| World object | Problem object |
|---|---|
| four wire birds | **four independent 64‑bit accumulators** `b0..b3` |
| "each fold's corner against **a different** bird's beak" | fold *i* goes to bird *i* mod 4 — round‑robin lanes, no lane ever sees the next mark |
| "at once" | the four lanes are in flight simultaneously (ILP), not one chain |
| "only when all four beaks agree does the crease count as set" | the crease is *committed* only at the end, by pressing the corner through all four beaks in turn (4 odd‑constant multiplies) |
| "refold tighter … markings line up **wrong on purpose**, scrambled, because a clean line‑up would mean two piles could look the same" | each fold must be **injective** (bijective in the accumulator) so a delta can never die; misaligned spiral constants = golden‑ratio odd constants, different rotation per bird |
| huge sheet, "pink or orange, doesn't matter, only that it's **large enough**" | two sheets = two regimes; the *only* selector is whether the sheet is large enough for the pile → runtime `len` check picking wide path vs. small path |

*Breaks:* **"the state is a single accumulator updated in place, one value"**, and — under the literal reading *"each fold's corner against **a different** bird"* — also **"each byte must be mixed into the running state before the next byte is read"**: byte 8 is read and absorbed by bird 1 while bird 0 is still absorbing byte 0. There is no single running state for a byte to be mixed into.

**SEED 3 — "thin the whole wad at the water's edge to one dense corner; throw away every scrap"**

| World object | Problem object |
|---|---|
| the bulky folded wad | wide (256‑bit) intermediate state |
| thinning at the water's edge | final compression 256→64 bits |
| "small, small, small … no bigger than a coin" | strong xorshift‑multiply finalizer (avalanche concentrated here) |
| scraps, failed gauge readings thrown into the mud | no state survives the call; nothing is carried between hashes |

*Breaks:* "mixing one byte requires a multiplication" and "more mixing rounds always means better mixing" — the pass is multiply‑free, all multiplication happens once at the water's edge.

## CHOSEN SEED

**SEED 2** (the four wire birds), with SEED 3's thinning as its *closing act* — the native describes one mechanism and the wad must be thinned to produce a 64‑bit token; the contract forces that. Seed 2 is the most literal mapping (four birds → four named accumulators, "a different bird" → round‑robin, "all four agree" → the commit step, "line up wrong on purpose" → injectivity) and the most different from the known way (which has exactly one accumulator).

Honest note on the tie‑break rule: SEED 2 breaks *"each byte must be mixed into the running state before the next byte is read"* only under the reading **"each fold → a different bird"** (round‑robin), not under the reading "every fold → all four birds". I chose the round‑robin reading deliberately and say so plainly; under the other reading **no seed breaks that assumption**, and SEED 2 would still be the pick as the most literal non‑textbook seed.

## ASSUMPTION BROKEN

1. **"The state is a single accumulator updated in place, one value."** → four birds, four independent chains.
2. **"Each byte must be mixed into the running state before the next byte is read."** → there is no single running state; word *i* and word *i+1* never touch the same bird, so they are absorbed concurrently.
3. **"Mixing one byte requires a multiplication."** → the pass is `xor → rotate → add` only (3 ops per 8 bytes); every multiply is deferred to the water's edge.
4. **"More mixing rounds always means better mixing."** → one fold per word, ever; diffusion comes from injectivity + one thinning.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- spiral & snail-shell markings: golden-ratio odd constants, deliberately misaligned ---- */
#define S0 0x9E3779B97F4A7C15ULL
#define S1 0xC2B2AE3D27D4EB4FULL
#define S2 0x165667B19E3779F9ULL
#define S3 0x27D4EB2F165667C5ULL

/* ---- the four wire birds' beaks: the gauges the crease is pressed against ---- */
#define BEAK0 0xFF51AFD7ED558CCDULL
#define BEAK1 0xC4CEB9FE1A85EC53ULL
#define BEAK2 0xBF58476D1CE4E5B9ULL
#define BEAK3 0x94D049BB133111EBULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> ((64u - r) & 63u));
}
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ONE FOLD PER MARK.  angle = f(mark, crease before it).  bijective in b for fixed w,
   so two different piles can never fold down onto the same crease: "line up wrong on purpose". */
#define FOLD(b, w, r) ((b) = rotl64((b) ^ (w), (r)) + (w))

/* "only when all four beaks agree does the crease count as set", then the water's edge:
   hold it down until it is small, small, small -> one hard dense corner, the coin. */
static inline uint64_t set_and_thin(uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                    uint64_t len)
{
    uint64_t t = S0 ^ rotl64(len, 27);
    t = (t ^ b0) * BEAK0; t = rotl64(t, 31);
    t = (t ^ b1) * BEAK1; t = rotl64(t, 29);
    t = (t ^ b2) * BEAK2; t = rotl64(t, 27);
    t = (t ^ b3) * BEAK3;
    t ^= t >> 33;  t *= BEAK0;
    t ^= t >> 29;  t *= BEAK1;
    t ^= t >> 32;
    return t;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;

    /* the four wire birds, seeded with the size of the pile */
    uint64_t b0 = S0 ^ (uint64_t)len;
    uint64_t b1 = S1;
    uint64_t b2 = S2;
    uint64_t b3 = S3 ^ rotl64((uint64_t)len, 32);

    /* choose the sheet: pink or orange, doesn't matter - only that it is large enough
       for the pile of marks.  This is the runtime regime test. */
    if (n >= 64) {
        /* ---- LARGE SHEET: one cache line per pass, each fold to a different bird ---- */
        do {
            uint64_t w0 = ld64(p),      w1 = ld64(p + 8),  w2 = ld64(p + 16), w3 = ld64(p + 24);
            uint64_t w4 = ld64(p + 32), w5 = ld64(p + 40), w6 = ld64(p + 48), w7 = ld64(p + 56);
            FOLD(b0, w0, 13); FOLD(b1, w1, 29); FOLD(b2, w2, 41); FOLD(b3, w3, 53);
            FOLD(b0, w4, 17); FOLD(b1, w5, 31); FOLD(b2, w6, 43); FOLD(b3, w7, 59);
            p += 64; n -= 64;
        } while (n >= 64);

        if (n >= 32) {
            uint64_t w0 = ld64(p), w1 = ld64(p + 8), w2 = ld64(p + 16), w3 = ld64(p + 24);
            FOLD(b0, w0, 13); FOLD(b1, w1, 29); FOLD(b2, w2, 41); FOLD(b3, w3, 53);
            p += 32; n -= 32;
        }
        /* at most three whole marks left, still one to each bird in turn */
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b0, w, 23); p += 8; n -= 8; }
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b1, w, 37); p += 8; n -= 8; }
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b2, w, 47); p += 8; n -= 8; }
        /* the trimmed scrap: re-read the last whole mark (safe, len >= 64), never skip it */
        if (n)      { uint64_t w = ld64(data + len - 8); FOLD(b3, w, 7); }

        return set_and_thin(b0, b1, b2, b3, (uint64_t)len);
    }

    /* ---- SMALL SHEET: same folds, same four birds, no wide set-up, marks read one at a
       time to the end.  Fallback path for a pile too small to need the big sheet. ---- */
    {
        static const unsigned R[4] = { 13u, 29u, 41u, 53u };
        uint64_t bb[4];
        unsigned k = 0;
        bb[0] = b0; bb[1] = b1; bb[2] = b2; bb[3] = b3;

        while (n >= 8) {
            uint64_t w = ld64(p);
            bb[k] = rotl64(bb[k] ^ w, R[k]) + w;
            k = (k + 1u) & 3u; p += 8; n -= 8;
        }
        if (n) {                       /* 1..7 loose marks, gathered then folded once */
            uint64_t tw = 0;
            for (size_t i = 0; i < n; i++) tw = (tw << 8) ^ (uint64_t)p[i];
            tw ^= ((uint64_t)n << 56);  /* the count is part of the fold: no two piles alias */
            bb[k] = rotl64(bb[k] ^ tw, 11) + tw;
        }
        return set_and_thin(bb[0], bb[1], bb[2], bb[3], (uint64_t)len);
    }
}
```

**Which code is which part of the native's mechanism**

| Native | Code |
|---|---|
| one fold per mark, angle from mark + crease before it | `FOLD(b,w,r)` — `rotl64(b ^ w, r) + w`, serial *within* a bird |
| four wire birds, each fold to a *different* beak | `b0..b3`, round‑robin over consecutive 8‑byte words; four independent dependency chains |
| "refold tighter … line up wrong on purpose, scrambled, so two piles can't look the same" | `FOLD` is a bijection in `b`; a one‑bit delta provably never dies, no matter how early it enters. Distinct rotation per bird breaks lane symmetry |
| spiral / snail‑shell markings | `S0..S3`, golden‑ratio‑derived odd constants |
| "only when all four beaks agree does the crease count as set" | `set_and_thin`'s four `(t ^ b_j) * BEAK_j` stages — the corner is pressed through every beak in turn before it counts |
| carry the wad to the water's edge, hold it down small, small, small, one coin | the `t ^= t>>33; t *= …` finalizer — *all* avalanche work concentrated here, once |
| throw every scrap and misreading into the mud | no `static`/global state, no scratch buffer, nothing survives the call |
| "a huge sheet, pink or orange, doesn't matter, only large enough" | `if (n >= 64)` — the regime test, with the small‑sheet path as fallback |

**Risk guarded (step 4):** the only condition where four birds could lose to one accumulator is a pile too small to fill the sheet (lane set‑up + 6 finalizer multiplies amortized over a handful of bytes). That is exactly what the `len >= 64` sheet‑choice guards, with the small‑sheet path doing no wide set‑up. **No thread parallelism:** the native's unit of work is a single fold — tens of picoseconds; OpenMP fork cost exceeds the entire hash at any plausible benchmark size, and the metaphor gives no basis for splitting the pile across sheets (one sheet, one reader, "never skipping").

## PREDICTION

PREDICTION: speedup_vs_dp = 12.0

Reasoning stated before any measurement: FNV‑1a is latency‑bound at one dependent `imul` (3–5 cycles) per **byte** → ≈0.25 B/cycle. This kernel costs 3 ALU ops per **8 bytes** across 4 independent chains → ALU‑bound at ≈8–10 B/cycle for L1/L2‑resident buffers (~30× there), falling to DRAM bandwidth (~3–5 B/cycle) for multi‑MB buffers and to ~2× at 16‑byte inputs. Averaged over a typical size sweep I expect ~12×. Quality prediction: avalanche flip fraction ≈0.50 (near‑ideal) — every fold is injective so no single‑bit delta can be lost, and every delta passes through ≥1 beak multiply plus the full 2‑multiply thinning.

## MEASUREMENT

**Not measured — no tools were available in this session.** `hash_bench` / `hash_contract` could not be invoked, so I ran zero of the four allowed improvement rounds and report that plainly rather than inventing numbers. The prediction above stands unrevised for the pipeline to check. Concretely falsifiable: if measured `speedup_vs_dp < 4` on L1‑resident sizes, the four‑bird ILP claim is wrong (likely load‑port or loop‑overhead bound, not ALU bound); if avalanche < 0.45, the multiply‑free fold is losing delta entropy and the thinning is not strong enough alone — the fix inside this mechanism would be a third beak pass, not a multiply per byte.

## VERDICT

The four wire birds are the whole content of the idea: they say *don't build a single running state*, and everything the standard solution assumes about per‑byte multiplication follows from having only one accumulator to protect. Splitting into four gauges makes the pass multiply‑free and the mixing deferrable, because the birds only have to "agree" once, at the end. The native's insistence that the markings "line up wrong on purpose" is the strongest part of the description — read literally it is a demand for **injectivity** of the fold, which is precisely what licenses the cheap 3‑op fold: a bijection cannot lose information, so no amount of extra rounds during the pass would buy anything, and one thinning at the water's edge suffices.

Expected failure modes, both addressed rather than hidden: (1) small inputs, where the four‑bird set‑up and the six finalizer multiplies dominate — guarded by the `len >= 64` sheet choice with a small‑sheet fallback; (2) very large inputs, where the kernel becomes memory‑bandwidth‑bound and the speedup compresses toward ~5× — not a defect of the mechanism, just the ceiling. The residual weakness I cannot claim away: each bird sees only words ≡ *j* (mod 4), so *collision resistance* against adversarial inputs is weaker than a full sequential multiply chain even though *avalanche* should be near ideal. The bench measures avalanche, not adversarial collisions, so a good score here must not be read as a claim of the latter.