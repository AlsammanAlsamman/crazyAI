## MAPPING

**SEED 1 — the never-pure cup**

| world object | problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| one mark | one `unsigned char` |
| the cup | the 64-bit running state |
| *pure* wine ("holds no memory") | XOR — an involution: pour the same drop twice and it is gone (`x^b^b == x`) |
| *cut / sickened* wine ("remembers every drop") | **addition mod 2⁶⁴** — carries are literally how a drop is remembered above its own bit; adding twice does not erase |
| "a pull on the cup" | one state step per byte |
| "taste the colour, let it become the colour waiting for the next pull" | **rotation**: the tasted part is turned away, a fresh part of the word faces the next mark → `c = rotl(c + b, tilt)` |
| "never pure" (cup starts tainted) | non-zero seed, so leading zero bytes still move the state |
| "single unbroken pour, no mark judged alone or twice" | one pass, each byte touched exactly once |
| **assumption broken** | *"mixing one byte requires a multiplication"* — a pull is `add` + `rotate`, 2 ops, 2 cycles of latency, no `imul` |

**SEED 2 — the seven organs**

| world object | problem object |
|---|---|
| the cup's last colour | the state after the absorb, before output |
| "my own organs", seven of them | a **fixed 7-round finalizer**, cost independent of `len` |
| "bent", a bending | shift (and a rotate, which is a bend around a circle) |
| "throws away half of what came before" | the shift **ladder halves**: 32, 16, 8, 4, 2, 1 — each bend discards half as much as the last |
| "keeping only what refuses to sit still" | `^` — only bits that *differ* between the word and its shifted copy survive |
| "a body carries a sickness until every organ **compensates**" | one organ gives back instead of discarding: `h += h << s` (carry = the sickness travelling upward) |
| "hold it beside the nightingale's last note; if the two don't trade beauty for beauty **exactly**, bend again" | stop when the measured flip-rate equals ½ — a **stopping rule**, not "more is better" |
| "small enough to knot into a jacket's collar" | 64-bit output |
| **assumption broken** | *"more mixing rounds always means better mixing"* — see below, provably |

**SEED 3 — the one-changed-mark test**

| world object | problem object |
|---|---|
| "change one mark, anywhere, even the quietest one" | flip one input bit at any position — including the **last** byte, whose difference reaches the organs as a single bit |
| "pour the whole thing through again from the first cup" | recompute the hash end-to-end |
| "if the new token still resembles the old one's shape, throw the method away" | avalanche ≈ 0.5 or the kernel is rejected |
| "a door someone forgot to turn" | a state that is a bijection but not a diffuser |
| **assumption broken** | same one as SEED 2, from the other side: the acceptance criterion is a *measured differential*, never a round count |

## CHOSEN SEED

**SEED 2** (the seven organs). Two of the three seeds attack the preferred assumption; SEED 2 is the one that is a *mechanism* rather than a test, so it is the one I build. SEED 1 is the same native's absorb — it supplies the material the organs act on, and I keep it literally (add + rotate, no multiply); SEED 3 is the acceptance test, which is exactly what `hash_bench`'s avalanche score computes. I did not substitute the textbook path anywhere: there is **no multiplication in the per-byte loop** and no `xxHash`-style word mixing.

Why "more rounds is better" is *false here*, provably, in the native's own idiom — a bend applied twice un-bends:

```
y = x ^ (x>>32);   y>>32 = x>>32   (x>>64 = 0)
y ^ (y>>32) = x ^ (x>>32) ^ (x>>32) = x        <-- two organs = ZERO mixing
```

And the reason **seven** is the number, not three and not twenty: each bend at most doubles the number of GF(2) terms landing in a 64-bit word, so after *r* organs ~2^r terms fall into 64 positions and cancel in pairs; expected flipped-bit count ≈ 32·(1−e^(−2^r/32)):

| organs r | 5 | 6 | **7** | 8 |
|---|---|---|---|---|
| expected avalanche | 0.33 | 0.43 | **0.49** | 0.4999 |

Seven is where the trade becomes "beauty for beauty" to within 2 %; the eighth organ buys 0.6 of a bit for 3 more cycles. The native's garden door is a saturation knee, not a round counter.

## ASSUMPTION BROKEN

Primary: **"more mixing rounds always means better mixing."** FNV-1a spends a full multiply-mix on *every* byte — `len` heavy rounds — and still needs its output to be good. Here the per-byte work is deliberately *too weak to avalanche* (2 ops), and **all** avalanche is bought once, at the end, in exactly seven bends. Total mixing work drops from Θ(len) heavy rounds to Θ(len) cheap pours + 7 fixed rounds.

Also broken: *"mixing one byte requires a multiplication"* (SEED 1: add+rotate) and *"the whole buffer must be read once, start to end, in order"* (the caravan regime reads it once but as 16 interleaved streams).

## ARTIFACT

Which code is which part of the native's mechanism:

* `organs7()` — **SEED 2, the whole of it.** Seven bends; shifts halve (32,16,8,4,2,1); `^` keeps "what refuses to sit still"; `h += h << 19` is organ 4, the one that *compensates* instead of discarding (its carry is the sickness travelling upward, and it is the only non-linear step, so the finalizer is a bijection with data-dependent differentials); a `rotl` after each bend is the "bend" turning the word so the next, smaller bend reaches bits the previous one could not (without it, six downward folds total exactly 63 and the low bits could never climb back).
* `ROTL64(c + p[i], tilt)` — **SEED 1, the pull on the never-pure cup**: `+` is the cut wine (carry = memory), `rotl` is "let the tasted colour become the colour waiting".
* `WINE[]` — the cup is never pure: pre-tainted seeds.
* `TILT[]`, all odd — each cup on the bar at its own tilt; odd ⇒ the tilt visits every bit position, and unequal tilts break the symmetry between cups.
* `h = ROTL64(h + cups[k], TILT[k])` — "the cups are poured into one another", using the cup's own gesture, nothing new invented.
* `ROTL64(h + len, 31)` — "when the pile ends": the tally of marks is poured as the closing mark. Without it, 64 zero bytes and 128 zero bytes collide (a rotation of period 64 returns the seed unchanged) — a flaw the native's own SEED-3 test finds immediately.
* **Three regimes, recognised at runtime by the size of the pile** (`len`), because the known way spans small and large buffers: one drinker with one cup (`len < 32`); the bar with eight cups (`32 ≤ len`, always available); the whole tavern with sixteen cups poured four at a time (`len ≥ 256`, only when AVX2 exists — otherwise it falls back to the bar). No thread parallelism: the metaphor's unit of work is one mark, and an OpenMP fork (~µs) costs more than the entire hash at these sizes.

```c
#include <stdint.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the cup is never pure: every cup starts already tainted */
static const uint64_t WINE[16] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD1B54A32D192ED03ULL, 0xA5CB3B6D4CF0E9C7ULL, 0x8EBC6AF09C88C6E3ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL, 0x85EBCA77C2B2AE63ULL, 0x9E3779B185EBCA87ULL,
    0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0xEB44ACCAB455D37DULL, 0xD6E8FEB86659FD93ULL
};
/* each cup sits at its own tilt; all odd, so a tilt visits every bit */
static const unsigned TILT[16] = { 11,23,37,53, 13,29,41,59, 17,31,43,61, 19,7,47,5 };

/* ---- SEED 2: the seven organs. bend, throw away half, keep what refuses to sit still ---- */
static inline uint64_t organs7(uint64_t h)
{
    h ^= h >> 32; h = ROTL64(h, 27);   /* organ 1: discards 32 -- half of the whole   */
    h ^= h >> 16; h = ROTL64(h, 13);   /* organ 2: half of what came before           */
    h ^= h >>  8; h = ROTL64(h, 41);   /* organ 3                                     */
    h += h << 19;                      /* organ 4: the organ that COMPENSATES (carry) */
    h ^= h >>  4; h = ROTL64(h,  7);   /* organ 5                                     */
    h ^= h >>  2; h = ROTL64(h, 53);   /* organ 6                                     */
    h ^= h >>  1; h = ROTL64(h, 31);   /* organ 7 -- the ladder is spent              */
    return h;                          /* small enough to knot into a collar          */
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *__restrict p = data;
    uint64_t h;
    size_t i, n;

    /* === regime 1: a small pile -- one drinker, one cup, one unbroken pour === */
    if (len < 32) {
        uint64_t c = WINE[0];
        for (i = 0; i < len; i++)
            c = ROTL64(c + (uint64_t)p[i], 11);
        c = ROTL64(c + (uint64_t)len, 31);
        return organs7(c);
    }

#if defined(__AVX2__)
    /* === regime 3: a caravan -- the whole tavern, sixteen cups, four poured at once === */
    if (len >= 256) {
        const __m256i T0L = _mm256_setr_epi64x(11,23,37,53);
        const __m256i T0R = _mm256_setr_epi64x(53,41,27,11);
        const __m256i T1L = _mm256_setr_epi64x(13,29,41,59);
        const __m256i T1R = _mm256_setr_epi64x(51,35,23, 5);
        const __m256i T2L = _mm256_setr_epi64x(17,31,43,61);
        const __m256i T2R = _mm256_setr_epi64x(47,33,21, 3);
        const __m256i T3L = _mm256_setr_epi64x(19, 7,47, 5);
        const __m256i T3R = _mm256_setr_epi64x(45,57,17,59);
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  0));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  4));
        __m256i a2 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  8));
        __m256i a3 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE + 12));
        uint64_t cups[16];
        int k;

        n = len & ~(size_t)15;
        for (i = 0; i < n; i += 16) {
            __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
            /* each mark poured into its own cup: byte j -> cup j, still one by one */
            a0 = _mm256_add_epi64(a0, _mm256_cvtepu8_epi64(b));
            a1 = _mm256_add_epi64(a1, _mm256_cvtepu8_epi64(_mm_srli_si128(b,  4)));
            a2 = _mm256_add_epi64(a2, _mm256_cvtepu8_epi64(_mm_srli_si128(b,  8)));
            a3 = _mm256_add_epi64(a3, _mm256_cvtepu8_epi64(_mm_srli_si128(b, 12)));
            /* the tasted colour becomes the colour waiting: each cup at its own tilt */
            a0 = _mm256_or_si256(_mm256_sllv_epi64(a0,T0L), _mm256_srlv_epi64(a0,T0R));
            a1 = _mm256_or_si256(_mm256_sllv_epi64(a1,T1L), _mm256_srlv_epi64(a1,T1R));
            a2 = _mm256_or_si256(_mm256_sllv_epi64(a2,T2L), _mm256_srlv_epi64(a2,T2R));
            a3 = _mm256_or_si256(_mm256_sllv_epi64(a3,T3L), _mm256_srlv_epi64(a3,T3R));
        }
        _mm256_storeu_si256((__m256i *)(void *)(cups +  0), a0);
        _mm256_storeu_si256((__m256i *)(void *)(cups +  4), a1);
        _mm256_storeu_si256((__m256i *)(void *)(cups +  8), a2);
        _mm256_storeu_si256((__m256i *)(void *)(cups + 12), a3);

        for (; i < len; i++)                        /* leftover marks -> the first cup */
            cups[0] = ROTL64(cups[0] + (uint64_t)p[i], 11);
        h = cups[0];
        for (k = 1; k < 16; k++)                    /* the cups poured into one another */
            h = ROTL64(h + cups[k], TILT[k]);
        h = ROTL64(h + (uint64_t)len, 31);          /* the pile ends: its tally is a mark */
        return organs7(h);
    }
#endif

    /* === regime 2: the bar -- eight cups, the pour goes down the row === */
    {
        uint64_t c0 = WINE[0], c1 = WINE[1], c2 = WINE[2], c3 = WINE[3];
        uint64_t c4 = WINE[4], c5 = WINE[5], c6 = WINE[6], c7 = WINE[7];

        n = len & ~(size_t)7;
        for (i = 0; i < n; i += 8) {
            c0 = ROTL64(c0 + (uint64_t)p[i + 0], 11);
            c1 = ROTL64(c1 + (uint64_t)p[i + 1], 23);
            c2 = ROTL64(c2 + (uint64_t)p[i + 2], 37);
            c3 = ROTL64(c3 + (uint64_t)p[i + 3], 53);
            c4 = ROTL64(c4 + (uint64_t)p[i + 4], 13);
            c5 = ROTL64(c5 + (uint64_t)p[i + 5], 29);
            c6 = ROTL64(c6 + (uint64_t)p[i + 6], 41);
            c7 = ROTL64(c7 + (uint64_t)p[i + 7], 59);
        }
        for (; i < len; i++)
            c0 = ROTL64(c0 + (uint64_t)p[i], 11);

        h = c0;
        h = ROTL64(h + c1, 23); h = ROTL64(h + c2, 37); h = ROTL64(h + c3, 53);
        h = ROTL64(h + c4, 13); h = ROTL64(h + c5, 29); h = ROTL64(h + c6, 41);
        h = ROTL64(h + c7, 59);
        h = ROTL64(h + (uint64_t)len, 31);
        return organs7(h);
    }
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 5.0**

Stated before any measurement, with the reasoning that produced it:

* FNV-1a is latency-bound on a serial chain: `xor`(1) + `imul`(3) = **4 cycles/byte**, ~1 GB/s at 4 GHz.
* Regime 1 (`len < 32`): 2 cycles/byte + ~20 fixed cycles of organs → **0.85× at 8 B, ~1.2× at 16 B, break-even near 12 B**.
* Regime 2 (8 cups): ~3 uops/byte across 8 independent chains → **~0.7–0.9 c/B → 4–6×**.
* Regime 3 (16 cups, AVX2): ~1.5 uops/byte, four independent vector chains → **~0.4 c/B → ~10×**, falling toward memory bandwidth past L2.
* Aggregated over a typical size sweep (8 B … 1 MB) the geometric mean lands near **5×**.
* Avalanche: **0.48–0.50**, from the saturation table above (7 organs → 0.49 expected), with organ 4's carries pushing the measured average toward 0.5. The weakest case is the *last* byte (the "quietest mark"), whose difference reaches `organs7` as a single bit; that case is precisely what the 7-bend ladder is sized for.

## MEASUREMENT

**Not measured — and I will not pretend otherwise.** This session was invoked with no tools available: `hash_bench` and `hash_contract` could not be called, so the four allotted improvement rounds were spent on paper (dependency-chain counting, the GF(2) cancellation estimate, the zero-run collision found by the native's own SEED-3 test and fixed by pouring `len`) rather than on the machine. Every number in PREDICTION is an analytic estimate. The pipeline's measurement stands; if it contradicts the table above, the table is wrong, not the measurement.

Specific falsifiable claims, so the measurement can actually kill them:
1. avalanche ≥ 0.47 — if it comes back at 0.30–0.43, the linear ladder is cancelling worse than the Poisson model says and organ count/rotations, not the mechanism, are at fault.
2. speedup ≥ 3 at large sizes.
3. speedup at 8 bytes in [0.8, 1.2].

## VERDICT

The native's claim is real and non-obvious: **the per-byte multiply in FNV/xxHash is not paying for avalanche — it is paying for avalanche too early and too often.** Avalanche is a property of the *last* step; buying it once, in seven bends, and making the per-byte pour as cheap as physically possible (add + rotate, 2 ops) is strictly the better division of labour, and the "seven" is a genuine saturation knee rather than mysticism — one more bend buys 0.6 of a bit, and one *repeated* bend provably buys nothing at all (`x ^= x>>32` twice is the identity). That is the textbook assumption broken cleanly.

Risks I am naming, each already addressed in the code rather than left as prose:

* **"Overhead if the pile is small."** Guarded: `len < 32` takes the single-cup path with no lane setup and no 16-way combine, which is the simplest path available. What remains is ~20 cycles of organs, which is the price of the avalanche the contract asks for, not a cost of my parallel machinery. I predict ~0.85× at 8 bytes and I am shipping that rather than hiding it; I will not drop the organs, because they *are* the chosen seed.
* **"Only helps if AVX2 exists."** Guarded by `#if defined(__AVX2__)` plus `len >= 256`, falling back to the eight-cup scalar bar, which itself never regresses below the single-cup path.
* **No thread parallelism**, deliberately: the metaphor's unit of work is one mark, and at benchmark sizes an OpenMP fork costs more than the whole hash.

The honest weakness, which the benchmark will *not* catch and which I state anyway: the absorb is a rotating **sum**. Its only non-linearity is carry propagation, so collisions are constructible by hand (adjust two marks in the same cup so the tilted sums agree) even though avalanche measures ~0.5. This is a fast non-cryptographic hash in the FNV/xxHash class — weaker than xxHash against a deliberate adversary, and it must never be used where collisions are chosen by an attacker. The native's own test ("change one mark, anywhere") is a differential test, and a method can pass it while still being additively invertible; that is a limit of the test, not a claim I am smuggling in.