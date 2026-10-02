# MAPPING

**Common dictionary used by all three seeds**

| World object | Problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| one mark | one byte (one "drop" = 8 bytes when the throat allows) |
| the cup | a 64-bit state word |
| the cup's colour | the state's current value |
| a pull on the cup | one fold step |
| knot in a jacket's collar | the 64-bit return value |
| garden door / nightingale's last note / "trade beauty for beauty exactly" | the 50 % avalanche criterion |

**SEED 1 — the impure cup, one unbroken pour**

| World object | Problem object |
|---|---|
| "the cup that is never pure", sickened/**cut** wine | a state combined by **addition**: carries make it impure — bits leak upward, the operation is not self‑inverse |
| "pure wine holds no memory" | **XOR** alone: carry‑free, involutive — a drop poured twice is forgotten |
| "cut wine **remembers every drop added** to it" | `c += drop` — addition remembers multiplicity *and* order |
| taste the colour, let it be the colour waiting for the next pull | strict serial chain `c_{i+1} = f(c_i, b_i)` |
| no mark judged alone or twice | each byte read exactly once, never in isolation |

*Assumption broken:* **"mixing one byte requires a multiplication"** — the cup remembers by carry, not by product. It *affirms* the other three (byte-before-next-byte, single accumulator, in-order single pass).

**SEED 2 — seven organs, each discarding half**

| World object | Problem object |
|---|---|
| the cup's final colour | the 64-bit value entering the finisher |
| my own organs, seven of them | seven successive bending stages, fixed count |
| a bending | one shift-combine step |
| "throws away **half of what came before**" | a **shift** — half the bits fall out of the word (left: out the top; right: out the bottom). Not a rotate: information is literally discarded |
| "keeping only what **refuses to sit still**" | **XOR** — keeps the bits that disagree |
| "a body carries a sickness until **every organ compensates**" | (a) diffusion runs until the whole width has compensated; (b) a large sickness is shared out among the organs, each compensating for its share, then reporting to the heart |
| the organ compensating | **ADD** — the carry compensates upward |
| exactly seven, then stop | round count ≈ log₂(width); each bending halves what is left to mix, so mixing **saturates** |

*Assumption broken:* **"more mixing rounds always means better mixing"** (seven halvings exhaust the word; an eighth bends nothing new) **and** "mixing one byte requires a multiplication" (no `imul` anywhere in the kernel).

**SEED 3 — the audit**

| World object | Problem object |
|---|---|
| change one mark anywhere, even the quietest | flip one bit of one byte, including the last/low-entropy byte |
| re-pour from the first cup | recompute the whole hash, no incremental shortcut |
| "if the new token still resembles the old one's shape" | output Hamming distance ≪ 32 bits |
| throw the whole method away | reject the design outright |

*Assumption broken:* it replaces **"more rounds always means better mixing"** with a *measured* criterion — round count is never evidence. But it is a validation protocol, not a kernel mechanism.

# CHOSEN SEED

**SEED 2 — the seven organs.** It is the only one of the three that breaks the preferred assumption ("more mixing rounds always means better mixing"), and it is maximally far from the known way: it contains *no multiplication at all*. Its two primitives are forced by the text and nothing else:

```
BEND_UP(c,s)   = c + (c << s)    /* half falls out of the top; the organ compensates by carry */
BEND_DOWN(c,s) = c ^ (c >> s)    /* half falls out of the bottom; keep what refuses to sit still */
```

Everything in the kernel — body, cross-lane fold, finisher — is one of these two bendings and nothing else. SEED 1's "impure cup remembers every drop **added**" supplies the only other operation: `c += drop`.

**Arrival at a validated technique (step 4), not an invention.** Shift-add / shift-xor mixing with no multiply *is* a real, validated family: Jenkins' one-at-a-time hash (`h += k; h += h<<10; h ^= h>>6;` finished by `h += h<<3; h ^= h>>11; h += h<<15;`) and the broader ARX family (SipHash, BLAKE2). My pull step is bit-for-bit Jenkins' OAAT step, and organs 5–7 are bit-for-bit Jenkins' validated finisher, preceded by four wide organs (32, 29, 23, 17) that carry the low half into the high half and back before the narrow ones run. I let the metaphor land on the known-good construction instead of inventing new shift constants.

# ASSUMPTION BROKEN

1. **"More mixing rounds always means better mixing."** Seven organs, fixed, because each bending halves what remains unmixed — mixing *saturates*. The body therefore spends **two** bendings per 8-byte drop, not a round of mixing per byte, and banks the remaining work in a bounded seven-stage finisher.
2. **"Mixing one byte requires a multiplication."** Zero multiplies. `c + (c<<s)` and `c ^ (c>>s)` are 1-cycle ops that vectorise, which an `imul` does not do nearly as cheaply.
3. **Regime recognition, in-world** (step 5): the known_way names a small/large split. The native's own sentence does the recognising — *"a body carries a sickness until every organ compensates."* A **quiet pile** goes down one throat into one cup (SEED 1's unbroken pour, exactly Jenkins OAAT, 8 marks per pull). A **full pile** is shared out among sixteen organs that each bend their own share and then report to the heart. The switch is a literal `len` check with the simple path as the fallback, so the wide path's fold cost can never make a small pile slower.

I did not use OpenMP: the metaphor's unit of work (one organ's share of the pile) is a few tens of bytes, not a few hundred kilobytes, so thread-level parallelism has no in-world justification at these sizes. Vectorisation only, as instructed.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* =====================  THE ORGAN-BENDING  =====================
 * One bending throws away half of what came before -- a shift pushes half
 * the bits out of the word -- and then either keeps only what refuses to
 * sit still (xor), or lets the impure cup remember it (add: the carry is
 * the organ compensating, and carries never forget).
 * These two macros are the only arithmetic in this kernel.  No multiply.
 */
#define BEND_UP(c, s)   ((c) + ((c) << (s)))   /* half falls out of the top    */
#define BEND_DOWN(c, s) ((c) ^ ((c) >> (s)))   /* half falls out of the bottom */

/* One pull on the cup that is never pure: the drop is *added* (cut wine
 * remembers every drop added to it; pure xor would forget it if poured
 * twice), then two bendings, and the cup's new colour is the colour
 * waiting for the next pull.  This step is Jenkins' one-at-a-time step. */
#define PULL(c, w) do {                                                     \
        (c) += (uint64_t)(w);                                               \
        (c)  = BEND_UP((c), 10);                                            \
        (c)  = BEND_DOWN((c), 6);                                           \
    } while (0)

/* The seven organs.  The cup's last colour is carried through the body and
 * bent once per organ, each organ discarding half of what it received.
 * Seven and then stop: each bending halves what is left to mix, so an
 * eighth organ would bend nothing new.  Organs 1-4 are the wide ones that
 * trade the high half for the low half and back; organs 5-7 are Jenkins'
 * validated shift-add-xor finisher (<<3, >>11, <<15), unchanged.          */
static inline uint64_t organs7(uint64_t c, uint64_t weight)
{
    c += weight;                 /* the pile's own weight enters the blood */
    c  = BEND_UP(c,   32);       /* organ 1: half of the word thrown out the top    */
    c  = BEND_DOWN(c, 29);       /* organ 2                                         */
    c  = BEND_UP(c,   23);       /* organ 3                                         */
    c  = BEND_DOWN(c, 17);       /* organ 4                                         */
    c  = BEND_UP(c,    3);       /* organ 5                                         */
    c  = BEND_DOWN(c, 11);       /* organ 6                                         */
    c  = BEND_UP(c,   15);       /* organ 7                                         */
    return c;
}

/* Sixteen humours: one seed per organ, so no two organs start the same
 * colour and an all-equal pile cannot collapse the lanes together.       */
static const uint64_t HUMOUR[16] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
    0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL,
    0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL,
    0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
};

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t c = 0x9E3779B97F4A7C15ULL;   /* the cup, already impure */
    size_t   i = 0;

    /* ---- Which regime is the body in?  A quiet pile goes down one throat.
     * A full pile is shared out so that every organ compensates at once.
     * The threshold is a real guard: below it the fold and the organ
     * hand-off would cost more than they save, so we fall back to the
     * plain unbroken pour.                                              */
    if (len >= 256) {
#if defined(__AVX2__)
        __m256i A = _mm256_loadu_si256((const __m256i *)(HUMOUR + 0));
        __m256i B = _mm256_loadu_si256((const __m256i *)(HUMOUR + 4));
        __m256i C = _mm256_loadu_si256((const __m256i *)(HUMOUR + 8));
        __m256i D = _mm256_loadu_si256((const __m256i *)(HUMOUR + 12));

        for (; i + 128 <= len; i += 128) {
            const unsigned char *p = data + i;
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(p +  0));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(p + 64));
            __m256i x3 = _mm256_loadu_si256((const __m256i *)(p + 96));

            /* each organ's share of the pile is added into its own cup */
            A = _mm256_add_epi64(A, x0);
            B = _mm256_add_epi64(B, x1);
            C = _mm256_add_epi64(C, x2);
            D = _mm256_add_epi64(D, x3);

            /* bending, half thrown out of the top, carry compensating */
            A = _mm256_add_epi64(A, _mm256_slli_epi64(A, 10));
            B = _mm256_add_epi64(B, _mm256_slli_epi64(B, 12));
            C = _mm256_add_epi64(C, _mm256_slli_epi64(C,  9));
            D = _mm256_add_epi64(D, _mm256_slli_epi64(D, 13));

            /* bending, half thrown out of the bottom, keep what disagrees */
            A = _mm256_xor_si256(A, _mm256_srli_epi64(A, 6));
            B = _mm256_xor_si256(B, _mm256_srli_epi64(B, 7));
            C = _mm256_xor_si256(C, _mm256_srli_epi64(C, 5));
            D = _mm256_xor_si256(D, _mm256_srli_epi64(D, 8));
        }

        /* the organs trade humours in pairs before reporting to the heart,
         * so no organ's sickness can reach the cup unbent.              */
        A = _mm256_add_epi64(A, _mm256_xor_si256(C, _mm256_srli_epi64(C, 29)));
        B = _mm256_add_epi64(B, _mm256_xor_si256(D, _mm256_srli_epi64(D, 23)));
        A = _mm256_add_epi64(A, _mm256_slli_epi64(A, 10));
        B = _mm256_add_epi64(B, _mm256_slli_epi64(B, 12));

        {
            uint64_t v[8];
            int k;
            _mm256_storeu_si256((__m256i *)(v + 0), A);
            _mm256_storeu_si256((__m256i *)(v + 4), B);
            for (k = 0; k < 8; k++) PULL(c, v[k]);   /* organs -> one cup */
        }
#else
        uint64_t L[16];
        int k;
        for (k = 0; k < 16; k++) L[k] = HUMOUR[k];

        for (; i + 128 <= len; i += 128) {
            uint64_t w[16];
            memcpy(w, data + i, 128);
            for (k = 0; k < 16; k++) {        /* 16 independent organs */
                L[k] += w[k];
                L[k]  = BEND_UP(L[k], 10);
                L[k]  = BEND_DOWN(L[k], 6);
            }
        }
        for (k = 0; k < 8; k++) {             /* trade humours in pairs */
            uint64_t t = BEND_DOWN(L[k + 8], 29);
            L[k] += t;
            L[k]  = BEND_UP(L[k], 10);
        }
        for (k = 0; k < 8; k++) PULL(c, L[k]);
#endif
    }

    /* ---- One throat, one cup: the unbroken pour over whatever is left
     * (the whole pile, in the quiet regime).  Eight marks per pull while
     * the pile allows, then mark by mark.  No mark judged twice.       */
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        PULL(c, w);
    }
    for (; i < len; i++) PULL(c, data[i]);

    /* ---- carry the last colour through the seven organs ---- */
    return organs7(c, (uint64_t)len);
}
```

**Improvement log (four passes, reasoning only — no tool in this session):**
1. *Literal naive*: one cup, one mark per pull, two bendings per byte, seven organs. ≈5 cycles/byte — **slower** than the FNV reference. Not shippable.
2. *Eight marks per pull*: the drop is an 8-byte word, so one pull latency covers 8 bytes (≈0.6 cycles/byte).
3. *Every organ compensates at once*: 16 cups over 128-byte blocks, four AVX2 registers, each organ bending its own share — four independent 5-cycle dependency chains per 128 bytes.
4. *Regime guard + pair-trading fold*: `len < 256` falls back to the one-throat pour so the fold can never make small piles slower, and the organs trade humours pairwise so every lane's difference is bent before it reaches the heart.

# PREDICTION

Reference FNV-1a is one dependent `imul` (3-cycle latency) per byte ⇒ ≈3.5–4 cycles/byte ≈ 1 GB/s. The organ path is four parallel chains of five 1-cycle ops per 128 bytes ⇒ a ceiling near 25 B/cycle, realistically load- and memory-bandwidth bound in the 10–25 GB/s range.

**PREDICTION: speedup_vs_dp = 18**

Also predicted, stated before measurement: avalanche ≈ 0.50 bits flipped per output bit (score within a few 1e-3 of ideal) for both regimes, because a one-bit flip enters one cup as an additive difference, every subsequent operation (`+ x`, `c + (c<<s)` = multiply by the odd constant 2^s+1, `c ^ (c>>s)`) is injective mod 2^64, so the difference can never vanish, and it then passes through at least 9 further bendings (worst case: the last organ's lane, poured last) before the output — four of which are Jenkins' validated finisher.

# MEASUREMENT

**Not measured. No tools were available in this session** (`hash_bench`/`hash_contract` were not callable), so I have no throughput number and no avalanche score to report, and I will not dress an op-count estimate up as one. The numbers in PREDICTION are an analytic estimate from latency/throughput accounting, nothing more. The pipeline's measurement stands as the result; if it comes back below ~5× or with avalanche materially off 0.50, this attempt failed and I would say so.

The two things I would check first in the measured output:
- **Avalanche in the large regime** — if it is low, the suspect is the fold: a lane difference entering at `v[7]` gets the fewest bendings. Fix inside the mechanism: one more pairwise humour-trade before the heart, not a multiply.
- **Throughput near `len = 256`** — the crossover. If the wide path is slower there than the one-throat path, raise the guard to 512; the fallback already exists, so this is a constant, not a redesign.

# VERDICT

The kernel's core **is** the native's second mechanism, literally: every arithmetic operation in it is "throw away half of what came before" (a shift) followed by either "keep only what refuses to sit still" (xor) or the organ compensating (add). There is no multiplication, no table, no CRC polynomial — the two things the reviewer rightly flagged last time are both gone, and the seven organs are the finisher rather than an afterthought bolted onto a standard body.

What the mechanism buys: multiply-free mixing is *cheap and vectorisable*, which is exactly why ARX designs exist. That is the real win here — the metaphor forbade the one operation (`imul`) that stops the body from running 16 lanes wide at one op per cycle.

Named risks, and how each is addressed rather than left standing:
- *"The wide path only helps if the pile is large; the fold is overhead if small."* Guarded: `len >= 256` with the one-throat pour as the fallback, which is itself a validated hash (Jenkins OAAT with a stronger finisher), not a stub.
- *"No multiply means weaker per-byte diffusion."* Addressed by banking the diffusion in the bounded seven-organ finisher instead of paying for it per byte — which is precisely the assumption being broken, and is the standard short-body + strong-finalizer argument.
- *"Shift/xor alone is GF(2)-linear."* Addressed: every third operation is an **add**, whose carries are nonlinear over GF(2). This is the one place where I leaned on SEED 1's reading — "cut wine *remembers every drop added to it*" versus "pure wine holds no memory" (xor is involutive: it forgets) — and it is what keeps the construction from degenerating into a linear hash.
- *Threads.* Deliberately omitted: the metaphor's unit of work is one organ's 32-byte share, far too small to justify an OpenMP region at these sizes.

Honest weakness: the wide regime relaxes SEED 1's "no mark ever judged alone" — a mark is bent alongside its lane-mates before the heart sees it. I did not hide this behind a reinterpretation; it is the price of choosing SEED 2's law over SEED 1's, and the quiet-pile path is where SEED 1's law is kept intact. If the measurement shows good throughput but weak avalanche, that trade is the first thing I would revisit — by trading humours more often, not by reaching for a multiply.