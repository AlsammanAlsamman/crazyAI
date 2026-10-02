## MAPPING (step 1 — one table per SEED)

**SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."**

| world object | problem object |
|---|---|
| sailor-chain, only ever lengthened, never a knot broken | the hash state, updated only by *invertible* operations (add / rotate / xor-transvection) — no byte's contribution is ever destroyed |
| one more knot per mark | one state injection per input unit |
| shadow is of the *whole* chain between fire and wall | every step's output depends on all six state words at once (a braid couples all lanes each step), not on one slot |
| "the whole line hangs together or not at all" | the state is a **row of six words**, read as a unit |
| six fixed cracks in the wall, the same six every time | six fixed rotation constants — the only rotation amounts the ritual ever uses (and six lanes the token reads from) |

*Breaks:* "**the state is a single accumulator updated in place, one value**" (and partly "the whole buffer must be read once, start to end, in order" — the six walk abreast).

**SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."**

| world object | problem object |
|---|---|
| mark | input byte |
| its strokes, counted as pebbles into a small cup | `popcount` of the byte |
| the cup's *fill* (cumulative — it is only emptied at the end) | running sum of popcounts of all bytes seen so far |
| how far the brick-incense swings on its short chain | **rotation amount**, `(cup + crack) mod 64` |
| smoke bending the fire's throw before it lands | the rotation is applied *after* the knot is added, so the data chooses how the state is re-aligned |
| fire (the only thing whose shadow is steady) | the mixing operator itself — one swing per step, nothing else |

*Breaks:* "**mixing one byte requires a multiplication**" — the whole ritual is add + rotate + xor, with the data choosing the rotation. No multiply anywhere.

**SEED 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."**

| world object | problem object |
|---|---|
| waiting for the smoke to thin to one straight thread | finalize **once**, at the end — no extra mixing round per byte |
| pressing the clay to the wall's six cracks | reading the six state words into the print |
| the stream running over the stub until only ridges remain | the finalizer: a few ARX washes, then truncation 384 → 128 → 64 bits |
| throwing away the smoke, the cup, the old print | intermediate state is discarded; the output is strictly smaller than the state (one-way, "no hooded reader works backward from print to pile") |
| "small enough for one fist" | the 64-bit return value |

*Breaks:* "**more mixing rounds always means better mixing**" — during absorption there are *zero* extra rounds; diffusion is paid for exactly once, at the print, and the wash count is read off how much smoke there was (pile size).

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks the preferred assumption (multiplication), and it is maximally unlike FNV-1a/xxHash: the data does not get multiplied into the state, it *chooses the geometry of the state's re-alignment*. SEEDs 1 and 3 are not discarded — they are the same native's single ritual, and they fix the shape of the state (a six-word row, braided every step) and the shape of the finalizer (press once, wash, truncate). I built all three; SEED 2 is the load-bearing one.

## ASSUMPTION BROKEN

Primary: **"mixing one byte requires a multiplication."** Replaced by *cumulative-popcount-driven data-dependent rotation* + add + xor.

Also broken: **"the state is a single accumulator, one value"** (six-word braided row), **"the whole buffer must be read once, start to end, in order"** (six abreast, 48 bytes per step, order still fully significant), **"each byte must be mixed before the next is read"** (eight marks are knotted as one handful — see the honesty note below), **"more rounds is better"** (zero rounds during absorb).

**Step-4 check — does the mechanism land on a validated technique?** Yes, deliberately, and I let it: multiplication-free mixing *is* the ARX family, and the canonical validated instance is **SipHash**. So the six cracks are literally SipRound's six rotation constants {13, 32, 16, 21, 17, 32}; the wash is SipRound generalized to six words; the padded final block is Merkle–Damgård length padding; the 128→64 fold is sponge-style truncation. The *only* element I kept that is not in SipHash is the native's own one: the rotation amount is data-driven (RC5/RC6 lineage) rather than fixed. That is the seed, and it is the part that is actually being tested.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================================================================= *
 *  The six fixed cracks in the wall — the same six every time.      *
 *  (These are exactly SipRound's six rotation amounts.)             *
 * ================================================================= */
#define CRK_A 13u
#define CRK_B 32u
#define CRK_C 16u
#define CRK_D 21u
#define CRK_E 17u
#define CRK_F 32u

/* the incense swings across the mouth of the piazza: rotate left */
static inline uint64_t swing(uint64_t x, unsigned c) {
    c &= 63u;
    return (x << c) | (x >> ((64u - c) & 63u));   /* UB-free; gcc emits rolq */
}

/* a mark's own strokes, counted as pebbles into the cup */
static inline unsigned strokes(uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcountll(w);     /* POPCNT under -march=native */
#else
    uint64_t y = w - ((w >> 1) & 0x5555555555555555ULL);
    y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
    y = (y + (y >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    y += y >> 8; y += y >> 16; y += y >> 32;
    return (unsigned)(y & 0x7fu);
#endif
}

static inline uint64_t handful(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;  /* 8 marks, in given order */
}

/* ----------------------------------------------------------------- *
 *  Walk 48 marks past the fire: the cup is filled with their strokes,
 *  each of the six sailors takes one more knot, each swings by the
 *  cup's fill bent through his own crack, then the whole line is
 *  braided so no shadow belongs to one knot alone.
 *  The braid is a composition of transvections => invertible: an old
 *  knot is lengthened, never broken.
 * ----------------------------------------------------------------- */
#define KNOT_BLOCK(src) do {                                                 \
    const unsigned char *s_ = (src);                                         \
    uint64_t w0 = handful(s_),      w1 = handful(s_ +  8),                    \
             w2 = handful(s_ + 16), w3 = handful(s_ + 24),                    \
             w4 = handful(s_ + 32), w5 = handful(s_ + 40);                    \
    cup += strokes(w0) + strokes(w1) + strokes(w2)                           \
         + strokes(w3) + strokes(w4) + strokes(w5);                          \
    v0 = swing(v0 + w0, cup + CRK_A);                                        \
    v1 = swing(v1 + w1, cup + CRK_B);                                        \
    v2 = swing(v2 + w2, cup + CRK_C);                                        \
    v3 = swing(v3 + w3, cup + CRK_D);                                        \
    v4 = swing(v4 + w4, cup + CRK_E);                                        \
    v5 = swing(v5 + w5, cup + CRK_F);                                        \
    v0 ^= v1; v1 ^= v2; v2 ^= v3; v3 ^= v4; v4 ^= v5; v5 ^= v0;              \
} while (0)

/* one wash of the stream over the print; every rotation is a crack */
#define WASH do {                                                            \
    v0 += v1; v1 = swing(v1, CRK_A); v1 ^= v0; v0 = swing(v0, CRK_B);        \
    v2 += v3; v3 = swing(v3, CRK_C); v3 ^= v2; v2 = swing(v2, CRK_B);        \
    v4 += v5; v5 = swing(v5, CRK_E); v5 ^= v4; v4 = swing(v4, CRK_B);        \
    v2 += v1; v1 = swing(v1, CRK_D); v1 ^= v2;                               \
    v4 += v3; v3 = swing(v3, CRK_A); v3 ^= v4;                               \
    v0 += v5; v5 = swing(v5, CRK_C); v5 ^= v0;                               \
} while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;
    unsigned cup = 0;                       /* the small cup of counted pebbles */

    /* the six sailors, each with his own first knot */
    uint64_t v0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL ^ swing((uint64_t)len, 32);
    uint64_t v3 = 0x7465646279746573ULL;
    uint64_t v4 = 0x7472757374746865ULL;    /* "trustthe" */
    uint64_t v5 = 0x736861646f777300ULL;    /* "shadows"  */

    /* REGIME CHECK, at the mouth of the piazza: is the pile wider than
       the six sailors standing abreast?  If so they walk the long pile
       in 48-mark stretches.  If not, this loop never runs and the pile
       goes straight to the tray below — one O(1) pass, no loop at all. */
    if (n >= 48) {
        do { KNOT_BLOCK(p); p += 48; n -= 48; } while (n >= 48);
    }

    /* The leftover marks (0..47) are laid on a padded tray of 48 slots.
       Empty slots cost nothing and can never be mistaken for marks,
       because the chain's length is knotted in separately.  Taken
       always, even when empty: that is the length-padding block. */
    {
        uint64_t tray[6] = { 0, 0, 0, 0, 0, 0 };
        unsigned char *t = (unsigned char *)tray;
        size_t m = n;
        while (m >= 8) { memcpy(t, p, 8); t += 8; p += 8; m -= 8; } /* inlined */
        while (m)      { *t++ = *p++; m--; }
        KNOT_BLOCK((const unsigned char *)tray);
    }

    /* press the print: the chain's length and the cup go in, then both
       are thrown away with the smoke */
    v1 ^= (uint64_t)len;
    v4 += (uint64_t)cup;

    /* read only when the smoke has thinned to one thread: a small pile
       makes less smoke, so it needs one wash fewer */
    {
        unsigned washes = (len < 32u) ? 3u : 4u;
        do { WASH; } while (--washes);
    }

    /* harden in the stream until only the ridges remain: 384 -> 128 -> 64,
       so nothing can be walked backward from print to pile */
    {
        uint64_t a = v0 ^ v2 ^ v4;
        uint64_t b = v1 ^ v3 ^ v5;
        a += b; b = swing(b, CRK_A); b ^= a; a = swing(a, CRK_B);
        a += b; b = swing(b, CRK_E); b ^= a; a = swing(a, CRK_B);
        return a ^ b;
    }
}
```

**Honest note on literalness.** A "mark" is a byte and the cup is filled one mark at a time — and it still is, *bit-exactly*: `popcount` is additive over bytes, so the cup's fill after a handful of eight is identical to counting those eight singly. What I coarsened is only the *knotting*: eight marks join the chain as one handful instead of eight successive swings. That is precisely the assumption "each byte must be mixed into the running state before the next byte is read", which the brief flags as a silent assumption to break, so I broke it on purpose rather than by accident. Everything the seed actually claims — the cup sets the swing, the swing is the only mixing, the whole row casts the shadow — is literal.

## PREDICTION

**PREDICTION: speedup_vs_dp = 10.0**

Written before any measurement. Reasoning, so it is falsifiable:

- FNV-1a baseline is latency-bound on `imul` (3c) + `xor` (1c) ⇒ ~4 cycles/byte ⇒ ~0.9 GB/s at 3.5 GHz.
- Mine is throughput-bound, not latency-bound: per 48 bytes ≈ 6 loads + 6 popcnt + 6 cup-adds + 6 lea + 6 adds + 6 rol + 6 xor + loop ≈ 45–50 uops ⇒ ~0.25 cycles/byte ⇒ ~14 GB/s (L1/L2-resident). Loop-carried chains are short (cup: ~3c; state: add→rol→braid ≈ 4c per 48 bytes), six-way ILP hides everything.
- So **12–16×** if the benchmark buffer is cache-resident, **~8–10×** if it is DRAM-bandwidth-bound. I predict 10.0 as the central estimate. Below ~12-byte inputs I predict a *loss* of a bounded ~10–15 cycles (fixed tray + wash cost).
- Avalanche: I predict a mean output-bit flip fraction of **0.50 ± 0.01** (near-maximal score). A single input bit flip always changes `popcount` by ±1, hence changes `cup`, hence changes the rotation amount of **all six lanes for every subsequent block** — so even a flip in the final tray block scrambles all six words before 3–4 SipRound-grade washes and the 128→64 fold.

Falsifiers I would accept: measured avalanche < 0.48 ⇒ the wash count is too low, raise to 4 unconditionally; measured speedup < 3× on a ≥64 KB buffer ⇒ `popcnt` is the real bottleneck (1/cycle on Intel ports) and the cup must be computed over fewer, wider units.

## MEASUREMENT

**Not measured.** No tools were available in this session — `hash_bench` and `hash_contract` could not be called, so I am reporting this as unmeasured rather than inventing numbers. The prediction above stands untested and the pipeline's numbers govern. What it should be checked against: `gcc -O3 -march=native -fopenmp -lm`, throughput on a buffer of at least 64 KB, avalanche at both a small (≤32 B, 3-wash path) and a large (≥48 B, 4-wash path) size, since those are two different code paths and I am less certain about the 3-wash one.

Zero of my four allowed improvement rounds were spent, because none could be informed by data. The design decisions I *would* have spent them on, in order, if the numbers disagreed: (1) wash count 3→4 unconditionally if small-input avalanche is weak; (2) drop the shared cup to six per-lane cups if the cup's add-tree shows up as the critical path; (3) replace six scalar lanes with AVX2 `vpsllvq`/`vpsrlvq`/`vpor` + a `pshufb` nibble-table popcount if `popcnt` port pressure dominates; (4) nothing else.

## VERDICT

The native's ritual is, read literally, a **length-padded ARX sponge with data-dependent rotation**: six-word state, one handful knotted per sailor per step, rotation amount taken from the cumulative popcount of everything seen so far, an invertible braid so the row casts one shadow, no absorb-time rounds at all, and a single press-and-wash at the end that truncates 384 bits down to 64. It is multiplication-free end to end. That is not a hole in the metaphor — it is the metaphor's whole point, and it lands on the validated ARX/SipHash family rather than on something I invented, with SipRound's actual six constants as the six cracks.

Named risks, and how each is addressed rather than left standing:

- **Fixed cost dominates tiny inputs.** For `len ≲ 12` the tray block plus washes (~35 cycles) exceeds FNV's whole run. Guarded by a size check: `len < 48` skips the block loop entirely (one O(1) tray pass, no loop, no per-byte serial path), and `len < 32` drops to three washes. The residual loss is a *bounded constant* of ~10–15 cycles, not a growing factor, and the crossover is ~10–12 bytes. I chose not to fall back to a multiplying hash there, because that would forfeit the exact assumption under test, and the bounded constant is the honest price.
- **The wide path needs a full 48-byte stretch.** Guarded by `n >= 48`, with the padded tray as the fallback path for every shorter case.
- **Thread parallelism: deliberately absent.** The metaphor's unit of work is 48 bytes; at the sizes a hash benchmark uses, OpenMP fork/join would swamp a kernel that runs at ~0.25 cycles/byte. The native's six sailors are my parallelism, and they are instruction-level: six independent lanes, `restrict`, and one sequential 48-byte stride that the hardware prefetcher likes. Per the brief, vectorization-shaped hints first; threads not added.
- **SIMD not hand-written, and I will say why rather than pretend.** Six lanes map onto neither AVX2 (4×64) nor AVX512 (8×64), and per-lane *variable* rotation plus per-lane 64-bit popcount wants AVX512-VPOPCNTDQ, which `-march=native` may not provide. Truncating the native's six cracks to four to fit a register would be exactly the silent substitution I was told not to make. Six-way scalar ILP already saturates a 4-wide front end.
- **Where this would genuinely lose to xxHash:** xxHash gets ~8 bytes per multiply-add with 4 lanes and no per-word popcount; my per-handful popcount pins me near 1 uop/byte. Against the stated `dp` baseline (FNV-1a) I expect a large win; against a tuned xxHash I would expect to be slower, and I am not claiming otherwise.

The claim I am actually making, and which the pipeline can refute: *a data-dependent rotation whose amount is the running popcount of the input is enough nonlinearity to replace the multiply entirely*, provided the state is a row that hangs together and the print is pressed, washed and truncated exactly once.