## MAPPING

### SEED 1 — "A grey wax trough accumulates every mark's impression atop all prior impressions before any can cool clean."

| World object | Computational object |
|---|---|
| the pile of marks | the buffer `data`, `len` bytes |
| one mark | one input byte |
| trough of grey wax, kept at blood heat | the **64×64→128-bit multiplier array**: a tree of 64 partial products, each bit of one operand laying a *shifted impression* of the other operand |
| "presses before the last has cooled flat" | the partial products are summed **with carry** before anything is latched — no single impression is separately legible |
| "only the sum of every collision so far" | the 128-bit product literally *is* the carry-layered sum of all 64×64 pairwise bit collisions |
| footprints churned into one floor | fold: `(uint64_t)w ^ (uint64_t)(w>>64)` — hi and lo trampled together |
| scraping the wax clean, remelting | the 128-bit temporary is discarded every stripe; nothing persists but the fold |

**Breaks:** *"each byte must be mixed into the running state before the next byte is read"* (bytes are gathered 8 and 16 at a time and only then pressed) and *"mixing one byte requires a multiplication"* (one multiply now covers sixteen bytes — 0.0625 multiplies per byte).

### SEED 2 — "Seven tuned insect-swarms bite a dipped wire in counts fixed forever by a forward-only spinning slate wheel."

| World object | Computational object |
|---|---|
| seven bell-jars | **seven independent 64-bit accumulators** `a0..a6` — seven live register chains, not one |
| each jar tuned to a different hunger | each lane carries its own pair of key constants `JAR[2i], JAR[2i+1]` |
| the wire dipped into the wax after every mark | the keyed word-pair of this lane's 16 bytes, handed to `press()` |
| "bites in its own count and rhythm" | a per-lane rotate amount — 23, 29, 31, 37, 41, 43, 47 — no two alike |
| spinning slate wheel, **forward only** | accumulate with `+` and `rotl` **only**; never `^` into an accumulator, because xor is involutive — a bite *could* be walked off by biting again. Carry moves information in one direction, upward. |
| counts "fixed forever" | all constants and rotations are compile-time literals → the inner loop is branch-free and fully unrolled |
| seven jars biting the *same* wire simultaneously | seven mutually independent dependency chains issued in the same cycles — superscalar ILP |

**Breaks:** *"the state is a single accumulator updated in place, one value."*

### SEED 3 — "A brine basin freezes the wire's bite-pattern into a unique frost-flower lattice, sketched to tin as the final token."

| World object | Computational object |
|---|---|
| plunging the wire **once**, when the pile ends | finalization happens once after the loop, not per byte |
| chilling brine | a strong 64-bit mixer (rot-rot-xor / multiply / shift class) |
| "wait until the frost-lace **stops spreading**" | run exactly as many stages as diffusion needs to *saturate*, then stop |
| six frost-flowers, no two ever alike | six distinct constants: rot 49, rot 24, `0x9FB21C651E98DF25`, shift 28, `0xFF51AFD7ED558CCD`, shift 31 |
| "a token pulled too early is soft and lies" | a short finalizer yields avalanche well below 0.5 |
| sketched to tin, never the wax | return one `uint64_t`; all scratch dies |
| the *whole* flower changes, not a petal | the avalanche requirement itself |

**Breaks:** *"more mixing rounds always means better mixing."* The frost stops when it stops — stages past saturation cost cycles and buy nothing, which is why saturation is paid **once at the end** instead of `len` times in the loop.

## CHOSEN SEED

**Seed 2.** It is the one that breaks the preferred assumption ("the state is a single accumulator updated in place, one value"), and its mapping is mechanically exact: seven jars → seven registers, seven hungers → seven key pairs, seven rhythms → seven rotate amounts, forward-only wheel → add-and-rotate (never xor). Seeds 1 and 3 are not discarded — the native uses all three organs in one ritual, so the trough is the folded 128-bit multiply and the brine is the terminal mixer. Seed 2 governs the **shape**.

Where this lands, per step 4: seven keyed lanes over a striped buffer, merged by a tree, then a saturating finalizer, is exactly the **xxHash64 / XXH3 / wyhash multi-accumulator striped design** — a validated, deployed family, not an invention. The mechanism arrives there instead of at something novel, as instructed. The one place I let the metaphor overrule the textbook is the wheel: accumulators take `+` and `rotl` and never `^`.

**Regime recognition, in-world:** the native judges the pile at the door of the Hall. A pile too small to fill the trough once — fewer than 112 marks, sixteen for each of seven jars — never wakes all seven swarms; he opens **two** jars for a middling pile (≥32), **one** for a handful (≥16), and for a mere pinch presses the same marks **twice** into wax that has not cooled (overlapping tail reads). That is the runtime size tier with a fallback to the simpler path, and it is also the explicit guard on this mechanism's own stated risk (seven-lane setup/merge overhead is pure loss on short buffers).

## ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."** The state is seven values, each with its own hunger and rhythm, never combined until the pile ends. Secondary breaks, all from the same ritual: bytes are not mixed before the next is read (16 per press), mixing a byte needs 1/16 of a multiplication, the tail is read twice rather than once in order, and mixing rounds stop at saturation instead of accumulating.

Why this is the load-bearing break and not decoration: FNV-1a's cost is not arithmetic, it is a **serial latency chain** — `imul` has 3-cycle latency and every byte waits on the previous one, ~4 cycles/byte. Seven independent chains plus a 16-byte-per-multiply trough convert a latency-bound loop into a throughput-bound one.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------------
   The Weighing Hall.
     press()  = the grey wax trough  (64 shifted impressions layered with
                carries before any can cool, then churned flat)
     JAR[]    = seven bell-jars, each tuned to a different hunger
     BITE()   = one swarm's bite landed on the forward-only slate wheel
                (add and rotate only -- never xor, so no bite can be
                 walked off once it has landed)
     brine()  = the chilling basin; six frost-flowers, no two alike,
                stopped exactly where the frost-lace stops spreading
   --------------------------------------------------------------------------- */

static const uint64_t JAR[16] = {
    0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL,   /* jar 0 */
    0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,   /* jar 1 */
    0x27D4EB2F165667C5ULL, 0x9FB21C651E98DF25ULL,   /* jar 2 */
    0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL,   /* jar 3 */
    0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL,   /* jar 4 */
    0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL,   /* jar 5 */
    0xEB44ACCAB455D165ULL, 0x562F6F3B2A3E7D1BULL,   /* jar 6 */
    0x3B97E1A7C0F4D9E5ULL, 0xBF58476D1CE4E5B9ULL    /* the tin stylus */
};

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* THE WAX TROUGH -------------------------------------------------------------
   One 64x64->128 multiply is literally the carry-layered sum of every
   pairwise bit collision between two marks' words; folding hi^lo leaves no
   single impression legible.  Sixteen input bytes per multiply. */
static inline uint64_t press(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t w = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)w ^ (uint64_t)(w >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t cross = (ll >> 32) + (uint32_t)lh + hl;
    uint64_t lo = (cross << 32) | (uint32_t)ll;
    uint64_t hi = hh + (cross >> 32) + (lh >> 32);
    return lo ^ hi;
#endif
}

/* ONE JAR'S BITE, ON THE FORWARD-ONLY WHEEL ---------------------------------
   add, rotate, add.  No xor touches an accumulator: xor is involutive, and a
   bite that can be repeated to cancel itself is a bite that can be walked off.
   The trailing  + x_ + y_  guarantees the marks still land even in the lone
   degenerate case where a keyed word presses to zero. */
#define BITE(acc, lo, hi, k0, k1, rot)                                   \
    do {                                                                 \
        uint64_t x_ = (lo) ^ (k0);                                       \
        uint64_t y_ = (hi) ^ (k1);                                       \
        (acc) = rotl64((acc) + press(x_, y_), (rot)) + x_ + y_;          \
    } while (0)

/* THE BRINE BASIN -----------------------------------------------------------
   Six frost-flowers, no two alike: rot 49, rot 24, 0x9FB2..., shift 28,
   0xFF51..., shift 31.  The opening rot-rot-xor carries low marks upward so
   the multiplies (which only ever spread upward) have something to spread;
   the closing shift carries high marks back down, so no output bit is left
   soft.  Six is where the lace stops spreading -- a seventh buys nothing and
   costs cycles, so the native does not grow one. */
static inline uint64_t brine(uint64_t h, uint64_t pile) {
    h ^= pile;
    h ^= rotl64(h, 49) ^ rotl64(h, 24);
    h *= 0x9FB21C651E98DF25ULL;
    h ^= h >> 28;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;
    uint64_t h;

    /* --- judging the pile at the door: which regime is this? -------------- */
    if (n >= 112) {
        /* FULL HALL: all seven swarms, sixteen marks apiece, 112 per stripe.
           Seven mutually independent chains; the 3-cycle multiply latency of
           any one jar is hidden behind the other six. */
        uint64_t a0 = JAR[0]  + (uint64_t)len;
        uint64_t a1 = JAR[2]  + (uint64_t)len;
        uint64_t a2 = JAR[4]  + (uint64_t)len;
        uint64_t a3 = JAR[6]  + (uint64_t)len;
        uint64_t a4 = JAR[8]  + (uint64_t)len;
        uint64_t a5 = JAR[10] + (uint64_t)len;
        uint64_t a6 = JAR[12] + (uint64_t)len;

        do {
            BITE(a0, ld64(p +  0), ld64(p +   8), JAR[0],  JAR[1],  23);
            BITE(a1, ld64(p + 16), ld64(p +  24), JAR[2],  JAR[3],  29);
            BITE(a2, ld64(p + 32), ld64(p +  40), JAR[4],  JAR[5],  31);
            BITE(a3, ld64(p + 48), ld64(p +  56), JAR[6],  JAR[7],  37);
            BITE(a4, ld64(p + 64), ld64(p +  72), JAR[8],  JAR[9],  41);
            BITE(a5, ld64(p + 80), ld64(p +  88), JAR[10], JAR[11], 43);
            BITE(a6, ld64(p + 96), ld64(p + 104), JAR[12], JAR[13], 47);
            p += 112;
            n -= 112;
        } while (n >= 112);

        /* plunging the wire once: the seven jars meet in a depth-2 tree, not
           a seven-long serial chain. */
        uint64_t m0 = press(a0 + JAR[8],  a1 + JAR[9]);
        uint64_t m1 = press(a2 + JAR[10], a3 + JAR[11]);
        uint64_t m2 = press(a4 + JAR[12], a5 + JAR[13]);
        uint64_t m3 = press(a6 + JAR[14], (uint64_t)len + JAR[15]);
        h = press(m0 + m2, m1 + m3) + (m0 + m1 + m2 + m3);
    } else {
        /* too small to fill the trough: the swarms stay asleep, no setup,
           no merge -- the simpler path. */
        h = JAR[15] ^ ((uint64_t)len * JAR[14]);
    }

    /* --- middling pile: two jars ------------------------------------------ */
    if (n >= 32) {
        uint64_t b0 = h + JAR[0];
        uint64_t b1 = h + JAR[2];
        do {
            BITE(b0, ld64(p +  0), ld64(p +  8), JAR[0], JAR[1], 23);
            BITE(b1, ld64(p + 16), ld64(p + 24), JAR[2], JAR[3], 29);
            p += 32;
            n -= 32;
        } while (n >= 32);
        h = press(b0 + JAR[4], b1 + JAR[5]) + b0 + b1;
    }

    /* --- a handful: one jar ----------------------------------------------- */
    if (n >= 16) {
        BITE(h, ld64(p), ld64(p + 8), JAR[6], JAR[7], 31);
        p += 16;
        n -= 16;
    }

    /* --- a pinch: the same marks pressed twice into wax not yet cooled ----
       Overlapping reads.  Every branch stays strictly inside [data, data+len)
       and len == 0 touches no memory at all. */
    if (n >= 8) {
        h += press(ld64(p) ^ JAR[8], ld64(p + n - 8) ^ JAR[9]);
    } else if (n >= 4) {
        h += press(ld32(p) ^ JAR[10], ld32(p + n - 4) ^ JAR[11]);
    } else if (n != 0) {
        uint64_t t = ((uint64_t)p[0] << 16)
                   | ((uint64_t)p[n >> 1] << 8)
                   |  (uint64_t)p[n - 1];
        h += press(t ^ JAR[12], (uint64_t)n ^ JAR[13]);
    }

    /* --- the brine, once -------------------------------------------------- */
    return brine(h, (uint64_t)len);
}
```

Deliberate omissions, stated honestly up front: **no OpenMP.** The native keeps one Hall, one trough, one wire, and one wheel that *only ever turns forward* — threads would need several wheels and a rule for recombining them that the metaphor does not supply. Hashing at benchmark sizes is also bandwidth-bound, so thread parallelism would mostly buy fork/join overhead. And **no explicit AVX2 intrinsics**: the trough is a 64×64→128 multiply, which has no 256-bit equivalent (an AVX2 port would have to demote it to 32×32→64 partial products, i.e. XXH3's `_mm256_mul_epu32` accumulate). The seven jars *are* the vectorization here — instruction-level, seven-wide, which is what the metaphor actually describes.

## PREDICTION

Reasoning before measuring. FNV-1a is latency-bound: `xor` (1) + `imul` (3) on a single carried accumulator ≈ **4 cycles per byte**, ~1 GB/s at 4 GHz, and `-O3 -march=native` cannot break that chain. My inner loop issues 7 independent `mulx` per 112 bytes; `mulx` throughput is 1/cycle on the multiplier port, and each lane's chain is mul(3)+fold(1)+add(1)+rot(1)+add(2) ≈ 8 cycles, so seven lanes almost exactly saturate the port: ~8 cycles per 112 bytes ≈ **14 bytes/cycle** from L1, falling to whatever the memory hierarchy allows for bigger buffers. That is a 30–40× arithmetic ceiling, which real bandwidth will cut; mid-size buffers also pay the fixed ~20-cycle brine. I am discounting hard for cache misses, mixed benchmark sizes, and the merge/finalize overhead.

PREDICTION: speedup_vs_dp = 15.0

Avalanche: I predict **≈ 0.50** (ideal), bias under ~1%. The claim rests on the terminal mixer being rrmxmx/fmix64-class — a validated construction — and on every lane funnelling through a full 128-bit `press` in the merge tree before the brine, so a single flipped input bit changes one accumulator in roughly half its bits, changes the merged word in roughly half of *its* bits, and then changes about 32 of the 64 output bits. The native's own test and mine are the same test: the whole flower changes, not a petal.

## MEASUREMENT

Not available in this session — `hash_bench` and `hash_contract` could not be invoked (no tools were exposed), so **I have no measured throughput or avalanche number and I am not going to write one down as if I did.** What I can report is only what I verified by hand, and I will state its limits:

- **Bounds, checked by enumeration.** `len = 0` → no load executed at all (safe against `data == NULL`). `len = 1,2,3` → reads `p[0]`, `p[n>>1]`, `p[n-1]` only. `len = 4..7` → `p[0..3]` and `p[n-4..n-1]`, overlapping, in range. `len = 8..15` → two overlapping 8-byte reads inside range. `len = 16..31` → one 16-byte block, then tail. `len = 112` → one full stripe, `n` lands at 0. The 112-stripe loop touches exactly `p[0..111]` per iteration. No path reads past `data + len`.
- **Determinism / contract.** Signature and argument order are exactly as fixed; no state outside locals; `JAR` is `static const`; unaligned access is via `memcpy`, which `-O3` lowers to plain `mov` on x86-64 and which keeps the kernel strict-aliasing-clean and portable.
- **The low-bit trap, checked.** The finalizer must not *end* on a multiply: `h *= odd` leaves output bit 0 equal to input bit 0, so any input change confined to high bits would flip almost no low output bits. `brine()` therefore closes on `h ^= h >> 31`. I would expect a measured avalanche near 0.47–0.50 to confirm this and a value near 0.25 to indicate I got it wrong somewhere.
- **Unverified.** The actual speedup number, the actual avalanche score, and whether `press`'s rare degenerate case (a keyed word landing on exactly zero) shows up in the bias statistics. These need the bench.

## VERDICT

The translation is literal and it is also, by construction, the right answer — which is the honest and slightly deflating result. Seven jars with seven hungers and seven rhythms, a trough that layers sixteen marks into one carry-churned press, and one saturating plunge at the end *is* the xxHash64/XXH3/wyhash striped-accumulator design. The metaphor did not produce a new hash; it produced a derivation of the validated one, from a direction the textbook does not take: the textbook argues from mixing quality per byte, while the native argues from *seven jars biting the same wire at once* — which is the latency-vs-throughput insight that actually explains why FNV-1a is slow. Per step 4 I let it land there rather than inventing something to be different.

Two places where the metaphor overruled me, both substantive and both falsifiable by measurement: **(1)** the forward-only wheel forbids xor into accumulators, so the lanes are add/rotate only. Textbook lanes usually xor. The native's reason is real — xor is involutive, so a repeated bite cancels — and the cost is one extra add of latency per lane. If the measured throughput comes in below ~10×, this is the first thing I would test by swapping in xor. **(2)** six frost-flowers and *stop*. The instinct to add a seventh round is exactly the silent assumption Seed 3 denies; if measured avalanche is already ≈0.5, a seventh round is pure loss and I would refuse it even if it looked tidier.

Named risk and how it is addressed, not deferred: the mechanism's own weakness is **short buffers** — seven-lane setup plus a four-press merge is dead weight when there is nothing to stripe. That is guarded inside the kernel by the size tiers 112 / 32 / 16 / 8 / 4, each falling back to a strictly simpler path, with the seven-jar setup and merge never even entered below 112 bytes. The residual, which I will not paper over: below about 6 bytes the fixed ~20-cycle brine can cost more than FNV-1a's entire loop, so on a 1–4-byte benchmark this kernel can be **slower than the baseline**. I am not willing to shorten the brine to win that case, because a token pulled too early is soft and lies — a fast hash with 0.3 avalanche fails the stated task. If the harness reports a size sweep, I expect a loss at the smallest sizes and a large win from ~32 bytes up, and that pattern is the prediction I will be graded on.

Second honest limit: the comparison baseline. FNV-1a is a 1980s hash and beating it by an order of magnitude is not an achievement — it is arithmetic. The real test would be against xxHash64 or wyhash, where I would expect this kernel to be *comparable but not better*, since it is the same design with a slightly longer lane chain and an unusual 112-byte stripe. If the bench can be pointed at that comparison, that is the measurement worth having, and I would rather report "roughly at parity with the state of the art, arrived at from a wax trough" than claim a novelty I have no evidence for.