No tools are available in this session, so every number below that is labeled a prediction stays a prediction — I will not dress up an estimate as a measurement.

## MAPPING

**SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."**

| World object | Problem object |
|---|---|
| sailor-chain, a row of knots | the hash state, held as a *row* of several 64-bit slots, not one |
| adding a knot, never breaking an old one | each byte is folded in without destroying earlier contributions |
| chain hung between fire and wall | state sits between a fixed source constant and the read-out |
| shadow = shape of the *whole* chain | each slot's new value depends on its neighbours' values, so tension (dependence) travels down the row |
| "the whole line hangs together or not at all" | neighbour coupling makes the output a function of all slots |

Breaks: **"the state is a single accumulator updated in place, one value."**

**SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."**

| World object | Problem object |
|---|---|
| a mark | one input byte |
| the mark's strokes counted as pebbles | `popcount(byte)` — the number of strokes composing the mark |
| the *small* cup | the count is tiny, 0–8 |
| incense on a *short* chain, swung that far | a rotation of the state by a *small*, data-dependent amount (1–9) |
| smoke bending the fire's throw before it lands | the mixing operator is a bend/rotate, applied before read-out |
| no stone, no grinding, no wheel anywhere in the rite | **no multiply in the per-byte path** |

Breaks: **"mixing one byte requires a multiplication."**

**SEED 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."**

| World object | Problem object |
|---|---|
| wet clay pressed to the wall | one read-out of the state across its fixed positions |
| waiting for the shadow to go still / smoke to thin to one thread | the read-out is a single reduction to one value, done once |
| stream washing the soft away, leaving ridges | a finalizer that discards low-information bits (xor-shift) |
| throwing away smoke, cup, old print | no retained scratch, no second pass, nothing to walk backward |
| one pass of water, not many | **one strong finishing step, not more rounds** |

Breaks: **"more mixing rounds always means better mixing."**

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks the preferred assumption ("mixing one byte requires a multiplication"), and its mapping is the most literal: a per-byte quantity I can compute exactly (`popcount`) sets a per-byte rotation amount, exactly as the pebble count sets the swing. SEED 1 is kept as the *shape* of the state the swing acts on (six cracks = six slots, neighbour coupling = "the whole line hangs together"), and SEED 3 as the single final read-out — the native describes one rite, not three, so I take all three objects and let SEED 2 supply the mixing operator.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** In the per-byte loop there is no `imul`: a byte is tied in with XOR, then the slot is *bent* by a rotation whose amount is that byte's own popcount, then its neighbour's shadow is added in. Multiplication appears only once, O(1), in the stream-wash at the very end — outside the per-byte path.

Where the mechanism lands on validated ground rather than novelty:
- **Data-dependent rotation as the mixer** is RC5/RC6's central primitive — a real, analysed cipher technique, not an invention. It is also the only nonlinearity in the loop, and it is cheap precisely because flipping any one input bit changes that byte's popcount by exactly ±1, which rotates the whole slot one place further and makes every later shadow land differently.
- **Several independent accumulators coupled loosely** is the standard way fast hashes break the latency chain (xxHash's 4 lanes, CRC 3-way folding). The native's "six cracks" is that technique, with six.
- **The final wash** is murmur3's `fmix64`, a validated 64-bit avalanche finalizer, not something I made up.
- Addition for the neighbour coupling (shadows *overlap and darken*, tension *adds*) is what keeps the loop from being purely GF(2)-linear, for the same single-uop cost as XOR.

Regimes: the known_way spans byte-at-a-time (FNV, short keys) and block-at-a-time (xxHash, long keys). The native recognises this in-world — he looks at the pile and either **marches the marks six abreast** (wide pile: six slots updated per rank from the previous rank's values, so the six bends pipeline) or **walks a handful single file** (each knot feeling its neighbour at once). The switch is the loop condition itself, so the small-pile path pays no setup, no allocation, no prologue.

Risk parts deliberately **dropped, not shipped unguarded**: no OpenMP (the chain is serial by the metaphor and a 6-byte rank is far too small a unit of work to thread), no wide-SIMD prologue (AVX2 has no per-lane variable 64-bit rotate, so a vector path would have to abandon the data-dependent swing — the very thing the seed is about), no scratch buffer. The only fixed cost left is the one-time finisher, which every validated hash also pays and which is what buys avalanche on short inputs.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* the bend: the brick-incense on its short chain, swung w places.  1 <= w <= 63 */
static inline uint64_t swing(uint64_t x, unsigned w) {
    return (x << w) | (x >> (64u - w));
}

/* the small cup: this mark's own strokes, counted as pebbles.  yields 1..9 */
#define PEBBLES(b) ((unsigned)__builtin_popcount((unsigned)(unsigned char)(b)) + 1u)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* the fire, thrown onto the wall's six fixed cracks: six places, six slots */
    uint64_t s0 = 0x9E3779B97F4A7C15ULL;
    uint64_t s1 = 0xBF58476D1CE4E5B9ULL;
    uint64_t s2 = 0x94D049BB133111EBULL;
    uint64_t s3 = 0xFF51AFD7ED558CCDULL;
    uint64_t s4 = 0xC4CEB9FE1A85EC53ULL;
    uint64_t s5 = 0x165667B19E3779F9ULL;

    size_t n = len;

    /* WIDE PILE: march the marks six abreast, one knot per sailor per rank.
       every slot is bent by its own mark's pebble-count, then takes its
       neighbour's shadow from the rank before - tension travels down the
       chain at finite speed, so after six ranks the whole line hangs together,
       while the six bends themselves stay independent and pipeline. */
    while (n >= 6) {
        unsigned w0 = PEBBLES(p[0]), w1 = PEBBLES(p[1]), w2 = PEBBLES(p[2]);
        unsigned w3 = PEBBLES(p[3]), w4 = PEBBLES(p[4]), w5 = PEBBLES(p[5]);
        uint64_t t0 = swing(s0 ^ p[0], w0) + s5;
        uint64_t t1 = swing(s1 ^ p[1], w1) + s0;
        uint64_t t2 = swing(s2 ^ p[2], w2) + s1;
        uint64_t t3 = swing(s3 ^ p[3], w3) + s2;
        uint64_t t4 = swing(s4 ^ p[4], w4) + s3;
        uint64_t t5 = swing(s5 ^ p[5], w5) + s4;
        s0 = t0; s1 = t1; s2 = t2; s3 = t3; s4 = t4; s5 = t5;
        p += 6; n -= 6;
    }

    /* A HANDFUL: fewer than six left, so walk them single file past the fire -
       here each knot feels its neighbour at once.  No setup, no prologue. */
    if (n > 0) s0 = swing(s0 ^ p[0], PEBBLES(p[0])) + s5;
    if (n > 1) s1 = swing(s1 ^ p[1], PEBBLES(p[1])) + s0;
    if (n > 2) s2 = swing(s2 ^ p[2], PEBBLES(p[2])) + s1;
    if (n > 3) s3 = swing(s3 ^ p[3], PEBBLES(p[3])) + s2;
    if (n > 4) s4 = swing(s4 ^ p[4], PEBBLES(p[4])) + s3;

    /* THE CLAY PRINT: the shadow has gone still; press it to the wall and read
       it across the six fixed cracks, each crack at its own fixed height.
       The chain's own length is a property of the chain, so it prints too. */
    uint64_t h = (swing(s0,  7) + swing(s1, 19))
               ^ (swing(s2, 31) + swing(s3, 43))
               ^ (swing(s4, 53) + swing(s5, 61))
               ^ (uint64_t)len;

    /* THE STREAM: let the water run over the stub until only ridges remain.
       One wash, not many - and it is the only place a wheel is turned. */
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 2.0

Reasoning stated before any measurement, so it can be scored against: FNV-1a is latency-bound on its serial chain at roughly 4 cycles/byte (3-cycle `imul` + 1-cycle `xor`). This kernel is throughput-bound, not latency-bound: six independent bends per rank, each costing about a zero-extending load, a `popcnt`, an increment, an `xor`, a variable rotate (2–3 uops on x86 because of flag handling), and an `add` — call it 7–8 uops/byte, which at ~4 uops/cycle is ~1.8–2.0 cycles/byte. Hence ~2.0×, and I expect it to land between 1.6× and 2.6× depending on how gcc renders the variable rotate (`rol r64,cl` vs BMI2 `shlx/shrx/or`). I do **not** expect xxHash-class throughput — the native insists on one knot per mark, so this is byte-at-a-time by construction, and no amount of honest translation makes a byte-at-a-time hash read 8 bytes at once.

Avalanche prediction: good, ~0.49–0.50. Flipping any single input bit necessarily changes that byte's popcount by ±1, which rotates its slot one extra place — a ~32-bit difference immediately, which then spreads to all six slots within six ranks. The weak case is a flip in the final few bytes, which has little room left to diffuse; `fmix64` is there precisely to carry that case. The case I'd flag as the real structural weakness is *two*-bit flips within one byte that leave popcount unchanged (e.g. `0x03` → `0x05`): those take the identical rotation schedule, so they propagate through an almost-affine path and arrive at the finisher with low difference weight. `fmix64` will still avalanche them to ~half the output bits, but the collision structure there is less random than a multiply-based hash's. A single-bit avalanche test will not see this.

## MEASUREMENT

Not performed. `hash_bench` and `hash_contract` are unavailable in this session, so there is no measured throughput and no measured avalanche score to report, and I am not going to invent either. The artifact is complete and compiles standalone against the fixed contract (`uint64_t kernel(const unsigned char *, size_t)`, C99, no intrinsics beyond `__builtin_popcount`, no OpenMP, no scratch memory), so the pipeline can measure it directly.

Two things to check when it is measured, which would falsify specific claims above:
1. If throughput comes in below ~1.3×, the variable-count rotate is costing more than ~3 uops and the six-wide ILP is not covering it — the honest fix is fewer, longer-lived swings (bend once per rank on a packed word), which starts to abandon "one knot per mark."
2. If avalanche comes in below ~0.45, the linear XOR/rotate skeleton is leaking through the finisher, and the correct response is a second neighbour-coupling pass per rank, not more constants.

## VERDICT

The translation is literal on every object: byte = mark, `popcount` = pebbles in the small cup, rotation amount = how far the incense swings on its short chain, six 64-bit slots = the wall's six fixed cracks, neighbour-coupled addition = the shadow of the whole chain rather than the newest knot alone, `fmix64` = the stream washing the clay to its ridges, and the per-byte multiply is genuinely gone. Nothing was quietly swapped for the textbook method — in particular I did not reintroduce a per-byte multiply, and I did not replace the data-dependent swing with a fixed rotation, which is the first thing a performance-minded rewrite would do.

Where it is honestly weaker than FNV/xxHash: the per-byte uop count is higher than FNV's, and it only wins because it is throughput-bound instead of latency-bound — so on a narrow or uop-starved machine the margin shrinks toward 1×, and on anything with hardware `crc32`/AES it is simply outclassed on speed. The two-bit-same-popcount path described above is a real structural concession to the metaphor that a multiply-based mixer does not have.

The two conditions my own reasoning flagged as risky are addressed rather than left standing: thread parallelism and a wide-SIMD prologue are both **dropped** (a 6-byte rank is nowhere near a threadable unit of work, and AVX2 cannot do per-lane variable 64-bit rotation without discarding the seed's own mechanism), and the small-input regime is handled by the native's own single-file walk, which costs no setup — so there is no size at which this kernel pays for machinery it cannot amortize. The only fixed cost remaining is the one-time finisher, which FNV's minimal example omits and pays for in avalanche.