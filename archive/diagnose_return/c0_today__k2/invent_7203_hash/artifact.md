## MAPPING

### SEED 1 — the chained prisoners sharpening the blank pillar

| World object | Computational object |
|---|---|
| the pile of marks, in their given order | the byte buffer `data[0..len)`, consumed in order |
| one mark | one 8-byte word pulled from the buffer |
| the *nearest* prisoner (only he touches the mark) | lane 0 of a 3-lane state; a mark is XOR-ed into exactly one lane, never broadcast |
| "does not look at it, only feels how it changes the angle of what he was already sharpening" | the lane does not store the mark; it absorbs it as a *perturbation of its own running remainder* — a shift-register/CRC step, not a multiply-accumulate |
| "passes that angle, not the mark, to the next in the chain" | lane *i*'s **state** (not the data) is XOR-ed into lane *i+1*'s input word |
| "the angle carries backward through the line" | lane 2's state is fed back into lane 0 — the three lanes form a closed ring, not three independent stripes |
| "a mark dropped at the very start still trembles in the hand of the last one" | complete dependency: every byte reaches every lane after ≤2 further passes |
| the blank pillar, never read whole | the 96-bit lane state, never sampled mid-stream |
| **assumption broken** | *"the state is a single accumulator updated in place, one value"* and *"mixing one byte requires a multiplication"* — the per-byte work here is a shift-register chain with **zero multiplies** |

### SEED 2 — the flood that never recedes, bending anchored wires

| World object | Computational object |
|---|---|
| the flood rising on each pass | a secondary accumulation driven once per absorbed stripe |
| "does not recede until every mark has gone through" | no value may be extracted before the loop ends |
| "a token pulled while the water is still up is worthless, the shapes beneath it still swimming" | the raw accumulator is *not* the hash — unfinalised state is explicitly declared garbage |
| the tiny wire shapes, **anchored** in the riverbed | three fixed, register-resident 64-bit accumulators (`w0,w1,w2`), seeded from fixed secrets ("anchors"), never from data |
| "they do not move themselves; the flood moves them" | the wires have no input of their own; they are driven only by the chain's output |
| "bending each wire by **exactly the angle** the last chained prisoner ground" | **data-dependent rotation**: `w0 = rotl64(w0 ^ angle, angle & 63)` — the rotate amount *is* the state. This is the only nonlinearity in the absorb loop, and it costs one variable shift |
| **assumption broken** | *"each byte must be mixed into the running state before the next byte is read"* — the wires are bent by a one-pass-lagged angle; the strong mixing of a byte completes *after* later bytes have already been read |

### SEED 3 — the silhouette against the sun, read once

| World object | Computational object |
|---|---|
| climbing to the pit's rim only after the water goes down | the finaliser runs exactly once, after the loop, outside it |
| "the shadows stop their chitter — the walls are still settling" | wait for the dependency chains to retire; do not interleave finalisation with absorption |
| the silhouette: "which lean, which stand straight, which cross another" | the projection takes **rotated** lanes (lean), **raw** lanes (straight), and **products** of lane pairs (cross) — a 64×64→128 multiply folded down |
| "small and fixed, drawn only once" | a constant-cost 3-multiply projection, independent of `len` |
| "I keep nothing but that shadow-shape" | return the projection; the lanes and wires are discarded |
| "the pillar keeps sharpening onward as if nothing had ever passed through it" | **no per-byte finalisation round at all** — the absorb step is deliberately weak |
| "I never ask the river where its source lies" | no inversion, no reordering, no parallel re-association of the stream |
| **assumption broken** | *"more mixing rounds always means better mixing"* — all strength is moved out of the per-byte loop into a single read |

## CHOSEN SEED

**SEED 3**, the read-once silhouette. It is the only one of the three that breaks the preferred assumption (*more mixing rounds always means better mixing*), and it is maximally different from FNV-1a, which spends a full strong round (xor + 64-bit multiply, a 4-cycle serial dependency) on *every single byte* and then returns the accumulator **raw**, with no finalisation at all. Seed 3 inverts both halves of that: the per-byte round becomes as weak and as parallel as possible, and the *only* strong round is one fixed projection at the end.

Seeds 1 and 2 are not discarded — the native's account is one machine, and seed 3 is only coherent if something cheap does the absorbing. They supply the loop body that seed 3's finaliser reads from.

**Letting the mechanism land on a validated technique (step 4).** A weak multiply-free absorb into several coupled lanes plus one strong read-once finaliser is not a new idea — it is the **sponge absorb/squeeze split**, and on x86 its validated instantiation is **hardware CRC32C (`_mm_crc32_u64`) lanes followed by a strong avalanche**, as used in FarmHash's CRC variants and in numerous production hash tables; the finaliser is **wyhash's `wymix` folded 128-bit multiply** plus a Murmur3-style tail, both SMHasher-validated. CRC32C is *literally* the native's chained prisoners: a shift register in which each input bit perturbs a remainder and the perturbation is handed down the chain, bit by bit, with every earlier input still present in the last position. I therefore let the metaphor arrive there rather than inventing a fresh ARX absorb. CRC32C alone is GF(2)-linear and would be a bad hash; the native already forbids keeping it ("a token pulled while the water is still up is worthless") and supplies both fixes — the data-dependent wire rotation inside the loop, and the multiplicative silhouette at the end.

## ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** Here the per-stripe round is 3 `crc32q` + 3 XOR + a rotation — no multiply, nothing resembling a mixing round — and it is run once per 24 bytes. The total number of strong (multiplicative) rounds in the whole hash is **three, regardless of input length**. Secondarily broken: the single-accumulator state (replaced by a 3-lane ring + 3 anchored wires), and the per-byte multiply.

**Two regimes, recognised in-world (step 5).** The native counts the pile against the depth of the pit *before* descending: a pile too small to raise the flood gets no ceremony at all — the shadows would never settle, the shapes would still be swimming, and the token would be worthless. So `len < 24` (one pass's worth) skips the chain, the flood and the wires entirely and presses the marks straight onto the shadow-stone: 2–3 overlapping word loads into the same silhouette. That is also exactly the guard my own verdict demands — the loop machinery has setup overhead that would lose to FNV-1a on short keys, so the short path is not optional. A second regime is hardware, not size: a pit with no chained prisoners (no SSE4.2) falls back to an identical ring in which each prisoner sharpens by add-rotate-xor instead of by shift register.

**No thread parallelism.** The native forbids it explicitly — the marks go down one by one in their given order, and the flood may not be read early. Threads would also require re-associating the stream, which changes the hash value. Only vectorisation-level hints are used: `restrict`, `memcpy` word loads, three independent 3-cycle-latency chains sized to saturate the CRC port, and a 24-byte stride.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__)
#  include <immintrin.h>
#  define CHAINED_PRISONERS 1
#else
#  define CHAINED_PRISONERS 0
#endif

/* the fixed anchors in the riverbed: the wires are seeded from these, never from data */
#define A0 0xa0761d6478bd642fULL
#define A1 0xe7037ed1a0b428dbULL
#define A2 0x8ebc6af09c88c6e3ULL
#define A3 0x2d358dccaa6c78a5ULL
#define A4 0x8bb84b93962eacc9ULL
#define A5 0x4b33a62ed433d4a3ULL

static inline uint64_t rd8(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t rd4(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

static inline uint64_t rotl64(uint64_t x, unsigned r){
    r &= 63u;                       /* r == 0 is well defined under this idiom */
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* two wires crossing in the silhouette: 64x64 -> 128, folded (wyhash's wymix) */
static inline uint64_t cross(uint64_t x, uint64_t y){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)x * (__uint128_t)y;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t xl = (uint32_t)x, xh = x >> 32, yl = (uint32_t)y, yh = y >> 32;
    uint64_t lo = x * y;
    uint64_t hi = xh * yh + ((xl * yh) >> 32) + ((xh * yl) >> 32);
    return lo ^ hi;
#endif
}

/* THE SILHOUETTE: read once, after the flood has drained and the shadows settled.
   `lean` are rotated lanes, `straight` are raw lanes, `crossing` is the term that
   crosses another.  Constant cost, three multiplies, independent of len.        */
static inline uint64_t silhouette(uint64_t lean, uint64_t straight,
                                  uint64_t crossing, uint64_t len)
{
    uint64_t h = cross(lean ^ A3, straight ^ A4);
    h = cross(h ^ crossing ^ A5, len ^ A2);
    h ^= h >> 32; h *= A0; h ^= h >> 29;
    return h;
}

/* REGIME 1: the pile is too small to raise the flood at all.  No descent, no
   chain, no wires -- the marks are pressed straight onto the shadow-stone.      */
static uint64_t pressed_flat(const unsigned char *p, size_t len)
{
    uint64_t A = 0, B = 0, C = 0;
    if (len >= 16)      { A = rd8(p); B = rd8(p + 8);       C = rd8(p + len - 8) ^ rotl64(A, 27); }
    else if (len >= 8)  { A = rd8(p); B = rd8(p + len - 8); C = rotl64(A, 32) ^ B; }
    else if (len >= 4)  { A = rd4(p); B = rd4(p + len - 4); C = (A << 32) ^ B; }
    else if (len > 0)   { A = p[0];   B = p[len >> 1];      C = p[len - 1]; }
    return silhouette(A, B, C, (uint64_t)len);
}

#if !CHAINED_PRISONERS
/* a pit with no chained prisoners: each one sharpens by add-rotate-xor instead
   of by shift register.  Same ring, same wires, same single read.               */
static inline uint64_t sharpen(uint64_t s, uint64_t m){
    s += m;
    s ^= rotl64(s, 23) ^ rotl64(s, 47);
    return s;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;

    /* count the pile against the depth of the pit before descending */
    if (len < 24) return pressed_flat(p, len);

    /* the prisoners, chained in a ring; each holds an angle, not a mark */
    uint64_t a = (uint32_t)(0x9E3779B9u ^ (uint32_t)len);
    uint64_t b = (uint32_t)(0x85EBCA77u ^ (uint32_t)(len >> 32));
    uint64_t c = (uint32_t)0xC2B2AE3Du;

    /* the wires, anchored in the riverbed -- data never seeds them */
    uint64_t w0 = A0, w1 = A1, w2 = A2;

    const unsigned char *last = p + len - 24;   /* the final full pass */

    for (;;) {
        uint64_t m0 = rd8(p), m1 = rd8(p + 8), m2 = rd8(p + 16);

        /* Each mark is fed singly to the nearest prisoner.  What travels down the
           chain is the angle -- the previous pass's state -- never the mark; and
           the last one's angle carries backward to the first, closing the ring,
           so a mark dropped at the very start still trembles in the last hand.  */
#if CHAINED_PRISONERS
        uint64_t na = _mm_crc32_u64(a, m0 ^ c);
        uint64_t nb = _mm_crc32_u64(b, m1 ^ a);
        uint64_t nc = _mm_crc32_u64(c, m2 ^ b);
#else
        uint64_t na = sharpen(a, m0 ^ c);
        uint64_t nb = sharpen(b, m1 ^ a);
        uint64_t nc = sharpen(c, m2 ^ b);
#endif
        a = na; b = nb; c = nc;

        /* The flood rises and bends each anchored wire by exactly the angle the
           last chained prisoner ground into the pillar on this pass.            */
        uint64_t angle = (nc << 32) ^ (nb << 16) ^ na;
        w0 = rotl64(w0 ^ angle, (unsigned)(nc & 63u));   /* bent BY the angle */
        w1 = rotl64(w1 + angle, 23);
        w2 ^= rotl64(angle, 47);

        if (p == last) break;                 /* the water recedes only here */
        p += 24;
        if (p > last) p = last;               /* the last marks overlap into one full pass */
    }

    /* climb to the rim and read the silhouette, once */
    return silhouette(w0 ^ rotl64(w1, 19),
                      w2 ^ rotl64(w0, 41) ^ (a << 32) ^ b,
                      (c << 32) ^ (a ^ b ^ c) ^ rotl64(w1 ^ w2, 7),
                      (uint64_t)len);
}
```

Why this should be fast, stated before measuring: FNV-1a's cost is a serial `xor`+`imul` chain, 4 cycles per **byte** (≈0.25 B/cycle). This kernel's recurrence is three *independent* `crc32q` chains of 3-cycle latency each, so one retires per cycle and the per-pass critical path is ~3 cycles for **24 bytes**; the cross-lane coupling is lagged one pass and so sits off the critical path, feeding only the early-ready data operand. Front-end uop count (~22 per pass) puts the realistic figure near 4 B/cycle, i.e. ~16× FNV-1a's dependency-bound rate, falling back toward memory bandwidth for buffers past L2.

Why avalanche should still be ~0.5: the absorb is deliberately weak and mostly linear, so a one-bit input flip produces a *fixed, nonzero* 96-bit lane delta plus a wire delta. `silhouette` is a bijective-ish strong mix (folded 128-bit multiply, twice, then a shift-multiply-shift), so a nonzero pre-read delta becomes a pseudorandom 64-bit output delta — ~32 of 64 output bits flip. The data-dependent rotation `rotl64(w0 ^ angle, nc & 63)` is the one nonlinear element in the loop and exists to stop the whole absorb being an affine function of the input.

## PREDICTION

PREDICTION: speedup_vs_dp = 12.0

Secondary predictions, stated now so they can be falsified: avalanche ≈ 0.50 bit-flip probability (score ≥ 0.98 on a normalised scale); the figure above refers to buffers of a few hundred bytes upward, and on very short keys (len < 24, the `pressed_flat` path) I expect only ~2–3×, since FNV-1a is itself cheap there.

## MEASUREMENT

**Not measured.** No tools are available in this session — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all listed but none can be invoked here, so the prediction above stands untested by me. I am reporting that plainly rather than reporting a number I did not obtain. The pipeline will compile and measure the block above; the honest status of the prediction is *open*.

What a measurement would have to show for the mechanism to be judged working, and what would falsify it:

- **throughput**: ≥ 8× FNV-1a on ≥256-byte buffers. Below ~4× would mean the three CRC chains are not in fact overlapping (CRC port contention, or the variable rotate serialising with the lane ring) and the seed-3 bet — weak absorb, one strong read — is paying for itself in setup without earning it back.
- **avalanche**: ≥ 0.49. A score materially below that would falsify the central claim that a *single* read-once projection can substitute for per-byte mixing rounds, and would vindicate the assumption I broke.
- **the short-key regime**: if `pressed_flat` is *slower* than FNV-1a for len < 24, the 24-byte threshold is wrong and should rise, not the long path change.

I made zero refinement passes (budget was four) because I could not measure; refining blind would be fabrication.

## VERDICT

The native's account maps onto a real, coherent machine with nothing left over, and the mapping is literal at every point: marks are words, prisoners are shift-register lanes in a ring, the angle is state passed instead of data, the anchored wires are secret-seeded accumulators bent by a data-dependent rotation whose amount *is* the ground angle, the flood's refusal to recede is the ban on sampling state mid-stream, and the silhouette is one constant-cost multiplicative projection. The native's own prohibitions did real work rather than being decoration — "a token pulled while the water is still up is worthless" is precisely the statement that a linear absorb must never be returned raw, and "I never ask the river where its source lies" is precisely why there is no OpenMP here.

Honest assessment of where this is weaker than the known way:

1. **It is not cryptographic and the absorb is nearly linear.** The benchmark scores throughput and avalanche, and a linear absorb behind a strong finaliser scores well on *both* while remaining vulnerable to crafted differentials — a CRC32C lane is a GF(2)-linear map, so an adversary can construct multi-block collisions in the lane state that the finaliser then maps to identical output. The single data-dependent rotation per pass raises the cost of that attack but does not close it. FNV-1a is no better here, but xxHash/wyhash are, and a reader should not take a good avalanche score as a claim of collision resistance.
2. **The hash value is platform-dependent.** The SSE4.2 and portable rings produce different values for the same input. That is acceptable for an in-process hash table and unacceptable for anything persisted or sent over a wire. I am flagging it rather than hiding it.
3. **The one risk my own reasoning named — overhead on small inputs — is guarded**, not merely noted: `len < 24` takes `pressed_flat`, which touches no lanes, no wires and no loop, and which cannot be slower than three FNV-1a iterations. Nothing in the kernel ships with an unaddressed stated risk.
4. **The claimed speedup is bounded by memory, not by this design.** Past L2 the kernel will flatten to bandwidth, so the ratio against FNV-1a will *grow* with size only until that point and then stop. A measurement that shows 12× at 1 KB and 12× at 64 MB would be evidence the benchmark is cache-resident, not evidence of a better kernel.

The interesting result, if the measurement holds, is the negative one the native insisted on: the per-byte mixing round that FNV-1a treats as mandatory is almost entirely wasted work. One read at the end, done properly, is enough — and the loop that feeds it can be weak, multiply-free, and three-wide.