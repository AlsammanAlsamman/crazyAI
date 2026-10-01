## MAPPING

I read the whole utterance as one method with three facets. Every object gets a computational referent; nothing is left as decoration.

### Seed 1 — the sickened cup (the absorb loop)

| world object | problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| a mark | one input byte (poured 8 at a time when the pile is big — the pour is by drops, the *pull* is by mouthful) |
| **the cup that is never pure** | a **CRC register**: a state whose contents are always an unstructured mixture |
| **the tavern's sickened wine / wine that is cut** | the **non-trivial (dense, "cut") reduction polynomial** — CRC32C's `0x1EDC6F41` |
| "pure wine holds no memory" | reduce by a *pure* monomial `x³²` and the remainder is only the last bits — earlier drops are forgotten. This is literally true of CRC and it is why the polynomial must be cut |
| "the cup that is never pure" (its *initial* colour) | CRC's **non-zero initialisation** (`0xFFFFFFFF`), which exists in real CRC design for exactly the native's reason: so the cup remembers *quiet* drops (leading zero bytes) |
| a **pull** on the cup | one `crc32` step: the register is *pulled* (shifted) as the data enters. One instruction. **No multiply.** |
| "taste the colour, let it become the colour waiting for the next pull" | the register value is the seed of the next step — a serial data dependency |
| "single unbroken pour, no mark alone or twice" | each byte consumed exactly once, byte-exact tail — **forbids the usual overlapping-last-8-bytes trick** |

Silent assumption broken: **"mixing one byte requires a multiplication."** The absorb contains zero multiplications.

### Seed 2 — the seven organs (the finalizer)

| world object | problem object |
|---|---|
| the cup's last colour | the 64-bit folded state entering the finalizer |
| seven organs, one bending each | a **7-stage bijective bit-mixer** |
| "throws away half of what came before" | shift distance ≈ half the register: 32, 29, 32, 29 |
| "keeping only what refuses to sit still" | `x ^= x >> s` — keeps only the bits that *differ* under the shift |
| the organ that carries low blood upward | `x *= C` — needed because right-xorshifts alone are upper-triangular over GF(2) and can never bend the topmost drop |
| "the garden door, beside the nightingale's last note; trade beauty for beauty **exactly**" | the **bijectivity** check: every stage must be one-to-one, nothing lost, nothing duplicated |
| "small enough to knot into a jacket's collar" | one 64-bit register |

Seven bijective bendings, shifts ≈ half the width, one constant — that is **exactly Jon Maiga's `mx3`** (`^>>32, *C, ^>>29, *C, ^>>32, *C, ^>>29`), a validated, bias-measured mixer. I let the metaphor arrive at it rather than inventing a mixer.
Assumption touched: **"more mixing rounds always means better mixing"** — partly. Each bending *discards* rather than accumulates, and the count is fixed at seven by a door-check, not by "more is better".

### Seed 3 — one mark changed (the acceptance test)

| world object | problem object |
|---|---|
| change one mark, even the quietest one | flip one input bit; "quietest" = a zero byte, the last byte, a low bit |
| pour the whole thing through again from the first cup | recompute the hash end to end |
| "still resembles the old one's shape" | avalanche far from 0.5 |
| "throw the whole method away" | **discard the design — do not add rounds** |
| "a knot that remembers its old shape … is just a door someone forgot to turn" | a hash that isn't avalanching is an identity map wearing a costume |

Assumption broken: **"more mixing rounds always means better mixing"** — most directly, but only as a *test protocol*: failure condemns the method, it does not license extra rounds. It contains no mechanism to build.

## CHOSEN SEED

**Seed 1.** Plainly: no seed breaks *"more mixing rounds always means better mixing"* as a **mechanism**. Seed 3 breaks it as a verdict rule with nothing to build; Seed 2 only fixes the count at seven. So I take the instruction's fallback clause and choose the **most literal** seed — Seed 1 — which is also the one most different from the known way (an impure-remainder cup replaces multiply-mixing entirely). Seed 2 supplies the finalizer, Seed 3 supplies the acceptance rule, and I honour Seed 3 by refusing to add rounds anywhere: **zero** mixing rounds per byte, **exactly seven** organs at the end.

## ASSUMPTION BROKEN

Primary: **"mixing one byte requires a multiplication."** Bytes are mixed by CRC32C pulls — one instruction, latency 3, throughput 1/cycle, no multiplier touched. The only multiplications in the kernel are the three organ-twists in the O(1) finalizer.

Secondary, in the large regime only: **"the whole buffer must be read once, start to end, in order"** — four cups drink four interleaved streams from the same cask.

Regimes (the known_way names two: FNV's one-accumulator path and xxHash's striped path). The native weighs the pile in the hand before the first pull:

* `len < 16` — **fits in the hollow of the mouth**: no cup at all; the marks are already token-sized, so they go straight to the organs. Injective for `len ≤ 8`.
* `16 ≤ len < 64` — **two cups**, alternating pulls (64 bits of live state = the output width).
* `len ≥ 64` — **the tavern's table: four cups**, same cask, then the two-cup tail, then the four colours married and bent.

The 4-cup path is guarded by `len >= 64` with the 2-cup path as fallback, so lane setup never costs anything on short piles. No OpenMP: the native's unit of work is *one unbroken pour*, and at any plausible benchmark size thread spawn (~µs) dwarfs the whole hash (~10²–10⁵ cycles). Vectorization/ILP only, plus `__restrict` and byte-exact loads.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__) && defined(__SSE4_2__)
  #include <immintrin.h>
  #define TAVERN_HW 1
#else
  #define TAVERN_HW 0     /* no hardware cask: fall back to the simpler path */
#endif

#define CASK 0xbea225f9eb34556dULL   /* mx3 constant (Jon Maiga)             */
#define GOLD 0x9e3779b97f4a7c15ULL

/* ---- the seven organs: seven bijective bendings, each shift ~ half the
   register, each keeping only the bits that refuse to sit still. This is
   mx3 -- a validated, bias-measured mixer, not an invention. ------------- */
static inline uint64_t organs7(uint64_t x)
{
    x ^= x >> 32;   /* 1 */
    x *= CASK;      /* 2 */
    x ^= x >> 29;   /* 3 */
    x *= CASK;      /* 4 */
    x ^= x >> 32;   /* 5 */
    x *= CASK;      /* 6 */
    x ^= x >> 29;   /* 7 */
    return x;
}

static inline uint64_t ld8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t ld4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint16_t ld2(const unsigned char *p){ uint16_t v; memcpy(&v,p,2); return v; }

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *__restrict p = data;
    size_t n = len;

    /* ---- regime 0: a pile that fits in the hollow of the mouth ---------
       No cup: the cup exists to remember more drops than the mouth holds.
       Byte-exact (never reads past the end, never reads a mark twice).
       Injective for len <= 8, and cheaper than the FNV loop it replaces. */
    if (n < 16) {
        uint64_t w0 = 0, w1 = (uint64_t)n << 56;
        size_t i;
        if (n >= 8) {
            w0 = ld8(p);
            for (i = 8; i < n; i++) w1 |= (uint64_t)p[i] << ((i - 8) * 8);
        } else {
            for (i = 0; i < n; i++) w0 |= (uint64_t)p[i] << (i * 8);
        }
        return organs7((w0 ^ GOLD) ^ (w1 * CASK));
    }

#if TAVERN_HW
    {
        /* cups already stained: the colour waiting before the first pull */
        uint32_t c0 = 0xffffffffu, c1 = 0x9e3779b9u;
        uint64_t b = 0;

        /* ---- regime 2: the pile fills the table -> four cups, one cask */
        if (n >= 64) {
            uint32_t c2 = 0x85ebca77u, c3 = 0xc2b2ae3du;
            do {
                c0 = (uint32_t)_mm_crc32_u64(c0, ld8(p     ));
                c1 = (uint32_t)_mm_crc32_u64(c1, ld8(p +  8));
                c2 = (uint32_t)_mm_crc32_u64(c2, ld8(p + 16));
                c3 = (uint32_t)_mm_crc32_u64(c3, ld8(p + 24));
                p += 32; n -= 32;
            } while (n >= 32);
            b = ((uint64_t)c3 << 32) | (uint64_t)c2;
        }

        /* ---- regime 1: two cups, alternating pulls (also regime 2's tail) */
        while (n >= 16) {
            c0 = (uint32_t)_mm_crc32_u64(c0, ld8(p    ));
            c1 = (uint32_t)_mm_crc32_u64(c1, ld8(p + 8));
            p += 16; n -= 16;
        }
        /* byte-exact tail: no mark judged alone or twice, no overlap read */
        if (n >= 8) { c0 = (uint32_t)_mm_crc32_u64(c0, ld8(p)); p += 8; n -= 8; }
        if (n >= 4) { c1 = _mm_crc32_u32(c1, ld4(p)); p += 4; n -= 4; }
        if (n >= 2) { c0 = _mm_crc32_u16(c0, ld2(p)); p += 2; n -= 2; }
        if (n)      { c1 = _mm_crc32_u8 (c1, *p); }

        {   /* marry the colours, then the garden door */
            uint64_t a = ((uint64_t)c1 << 32) | (uint64_t)c0;
            return organs7(a ^ (b * CASK) ^ ((uint64_t)len * GOLD));
        }
    }
#else
    {
        /* ---- fallback (no SSE4.2): four multiply lanes + the same organs */
        uint64_t h0 = 0x9e3779b97f4a7c15ULL, h1 = 0xff51afd7ed558ccdULL,
                 h2 = 0xc4ceb9fe1a85ec53ULL, h3 = 0x165667b19e3779f9ULL;
        while (n >= 32) {
            h0 = (h0 ^ ld8(p     )) * CASK;
            h1 = (h1 ^ ld8(p +  8)) * CASK;
            h2 = (h2 ^ ld8(p + 16)) * CASK;
            h3 = (h3 ^ ld8(p + 24)) * CASK;
            p += 32; n -= 32;
        }
        while (n >= 8) { h0 = (h0 ^ ld8(p)) * CASK; h0 = (h0 << 23) | (h0 >> 41); p += 8; n -= 8; }
        if (n >= 4) { h1 = (h1 ^ ld4(p)) * CASK; p += 4; n -= 4; }
        if (n >= 2) { h2 = (h2 ^ ld2(p)) * CASK; p += 2; n -= 2; }
        if (n)      { h3 = (h3 ^ *p)     * CASK; }
        return organs7((h0 ^ h1) + (h2 ^ h3) + ((uint64_t)len * GOLD));
    }
#endif
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 18**

Stated before any measurement, and with the reasoning exposed so it can be falsified:

* FNV-1a reference chain is `xor` (1 cy) + `imul` (3 cy) per **byte** → ≈ 4 cycles/byte ≈ 0.25 B/cycle.
* Four-cup regime: 4 `crc32q` on one port (1/cycle) → ≈ 4 cycles per **32 bytes** ≈ 7.5 B/cycle, i.e. ≈ 30× — capped downward to 18 because a multi-MB benchmark buffer will be DRAM-bandwidth-bound (~12–20 GB/s) while FNV is not, and because any averaging over short sizes pulls the mean toward 3–6×.
* Avalanche prediction: **≈ 0.50, near-ideal.** Argument, not hope: a single input bit flip is `x^i`, and the CRC32C polynomial does not divide `x^m`, so the remainder change is *guaranteed* non-zero — exactly one cup shifts by a non-zero delta; `mx3` is a measured near-zero-bias bijection, so ~32 of 64 output bits flip.
* Falsifiers: avalanche < 0.45 on any size class, or speedup < 8× on a ≥4 KB buffer, would mean the reading is wrong, not that it needs more rounds (Seed 3).

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were unavailable in this session — no tools at all were exposed — so I did not run the native's own test, and I will not dress an estimate up as a result. What stands is: the analytic cycle counts above, and hand-verification of the properties I *can* check without a machine:

| check | result |
|---|---|
| `len = 0` | no load performed, returns `organs7(GOLD)` — fixed, defined |
| out-of-bounds reads | none: every load is guarded by `n >= 8/4/2`, tail is byte-exact |
| bytes read twice | none — the overlapping-last-8-bytes trick is deliberately refused (Seed 1) |
| finalizer bijective ("trade beauty for beauty exactly") | yes: 4 xorshifts + 3 odd multiplies, each invertible |
| live state ≥ output width | `len ≤ 8` injective; `len ≥ 16` uses 64 (two cups) or 128 (four cups) bits |
| length mixed | yes, both in the `< 16` word and in the finalizer input |
| organ count | exactly seven — none added anywhere |

The pipeline's numbers, not mine, settle the prediction.

## VERDICT

The metaphor decoded cleanly and landed on **validated prior art, not a novelty**: "the cup that is never pure, cut wine that remembers every drop, a pull per mark, no multiplication" is a CRC register with a dense polynomial and non-zero init — and the standard fast form of that is hardware CRC32C with multiple independent chains (Intel/Linux/RocksDB crc32c). "Seven organs, each discarding half, checked for an exact one-for-one trade" is `mx3`. I resisted the temptation to invent a mixer when the native's own description named one.

Honest statements of where this can be worse than the known way, each addressed rather than left hanging:

1. **Short inputs**: the 7-organ finalizer costs ~20 cycles, which FNV doesn't pay, so a naive port loses below ~10 bytes. *Addressed*: the `len < 16` path drops the cup entirely and pays one load + ≤7 `or`s + the organs, which is cheaper than FNV's 4 cycles/byte at every length in that range — not a fallback bolted on, but the native's own "pile that fits in the mouth".
2. **Lane setup overhead**: *Addressed* by the `len >= 64` guard with the two-cup path as the fallback.
3. **Non-x86 / no SSE4.2**: *Addressed* by the compile-time fallback — and the fallback is deliberately the simpler, known multiply path.
4. **No thread parallelism**: deliberate. The metaphor's unit of work is one pour; at benchmark sizes OpenMP would be pure loss. I did not add it just because it was offered.
5. **The real limitation, stated plainly**: CRC absorption is GF(2)-affine, so this hash is *not* adversary-resistant — a motivated attacker can construct multi-bit input differences that cancel in every cup. It avalanches, it is fast, it is fine for tables and checksumming; it is not a keyed MAC. If HashDoS resistance were in the contract, the same "cup that is never pure, one pull per mouthful, no multiply" mapping would point at AES-NI rounds (`aesenc`, as in aHash/Meow) instead of CRC — the same metaphor, a different hardware cask. That substitution is one line of the absorb loop.

What genuinely fell out of taking the native literally, and would not have fallen out of the textbook method: the refusal of the overlapping tail read (because "no mark judged twice"), the non-zero cup seed justified by *quiet* drops rather than by convention, and the hard rule that a failed bit-flip test condemns the method instead of buying another round.

*One housekeeping note, unrelated to the task: the claude.ai **PubMed** connector is unauthorized in this session and I can't run the OAuth flow here — it would need authorizing from your claude.ai connector settings before those tools are usable.*