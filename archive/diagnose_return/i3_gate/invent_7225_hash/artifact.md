## MAPPING

**SEED 1 — the sickened cup, one unbroken pour**

| World object | Problem object |
|---|---|
| a *mark* in the pile | one input byte (poured 8 at a time when the pile is deep) |
| the *cup* | the running hash state |
| the cup is **never pure** / the wine is **cut** | the CRC‑32C generator polynomial `0x11EDC6F41` has **18 terms — an even count — so it is divisible by (x+1): provably reducible.** A "pure" (irreducible) generator makes a clean field; the cut one keeps the (x+1) factor, which is exactly why "it remembers every drop added to it" — the register permanently carries the parity of everything poured in. The cup is also never pure in a second, literal sense: it holds *sediment* from the previous pull. |
| a *pull* on the cup | one CRC step: shift the register, feed back through the cut polynomial. **Carry‑less — no integer multiply anywhere.** |
| *tasting the cup's colour* | reading the 32‑bit register that falls out of the pull |
| *that colour becomes the colour waiting for the next pull* | the pull's output is the next pull's initial register, and the colour before it is the sediment XORed into the next mark |
| *single unbroken pour, no mark judged alone or twice* | one strictly serial dependency chain; every byte enters exactly one pull |

Assumption broken: **"mixing one byte requires a multiplication."** The whole pour is GF(2) shift‑and‑xor. (It also dents "the state is a single accumulator, one value": the cup is one vessel holding two coupled colours, live and sediment — they cannot be split into two independent pours.)

**SEED 2 — seven organs**

| World object | Problem object |
|---|---|
| *the body carries the sickness until every organ compensates* | the finalizer, applied once after the pour, not per byte |
| *seven bendings, once per organ* | exactly seven avalanche stages — a fixed, tuned count |
| *each bending throws away half of what came before* | `x ^= x >> 32/29` discards the shifted‑out half; `x *= C` discards the high 64 bits of the 128‑bit product. Every stage literally loses half of what it received. |
| *keeping only what refuses to sit still* | xorshift keeps only bit positions that **differ** from their shifted partner |
| *the garden door / the nightingale's last note / trade beauty for beauty exactly* | stop when output bits flip with probability exactly ½; bend again only if not |

Assumption broken: **"more mixing rounds always means better mixing."** The count is *seven*, terminated by a measured condition, and each round is deliberately lossy rather than additive.

**SEED 3 — change one mark, pour again**

| World object | Problem object |
|---|---|
| *change one mark, even the quietest one* | flip one bit anywhere, including in the tail/short input |
| *pour the whole thing through again from the first cup* | full re‑evaluation, no incremental shortcut |
| *if the new token resembles the old, throw the method away* | the avalanche acceptance test; resemblance = disqualification, not a warning |

Assumption broken: none of the five directly — this is the *verification protocol*, not a mixing rule.

## CHOSEN SEED

**SEED 1 as the core, with SEED 2 shipped intact as its mandated closure.**

Plainly: the preference rule points at SEED 2, which is the one that breaks *"more mixing rounds always means better mixing."* But SEED 2 **consumes no input** — seven bendings of a number cannot hash a buffer. So I take SEED 1 as the core (it is the most literal *and* the furthest from the known way: carry‑less LFSR instead of multiply‑accumulate) and implement SEED 2 exactly as written, seven organs, as the closing stage. The shipped artifact therefore breaks both assumptions; I am not smuggling SEED 2 out.

## ASSUMPTION BROKEN

Primary: **mixing one byte requires a multiplication.** The pour contains zero multiplies — only carry‑less polynomial division by a deliberately *reducible* generator. Secondary: **more mixing rounds always means better mixing** — seven, fixed, each lossy.

Per step 4, the mechanism is allowed to land on a validated technique rather than a novel one, and it does, twice: the pour is hardware **CRC‑32C** (`crc32q`, SSE4.2), the closure is **mx3**, which is *exactly* seven stages of xorshift/multiply.

Two regimes, both recognised at runtime by the native's own terms:
- *"Does this tavern's cup come already sickened, or must I sicken it myself?"* → `__SSE4_2__` / `__builtin_cpu_supports` dispatch to the hardware pour, else to a hand‑built 256‑entry sediment table using the same cut polynomial. Same mechanism either way.
- *"Is the pile deep, or does it run thin?"* → 32‑byte stride, then 8‑byte, then drop‑by‑drop. No multiply‑based fast path is smuggled in for small sizes.

Stated risk I must guard (step 4): the single unbroken pour is **latency‑bound**, so it wins only where per‑call overhead is amortised and the CPU has `crc32`. Both are guarded: the ISA by runtime dispatch with a working fallback, the size by the tiered pour with no fixed setup cost (the table is built lazily and only on the fallback path).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

#if defined(__x86_64__)
  #define CUP_X86 1
  #include <immintrin.h>
  #if defined(__SSE4_2__)
    #define CUP_TGT
  #else
    #define CUP_TGT __attribute__((target("sse4.2")))
  #endif
#endif

/* ---------------- the seven organs ----------------
   The cup's last colour is carried through seven bendings. Each bending is
   invertible and each throws away half of what it received: a right xorshift
   discards the half shifted out and keeps only the bits that refuse to sit
   still (those differing from their shifted partner); a 64x64 multiply
   discards the high half of the 128-bit product. Seven, not more: the count
   was closed at the garden door, where output bits flip at exactly one half. */
static inline uint64_t organs(uint64_t x) {
    const uint64_t C = 0xbea225f9eb34556dULL;
    x ^= x >> 32;   /* organ 1 */
    x *= C;         /* organ 2 */
    x ^= x >> 29;   /* organ 3 */
    x *= C;         /* organ 4 */
    x ^= x >> 32;   /* organ 5 */
    x *= C;         /* organ 6 */
    x ^= x >> 29;   /* organ 7 */
    return x;
}

/* ------------- sickening the cup by hand -------------
   Reflected CRC-32C generator 0x82F63B78. Its unreflected form 0x11EDC6F41
   has 18 terms - an even count - hence it is divisible by (x+1): the wine is
   cut. That impurity is precisely what makes the cup remember every drop. */
static uint32_t cup_tab[256];
static int cup_ready = 0;

static void cup_sicken(void) {
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0x82F63B78u & (uint32_t)(-(int32_t)(c & 1u)));
        cup_tab[i] = c;
    }
    cup_ready = 1;
}

/* Software pour: same cup, same cut wine, poured drop by drop. */
static uint64_t pour_soft(const unsigned char *data, size_t len,
                          uint32_t a, uint32_t b) {
    const unsigned char *restrict p = data;
    if (!cup_ready) cup_sicken();
    for (size_t i = 0; i < len; i++) {
        uint32_t t = (uint32_t)p[i] ^ (a & 0xFFu);          /* mark meets sediment */
        uint32_t c = (b >> 8) ^ cup_tab[(b ^ t) & 0xFFu];   /* one pull            */
        a = b; b = c;                                       /* colour becomes seed */
    }
    return ((uint64_t)a << 32) | (uint64_t)b;
}

#if CUP_X86
/* Hardware pour. One unbroken serial chain: b_{n+1} = crc32c(b_n, w_n ^ b_{n-1}).
   The only loop-carried latency is the crc32 itself (3 cycles / 8 bytes); the
   sediment XOR and the loads hang off the chain, not in it. No multiply. */
CUP_TGT
static uint64_t pour_hw(const unsigned char *data, size_t len,
                        uint32_t a, uint32_t b) {
    const unsigned char *restrict p = data;
    size_t i = 0;

    #define PULL(W) do {                                                  \
        uint64_t s_ = (uint64_t)a; s_ ^= s_ << 32;   /* sediment colours */ \
        uint32_t c_ = (uint32_t)_mm_crc32_u64((uint64_t)b, (W) ^ s_);     \
        a = b; b = c_;                                                    \
    } while (0)

    /* the pile is deep: pour in eights, four pulls to a breath */
    for (; i + 32 <= len; i += 32) {
        uint64_t w0, w1, w2, w3;
        __builtin_memcpy(&w0, p + i,      8);
        __builtin_memcpy(&w1, p + i +  8, 8);
        __builtin_memcpy(&w2, p + i + 16, 8);
        __builtin_memcpy(&w3, p + i + 24, 8);
        PULL(w0); PULL(w1); PULL(w2); PULL(w3);
    }
    /* the pile thins */
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        __builtin_memcpy(&w, p + i, 8);
        PULL(w);
    }
    /* the last drops, one at a time - every mark poured, none twice */
    for (; i < len; i++) {
        uint32_t t = (uint32_t)p[i] ^ (a & 0xFFu);
        uint32_t c = (uint32_t)_mm_crc32_u8((unsigned int)b, (unsigned char)t);
        a = b; b = c;
    }
    #undef PULL
    return ((uint64_t)a << 32) | (uint64_t)b;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the cup is rinsed with the measure of the pile */
    uint32_t a = 0x9E3779B9u ^ (uint32_t)len;
    uint32_t b = 0x85EBCA6Bu ^ (uint32_t)(len >> 32)
                             ^ (uint32_t)(len << 7) ^ (uint32_t)(len >> 3);
    uint64_t cup;

#if defined(__SSE4_2__)
    cup = pour_hw(data, len, a, b);
#elif CUP_X86
    cup = __builtin_cpu_supports("sse4.2") ? pour_hw(data, len, a, b)
                                           : pour_soft(data, len, a, b);
#else
    cup = pour_soft(data, len, a, b);
#endif

    return organs(cup);   /* carry the last colour through seven organs */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 7.5**

Reasoning stated before measuring: FNV‑1a's loop is `xor` + `imul` on one accumulator, a 4‑cycle loop‑carried chain per byte ≈ 0.25 B/cycle. The pour's loop‑carried chain is one `crc32 r64,r64` per 8 bytes, latency 3 ≈ 2.67 B/cycle — about 10.7× at large sizes, derated to ~7.5 by call overhead and byte‑wise tails at small sizes.

Avalanche: I predict a near‑perfect score (≈0.500 flip fraction). The pour is GF(2)‑affine, so a single input bit flip produces a fixed nonzero state delta — nonzero because a weight‑1 error polynomial can never be a multiple of a degree‑32 generator whose minimum‑weight multiple over 64 bits is ≥ 4 — and the seven organs carry any nonzero 64‑bit delta to half the output bits.

Honest caveat I expect the numbers to confirm: linearity of the pour means this is strong on *avalanche* and *speed* but is **not** collision‑resistant against an adversary who can solve a linear system. That is a property of the native's mechanism, not a bug I introduced, and I am not patching it out.

## MEASUREMENT

No tools were available in this session, so I report this plainly rather than fabricating numbers: **the artifact is unmeasured by me.** The prediction above is stated ahead of the pipeline's run, which is where the real numbers will come from. What I can verify by hand, and did:

- CRC‑32C generator term count = 18 (even) ⇒ divisible by (x+1) ⇒ the cup is provably impure, as the native insists. Checked by popcount of `0x1EDC6F41` (17) plus the implicit `x^32` term.
- The loop‑carried dependency is exactly one `crc32` per 8 bytes: `a_{n} = b_{n-1}`, so the data operand `w ^ a` is ready three cycles before it is needed and never lengthens the chain.
- Single‑bit deltas cannot be annihilated: cancellation between the register path and the sediment path two pulls later would require the generator to divide `x^{32}(x^{96}+1) = x^{32}((x+1)(x^2+x+1))^{32}`, which it cannot, since its degree‑31 irreducible factor is neither `x+1` nor `x^2+x+1`.
- Organ count is seven, and they are the mx3 sequence exactly — a validated finalizer arrived at, not invented.

## VERDICT

The core of this kernel **is** the native's mechanism, not a repainted FNV: there is no multiplication anywhere in the byte‑consuming loop. Mixing is carry‑less division by a deliberately reducible polynomial, the state is one vessel carrying a live colour and its sediment, and the pour is a single unbroken serial chain in which every byte is used exactly once.

What the metaphor costs me, stated openly: *"a single unbroken pour"* forbids interleaving three independent CRC chains, which is the standard way to reach ~8 B/cycle on this instruction. I am leaving roughly a 3× speedup on the table because the native was explicit that the pile passes as one pour, and step 3 says to translate rather than substitute. I flag this as the single largest measurable concession to fidelity.

Every risk my own reasoning named is guarded rather than hoped away: the ISA dependency by runtime dispatch to a table‑driven pour of the identical polynomial; the small‑size regime by a tiered 32/8/1 pour with no setup cost on the fast path; the thread‑parallelism temptation declined outright, since a latency‑bound serial chain is the metaphor's own unit of work and cannot be split without becoming several pours.

SEED 3 is the reason I would throw this away rather than defend it: if the pipeline reports an avalanche score materially below ~0.49, the knot remembers its old shape after one thread was pulled, and the method goes in the fire regardless of how fast it ran.