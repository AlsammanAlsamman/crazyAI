No tools are available in this session, so every number below that is labelled a prediction is a prediction, and the MEASUREMENT section says plainly that nothing was measured here.

---

## MAPPING

### SEED 1 — the sickened cup

| World object | Problem object |
|---|---|
| the pile of marks | the byte buffer `data[0..len)` |
| one mark | one byte `data[i]` |
| the cup | the running hash state |
| **"never pure" / "the tavern's sickened wine"** | the state update must be **cut** — built from *incompatible algebras* (GF(2) xor/rotate **and** Z/2⁶⁴ addition-with-carry) |
| "pure wine holds no memory" | a single-algebra update (xor-only, or add-only) is order-blind / self-cancelling — no diffusion |
| "wine that is cut remembers every drop" | carries + rotation make the state order-sensitive and position-sensitive |
| a pull on the cup | `c = rotl64(c + b, R)` |
| "taste the colour, let it become the colour waiting for the next pull" | `c_{i+1} = f(c_i, b_i)` — serial feedback |
| "no mark judged alone or twice" | each byte read exactly once |

**Breaks:** assumption 3, *"mixing one byte requires a multiplication."* The native's requirement is **impurity**, not multiplication: any two mutually non-distributive operations will do, and add+rotate costs 2 cycles of latency where xor+multiply costs 4. It **affirms** assumptions 1, 2 and 4 — this seed *is* essentially the known way, minus the multiply.

### SEED 2 — the seven organs

| World object | Problem object |
|---|---|
| the cup's last colour | the state arriving at finalization |
| the body | the finalizer |
| **seven organs, in turn** | **exactly 7** finalization rounds (not the textbook 2–3 of `fmix64`) |
| "each organ **bends**" | multiply by a distinct odd 64-bit constant (propagates low bits → high bits) |
| "**throws away half** of what came before" | `>> ~32` — literally discard half the width |
| "keeping only what **refuses to sit still**" | `^` of the shifted copy with the original: the bits that *differ* survive (propagates high → low) |
| "until every organ **compensates**" | bend and discard alternate direction so nothing is one-way |
| the garden door | the avalanche measurement point |
| the nightingale's last note | the 50 % reference — half the output bits |
| "trade beauty for beauty **exactly**" | avalanche score = 0.50 |
| "if not, I bend again" | round count is a **measured** quantity, not a constant |
| "small enough to knot into a jacket's collar" | 64-bit output |

**Breaks:** assumption 5, *"more mixing rounds always means better mixing."* Two ways: (a) every round is described as **lossy** — "throws away half" — so rounds are not monotone accumulation; (b) the count is arbitrated at the garden door by comparison against an external standard, not by "add more to be safe." Secondarily it dents assumption 2: the state at finalization is a **body of seven organs**, not one accumulator.

### SEED 3 — the pulled thread

| World object | Problem object |
|---|---|
| change one mark, even the quietest one | flip one input bit, including in the last byte |
| pour the whole thing through again from the first cup | recompute the full hash |
| "resembles the old token's shape" | Hamming distance between old and new digest ≪ 32 |
| "a door someone forgot to turn" | an identity-ish map: a hash that leaks its input's shape |
| throw the whole method away | reject the design |

**Breaks:** none of the five. This is not a mechanism, it is the **acceptance test** — it is what makes SEED 2's garden-door check falsifiable. Stated plainly: SEED 3 breaks no listed assumption.

---

## CHOSEN SEED

**SEED 2** — the seven organs. It is the only one of the three that breaks the preferred assumption (*"more mixing rounds always means better mixing"*), and its mapping is the most literal: seven is a count, "throws away half" is a shift width of 32, "keeps what refuses to sit still" is an xor with the shifted copy, "trade beauty for beauty exactly" is the number 0.50. SEED 1 is retained as the cup that feeds the body (the native describes one method, and the organs need something to receive), but the design decision under test is the organs.

**Two regimes, recognized in-world.** The assumption list describes two: a pour that is read strictly in order through one cup (short piles), and a pour long enough that "every organ compensates" simultaneously (long piles). So the tavern keeper **hefts the pile before pouring**: under 64 marks it goes into one cup, one unbroken pour; at 64 or more the body divides it among seven cups, one organ per cup, marks dealt out in strict order so the pile is still read front to back exactly once. That `len < 64` test is the runtime regime detector, and the single-cup path is the fallback.

---

## ASSUMPTION BROKEN

*"More mixing rounds always means better mixing."*

The native replaces "more is safer" with **a fixed anatomy plus a measured door**. Seven organs is the body's full complement; each organ is *lossy by construction* (half thrown away); and the stopping rule is an external comparison, not a monotone belief. Consequence in code: the round count is an audited number, and at `len ≤ 16` the seven-organ chain costs more cycles than the entire known-way loop, so the door is reached with **three** organs there — licensed by the seed's own "bend again *if*" rather than "bend seven times regardless."

Secondary break (SEED 1, carried along): per-byte mixing uses **no multiplication** — `rotl64(c + b, R)`, 2-cycle chain instead of FNV's 4-cycle `xor; mul`.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* "The Cup and the Seven Organs"
 *
 *   the cup is never pure : state mixed in two algebras at once
 *                           (Z/2^64 addition-with-carry, and rotation)
 *                           -- no multiply is spent per mark.
 *   one pull per mark     : c = rotl64(c + byte, R), each byte once, in order.
 *   the tavern keeper     : hefts the pile first -- < 64 marks go to one cup,
 *     hefts the pile        >= 64 are dealt to the body's seven cups,
 *                           one mark to each organ in turn, still front to back.
 *   the seven organs      : z *= C_j   (the bending)
 *                           z ^= z>>s_j (throws away half; keeps only the bits
 *                                        that refuse to sit still)
 *   the garden door       : 3 organs answer a pile of 16 or fewer, 7 otherwise.
 */

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t z;
    int organs;

    if (len < 64u) {
        /* ---- small pile: one cup, one unbroken pour ---- */
        uint64_t c = 0x9e3779b97f4a7c15ULL ^ (uint64_t)len;
        size_t i;
        for (i = 0; i < len; i++)
            c = rotl64(c + (uint64_t)p[i], 11);
        z = c;
        organs = (len <= 16u) ? 3 : 7;
    } else {
        /* ---- large pile: seven cups, one per organ ----
           seven independent 2-cycle chains; the buffer is still walked
           strictly front to back, so every cache line is consumed whole. */
        uint64_t c0 = 0x9e3779b97f4a7c15ULL;
        uint64_t c1 = 0xbf58476d1ce4e5b9ULL;
        uint64_t c2 = 0x94d049bb133111ebULL;
        uint64_t c3 = 0x2545f4914f6cdd1dULL;
        uint64_t c4 = 0xd6e8feb86659fd93ULL;
        uint64_t c5 = 0xa3b195354a39b70dULL;
        uint64_t c6 = 0x1b03738712fad5c9ULL;
        uint64_t x, y;
        size_t i = 0, r;

        for (; i + 7 <= len; i += 7) {
            c0 = rotl64(c0 + (uint64_t)p[i + 0],  7);
            c1 = rotl64(c1 + (uint64_t)p[i + 1], 11);
            c2 = rotl64(c2 + (uint64_t)p[i + 2], 13);
            c3 = rotl64(c3 + (uint64_t)p[i + 3], 17);
            c4 = rotl64(c4 + (uint64_t)p[i + 4], 19);
            c5 = rotl64(c5 + (uint64_t)p[i + 5], 23);
            c6 = rotl64(c6 + (uint64_t)p[i + 6], 29);
        }
        r = len - i;                      /* 0..6 last drops */
        if (r > 0) c0 = rotl64(c0 + (uint64_t)p[i + 0],  7);
        if (r > 1) c1 = rotl64(c1 + (uint64_t)p[i + 1], 11);
        if (r > 2) c2 = rotl64(c2 + (uint64_t)p[i + 2], 13);
        if (r > 3) c3 = rotl64(c3 + (uint64_t)p[i + 3], 17);
        if (r > 4) c4 = rotl64(c4 + (uint64_t)p[i + 4], 19);
        if (r > 5) c5 = rotl64(c5 + (uint64_t)p[i + 5], 23);

        /* the seven colours are married; each cup enters exactly once under
           its own rotation, so the marriage is injective in every cup. */
        x = c0 ^ rotl64(c1,  9) ^ rotl64(c2, 18) ^ rotl64(c3, 27);
        y = c4 ^ rotl64(c5, 36) ^ rotl64(c6, 45) ^ (uint64_t)len;
        z = rotl64(x, 23) + y;
        organs = 7;
    }

    /* ---- the organs: bend, then throw away half ---- */
    z *= 0xff51afd7ed558ccdULL; z ^= z >> 33;   /* 1 */
    z *= 0xc4ceb9fe1a85ec53ULL; z ^= z >> 29;   /* 2 */
    z *= 0xbf58476d1ce4e5b9ULL; z ^= z >> 32;   /* 3 */
    if (organs == 3) return z;                  /* the door, reached early */
    z *= 0x94d049bb133111ebULL; z ^= z >> 31;   /* 4 */
    z *= 0xd6e8feb86659fd93ULL; z ^= z >> 30;   /* 5 */
    z *= 0xa3b195354a39b70dULL; z ^= z >> 27;   /* 6 */
    z *= 0x1b03738712fad5c9ULL; z ^= z >> 33;   /* 7 */
    return z;
}
```

Design notes I owe you, since none of this was measured:

- **No threads.** The metaphor's unit of work is *one pull of the cup* — one byte. That is far below the size at which handing work to another body pays. The seven organs are one body's instruction-level parallelism, not seven bodies. Per the brief's ordering, I used `restrict`, seven independent dependency chains, and strictly sequential cache-line consumption instead.
- **No SIMD.** Seven 64-bit lanes does not fit a 4-wide `ymm`, and rounding seven up to eight to make it fit would be replacing the native's number with a convenient one. Seven scalar GPR chains fit comfortably in the register file.
- **Where the nonlinearity lives.** The cup is nonlinear over GF(2) (carries); the organs are the diffuser. Every step in both is a bijection in the value being changed, so a single flipped input bit is *guaranteed* to reach the first organ as a nonzero difference — which is the precondition under which a multiply/xorshift chain gives ~50 % output flip.
- **Rejected during design (four unmeasured refinements):** (1) xorshift-only organs — rejected, right-shift-only composition is GF(2)-linear and output bit 63 would depend on input bit 63 alone, i.e. SEED 3's "door someone forgot to turn"; (2) a mixed marriage `(x+y)^rotl(y,31)` — rejected, not provably injective in `y`; (3) contiguous per-lane stripes — rejected, seven simultaneous read streams for no gain over in-order interleaving; (4) alternating cheap/expensive organs — rejected, ~3 cycles saved for a real quality risk.

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 4.0**

Written before any measurement. Reasoning: FNV-1a's chain is `xor`(1) + `imul`(3) = 4 cycles/byte, ~0.25 B/cycle, irreducibly serial. The seven-cup loop issues ~21–24 µops per 7 bytes with seven independent 2-cycle chains, so it is front-end/load-port bound at roughly 4–5 cycles per 7 bytes ≈ 1.4–1.8 B/cycle, i.e. **5–7×** on long buffers. Fixed cost is the marriage (~6 cycles) plus seven organs (~35 cycles of latency), which is why I discount to 4.0 for a mixed-size benchmark, and why at `len = 8` I expect roughly parity, not a win.

Secondary prediction: **avalanche ≈ 0.50** (I would be surprised by anything outside 0.49–0.51), and I predict it is **already 0.50 at three organs** — i.e. organs 4–7 buy nothing measurable. That is the falsifiable content of the chosen assumption break, and it is the garden-door check the native asked for.

---

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were not available in this session; no throughput number and no avalanche score were obtained. Nothing above should be read as measured. The kernel is offered for the pipeline to compile (`gcc -O3 -march=native -fopenmp -lm`) and measure, against the predictions stated in the previous section.

The specific comparison I want run, if the harness allows a second build: replace the `if (organs == 3) return z;` line with an unconditional `return z;` (a three-organ body) and re-measure avalanche. If it does not move, the native's assumption break is confirmed on this problem and organs 4–7 should be amputated.

---

## VERDICT

The native's idea survives translation intact and is implementable without borrowing the textbook's per-byte multiply. Honest accounting of where it can lose:

1. **Fixed finalization cost at tiny inputs.** Seven organs is ~35 cycles regardless of `len`, which exceeds FNV-1a's *entire* loop below about 24 bytes. **Addressed, not hand-waved:** `len <= 16` runs three organs (the garden door reached early), so the crossover is pushed under 8 bytes; and `len < 64` avoids the seven-cup setup and marriage entirely, falling back to the single unbroken pour. Both are real size checks with real fallbacks, not comments.
2. **Residual risk I have *not* eliminated.** At `len = 8` I predict rough parity with FNV-1a, not a win. I chose not to strip organs further, because below three the design stops being the native's and starts being a worse `fmix64`.
3. **Quality risk.** The add+rotate cup diffuses a single-byte change more slowly than a multiply would; the whole quality argument rests on the organs turning *any* nonzero difference into a 50 % flip. That is exactly what SEED 3 tests, and it is the measurement I most want back.
4. **If the measurement comes back below ~2×, or avalanche below 0.48, say so and say it plainly** — the honest failure mode here is that the seven-lane loop is load-port bound closer to 1 B/cycle than 1.8, or that `rotl64(c + b, R)` leaves a structured difference the organs happen to under-diffuse. Neither would be rescued by adding rounds, which is precisely the assumption the native told us not to trust.