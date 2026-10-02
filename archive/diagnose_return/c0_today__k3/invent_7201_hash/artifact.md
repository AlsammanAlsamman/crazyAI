## MAPPING

### SEED 1 — the grey wax trough

| World object | Problem object |
|---|---|
| a "mark" in the pile | one input byte (`data[i]`) |
| the pile | the byte buffer + `len` |
| trough of grey wax at blood heat | the live register state, warm/mutable, never written to memory |
| "presses in before the last has cooled flat" | the next byte is absorbed **before the previous mix's latency has retired** — overlapping, pipelined absorption |
| "never a clean picture of any single mark, only the sum of every collision" | state is a superposition, not a per-byte record |
| scraped clean and remelted | no scratch allocation; state dies with the call |

**Assumption broken:** *"each byte must be mixed into the running state before the next byte is read."* The wax explicitly does **not** wait for the previous impression to set.

### SEED 2 — seven tuned swarms and the forward-only wheel

| World object | Problem object |
|---|---|
| seven bell-jars | **seven independent 64-bit accumulators** `a0..a6` |
| "each jar tuned to a different hunger" | a distinct odd 64-bit multiplier constant per lane |
| "bites in its own count and rhythm" | a distinct rotation amount per lane |
| the dipped wire | the 8-byte word handed to each lane (`56` bytes per dip = 7 lanes × 8) |
| one shared spinning slate wheel | the shared rotate-then-multiply stage all lanes pass through |
| "only ever turns forward, never back" | **rotation, not shift** — information-preserving; nothing discarded |
| "no bite can be walked off once it's landed" | accumulate with `+`/`^` into an invertible state; no lossy truncation |
| weighing the pile before the jars are opened | a runtime `len` check choosing how many jars to wake |

**Assumption broken:** *"the state is a single accumulator updated in place, one value."* Seven jars, seven states, seven independent dependency chains.

### SEED 3 — the brine basin and the six frost-flowers

| World object | Problem object |
|---|---|
| plunge the wire **once**, at the end | all heavy avalanche work is deferred to a single finalization, not paid per byte |
| "wait until the frost-lace stops spreading" | run the finalizer to saturation (full xor-shift-multiply avalanche), don't cut it short |
| "a token pulled too early is soft and lies" | a truncated finalizer gives poor avalanche |
| **six** frost-flowers from the wire | folding **seven** lanes to one takes exactly **six** binary merges |
| "no two ever alike" | each of the six merges uses its own constant and its own rotation |
| the tin sketch | the returned `uint64_t` — the only survivor |
| wax/wire/hunger thrown back | zero persistent state, zero allocation |

**Assumption broken:** *"more mixing rounds always means better mixing"* and *"mixing one byte requires a multiplication."* Per-byte work is made cheap; mixing quality is bought once, at the basin.

## CHOSEN SEED

**SEED 2.** It is the one of the three that breaks *"the state is a single accumulator updated in place, one value"*, which I was told to prefer — and it is the farthest from FNV-1a, whose entire identity is one accumulator. I use the other two seeds as the machine they clearly belong to: SEED 1 is the overlapping (latency-hiding) absorption that seven jars make possible, SEED 3 is the merge-and-finalize that seven jars make *necessary*.

Per step 4, I let the mechanism land on validated ground rather than invent: "seven accumulators, each with its own hunger, all passing through one forward-only wheel, then folded and frozen once" **is** the striped multi-accumulator design of xxHash64 (4 lanes → here 7) with a murmur3 `fmix64` finalizer. Every round, constant, and the tail ladder are taken from that validated family. The native's contribution is the lane count, the per-lane distinct hungers/rhythms, the exactly-six distinct merges, and the pile-weighing regime check.

## ASSUMPTION BROKEN

*The state is a single accumulator updated in place, one value.* → Seven accumulators. The payoff is mechanical, not aesthetic: FNV-1a's `h *= prime` is a **serial multiply chain**, so it pays full multiply *latency* (~3–4 cycles) per byte. Seven independent chains pay multiply *throughput* (1/cycle) instead, so the wax really does take the next impression before the last has cooled.

Also broken, as a consequence: the forward-only wheel means **rotate, never shift** in the absorbing path (a shift walks bites off the wheel), and the basin means multiplication is spent per 4 bytes instead of per 1 byte.

## ARTIFACT

Regime recognition (step 5): the known_way section describes both a per-byte small-input regime and a bulk regime, so the native **weighs the pile first**. A pile under 112 marks cannot cover the wire for all seven jars — the six frost-flowers would cost more than they mix — so he keeps **one jar awake** and reads marks one at a time through the same wheel and the same basin. No threads: the metaphor's unit of work is one dip of a wire (56 bytes); one Weighing Hall, seven jars — thread parallelism at that granularity would be pure overhead, so I stop at register-level ILP plus `restrict` and `memcpy` loads.

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the wheel (forward only: rotate, never shift) ---- */
static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t load64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t load32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

#define BASIS 1469598103934665603ULL          /* FNV-1a offset basis, kept as a nod */
#define P1    0x9E3779B185EBCA87ULL           /* the shared slate wheel's multiplier */

/* seven hungers, one per bell-jar (all odd) */
#define PL0 0xC2B2AE3D27D4EB4FULL
#define PL1 0x165667B19E3779F9ULL
#define PL2 0x85EBCA77C2B2AE63ULL
#define PL3 0x27D4EB2F165667C5ULL
#define PL4 0xD6E8FEB86659FD93ULL
#define PL5 0xA3B195354A39B70DULL
#define PL6 0x589965CC75374CC3ULL

/* six frost-flowers, no two ever alike */
#define M1 0x9E3779B97F4A7C15ULL
#define M2 0xBF58476D1CE4E5B9ULL
#define M3 0x94D049BB133111EBULL
#define M4 0x2545F4914F6CDD1DULL
#define M5 0x7FB5D329728EA185ULL
#define M6 0x1C69B3F74AC4AE35ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t rem = len;
    uint64_t h;

    /* ---- weigh the pile: is it thick enough to cover the wire for seven jars? ---- */
    if (rem >= 112) {
        /* seven troughs, seven hungers */
        uint64_t a0 = BASIS + PL0 + PL1;
        uint64_t a1 = BASIS + PL1;
        uint64_t a2 = BASIS;
        uint64_t a3 = BASIS - PL0;
        uint64_t a4 = BASIS + PL2;
        uint64_t a5 = BASIS - PL2;
        uint64_t a6 = BASIS ^ PL3;

        /* one dip of the wire = 56 marks; no lane waits for another to cool */
        do {
            a0 = rotl64(a0 + load64(p +  0) * PL0, 31) * P1;
            a1 = rotl64(a1 + load64(p +  8) * PL1, 29) * P1;
            a2 = rotl64(a2 + load64(p + 16) * PL2, 33) * P1;
            a3 = rotl64(a3 + load64(p + 24) * PL3, 27) * P1;
            a4 = rotl64(a4 + load64(p + 32) * PL4, 37) * P1;
            a5 = rotl64(a5 + load64(p + 40) * PL5, 23) * P1;
            a6 = rotl64(a6 + load64(p + 48) * PL6, 35) * P1;
            p += 56; rem -= 56;
        } while (rem >= 56);

        /* ---- the lattice: seven jars fold to one in exactly six frost-flowers ---- */
        h = a0;
        h = (h ^ rotl64(a1 * M1, 29)) * P1;
        h = (h ^ rotl64(a2 * M2, 31)) * P1;
        h = (h ^ rotl64(a3 * M3, 33)) * P1;
        h = (h ^ rotl64(a4 * M4, 27)) * P1;
        h = (h ^ rotl64(a5 * M5, 35)) * P1;
        h = (h ^ rotl64(a6 * M6, 25)) * P1;
    } else {
        /* thin pile: keep one jar awake, same wheel, same basin */
        h = BASIS + PL3;
    }

    h += (uint64_t)len;                       /* the pile's own weight is a mark too */

    /* ---- the last marks, read one at a time ---- */
    while (rem >= 8) {
        h ^= rotl64(load64(p) * PL0, 31) * P1;
        h  = rotl64(h, 27) * PL2 + PL3;
        p += 8; rem -= 8;
    }
    if (rem >= 4) {
        h ^= load32(p) * P1;
        h  = rotl64(h, 23) * PL1 + PL2;
        p += 4; rem -= 4;
    }
    while (rem > 0) {
        h ^= (uint64_t)(*p) * PL4;
        h  = rotl64(h, 11) * P1;
        p++; rem--;
    }

    /* ---- the brine basin: wait until the frost-lace stops spreading ---- */
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;                                 /* only the tin sketch survives */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Written before any measurement, and the arithmetic behind it, so it can be judged:

- FNV-1a is a serial `xor; imul` chain: `imul r64` latency ≈ 3 cycles, so ≈ 4 cycles **per byte** ⇒ ≈ 0.25 B/cycle ≈ 0.75 GB/s at 3 GHz.
- Seven jars cost 14 multiplies per 56 bytes = 1 multiply per 4 bytes, and the seven chains are independent, so the limit is multiply *throughput* (1/cycle) not latency ⇒ ≈ 4 B/cycle ≈ 12 GB/s at 3 GHz.
- 4 ÷ 0.25 = **16×** compute-bound; derated to **12×** for loads, DRAM bandwidth on large buffers, and finalizer overhead.

Avalanche: I predict **≈ 0.5** (score ≈ 0.49–0.51, or ≈ 1.0 if normalized). The reasoning I'd want checked is specifically the high bits: flipping bit 63 of an input word is *not* spread by `w * PL_k` (carries have nowhere to go), and is rescued only by the `rotl` that precedes the next multiply — which is exactly why the native's wheel turns before the jar bites, and why a shift there would have failed.

## MEASUREMENT

**No measurement was taken.** `hash_bench` and `hash_contract` were listed as available but this session was launched with no tool access, so I could not run them and I did not get my four improvement rounds. I am reporting that plainly rather than inventing numbers: the MEASUREMENT section above the line is empty, and the 12× is a prediction, not a result.

What I would check first, in order, and what each outcome would mean:

1. **Throughput < 4×** → the benchmark is dominated by inputs under ~112 bytes and never enters the seven-jar hall; the win is then only word-wise-vs-byte-wise, not lane parallelism.
2. **Throughput ≈ 8–16×** → prediction confirmed; the wax hypothesis (latency hiding, not fewer operations, is the whole prize) is correct.
3. **Throughput > 20×** → I under-derated; probably everything stayed in L1 and the 56-byte stripe is also winning on load ports.
4. **Avalanche < 0.45** → the suspect is the high-bit case above; the fix is a third round of the basin (`h ^= h >> 29; h *= M1;`) before the final shift, cheap because it is paid once per call, not once per byte.

## VERDICT

The native's machine is not a curiosity — it is a correct and specific diagnosis of what is actually wrong with FNV-1a. The single in-place accumulator is a **serial latency chain**, and "each mark presses in before the last has cooled flat" is the precise statement that this is the bottleneck. Seven jars is the fix; the six frost-flowers are the price of the fix (seven states must become one); the brine basin is where the quality that FNV pays for per byte gets bought once instead.

Honestly stated: this mechanism **arrives at xxHash64's striped design** rather than beating it. That is the intended outcome under step 4 — a validated technique reached by the metaphor is worth more than a novel one invented to look original. The genuinely native parts are the lane count of seven (odd, SIMD-hostile, but ideal for scalar ILP — 7 chains fully cover a 3–4 cycle multiply latency at 1/cycle throughput), the per-lane distinct hunger *and* rhythm, and the exactly-six-merge fold.

**The risk my own verdict names, and how it is addressed:** seven jars plus six frost-flowers is a fixed overhead of ~24–30 cycles that a short pile cannot amortize — below roughly 64–112 bytes the seven-jar path is *slower* than one accumulator. I did not leave that as a caveat. The pile is weighed at runtime (`len >= 112`), and a thin pile takes the one-jar path, which is still word-wise and still strictly faster than per-byte FNV. The crossover was derived (big path ≈ 0.25n + 30 cycles vs. one jar ≈ 0.625n cycles ⇒ n ≈ 64) and the threshold set at 112 to clear it with margin and guarantee at least two full dips.

**What I deliberately did not do:** no OpenMP. The metaphor's unit of work is one dip of a wire — 56 bytes. There is one Weighing Hall. Spawning threads over 56-byte units at benchmark sizes would be overhead, and splitting the pile across halls would change the token, which the native forbids ("change one mark anywhere and the *whole* flower comes out changed"). I stopped at `restrict`, `memcpy` loads, and seven-way ILP. A wider path exists — xxh3 folds 16 bytes per multiply via 32×64 products — but x86 has no 64×64 SIMD multiply, so that path would have abandoned the seven jars for four, and I was not willing to quietly substitute it.