## MAPPING

### Seed 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."

| World object | Problem object |
|---|---|
| the pile of marks, in their given order | `data[0 .. len)`, streamed in index order |
| a mark | one input byte |
| the sphere at the trail's high mouth | the initialised running state |
| "my only memory" | no side table, no history buffer, no second pass |
| *a sphere*, not a pebble — it has **faces** | the state is one object but not one scalar: a small bundle of lanes |
| letting the old face shrink and go | bits leave the low end (right shift) and are never recovered |

**Assumption broken:** essentially none — this seed *restates* "the state is a single accumulator updated in place." Its only dissent is geometric: a sphere has many faces, so "one state" need not mean "one 64-bit number." Weak breaker.

### Seed 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."

| World object | Problem object |
|---|---|
| a tumble / "turn" of the sphere | a **bit rotation** of a state word (`rotl64`) |
| the mason trail — cut stone **steps** | fixed shift/rotation distances; the step heights are constants |
| striking a stalk | one round of the mixing permutation |
| "it sobs once, cracking a hairline into itself" | the word is combined **with a displaced copy of itself** (xor/add after rotate) |
| "the angle of that crack" is the next weight | the round's output *is* the new state — nothing else survives |
| pressing the mark into the sphere's face | XOR the input word into a state lane |
| **a fixed count — no more, no fewer** | a constant, small number of rounds per absorbed chunk |
| the footprint at the stalk, "a woman's business, not mine" | intermediate state is never read back, stored, or branched on |
| the shadow that "already runs white and arching" | the mixing schedule is fixed ahead of time, data-independent — no data-dependent branches, no table lookups |

**Assumption broken:** **"mixing one byte requires a multiplication."** A tumble is a rotate and a strike is an add/xor. There is no multiply anywhere in this world — stone does not multiply, it rolls and chips.
Secondary break: "more mixing rounds always means better mixing" is denied outright by *no more, no fewer*.

### Seed 3 — "Only the final hairline crack in the last stalk is kept as the token; all intermediate cracks and dust are swept away."

| World object | Problem object |
|---|---|
| intermediate cracks at every stalk | per-chunk intermediate states |
| swept into the jungle's open plumbing (the cenote) | discarded scratch; never observed, so never required to be strong |
| the last, smallest crack in the final stalk | the **finalizer**: a strong terminal avalanche step |
| the sphere itself is not kept | the raw accumulator is never returned |
| "eggshells the trail sheds along the way" | the ragged tail bytes that don't fill a block |

**Assumption broken:** "each byte must be mixed into the running state before the next byte is read" (intermediate states are unobservable, so per-byte *full* mixing is waste), and again "more mixing rounds always means better mixing."

---

## CHOSEN SEED

**Seed 2.** The selection rule is explicit: prefer a seed that breaks *"mixing one byte requires a multiplication."* Seed 2 is the only one of the three that breaks it, and it breaks it literally — the entire world contains rotation (tumble), displacement (stone steps), and collision (add/xor), and contains no scaling operation at all. It is also the most mechanically specific: it names the operation (turn), the operand (self, displaced), and the schedule (fixed count).

Seeds 1 and 3 are not discarded, they are subordinate: Seed 1 fixes the state's shape (one object, several faces) and Seed 3 fixes the output rule (finalizer only). Seed 2 supplies the actual arithmetic.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."**

Replaced by: *mixing is Add–Rotate–Xor with fixed, data-independent constants.*

Per step 4, I let the mechanism land on a validated technique instead of inventing one. Multiply-free ARX mixing with a fixed small round count, a 4-word state, and a separate stronger finalization is not a new idea I need to justify from scratch — it is **SipHash** (Aumasson & Bernstein, 2012): `SipRound` is exactly "tumble a fixed count of turns, cracking a hairline into yourself at each stalk"; SipHash‑2‑4's *2 compression rounds / 4 finalization rounds* is exactly "a fixed count, no more, no fewer" plus "only the last crack is kept"; and SipHash's tail word `(len<<56) | leftover` is exactly "the eggshells the trail sheds." SipHash is the standard multiplication-free hash (Rust's default hasher, Python dicts, Perl, OpenBSD, the Linux kernel). So the narrow path is **verbatim SipHash‑2‑4 with a zero key**, not a lookalike.

**Two regimes (step 5).** The prompt's known_way names two: FNV‑1a (serial, short) and xxHash (blocked, long). The native's own trail is *coiled*, so it has a circumference, and the metaphor recognises the regime by comparing the pile against one turn of the coil:

* **Short pile (`len < 32`) — narrow coil.** The sphere walks the stairs; one mark-word per stalk, two tumbles per stalk. This is exactly SipHash‑2‑4 — the validated, low-fixed-cost fallback.
* **Long pile (`len ≥ 32`) — wide coil.** Once the pile is long enough to wrap the coil, the sphere is rolling fast enough that **four marks land on four different faces between one stalk and the next** (Seed 1's geometry: one sphere, four faces). This is a full-state sponge absorb of 32 bytes per two tumbles — 4× the data per unit of critical path, same permutation, same fixed round count.

The regime test is a single `len >= 32` branch, and the long path *falls into* the short path for its remainder, so there is exactly one permutation and one finalizer. No thread parallelism: the sphere is the *sole* memory, which is a strictly serial chain; splitting it would mean several spheres, which the native forbids, and OpenMP fork cost (~5–20 µs) would exceed the entire hash at any plausible benchmark size. Vectorization hints only — `restrict`, `memcpy` word loads, four scalar lanes deliberately kept in GPRs (SipRound's `v0+=v3 / v2+=v1` cross-couplings are shuffle-hostile; AVX2 would cost more in permutes than it saves).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t x;
    memcpy(&x, p, sizeof x);   /* unaligned word load; -O3 emits one mov */
    return x;
}

/* ONE TUMBLE.  The sphere turns a fixed count, strikes a stalk, sobs once,
   and cracks a hairline into itself: every step is add / rotate / xor of the
   state against a displaced copy of itself.  No multiplication exists on this
   trail.  This permutation is SipRound (Aumasson & Bernstein, 2012). */
#define TUMBLE(v0, v1, v2, v3)                                            \
    do {                                                                  \
        v0 += v1;  v1 = rotl64(v1, 13);  v1 ^= v0;  v0 = rotl64(v0, 32);  \
        v2 += v3;  v3 = rotl64(v3, 16);  v3 ^= v2;                        \
        v0 += v3;  v3 = rotl64(v3, 21);  v3 ^= v0;                        \
        v2 += v1;  v1 = rotl64(v1, 17);  v1 ^= v2;  v2 = rotl64(v2, 32);  \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* The one sphere, four faces, set at the trail's high mouth.
       (SipHash initial state with a zero key.) */
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;

    size_t i = 0;

    /* ---- REGIME TEST: is the pile longer than one turn of the coil? ---- */
    if (len >= 32u) {
        /* WIDE COIL.  The sphere is rolling fast enough that four marks land
           on four different faces between one stalk and the next: absorb 32
           bytes, then the same fixed two tumbles -- no more, no fewer. */
        size_t nblk = len & ~(size_t)31;
        for (; i < nblk; i += 32) {
            uint64_t m0 = ld64(p + i);
            uint64_t m1 = ld64(p + i +  8);
            uint64_t m2 = ld64(p + i + 16);
            uint64_t m3 = ld64(p + i + 24);
            v0 ^= m0; v1 ^= m1; v2 ^= m2; v3 ^= m3;
            TUMBLE(v0, v1, v2, v3);
            TUMBLE(v0, v1, v2, v3);
        }
    }

    /* NARROW COIL.  The short pile, and the eggshells the wide coil shed:
       one mark-word per stalk, two tumbles per stalk.  For len < 32 nothing
       above ran and this is verbatim SipHash-2-4 with a zero key. */
    {
        size_t end = len & ~(size_t)7;
        for (; i < end; i += 8) {
            uint64_t m = ld64(p + i);
            v3 ^= m;
            TUMBLE(v0, v1, v2, v3);
            TUMBLE(v0, v1, v2, v3);
            v0 ^= m;
        }
    }

    /* The last eggshell: the leftover marks, with the pile's own length
       pressed into the top face so a short pile can never wear a long
       pile's token. */
    {
        uint64_t b = ((uint64_t)len) << 56;
        switch (len & 7u) {
            case 7: b |= (uint64_t)p[i + 6] << 48; /* fall through */
            case 6: b |= (uint64_t)p[i + 5] << 40; /* fall through */
            case 5: b |= (uint64_t)p[i + 4] << 32; /* fall through */
            case 4: b |= (uint64_t)p[i + 3] << 24; /* fall through */
            case 3: b |= (uint64_t)p[i + 2] << 16; /* fall through */
            case 2: b |= (uint64_t)p[i + 1] <<  8; /* fall through */
            case 1: b |= (uint64_t)p[i + 0];       /* fall through */
            case 0: break;
        }
        v3 ^= b;
        TUMBLE(v0, v1, v2, v3);
        TUMBLE(v0, v1, v2, v3);
        v0 ^= b;
    }

    /* Only the last, smallest crack in the final stalk is kept.  Four closing
       tumbles; then the sphere, the dust and the footprints all go into the
       cenote and only the token is handed over. */
    v2 ^= 0xffu;
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    TUMBLE(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 6.0**

Stated before any measurement, with the reasoning that makes it falsifiable:

* FNV‑1a's loop-carried chain is `xor` (1 cy) + `imul` (3 cy) = **≈4 cycles/byte**, ≈0.9 GB/s at 3.5 GHz. It cannot be vectorized or pipelined away; `-march=native` does not help it.
* The wide coil's loop-carried chain is 2 × SipRound ≈ 10–11 cycles per **32 bytes** ≈ **0.33 cycles/byte**, ≈10–11 GB/s. Issue pressure is ~32 µops per 32 bytes ≈ 1 µop/byte, so ~0.25 c/B throughput-bound — latency is the binding constraint, not ports.
* That is ≈12× on a long buffer. I discount to **6.0** because (a) the tail + 4 finalization rounds are a fixed ~50-cycle cost that dominates below ~64 bytes, (b) at multi-MB sizes DRAM bandwidth caps the wide path well before 11 GB/s, and (c) the short-pile path (plain SipHash‑2‑4) is only ~1.3–1.5× FNV‑1a at 16 bytes. If hash_bench uses a single large buffer I expect 8–12×; if it averages a size sweep including ≤32-byte inputs I expect 2–4×. 6.0 is my honest midpoint.

Secondary prediction: **avalanche ≈ 0.50 ± 0.01** (essentially ideal). Every input bit, wherever it lands, is followed by at least 2 compression rounds plus the tail absorb (2 rounds) plus 4 finalization rounds = ≥8 SipRounds before output, and SipHash's 4-round finalization is designed to fully avalanche a 4-word state on its own. I expect this kernel to score at or above FNV‑1a, which has known weak avalanche in its high bits for short inputs.

## MEASUREMENT

**Not performed.** This session has no tool access — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were listed but none are callable here, so I did not run the compile-measure-improve loop and I did not get to use the four allowed revisions. I am reporting that plainly rather than inventing numbers; the artifact above is the un-tuned first submission and the prediction above is a genuine prior, not a post-hoc fit.

What I can assert without the bench, by construction rather than by measurement:

* **Contract:** signature is exactly `uint64_t kernel(const unsigned char *data, size_t len)`; argument order untouched.
* **Well-definedness:** `len == 0` reads no bytes (the `switch` falls to `case 0`, `p[i]` is never evaluated) and returns a fixed nonzero token. All loads are `memcpy`-based, so unaligned input is defined behaviour. No UB in the rotates (`r` is always 13/16/17/21/32, never 0 or 64).
* **Determinism / regime consistency:** the two paths are the same permutation and the same finalizer; the wide loop consumes only whole 32-byte blocks and hands its remainder to the narrow loop, so there is no seam and no length for which the result is ambiguous.
* **Reference agreement:** for every `len < 32` the function is bit-identical to SipHash‑2‑4 with `k0 = k1 = 0`, returning `v0^v1^v2^v3`. That is checkable against any SipHash test vector without this tool.

Falsification criteria I would accept: if measured avalanche is below ~0.48, the wide full-state absorb (capacity 0) is the suspect and the fix is to absorb only 24 bytes into `v0..v2` and keep `v3` as capacity, at a 25% throughput cost. If measured speedup is below ~1.5×, the benchmark is short-input dominated and the right move is dropping finalization from 4 rounds to 3 (SipHash‑1‑3, also validated), not touching the wide path.

## VERDICT

The translation is literal and it lands on a real, validated technique rather than a novelty: tumble → rotate, stalk → round, crack-into-itself → xor/add with a displaced copy, fixed count → fixed round schedule, last crack only → finalizer, eggshells → tail word, cenote → discarded scratch. Taking the native seriously removed the multiplication entirely, and the object that falls out of "a sphere with four faces, a fixed number of tumbles per stalk, keep only the last crack" is SipHash. I consider that a success condition, not a coincidence — the metaphor was pointing at something that already exists and works.

Two honest caveats, both already guarded in the code rather than left as prose:

1. **The wide coil is worse than nothing below 32 bytes** — its block loop cannot run and its only effect would be a mispredicted branch. Guarded: `if (len >= 32u)`, with the fallback being not a degraded version but the full validated SipHash‑2‑4. I did not ship an unguarded risk.
2. **Thread parallelism was considered and rejected**, not deferred. The metaphor's unit of work is one indivisible serial chain through one sphere; at any size where OpenMP would pay for itself the kernel is already DRAM-bound, and below that the fork cost exceeds the entire hash. Adding it would have been me overruling the native for no measured gain.

The one place I am genuinely exposed is the wide path's full-state absorb: rate 256 / capacity 0 is fine for avalanche but is trivially invertible, so this is a fast non-cryptographic hash for `len ≥ 32` and a real keyless MAC-grade permutation only for `len < 32`. If the target ever needs collision resistance against an adversary rather than avalanche against a test harness, the wide path must be dropped back to a 24-byte rate — and I would rather state that here than let the throughput number imply a guarantee the construction does not make. Unmeasured, so: predicted 6.0×, ~0.50 avalanche, and I will take the bench's answer over mine.