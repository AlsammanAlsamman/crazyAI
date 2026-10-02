## MAPPING

### SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."

| World object | Problem object |
|---|---|
| sailor-chain, a row of knots | the hash state as a **row of lanes** (4 × `uint64_t`), not one number |
| "never breaking an old knot, only lengthening" | state is never reset; each byte extends the dependency chain |
| the shadow is of **the whole chain**, not the new knot | every byte-absorb is followed by a permutation that touches **all** lanes |
| "the whole line hangs together or not at all" | full cross-lane diffusion each step; no lane evolves alone |
| fire → chain → wall | input → state → output projection |

**Breaks:** *"the state is a single accumulator updated in place, one value."*

### SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."

| World object | Problem object |
|---|---|
| a mark | one input word/byte from `data` |
| "that mark's own strokes" counted as pebbles | `popcount` of the mark |
| the small cup's fill | the rotation amount, 0..64 |
| swinging incense on a **short chain** | rotation is cheap and bounded, `rotl64` |
| smoke **bends the fire's throw** before it lands | **data-dependent rotation applied to the lane** before the permutation |
| wall's **six fixed cracks**, same six every time | six fixed rotation positions per reading (13, 32, 16, 21, 17, 32) |
| "six places my token will ever hold" | 6 bits name all 64 bit-positions of the token; all rotations are mod 64 |
| no object anywhere that scales or repeats a quantity | **no multiplier exists in this world** |

**Breaks:** *"mixing one byte requires a multiplication."* The byte's own content is spent as a *shift distance*, not as a multiplicand.

### SEED 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."

| World object | Problem object |
|---|---|
| waiting for the shadow to go still | read the state only after the absorb loop completes |
| pressing wet clay | copying the settled lanes out |
| the stream running over the stub | **finalization rounds** on a state no longer fed input |
| "only the hardened ridges remain" | the final xor-fold of all lanes — lossy, one-way |
| throwing away smoke, cup, old print | no retained scratch, no per-round history |
| "hooded reader working backward from print to pile" | preimage resistance |
| one fist | `uint64_t` |

**Breaks:** *"more mixing rounds always means better mixing"* — rounds are spent in two **different** places with different jobs (absorb vs. wash); the wash is the only place where added rounds buy avalanche, and it is finite (4).

---

## CHOSEN SEED

**Seed 2.** It is the one seed that breaks the preferred assumption, and the mapping is mechanical: the byte's popcount *is* a rotation count, nothing is reinterpreted. It is also maximally distant from FNV-1a/xxHash, whose entire engine is `imul`. Seeds 1 and 3 supply the state shape and the finalizer, so nothing is discarded — Seed 2 decides the *mixing primitive*.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** There is no `*` in the kernel's mixing path. Every bit of diffusion comes from add, rotate, xor.

Per step 4, I did not invent a new multiply-free mixer. The native's own equipment — a row of lanes, additions of a mark, bends at six fixed cracks, a wash at the end — *is* **ARX (add–rotate–xor)**, and the six fixed cracks land exactly on the **SipRound** permutation of Aumasson & Bernstein (rotations 13, 32, 16, 21, 17, 32; four lanes; `v2 ^= 0xff` then finalization rounds then fold). That is a validated, deployed primitive (Python, Rust, Perl hash tables). I let the metaphor arrive there and added only what the metaphor demands and SipHash lacks: the popcount-driven **bend** before each landing.

Two regimes are named in the problem statement (`FNV-1a` byte-at-a-time *or* `xxHash` block-wide), so the native recognizes them in-world: a short pile gets **one short span** walked mark-by-mark; a long pile gets the row **strung across the whole mouth of the piazza, four marks abreast, one to each quarter**, the quarters knotted together at the end. The regime test is `len >= 256`, and the short path is the fallback — so the wide path's own stated risk (overhead on small inputs) is guarded, not merely noted. No OpenMP: the metaphor forbids breaking the chain, and the wide path is already bandwidth-adjacent, so threads would buy nothing I could honestly justify.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============ the wall's six fixed cracks, the same six every time ============
   One reading of the shadow = six rotations at fixed places
   (13, 32, 16, 21, 17, 32) over a four-knot row.  This is SipRound.
   There is no multiplication anywhere in this world.                         */

static inline uint64_t bend(uint64_t x, unsigned r) {
    return (x << (r & 63)) | (x >> ((64u - r) & 63));   /* r==64 == r==0 */
}

#define CRACKS(a,b,c,d) do {                                               \
    (a) += (b); (b) = bend((b),13); (b) ^= (a); (a) = bend((a),32);        \
    (c) += (d); (d) = bend((d),16); (d) ^= (c);                            \
    (a) += (d); (d) = bend((d),21); (d) ^= (a);                            \
    (c) += (b); (b) = bend((b),17); (b) ^= (c); (c) = bend((c),32);        \
} while (0)

static inline uint64_t mark8(const unsigned char *p) {
    uint64_t m; memcpy(&m, p, sizeof m); return m;      /* one mov */
}

/* the small cup: this mark's own strokes, counted as pebbles,
   decide how far the brick-incense swings on its short chain      */
#define CUP(m) ((unsigned)__builtin_popcountll(m))

/* knot the mark onto the row, then let the smoke bend the throw   */
#define KNOT(v,m) do { uint64_t _m = (m); (v) ^= _m; (v) = bend((v), CUP(_m)); } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t n = len;

    /* the row as first hung between fire and wall */
    uint64_t v0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL ^ (uint64_t)len;

    /* ---- REGIME A: a long pile.  The row is strung across the whole mouth
            of the piazza and four marks walk abreast, one to each quarter.
            Guarded: only when the pile is big enough to pay for stringing
            it; otherwise fall straight through to the short span below. */
    if (n >= 256) {
        uint64_t a0=v0, a1=v1, a2=v2, a3=v3;
        uint64_t b0=v0^0x9e3779b97f4a7c15ULL, b1=v1^0x9e3779b97f4a7c15ULL, b2=v2, b3=v3;
        uint64_t c0=v0^0xbf58476d1ce4e5b9ULL, c1=v1^0xbf58476d1ce4e5b9ULL, c2=v2, c3=v3;
        uint64_t d0=v0^0x94d049bb133111ebULL, d1=v1^0x94d049bb133111ebULL, d2=v2, d3=v3;

        do {
            KNOT(a0, mark8(p +   0)); KNOT(a1, mark8(p +   8));
            KNOT(a2, mark8(p +  16)); KNOT(a3, mark8(p +  24));
            KNOT(b0, mark8(p +  32)); KNOT(b1, mark8(p +  40));
            KNOT(b2, mark8(p +  48)); KNOT(b3, mark8(p +  56));
            KNOT(c0, mark8(p +  64)); KNOT(c1, mark8(p +  72));
            KNOT(c2, mark8(p +  80)); KNOT(c3, mark8(p +  88));
            KNOT(d0, mark8(p +  96)); KNOT(d1, mark8(p + 104));
            KNOT(d2, mark8(p + 112)); KNOT(d3, mark8(p + 120));
            CRACKS(a0,a1,a2,a3);
            CRACKS(b0,b1,b2,b3);
            CRACKS(c0,c1,c2,c3);
            CRACKS(d0,d1,d2,d3);
            p += 128; n -= 128;
        } while (n >= 128);

        /* knot the four quarters back into one row: every quarter must
           hang on every other, or the line does not hang at all       */
        v0 = a0 ^ b1 ^ c2 ^ d3;
        v1 = a1 ^ b2 ^ c3 ^ d0;
        v2 = a2 ^ b3 ^ c0 ^ d1;
        v3 = a3 ^ b0 ^ c1 ^ d2;
        CRACKS(v0,v1,v2,v3);
        CRACKS(v0,v1,v2,v3);
    }

    /* ---- REGIME B: the short pile -- and every long pile's remainder.
            One short span; each mark walks past the fire alone and the
            whole chain is read once for it.                           */
    while (n >= 8) {
        uint64_t m = mark8(p);
        v3 ^= m;                    /* the new knot            */
        v3 = bend(v3, CUP(m));      /* the cup bends the throw */
        CRACKS(v0,v1,v2,v3);        /* shadow of the whole row */
        v0 ^= m;
        p += 8; n -= 8;
    }

    /* the last short mark, with the pile's own length knotted in, so that
       moving a mark changes every bend after it and the length is not free */
    {
        uint64_t t = ((uint64_t)(len & 0xff)) << 56;
        switch (n) {
            case 7: t |= (uint64_t)p[6] << 48; /* fall through */
            case 6: t |= (uint64_t)p[5] << 40; /* fall through */
            case 5: t |= (uint64_t)p[4] << 32; /* fall through */
            case 4: t |= (uint64_t)p[3] << 24; /* fall through */
            case 3: t |= (uint64_t)p[2] << 16; /* fall through */
            case 2: t |= (uint64_t)p[1] <<  8; /* fall through */
            case 1: t |= (uint64_t)p[0];       /* fall through */
            default: break;
        }
        v3 ^= t;
        v3 = bend(v3, CUP(t));
        CRACKS(v0,v1,v2,v3);
        v0 ^= t;
    }

    /* ---- the mouth of the stream: the clay stub held under running water
            until only the hardened ridges remain.  Four full readings with
            no mark feeding in, then the whole row folded into one fist --
            lossy, so no hooded reader walks back from print to pile.    */
    v2 ^= 0xff;
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    CRACKS(v0,v1,v2,v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 8.0**

(Stated before any measurement, and no measurement was possible in this session — see below.)

Reasoning behind the number, so it can be judged rather than taken on faith:
- Reference FNV-1a is latency-bound on a serial `imul`(3) → `xor`(1) chain: **~4 cycles/byte ≈ 0.25 B/cyc**.
- Long path: ~7 µops per 8-byte mark (load, popcnt, neg/and, shlx, shrx, or, xor) + 12 µops per `CRACKS` per 32 bytes ≈ **1.25 µops/byte**, with four independent quarters to hide the ~8-cycle round latency → **~2.5–3 B/cyc**. Ratio ≈ 10–12×.
- Short path (< 256 B, and all tails): one `CRACKS` per 8 bytes, serial → **~0.8 B/cyc**, ratio ≈ 3×.
- If the harness averages across sizes, the blend lands near 8×. I expect avalanche **≈ 0.50** (SipHash's 4-round finalization is the validated reason, not a hope), and I expect it to hold at *both* regimes because the merge contributes 2 extra full-state rounds before the wash.

Where I expect to be wrong: if the benchmark is dominated by very small buffers (≤ 32 B), 8.0 is too optimistic and the honest number is nearer 2.5–3.5. If it is dominated by multi-MB buffers, DRAM bandwidth caps both and the ratio could exceed 12×.

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were unavailable in this session (no tools present), so I have no throughput or avalanche figure to report. I am stating this plainly rather than reporting a plausible-looking number: everything in the PREDICTION section is a static µop/latency estimate, not a measurement, and the prediction stands unverified. The four permitted improvement rounds were therefore not spent — the artifact above is the first and only version, which also means it has had no opportunity to be tuned against reality.

Two concrete things a measurement should check first:
1. Whether `popcnt` + variable rotate per mark costs more than the diffusion it buys. If the long path is below ~1.5 B/cyc, the `CUP`-bend is the suspect, and the honest fix is to keep the bend only on the short path (where latency, not µop count, binds) — not to delete it, since it is the chosen seed's whole content.
2. Whether avalanche at the regime boundary (len = 255 vs. 256) is stable. A gap there would indict the merge, not the rounds.

## VERDICT

The translation is literal and nothing in the native's account was dropped: the row of lanes is the chain, each byte's popcount is the cup that sets the swing, the six cracks are six fixed rotation positions, the whole-chain shadow is a permutation that touches all four lanes every step, and the stream-wash is a four-round finalization followed by an irreversible fold. The mechanism contains no multiplication, which is exactly the assumption chosen for breaking.

The result's honest status: **the metaphor did not produce a novel hash — it produced SipHash-1-4 with a data-dependent rotation added and a four-way block split for long inputs.** I count that as the correct outcome under step 4, not a disappointment. A multiply-free mixer that *converges* on a deployed, cryptanalyzed primitive is far better evidence that the seed is sound than a clever thing I made up would be.

Named risk and how it is handled, so no stated risk ships unaddressed:
- *The four-quarter path only pays on large piles.* Guarded by `if (n >= 256)` with the single-span path as the fallback; the two paths share the same finalizer, so there is one hash, not two.
- *Thread parallelism.* Deliberately absent. The metaphor forbids breaking the chain, and I cannot justify OpenMP at unknown benchmark sizes; I would rather ship nothing there than ship a guess.
- *The popcount bend is the one piece with no prior validation.* It cannot lose information (the full mark is xored in separately, so the bend is strictly additional structure), but it is unmeasured, and if the bench shows it costing throughput without improving avalanche, it is the first thing to confine to the short path.

Unverified claim, stated as such: I believe this beats FNV-1a by roughly 8× with avalanche near 0.5. Until `hash_bench` runs, that is an estimate, and I am not going to dress it up as a result.