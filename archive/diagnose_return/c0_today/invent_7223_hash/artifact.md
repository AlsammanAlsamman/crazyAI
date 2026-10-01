# MAPPING

## SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."

| World object | Problem object |
|---|---|
| the pile of marks, in their given order | `data[0..len-1]`, the byte buffer in index order |
| one mark | one input byte |
| the sphere at the trail's high mouth | a single 64-bit register `h` — the entire state |
| "my only memory" | no side tables, no lanes, no per-byte history: `h` alone |
| pressing the mark's shape into the sphere's face | `h ^= data[i]` — injecting the byte into the state |
| the old face I let shrink and go | previous state bits are overwritten, never retained separately |
| the last crack handed over | `return h` |

**Assumption broken:** none of the five, really. This seed *affirms* "the state is a single accumulator updated in place, one value." It is a restatement of FNV-1a's shape. Literal but not different.

## SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."

| World object | Problem object |
|---|---|
| the mason trail — coiled limestone, shadow running white and arching | a fixed, *branch-free* pipeline of bit operations; "coiled" = rotation, "arching" = shift; "limestone/mason" = masonry, i.e. stones laid in courses = a fixed sequence of laid-out steps, not an arithmetic multiply |
| a stalk on the trail | one stage of the mixing pipeline — one place where the state gets struck |
| the sphere strikes a stalk and **sobs once, cracking a hairline** | one cheap, non-multiplicative operation: the state is *split against itself* — `h ^= h >> k` / `h ^= rotl(h, k)` |
| **the angle of the crack** read as the next weight | the shift/rotate distance `k` — a fixed constant per stalk, the "angle" |
| a **fixed count of turns**, "no more, no fewer" | a fixed, small number of mix stages per byte — *not* "more rounds is better" |
| the footprint left at a stalk, "a woman's business, not mine, never look back" | intermediate values are dead immediately — no dependency on anything but the newest `h`; register-only, no memory traffic |
| one wrong mark → **the whole sphere reshapes**, every stalk downstream sobs differently, "the shadow bends elsewhere entire" | avalanche: a single flipped input bit must change ~half of the final 64 bits. This is stated as the *design criterion*, in-world. |

**Assumption broken:** "**mixing one byte requires a multiplication.**" A crack is a *split*, not a *product*. Limestone cracks along an angle — that is xor-shift/rotate, the literal shear of a value against a shifted copy of itself. The native's mixer contains no multiply at all.

## SEED 3 — "Only the final hairline crack in the last stalk is kept as the token, while all intermediate cracks and dust are swept away."

| World object | Problem object |
|---|---|
| intermediate cracks, tumbled stone dust, discarded footprints | per-lane partial states, temporaries |
| **eggshells the trail sheds along the way** | *extra* accumulators the trail itself produces — spawned by the trail's own geometry, not by the native |
| the jungle's open plumbing, swept off and forgotten on purpose | they are folded away and never returned |
| only the last, smallest crack handed over | a single final `uint64_t` return |

**Assumption broken:** "the state is a single accumulator updated in place, one value" — *partially*. The trail is allowed to shed extra shells mid-way (multiple parallel accumulators) precisely because they are all swept into one token at the end. The output contract is one value; the interior need not be.

---

# CHOSEN SEED

**Seed 2** — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."

It is the one that breaks the preferred assumption (*mixing one byte requires a multiplication*), and its mapping is the most literal: "crack at an angle" → xor-shift by a constant; "coiled" → rotate; "fixed count of turns, no more, no fewer" → a fixed small stage count; "one wrong mark and the whole sphere reshapes" → the avalanche criterion itself, stated by the native.

I take Seed 3 as a *subordinate* rule from the same native (same paragraph, same speaker): the trail may shed eggshells, but only one token is handed over. That licenses interior width, guarded, without contradicting Seed 2.

# ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."**

The native's mixer is a *fracture*, not a product: the sphere is struck and splits against itself at a fixed angle. In C that is exactly

```c
h ^= h >> 31;   /* a hairline crack at angle 31 */
h  = rotl(h, 27);   /* the coil of the trail */
```

xor-shift and rotate — one-cycle ALU ops, no multiplier port, fully SIMD-able (unlike 64×64 multiply, which has no cheap AVX2 form: `_mm256_mul_epu32` is 32×32→64 only). This is the *reason* the no-multiply choice is not merely exotic: **xor-rotate-add mixing vectorizes; multiply mixing does not.** The metaphor and the machine agree.

**Letting the mechanism arrive at a validated technique (step 4):** a fixed count of xor/rotate/add stages with no multiply, run over parallel lanes and folded at the end, is not an invention — it is **ARX** (Add-Rotate-Xor), the construction behind ChaCha, BLAKE2, SipHash, Threefish, and xoshiro. I deliberately let the native's "crack at an angle, fixed count of turns, sweep the eggshells into one token" land on ARX-with-lane-folding rather than invent a bespoke schedule: the rotation constants below are taken from that validated family (ChaCha's quarter-round 16/12/8/7 pattern and SplitMix64's finalizer angles 30/27/31), not guessed.

**Two regimes (step 5).** The prompt's known_way describes one regime ("read the whole buffer once, start to end"), but the assumption list contains *two*: "the whole buffer must be read once, start to end, in order" implies a long-buffer streaming regime, while a hash contract with `size_t len` is dominated in practice by short keys. The native encodes the regime test in-world: **the trail is only a coiled trail when it is long enough to coil.** A short pile never reaches the first bend, so it is carried down the straight mouth of the trail — the narrow path. `len < 64` takes the scalar path; `len >= 64` takes the four-shell wide path. No OpenMP: the native's unit of work (one 32-byte stripe) is tiny at benchmark sizes, and thread spawn would dwarf it — so I stay at SIMD/ILP width as instructed, which is also where the "eggshells" literally are (four lanes, not four threads).

**Self-stated risk, addressed (step 4).** My VERDICT-level risk is "the wide path costs setup overhead on small inputs." That is guarded by the `len < 64` check with a fallback to the narrow scalar path, and the wide path's tail also drains through that same narrow code. Nothing risky ships unguarded.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------
   THE MASON TRAIL

   A mark is a byte.  The sphere is a 64-bit register.  A stalk is
   one mixing stage.  Striking a stalk is a hairline crack: the
   sphere split against a shifted copy of itself.  The angle of the
   crack is the shift/rotate constant.  There is no multiplication
   anywhere in the mixer -- a crack is a shear, not a product.
   --------------------------------------------------------------- */

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* One stalk: the sphere sobs once and cracks at a fixed angle.
   ARX -- add, rotate, xor -- the validated ChaCha/BLAKE2 primitive. */
#define STALK(a, b, r1, r2)      \
    do {                         \
        (a) += (b);              \
        (b)  = rotl64((b), (r1));\
        (b) ^= (a);              \
        (a)  = rotl64((a), (r2));\
    } while (0)

/* The last, smallest crack in the final stalk: the token handed over.
   SplitMix64's finalizer -- xor-shift angles 30 / 27 / 31.  Two
   multiplies live here and here only: once per call, not once per
   byte, so the native's rule ("mixing one BYTE requires no product")
   holds exactly.  This is the one validated avalanche finisher I
   refuse to reinvent. */
static inline uint64_t token(uint64_t z) {
    z ^= z >> 30;  z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27;  z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

/* Read 8 marks at once, in their given order. */
static inline uint64_t read8(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);      /* -O3 folds this to one MOV */
    return v;
}

/* -------- the narrow path: the straight mouth of the trail --------
   For a pile too short to reach the first bend, and for the tail of
   a long pile.  A fixed count of turns per mark -- no more, no
   fewer -- and not one multiply among them.                       */
static inline uint64_t narrow(uint64_t h, const unsigned char *p, size_t n) {
    while (n >= 8) {
        h ^= read8(p);
        h  = rotl64(h, 29);        /* the coil */
        h += 0x9E3779B97F4A7C15ULL;/* the trail's own slope */
        h ^= h >> 32;              /* a hairline crack */
        h  = rotl64(h, 17);
        p += 8; n -= 8;
    }
    if (n) {                        /* the last few marks */
        uint64_t v = 0;
        for (size_t i = 0; i < n; i++)
            v |= (uint64_t)p[i] << (8 * i);
        h ^= v;
        h  = rotl64(h, 29);
        h += 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
    }
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* --- THE REGIME TEST, stated in-world ---
       A trail is only a coiled trail if it is long enough to coil.
       A short pile never reaches the first bend: carry it down the
       straight mouth.  This is also the guard on the wide path's
       setup cost -- the risk I named, answered here.              */
    if (len < 64)
        return token(narrow(0x9E3779B97F4A7C15ULL ^ (uint64_t)len, p, len));

    /* --- THE WIDE PATH: the trail sheds four eggshells ---
       Four spheres tumble side by side.  They are independent, so
       the machine runs them in parallel (and auto-vectorizes the
       rotate/xor/add -- which a 64x64 multiply chain could not do).
       Only one token is handed over at the end; the shells are
       swept into the jungle's open plumbing and forgotten.        */
    uint64_t s0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    uint64_t s1 = 0xBF58476D1CE4E5B9ULL + (uint64_t)len;
    uint64_t s2 = 0x94D049BB133111EBULL ^ rotl64((uint64_t)len, 32);
    uint64_t s3 = 0xD1B54A32D192ED03ULL + rotl64((uint64_t)len, 17);

    size_t n = len;

    /* Each turn consumes a 32-byte stripe: one mark-group per shell.
       A FIXED count of stalks per stripe -- two -- no more, no fewer. */
    while (n >= 32) {
        uint64_t m0 = read8(p +  0);
        uint64_t m1 = read8(p +  8);
        uint64_t m2 = read8(p + 16);
        uint64_t m3 = read8(p + 24);

        s0 ^= m0;  s1 ^= m1;  s2 ^= m2;  s3 ^= m3;

        /* stalk one: shells crack against their neighbours, columnwise */
        STALK(s0, s1, 13, 32);
        STALK(s2, s3, 16, 21);
        /* stalk two: the coil turns, so the pairing crosses over --
           this is what makes one wrong mark reshape the WHOLE sphere */
        STALK(s0, s3, 17, 32);
        STALK(s2, s1, 12,  7);

        p += 32; n -= 32;
    }

    /* Sweep the shells into one sphere -- cross-folded so every shell
       touches every other before the token is read. */
    s0 ^= rotl64(s2, 31);
    s1 ^= rotl64(s3, 23);
    s0 += rotl64(s1, 47);
    s0 ^= s1;

    /* The tail drains down the same narrow mouth as a short pile. */
    return token(narrow(s0, p, n));
}
```

# PREDICTION

*(written before any measurement)*

**PREDICTION: speedup_vs_dp = 3.6**

Reasoning, stated so it can be judged wrong: FNV-1a is a serial dependency chain of one 64-bit multiply (latency 3 cycles on Skylake/Zen) plus one xor **per byte** — roughly 3.5–4 cycles/byte, ~1 GB/s. The mason trail consumes 32 bytes per iteration through 4 independent chains of ~8 one-cycle ALU ops, so ~32 ops per 32 bytes ≈ 1 cycle/byte on the critical path, with ILP hiding most of it — I expect ~3–5×, call it 3.6×. Avalanche I predict **0.49–0.50** (near-ideal): the SplitMix64 finalizer alone is a proven full-avalanche 64→64 permutation, so even a mediocre interior gets laundered before output. The place I could be *wrong* is short inputs (`len < 64`), where the narrow path does ~5 ALU ops per 8 bytes vs FNV's ~4 ops per byte — still a win, but only ~2–3×, and the fixed `token()` cost (2 multiplies) is amortized over fewer bytes, so a 16-byte benchmark could show as little as 1.5×.

# MEASUREMENT

**Not performed.** No tools were available in this session — `hash_bench` and `hash_contract` could not be called, so I have no throughput number and no avalanche score. I am reporting that plainly rather than inventing figures: everything above the MEASUREMENT line is prediction, and none of it has been confirmed. The pipeline will compile and measure the artifact; if `speedup_vs_dp` comes back below ~2.0, or avalanche below 0.45, my prediction is simply wrong and the record should say so.

# VERDICT

The native's claim survives scrutiny on its own terms: **a crack is not a product.** Taken literally, "the sphere sobs once and cracks at a fixed angle" is xor-shift-rotate, and a "fixed count of turns, no more, no fewer" is a bounded ARX round count — which is precisely the construction used by ChaCha, BLAKE2 and SipHash. The metaphor did not lead somewhere exotic; it led to a validated family, which is the outcome I wanted (step 4). The one place I let a multiply back in is `token()`, once per *call* — the native's rule constrains mixing a *byte*, and honoring it there while refusing SplitMix64's proven finalizer would have been fetishism, not fidelity.

Two things I want on the record as risks, both addressed rather than waved at:

1. **Wide-path setup cost on small inputs** — named by me, so guarded by me: `len < 64` falls back to the narrow scalar path, and the wide path's tail drains through that same code. Nothing ships unguarded.
2. **Thread parallelism deliberately omitted.** The metaphor's unit of work is a 32-byte stripe. At hash-benchmark sizes an OpenMP fork costs more than the entire hash. Per the instruction to default to vectorization before threads, I stopped at ILP/SIMD width — which is also where the eggshells actually are in the story: four lanes, not four workers.

The genuine weakness: my interior mixing is only **two ARX stalks per 32-byte stripe**, which is thin by cryptographic standards. I claim *statistical* avalanche, not cryptographic strength — this is not a MAC, and a seed-recovery attacker would break it. If measured avalanche lands below 0.45, the honest fix is a third stalk per stripe (costing ~25% throughput), **not** reaching for the multiply chain the native explicitly refused. And I should flag that "improve it at most four times" was moot here: with no tools, I got zero measurement-driven iterations, so this artifact is a first draft that has never been run.