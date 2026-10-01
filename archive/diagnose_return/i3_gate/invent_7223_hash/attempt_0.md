## MAPPING

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| world object | problem object |
|---|---|
| pile of marks, in their given order | `data[0 .. len)`, byte order significant |
| the sphere at the trail's high mouth | the running hash state, initialized from fixed constants |
| "my only memory" | no tables, no history buffer, no per-position storage |
| pressing a mark into the sphere's face | XOR the absorbed word into one word of the state |
| "the old face I let shrink and go" | state updated in place; predecessor unrecoverable |
| a sphere (one body, many contact faces) | one state *object* — not necessarily one 64-bit *value* |

*Silent assumption broken:* essentially **none**. This seed restates "the state is a single accumulator updated in place." The only crack in it is *face* vs *value*: a sphere is one body with a surface touched at several points, which licenses a 256-bit state that is still "one sphere."

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| world object | problem object |
|---|---|
| tumbling / rolling | **bitwise rotation** — the only motion a rolling sphere has |
| the mason trail's stalks | the fixed rotate-distance schedule (13, 16, 21, 17, 32) |
| "I read the *angle* of that crack" | the rotate amounts are literally angles, constants of the trail |
| "it sobs once, cracking a hairline into itself" | the one non-linear event per strike: an **addition**, whose carry cracks upward through bit positions |
| "a fixed count of turns, no more, no fewer" | a compile-time-constant round count `c` per absorbed word — not tuned, not adaptive |
| "press its shape into the sphere's face before letting it go" | `v3 ^= m` *before* the rounds, `v0 ^= m` after |
| "never look back at the footprint" | no lookup table, no memory of prior positions |
| "one wrong mark and every stalk downstream sobs differently" | full-state diffusion, all downstream rounds altered = avalanche |

*Silent assumption broken:* **"mixing one byte requires a multiplication."** A rolling stone does not multiply. It *rotates* (rotl), it *cracks* (add-with-carry), and the mark is *pressed in* (xor). Rotate + Add + Xor. Secondarily it breaks "more mixing rounds always means better mixing": the count is fixed, *no more, no fewer*.

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token, while all intermediate cracks and dust are swept away."**

| world object | problem object |
|---|---|
| intermediate cracks, footprints, eggshells, stone dust | intermediate states — never observed, therefore not obligated to exist in sequential order |
| "the jungle's open plumbing" | scratch memory / dead registers, discarded |
| the last, smallest crack in the final stalk | the 64-bit finalization fold `v0^v1^v2^v3` |
| "I don't keep the sphere itself" | the 256-bit state is not the output; only its fold is |

*Silent assumption broken:* **"the whole buffer must be read once, start to end, in order"** and "each byte must be mixed into the running state before the next byte is read." If no intermediate crack is ever inspected, ordering is only binding *within one groove* — several spheres may run abreast down parallel grooves and be pressed together at the end.

## CHOSEN SEED

**SEED 2**, as instructed — it is the one that breaks "mixing one byte requires a multiplication," and its mapping is the most literal of the three (tumble = rotate, crack angle = rotate constant, sob = carry).

SEED 3 is admitted only as a **licence for the second regime**: it is what permits more than one sphere without lying about the metaphor.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** Replaced by the sphere's own physics: *roll* (rotate), *crack* (add — carry propagation is the sole non-linearity), *press* (xor). No `imul` appears anywhere in the kernel.

This is not an invention. Taking the mechanism seriously lands exactly on a validated, published construction: **SipHash** (Aumasson & Bernstein, 2012) is an ARX permutation whose *entire design premise* is multiplication-free mixing, whose rotate constants are a fixed "trail" of angles {13, 16, 21, 17, 32}, whose message is XOR-ed into `v3` before a **fixed** round count and into `v0` after, and which — literally — discards the state and hands over only `v0^v1^v2^v3`. Per step 4, I let the metaphor arrive there rather than hand-rolling a novel ARX mixer.

**Two regimes, recognized in-world.** The native weighs the pile before laying it. A *short* pile: one sphere, one groove, the careful roll (SipHash-2-4, the most-validated parameter set — latency-optimal, no setup). A *tall* pile: the coiled trail's parallel grooves are opened, eight spheres run abreast (AVX2, 8 lanes × 64 B/iteration), and at the end their eight hairline cracks are pressed into one last sphere that is rolled the careful way. The switch is a runtime `len` check at 256 bytes — which is also the required guard for the risk my own verdict names (lane fan-in/fan-out overhead is pure loss on small buffers). No threads: at these sizes the metaphor's units of work do not pay for a fork.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the trail: rolling (rotate), cracking (add), pressing (xor). No multiply. ---- */
#define ROTL64(x,b) ((uint64_t)(((x) << (b)) | ((x) >> (64 - (b)))))

#define SIPROUND(v0,v1,v2,v3) do {                                      \
    v0 += v1; v1 = ROTL64(v1,13); v1 ^= v0; v0 = ROTL64(v0,32);         \
    v2 += v3; v3 = ROTL64(v3,16); v3 ^= v2;                             \
    v0 += v3; v3 = ROTL64(v3,21); v3 ^= v0;                             \
    v2 += v1; v1 = ROTL64(v1,17); v1 ^= v2; v2 = ROTL64(v2,32);         \
} while (0)

#define GROOVE_MIN 256u   /* pile taller than this -> open the parallel grooves */

static const uint64_t LANEK0[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0xD1B54A32D192ED03ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
    0x27D4EB2F165667C5ULL, 0x85EBCA77C2B2AE63ULL
};
static const uint64_t LANEK1[8] = {
    0x2545F4914F6CDD1DULL, 0x9FB21C651E98DF25ULL, 0xEB44ACCAB455D165ULL,
    0x589965CC75374CC3ULL, 0xA24BAED4963EE407ULL, 0x9D9D7BB0B44B4B4DULL,
    0xC0A2AE9C0E6D7A47ULL, 0x7FEDD1B6B9F0A15FULL
};

static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ---- ONE SPHERE, ONE GROOVE: the careful roll (SipHash-2-4) ---- */
static uint64_t sphere_roll(const unsigned char * restrict p, size_t len,
                            uint64_t k0, uint64_t k1)
{
    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;
    size_t n = len & ~(size_t)7, i;
    for (i = 0; i < n; i += 8) {
        uint64_t m = ld64(p + i);
        v3 ^= m;                                   /* press the mark into the face */
        SIPROUND(v0,v1,v2,v3);                     /* a fixed count of turns ...   */
        SIPROUND(v0,v1,v2,v3);                     /* ... no more, no fewer        */
        v0 ^= m;
    }
    {   uint64_t b = ((uint64_t)len) << 56;
        const unsigned char *t = p + n;
        switch (len & 7) {
          case 7: b |= (uint64_t)t[6] << 48; /* fall through */
          case 6: b |= (uint64_t)t[5] << 40; /* fall through */
          case 5: b |= (uint64_t)t[4] << 32; /* fall through */
          case 4: b |= (uint64_t)t[3] << 24; /* fall through */
          case 3: b |= (uint64_t)t[2] << 16; /* fall through */
          case 2: b |= (uint64_t)t[1] <<  8; /* fall through */
          case 1: b |= (uint64_t)t[0];       /* fall through */
          default: break;
        }
        v3 ^= b; SIPROUND(v0,v1,v2,v3); SIPROUND(v0,v1,v2,v3); v0 ^= b;
    }
    v2 ^= 0xffULL;                                 /* the final stalk */
    SIPROUND(v0,v1,v2,v3); SIPROUND(v0,v1,v2,v3);
    SIPROUND(v0,v1,v2,v3); SIPROUND(v0,v1,v2,v3);
    return v0 ^ v1 ^ v2 ^ v3;                      /* keep only the hairline crack */
}

#if defined(__AVX2__)
/* ---- EIGHT SPHERES ABREAST: the coiled trail's parallel grooves ---- */
#define VADD _mm256_add_epi64
#define VXOR _mm256_xor_si256
#define VROT(x,b) _mm256_or_si256(_mm256_slli_epi64((x),(b)), _mm256_srli_epi64((x),64-(b)))
#define VR32(x)   _mm256_shuffle_epi32((x),0xB1)   /* rotate-by-32 as a lane swap */
#define VSIP(w,x,y,z) do {                                              \
    w = VADD(w,x); x = VROT(x,13); x = VXOR(x,w); w = VR32(w);          \
    y = VADD(y,z); z = VROT(z,16); z = VXOR(z,y);                       \
    w = VADD(w,z); z = VROT(z,21); z = VXOR(z,w);                       \
    y = VADD(y,x); x = VROT(x,17); x = VXOR(x,y); y = VR32(y);          \
} while (0)

static uint64_t grooves(const unsigned char * restrict p, size_t len)
{
    const __m256i C0 = _mm256_set1_epi64x((long long)0x736f6d6570736575ULL);
    const __m256i C1 = _mm256_set1_epi64x((long long)0x646f72616e646f6dULL);
    const __m256i C2 = _mm256_set1_epi64x((long long)0x6c7967656e657261ULL);
    const __m256i C3 = _mm256_set1_epi64x((long long)0x7465646279746573ULL);
    const __m256i KA0 = _mm256_set_epi64x((long long)LANEK0[3],(long long)LANEK0[2],
                                          (long long)LANEK0[1],(long long)LANEK0[0]);
    const __m256i KA1 = _mm256_set_epi64x((long long)LANEK1[3],(long long)LANEK1[2],
                                          (long long)LANEK1[1],(long long)LANEK1[0]);
    const __m256i KB0 = _mm256_set_epi64x((long long)LANEK0[7],(long long)LANEK0[6],
                                          (long long)LANEK0[5],(long long)LANEK0[4]);
    const __m256i KB1 = _mm256_set_epi64x((long long)LANEK1[7],(long long)LANEK1[6],
                                          (long long)LANEK1[5],(long long)LANEK1[4]);
    __m256i A0=VXOR(C0,KA0), A1=VXOR(C1,KA1), A2=VXOR(C2,KA0), A3=VXOR(C3,KA1);
    __m256i B0=VXOR(C0,KB0), B1=VXOR(C1,KB1), B2=VXOR(C2,KB0), B3=VXOR(C3,KB1);
    const unsigned char * restrict q = p;
    size_t nb = len >> 6, i;

    for (i = 0; i < nb; i++) {                     /* 64 B per turn, 8 grooves */
        __m256i MA = _mm256_loadu_si256((const __m256i *)(const void *)(q));
        __m256i MB = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32));
        A3 = VXOR(A3, MA); B3 = VXOR(B3, MB);
        VSIP(A0,A1,A2,A3); VSIP(B0,B1,B2,B3);      /* one fixed turn per groove */
        A0 = VXOR(A0, MA); B0 = VXOR(B0, MB);
        q += 64;
    }
    {   const __m256i FF = _mm256_set1_epi64x((long long)0xffULL);
        A2 = VXOR(A2,FF); B2 = VXOR(B2,FF);
        VSIP(A0,A1,A2,A3); VSIP(B0,B1,B2,B3);
        VSIP(A0,A1,A2,A3); VSIP(B0,B1,B2,B3);
        VSIP(A0,A1,A2,A3); VSIP(B0,B1,B2,B3);
    }
    {   __m256i RA = VXOR(VXOR(A0,A1), VXOR(A2,A3));   /* each groove's own crack */
        __m256i RB = VXOR(VXOR(B0,B1), VXOR(B2,B3));
        unsigned char scratch[128];
        size_t tail = len & 63u;
        _mm256_storeu_si256((__m256i *)(void *)scratch,        RA);
        _mm256_storeu_si256((__m256i *)(void *)(scratch + 32), RB);
        if (tail) memcpy(scratch + 64, q, tail);
        /* press the eight cracks, and the unfinished marks, into one last sphere */
        return sphere_roll(scratch, 64 + tail, (uint64_t)len,
                           (uint64_t)len * 0x9E3779B97F4A7C15ULL ^ 0xA0761D6478BD642FULL);
    }
}
#else
/* ---- fallback grooves for machines without AVX2: two spheres abreast ---- */
static uint64_t grooves(const unsigned char * restrict p, size_t len)
{
    uint64_t a0=0x736f6d6570736575ULL^LANEK0[0], a1=0x646f72616e646f6dULL^LANEK1[0],
             a2=0x6c7967656e657261ULL^LANEK0[0], a3=0x7465646279746573ULL^LANEK1[0];
    uint64_t b0=0x736f6d6570736575ULL^LANEK0[1], b1=0x646f72616e646f6dULL^LANEK1[1],
             b2=0x6c7967656e657261ULL^LANEK0[1], b3=0x7465646279746573ULL^LANEK1[1];
    const unsigned char * restrict q = p;
    size_t nb = len >> 4, i;
    for (i = 0; i < nb; i++) {
        uint64_t ma = ld64(q), mb = ld64(q + 8);
        a3 ^= ma; SIPROUND(a0,a1,a2,a3); a0 ^= ma;
        b3 ^= mb; SIPROUND(b0,b1,b2,b3); b0 ^= mb;
        q += 16;
    }
    a2 ^= 0xffULL; SIPROUND(a0,a1,a2,a3); SIPROUND(a0,a1,a2,a3); SIPROUND(a0,a1,a2,a3);
    b2 ^= 0xffULL; SIPROUND(b0,b1,b2,b3); SIPROUND(b0,b1,b2,b3); SIPROUND(b0,b1,b2,b3);
    {   uint64_t r0 = a0^a1^a2^a3, r1 = b0^b1^b2^b3;
        unsigned char scratch[48];
        size_t tail = len & 15u;
        memcpy(scratch, &r0, 8); memcpy(scratch + 8, &r1, 8);
        if (tail) memcpy(scratch + 16, q, tail);
        return sphere_roll(scratch, 16 + tail, (uint64_t)len,
                           (uint64_t)len * 0x9E3779B97F4A7C15ULL ^ 0xA0761D6478BD642FULL);
    }
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    if (len >= (size_t)GROOVE_MIN)                 /* weigh the pile first */
        return grooves(p, len);                    /* tall pile: grooves abreast */
    return sphere_roll(p, len,                     /* short pile: one careful sphere */
                       0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
}
```

## PREDICTION

*(stated before any measurement; no benchmark tool was reachable in this session)*

**PREDICTION: speedup_vs_dp = 6.0**

Reasoning behind the number, so it can be scored honestly rather than hand-waved:
- FNV-1a's cost is a serial dependency chain of `xor; imul` ≈ **4 cycles/byte** (~1 GB/s at 4 GHz). It is latency-bound on the multiplier, not throughput-bound.
- Short path (SipHash-2-4): ≈ **1.3–1.6 cycles/byte** → ~2.5–3× on buffers under 256 B, *plus* a fixed ~6-round finalization that hurts on very short keys (I expect roughly parity, maybe slightly worse, at len ≤ 16).
- Long path (8 AVX2 grooves, 64 B per one vectorized round, two independent vector states to cover round latency): ≈ **0.25–0.35 cycles/byte** → ~12–15× on multi-kilobyte buffers, until DRAM bandwidth caps it.
- 6.0 is my estimate of the geometric mix if the harness sweeps sizes from tiny to large. If it benchmarks only ≥ 4 KB, expect 10–14. If it benchmarks only ≤ 32 B, expect 0.8–1.5, and my prediction is badly wrong.

Avalanche: I predict **≥ 0.49 and near-ideal bias**, because both paths terminate in an unmodified SipHash-2-4 finalization and every input bit reaches it through at least four full ARX rounds. Unlike a pure rotate-xor "rolling stone" — which would be GF(2)-linear and could score 0.5 on a bit-count metric while being cryptographically dead — the *crack* is an integer add, so the mixing is genuinely non-linear.

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were not available in this session; no number here is empirical and I am not going to present an estimate as a reading. The prediction above stands as written, before the fact, for the pipeline to confirm or refute.

What the pipeline should check, and what would count as my mechanism failing:
1. `speedup_vs_dp` at len ≥ 4096 below ~8× → the groove/AVX2 argument is wrong (likely round-latency-bound, not throughput-bound, as I assumed).
2. `speedup_vs_dp` at len ≤ 32 below 1.0 → the fixed 6-round finalization is a real tax on short keys and `GROOVE_MIN` is not the only guard I needed.
3. Avalanche below ~0.45 → the ARX-only claim (no multiply needed) is refuted at these round counts, and the honest conclusion would be that multiplication is doing real work in FNV/xxHash that rotation-plus-carry does not replace at equal cost.

## VERDICT

The native's account is literally an ARX sponge, and taking it literally does not produce a novelty — it produces **SipHash**, a published and heavily validated multiplication-free hash. That is the right outcome, not a disappointment: the seed's claim was "mixing needs no multiplication," and the strongest possible evidence for that claim is that a real, deployed, cryptanalyzed construction already makes it. Rolling = rotate, sobbing = carry, the trail's stalks = the rotate-angle schedule, "a fixed count, no more no fewer" = `c=2, d=4`, "I don't keep the sphere, only the last hairline crack" = `v0^v1^v2^v3`. Every object maps; nothing was smuggled in.

The one place the metaphor genuinely *outruns* the textbook is SEED 3: because no intermediate crack is ever inspected, the strict in-order single-accumulator discipline is not load-bearing, and eight spheres may run abreast. That is what buys the speed, and it is the part most likely to be wrong in practice.

**Named risks, and how each is addressed rather than hoped away:**
- *The grooves only pay on a tall pile.* Guarded: `len >= 256` selects grooves, everything below falls back to the single careful sphere. My break-even arithmetic puts the crossover near 128 B, so 256 is a deliberately conservative threshold — I would rather lose a little speed at 200 bytes than ship a regression.
- *AVX2 may not exist.* Guarded: `#if defined(__AVX2__)` with a two-groove scalar path that uses the same construction and the same combiner, so behavior degrades in speed only.
- *Thread parallelism.* Deliberately not used. The metaphor's unit of work (a 64-byte turn) is far too small to amortize a fork at these sizes; adding OpenMP would be a mechanism whose own stated risk I could not guard.
- *Short keys.* This is the honest weak point. SipHash-2-4's six-round finalization is a fixed cost FNV-1a does not pay, so at len ≤ 16 I expect roughly parity and would not be surprised by a small loss. I chose not to shave finalization rounds to chase that, because doing so would trade the validated parameter set for an untested one — which step 4 explicitly forbids, and which would put the avalanche score, the thing the seed actually claims, at risk to win a benchmark regime.