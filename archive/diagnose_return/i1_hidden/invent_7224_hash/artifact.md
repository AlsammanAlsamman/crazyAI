No tools were reachable in this session (the `hash_bench` / `symbolic_*` / `unconventional_*` set did not surface, and the claude.ai **PubMed** connector is unauthorized — it needs authorizing in your claude.ai connector settings before that capability is usable). So everything below is hand-derived, the prediction is stated blind, and the pipeline's measurement will be the first real number.

---

## MAPPING

**SEED 1 — "the servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded"**

| World object | Problem object |
|---|---|
| a mark | one input byte `data[i]` |
| the pile "in their given order" | the buffer `data[0..len)` |
| walking marks past the servant | the absorb loop |
| the inflated house swatted on the mouth (the Zen master's bladder-on-a-stick) | the *suppression of the spoken chain*: the sequential data dependency is forbidden |
| "the order never gets spoken aloud and lost" | the index is not carried by control flow but **written into the data**: each word is keyed by `blk*PHI ^ KK[j]` |
| the flat stone at the riverbank | position-keyed absorb: `v[j] ^= w + (KK[j]^kb)` |
| silence held *until* each mark is laid | no lane may depend on another lane's current value |

**SEED 2 — "each heap races the musk deer's mountain stride"**

| World object | Problem object |
|---|---|
| heaps of even count, fixed strides | 64-byte blocks, 8 × 64-bit lanes |
| musk deer's stride | fixed unrolled stride, no data-dependent advance |
| uphill twist | `ROTL64` |
| downhill fold | `x ^= x >> s` |
| doubling-back that eats its own trail | multiply by a large odd constant (low→high avalanche, non-recoverable) |
| "no heap keeps the shape it entered with" | every lane is *reassigned*, never accumulated |
| the final stone at the shrine, carried forward | one single cross-lane carry `v[0] += ROTL64(v[7],29)` |
| "the part I never let ossify — a pillar that brings the roof down" | the carry is re-mixed every block and never becomes a serial bottleneck |

**SEED 3 — "the owl calls the final shape inside the dreamer inside; the heaps are burned"**

| World object | Problem object |
|---|---|
| drifting man searching squares in every direction | the state read as a 2×4 grid of lanes |
| turning ninety degrees | lane transpose `(v0,v4,v1,v5,v2,v6,v3,v7)` |
| folding corners into the center | cross-diagonal `add` / `xor`-of-rotate pairs |
| "again and again until one changed mark smears across every square" | 4 `square_turn` rounds (reach: 1→2→4→8 lanes) |
| the owl / dreamer inside | `owl()` finalizer, xor-shift-multiply ladder |
| sealed to one fixed size regardless of marks | `uint64_t` return, independent of `len` |
| heaps burned, cannot be walked back | lanes are discarded; no reversible state escapes; `h>>s` folds destroy information |
| the butterfly who cannot recall being a man | one-wayness of the final seal |

## CHOSEN SEED

**SEED 1.** It is the only one of the three whose central image is an *instruction about reading order* rather than about arithmetic, and it is the most literal: the servant physically prevents the order from being **spoken** (serialized) and requires it be **laid on a stone** (materialized as data). SEED 2 and SEED 3 describe the mixing function and the finalizer — they are subordinate mechanisms, and I implement both, but neither challenges anything.

## ASSUMPTION BROKEN

> *"the whole buffer must be read once, start to end, in order."*

Taken literally, SEED 1 says the order must be **fixed as data before any folding happens**, precisely so that it never has to be *spoken* — i.e. never carried by the sequence of reads. Once each 8-byte word's position is folded into its own key (`KK[j] ^ blk*PHI`), the read order becomes semantically irrelevant: the result is identical whether the 8 lanes are walked in step, out of step, or on different cores. Two consequences, both shipped:

1. **Within a block** there is no chain across lanes — 8 independent xor-mul-rotate streams, so the multiplier port, not latency, is the limit (~8–12 cycles per 64 bytes instead of ~3 cycles per *byte*).
2. **Across the buffer**, for `len ≥ 4 MB`, the buffer is cut into fixed 1 MB chunks each keyed by its global block offset, hashed in parallel, and recombined with a *commutative* `+`. Fixed chunking (not thread-count chunking) keeps the value deterministic for any thread count.

The one thing the native insists must still travel — "the final stone at the shrine" — survives as a single cross-lane carry, and is deliberately kept off the critical path so it cannot "ossify into a pillar."

## ARTIFACT

```c
/* kernel.c — the riverbank stone, the musk deer's stride, the owl's seal.
   Contract: uint64_t kernel(const unsigned char *data, size_t len);
   gcc -O3 -march=native -fopenmp -lm                                    */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define C1  0xBF58476D1CE4E5B9ULL
#define C2  0x94D049BB133111EBULL
#define C3  0xFF51AFD7ED558CCDULL
#define PHI 0x9E3779B97F4A7C15ULL

/* eight stones of the riverbank: one per lane */
static const uint64_t KK[8] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0x27D4EB2F165667C5ULL, 0xD6E8FEB86659FD93ULL,
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* the owl takes the pen: one-way, sealed to one fixed size */
static inline uint64_t owl(uint64_t h)
{
    h ^= h >> 32; h *= C3;
    h ^= h >> 29; h *= C2;
    h ^= h >> 32; h *= C1;
    h ^= h >> 31;
    return h;
}

/* the drifting man searches squares: turn the 2x4 grid ninety degrees,
   fold its corners into its own centre, then let each cell eat its trail */
static inline void square_turn(uint64_t *v)
{
    uint64_t t0=v[0], t1=v[4], t2=v[1], t3=v[5], t4=v[2], t5=v[6], t6=v[3], t7=v[7];
    t0 += ROTL64(t7, 13); t7 ^= ROTL64(t0, 41);
    t1 += ROTL64(t6, 19); t6 ^= ROTL64(t1, 47);
    t2 += ROTL64(t5, 29); t5 ^= ROTL64(t2, 53);
    t3 += ROTL64(t4, 37); t4 ^= ROTL64(t3, 59);
    v[0]=t0*C1; v[1]=t1*C2; v[2]=t2*C3; v[3]=t3*C1;
    v[4]=t4*C2; v[5]=t5*C3; v[6]=t6*C1; v[7]=t7*C2;
}

/* one run of heaps. n >= 64 required. blk0 = this run's global block offset:
   the order, laid on the stone as data, never spoken as a dependency.      */
static uint64_t heap_run(const unsigned char *p, size_t n, uint64_t blk0, uint64_t tag)
{
    uint64_t v[8];
    const unsigned char *q = p;
    size_t m = n;
    uint64_t blk = blk0;

    v[0]=KK[0]^(tag*PHI); v[1]=KK[1]^blk0; v[2]=KK[2]; v[3]=KK[3];
    v[4]=KK[4];           v[5]=KK[5];      v[6]=KK[6]; v[7]=KK[7]^tag;

    while (m >= 64) {
        uint64_t kb = blk * PHI;                 /* the position, as data */
        v[0] = ROTL64((v[0] ^ (rd64(q+ 0) + (KK[0]^kb))) * C1, 23);
        v[1] = ROTL64((v[1] ^ (rd64(q+ 8) + (KK[1]^kb))) * C1, 27);
        v[2] = ROTL64((v[2] ^ (rd64(q+16) + (KK[2]^kb))) * C1, 31);
        v[3] = ROTL64((v[3] ^ (rd64(q+24) + (KK[3]^kb))) * C1, 35);
        v[4] = ROTL64((v[4] ^ (rd64(q+32) + (KK[4]^kb))) * C1, 39);
        v[5] = ROTL64((v[5] ^ (rd64(q+40) + (KK[5]^kb))) * C1, 43);
        v[6] = ROTL64((v[6] ^ (rd64(q+48) + (KK[6]^kb))) * C1, 47);
        v[7] = ROTL64((v[7] ^ (rd64(q+56) + (KK[7]^kb))) * C1, 51);
        v[0] += ROTL64(v[7], 29);   /* the one stone carried to the next heap */
        q += 64; m -= 64; blk++;
    }

    if (m) {   /* last heap, read overlapping its predecessor: always in bounds */
        const unsigned char *r = p + n - 64;
        uint64_t kb = blk * PHI, x;
        x = rd64(r+ 0) + (KK[7]^kb); x ^= x >> 31; v[0] = (v[0] ^ x) * C2;
        x = rd64(r+ 8) + (KK[6]^kb); x ^= x >> 31; v[1] = (v[1] ^ x) * C2;
        x = rd64(r+16) + (KK[5]^kb); x ^= x >> 31; v[2] = (v[2] ^ x) * C2;
        x = rd64(r+24) + (KK[4]^kb); x ^= x >> 31; v[3] = (v[3] ^ x) * C2;
        x = rd64(r+32) + (KK[3]^kb); x ^= x >> 31; v[4] = (v[4] ^ x) * C2;
        x = rd64(r+40) + (KK[2]^kb); x ^= x >> 31; v[5] = (v[5] ^ x) * C2;
        x = rd64(r+48) + (KK[1]^kb); x ^= x >> 31; v[6] = (v[6] ^ x) * C2;
        x = rd64(r+56) + (KK[0]^kb); x ^= x >> 31; v[7] = (v[7] ^ x) * C2;
    }

    square_turn(v); square_turn(v); square_turn(v); square_turn(v);

    uint64_t h = tag * PHI + blk0;
    h += (v[0] ^ ROTL64(v[1], 17)) * C1;
    h ^= (v[2] + ROTL64(v[3], 29)) * C2;
    h += (v[4] ^ ROTL64(v[5], 41)) * C3;
    h ^= (v[6] + ROTL64(v[7], 53)) * C1;
    return h;                       /* the heaps are burned here */
}

/* short buffers: a two-by-two square, so the eight-lane fold's fixed cost
   is never charged to an input too small to amortise it                  */
static uint64_t short_run(const unsigned char *p, size_t n)
{
    uint64_t a = KK[0] ^ (n * PHI), b = KK[1] ^ n, c = KK[2], d = KK[3];

    if (n >= 16) {
        const unsigned char *e = p + n - 16;
        a ^= rd64(p+0) + KK[4];
        b ^= rd64(p+8) + KK[5];
        c ^= rd64(e+0) + KK[6];
        d ^= rd64(e+8) + KK[7];
        if (n > 32) {                       /* 33..63: cover the middle too */
            const unsigned char *m1 = p + 16, *m2 = p + n - 32;
            a += rd64(m1+0) ^ KK[5];
            b += rd64(m1+8) ^ KK[4];
            c += rd64(m2+0) ^ KK[7];
            d += rd64(m2+8) ^ KK[6];
        }
    } else if (n >= 8) {
        a ^= rd64(p)         + KK[4];
        c ^= rd64(p + n - 8) + KK[6];
    } else if (n) {
        uint64_t w;
        if (n >= 4) w = rd32(p) | (rd32(p + n - 4) << 32);
        else        w = ((uint64_t)p[0])
                      | ((uint64_t)p[n >> 1] << 24)
                      | ((uint64_t)p[n - 1] << 48);
        a ^= w + KK[4];
        c ^= ROTL64(w, 32) + KK[6];
    }

    for (int i = 0; i < 3; i++) {
        uint64_t s0=a, s1=c, s2=b, s3=d;    /* turn the square ninety degrees */
        s0 += ROTL64(s3, 17); s3 ^= ROTL64(s0, 43);
        s1 += ROTL64(s2, 31); s2 ^= ROTL64(s1, 53);
        a = s0*C1; b = s1*C2; c = s2*C3; d = s3*C1;
    }

    uint64_t h = n * PHI;
    h += (a ^ ROTL64(b, 27)) * C1;
    h ^= (c + ROTL64(d, 45)) * C2;
    return owl(h);
}

#define PAR_MIN ((size_t)1 << 22)   /* 4 MB: below this, threads cost more than they buy */
#define CHUNK   ((size_t)1 << 20)   /* fixed chunking => value independent of thread count */

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (len < 64)      return short_run(data, len);
    if (len < PAR_MIN) return owl(heap_run(data, len, 0, (uint64_t)len));

    size_t nc = len / CHUNK;                 /* >= 4, every chunk >= 64 bytes */
    uint64_t acc = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:acc) schedule(static)
#endif
    for (size_t i = 0; i < nc; i++) {
        size_t off = i * CHUNK;
        size_t sz  = (i + 1 == nc) ? (len - off) : CHUNK;
        acc += heap_run(data + off, sz, (uint64_t)(off / 64), (uint64_t)len) * C1;
    }
    return owl(acc ^ ((uint64_t)len * PHI));
}
```

## PREDICTION

Stated before any measurement (and, in this session, unavoidably before any):

**PREDICTION: speedup_vs_dp = 12.0**

Reasoning behind the number: a byte-at-a-time reference chain is latency-bound at roughly one 3-cycle multiply per byte (≈3 c/B). This kernel issues 8 multiplies per 64 bytes with no cross-lane dependency, so it is multiplier-*throughput* bound at ≈8 cycles, realistically ≈12 cycles/64 B with loads, keys and rotates ⇒ ≈0.19 c/B. That is a ~16× ratio, discounted to 12 for the fixed cost of the 4 `square_turn` rounds (~35 cycles) which is not amortised on 64–512 byte inputs, and for cache/bandwidth effects at large sizes. Avalanche prediction: ≈0.50 (I expect a score in 0.49–0.51 and Δ-bit distributions near binomial(64, ½)); I would treat anything below 0.48 as a failure of the 4-round square fold, not as noise.

## MEASUREMENT

**Not performed.** `hash_bench` did not surface in this session, so I have no throughput or avalanche number and I will not invent one. What I can report honestly is only the static audit I was able to do by hand:

- **Bounds:** every read is provably in-bounds — `heap_run`'s overlapping tail needs `n ≥ 64` (guaranteed by both call sites); `short_run`'s front/back/middle windows are each checked against `n`; the 1–3 byte case touches only `p[0]`, `p[n>>1]`, `p[n-1]`.
- **Coverage:** every input byte reaches the state on every path — `[0,16) ∪ [n−16,n)` covers `n ≤ 32`; `[0,32) ∪ [n−32,n)` covers `33 ≤ n ≤ 63`; the 64-byte stride plus the overlapping final block covers all `len ≥ 64`.
- **No rotate by 0 or 64** anywhere (all amounts in 13…59), so no UB.
- **Zero-block fixed point closed:** the classic `v = (v ^ w)*C` trap (state sticks at 0 on a run of zero bytes) is broken by the position key `KK[j] ^ blk*PHI`, which is never zero and never repeats — this is the direct payoff of taking SEED 1 literally rather than decoratively.
- **Determinism under threads:** chunking is by fixed 1 MB offsets, not by thread count, and the recombination is integer `+`, so the same buffer gives the same 64-bit value at any `OMP_NUM_THREADS`.

## VERDICT

The native's picture translated more cleanly than I expected, and in one place it is *better* than the textbook default: the insistence that order be "laid on the stone" rather than "spoken" is exactly the trick that both (a) frees the absorb loop from a serial dependency and (b) kills the zero-run fixed point that a plain xor-multiply chain suffers. I did not substitute a standard construction anywhere — the 8 lanes are the "heaps of even count," the lane transpose is the literal ninety-degree turn, the single `v[0] += ROTL64(v[7],29)` is the one stone carried to the shrine, and `owl()` is the seal.

**Named risks, and what each is guarded by (per step 4 — nothing risky ships unaddressed):**

1. *The 4-round `square_turn` finalizer is a fixed ~35-cycle tax, ruinous on short inputs.* → Guarded by the `len < 64` size check routing to `short_run`, a 4-lane / 3-round path with no 8-lane fold.
2. *OpenMP thread spawn can exceed the work for medium buffers.* → Guarded by `len < PAR_MIN (4 MB)`; below it the parallel path is not entered at all, and the `#pragma` is `#ifdef _OPENMP`-gated so the code also compiles and runs correctly without `-fopenmp`.
3. *Chunked parallel hashing risks a thread-count-dependent result.* → Guarded by fixed 1 MB chunking plus a commutative reduction; the last chunk absorbs the remainder so no chunk is ever under 64 bytes.
4. *The carry could become the critical path — the native's own warning about the ossified pillar.* → It is a single add-of-rotate whose latency (~6 cycles/block) sits under the block's multiply-throughput floor (~8 cycles/block), so it costs nothing measurable.

**The honest weak point I could not guard, only flag:** the parallel path's combine is a plain sum of chunk digests. Single-bit avalanche is unaffected (one flipped bit destroys one whole chunk digest), but a *sum* is structurally weaker than a chain against adversarial multi-chunk collisions. The native licenses this — order is data, so heaps may be swallowed in any order — and the benchmark does not test it, but I will not pretend it is cryptographic. If a future measurement cared about that, the fix is to make the combine non-commutative for `len ≥ PAR_MIN`, which costs the parallelism and therefore the entire point of the chosen seed. That trade is a real one, and I am naming it rather than hiding it.