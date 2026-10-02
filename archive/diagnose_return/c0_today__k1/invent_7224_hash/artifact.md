No tools were reachable in this session (and the claude.ai PubMed / connector servers would need interactive OAuth anyway, which can't be done here). So: the prediction below is stated cold, and the MEASUREMENT section reports plainly that nothing was measured by me — the pipeline's numbers are the first real data point.

---

## MAPPING

### SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."

| World object | Problem object |
|---|---|
| mark | one input byte `data[i]` |
| pile of marks in their given order | the buffer `data[0..len)` |
| walking marks past the servant | the bulk load of a whole block into registers |
| inflated house on a stick, swatting the mouth | the *prohibition on speaking* = no state update during the load: zero arithmetic between consecutive byte reads |
| "so the order never gets spoken aloud and lost" | order is preserved **positionally** (lane index / row offset) instead of **temporally** (sequence of accumulator updates) |
| flat stone at the riverbank | the 4×8 register block (rows `a,b,c,d`) the block is laid into |
| riverbank | start of the current block; also the short-input path |

**Breaks:** *"each byte must be mixed into the running state before the next byte is read"* — directly, and with it the *"in order"* half of *"the whole buffer must be read once, start to end, in order"*: once order is carried by position, the mixer never has to consume byte *i* before byte *i+1*.

### SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."

| World object | Problem object |
|---|---|
| heap of even count | a fixed power-of-two block = 128 bytes |
| musk deer's fixed strides | fixed block granularity, constant stride through memory |
| uphill twist | `ROTL32` |
| downhill fold | `^=` (xor across rows) |
| doubling-back that eats its own trail | `+=` — addition, whose carry re-enters and destroys the lane's old shape; no multiply anywhere in the stride |
| "no heap keeps the shape it entered with" | full in-block diffusion before the block is reduced |
| nothing stays still except the final stone at the shrine | exactly **one** value survives a block: the chaining value |
| carried forward into the next heap's racing | the chain is injected into the next block's rows (serial Merkle–Damgård link) |
| "never let ossify — a calculation too satisfied with itself is a pillar that brings the roof down" | no fixed point / no absorbing state: a per-heap counter is mixed in so identical blocks never produce identical contributions, and block order is not commutative |

**Breaks:** *"the state is a single accumulator updated in place, one value"* (state is 32 lanes), and *"mixing one byte requires a multiplication"* (ARX only: add/rot/xor).

### SEED 3 — "The owl calls the final folded shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned."

| World object | Problem object |
|---|---|
| the owl taking the pen | the finalizer, run once, outside the hot loop |
| "inside the dreamer inside" | *nested* xorshift–multiply (murmur3 `fmix64`): two multiplies, one inside the other's output |
| sealed small and unchanging, size of any other token | 64-bit output regardless of `len` |
| intermediate heaps burned the moment the next swallows them | 512-bit block state reduced to 64 bits and discarded — lossy, nothing retained, no scratch kept |
| "cannot recall being a man once entirely a butterfly" | the compression is non-injective by construction; the token can't be walked back |
| "if even a corner survives untouched I throw the method away" | the round count is set by the avalanche test, at the **minimum** that passes |

**Breaks:** *"more mixing rounds always means better mixing"* — the owl means in-block rounds only need to reach full diffusion, not cryptographic margin. ChaCha20 uses 10 double-rounds; this uses **2**.

---

## CHOSEN SEED

**Seed 1.** It is the only one of the three that touches the read-order assumption, so per the instruction it wins the preference. Honest caveat: it breaks *"in order"* (the per-byte serial dependency), not *"read once"* — no seed lets the native skip marks; every mark is walked. The native's method is one method, so Seeds 2 and 3 supply the heap stride and the seal; Seed 1 governs the structure.

It is also the most *different* from FNV-1a, whose entire identity is "mix byte, then read next byte."

## ASSUMPTION BROKEN

Primary: **each byte must be mixed into the running state before the next byte is read**, and with it **"start to end, in order"** as a *dependency* (not as coverage). Secondary, from the other two seeds: single-accumulator state, multiplication-per-byte, and more-rounds-is-better.

**Letting the mechanism land on a validated technique (step 4).** Taken literally, the native's own objects *are* an existing design: a 4×4 square searched "in every direction", "turning the shape ninety degrees and folding its corners into its own center", mixed by uphill twist / downhill fold / doubling-back, is exactly **ChaCha's column round + diagonal round over 16 words, built from rotate/xor/add**. The owl's "inside the dreamer inside" is exactly **murmur3's `fmix64`**. The shrine's single surviving stone is **Merkle–Damgård chaining with a block counter**. The riverbank short path and the overlapping final heap are **xxHash's small-input and last-block tricks**. So I did not invent a mixer: the metaphor was decoded into ChaCha-core + fmix64 + MD chaining, all validated, with the round count cut to 2 double-rounds because the owl finishes the job. The only genuinely new thing is the *round budget*, which is the one place the native explicitly tells you to test and discard.

**Regime recognition (step 5).** The known-way text spans small vs. large buffers, so the native must know which it's in. His own words supply the test: heaps are of *even count*, so when there aren't enough marks to fill a heap you never leave the riverbank, and when a leftover can't fill a heap you walk back over marks already laid rather than invent new ones. Three runtime paths:
- `len < 32` — riverbank: no square at all, 64-bit ARX + double owl.
- `32 ≤ len < 128` — one heap, zero-padded, length folded into its tag.
- `len ≥ 128` — streaming heaps, final short remainder absorbed as an **overlapping** last 128 bytes (no padding, no branch in the hot loop).

**No threads.** The metaphor's unit of work is one 128-byte heap and the native insists on a *single* shrine stone — the chain is serial by his own account. Thread parallelism would require breaking that, and 128 bytes is far below any thread's worth. Per the instruction I stop at vectorization hints: `restrict`, 32-byte contiguous row loads, and lane-width-8 loops that `-O3 -march=native` packs into one AVX2 (or two SSE2) ops each, giving two independent squares per heap for free.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL32(x,n) ((uint32_t)(((uint32_t)(x) << (n)) | ((uint32_t)(x) >> (32 - (n)))))
#define ROTL64(x,n) ((uint64_t)(((uint64_t)(x) << (n)) | ((uint64_t)(x) >> (64 - (n)))))

/* fixed stones at the shrine */
static const uint32_t SIGMA[4] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u};
static const uint32_t TAU[4]   = {0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 0x165667b1u};

static inline uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* THE OWL: calls the shape inside the dreamer inside, seals it to one fixed size. */
static inline uint64_t owl(uint64_t z) {
    z ^= z >> 33; z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 29; z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 32;
    return z;
}

/* one lane-wide step over both squares at once (gcc -O3 -march=native -> one AVX2 op) */
#define LANE8(stmt) do { for (j_ = 0; j_ < 8; j_++) { stmt; } } while (0)

/* THE DEER'S STRIDE: uphill twist (rot), downhill fold (xor), doubling-back (add).
   No multiplication anywhere in the stride. */
#define STRIDE()                                   \
    LANE8(a[j_] += b[j_]);                         \
    LANE8(d[j_] ^= a[j_]);                         \
    LANE8(d[j_] = ROTL32(d[j_], 16));              \
    LANE8(c[j_] += d[j_]);                         \
    LANE8(b[j_] ^= c[j_]);                         \
    LANE8(b[j_] = ROTL32(b[j_], 12));              \
    LANE8(a[j_] += b[j_]);                         \
    LANE8(d[j_] ^= a[j_]);                         \
    LANE8(d[j_] = ROTL32(d[j_], 8));               \
    LANE8(c[j_] += d[j_]);                         \
    LANE8(b[j_] ^= c[j_]);                         \
    LANE8(b[j_] = ROTL32(b[j_], 7))

/* TURNING THE SQUARE NINETY DEGREES: rotate lanes inside each 4-lane square,
   so corners fold into the centre and the next stride crosses the diagonals. */
#define SPIN(v, s) do {                                                        \
    uint32_t q0 = v[((s)  )&3], q1 = v[((s)+1)&3],                             \
             q2 = v[((s)+2)&3], q3 = v[((s)+3)&3];                             \
    uint32_t r0 = v[4+(((s)  )&3)], r1 = v[4+(((s)+1)&3)],                     \
             r2 = v[4+(((s)+2)&3)], r3 = v[4+(((s)+3)&3)];                     \
    v[0]=q0; v[1]=q1; v[2]=q2; v[3]=q3;                                        \
    v[4]=r0; v[5]=r1; v[6]=r2; v[7]=r3;                                        \
} while (0)

/* ONE HEAP: 128 marks laid on the flat stone in silence, raced, then burned down
   to the single stone that survives at the shrine. */
static inline uint64_t absorb_heap(const unsigned char *restrict p,
                                   uint64_t chain, uint64_t tag)
{
    uint32_t a[8], b[8], c[8], d[8];
    uint32_t k0 = (uint32_t)chain, k1 = (uint32_t)(chain >> 32);
    uint32_t g0 = (uint32_t)tag,   g1 = (uint32_t)(tag   >> 32);
    uint64_t f = 0;
    int j_;

    /* SEED 1: the whole heap is laid down first, in silence. No mixing between
       byte reads; the order of the marks is held by lane position alone.
       Each row is one contiguous 32-byte load. */
    for (j_ = 0; j_ < 8; j_++) a[j_] = rd32(p +  0 + 4*j_) ^ SIGMA[j_ & 3];
    for (j_ = 0; j_ < 8; j_++) b[j_] = rd32(p + 32 + 4*j_) ^ (k0 + 0x9e3779b9u * (uint32_t)j_);
    for (j_ = 0; j_ < 8; j_++) c[j_] = rd32(p + 64 + 4*j_) ^ (k1 ^ TAU[j_ & 3]);
    /* the stone that never ossifies: the heap counter, different for every heap */
    for (j_ = 0; j_ < 8; j_++) d[j_] = rd32(p + 96 + 4*j_) ^ (g0 + (uint32_t)j_)
                                                           ^ (g1 + TAU[(j_ + 1) & 3]);

    /* Two double-strides: column, diagonal, column, diagonal. One double-stride
       already touches all sixteen words of each square; two give bit-level
       smearing. The owl finishes the rest -- more strides here would only cost
       throughput, which is why the method is not raced ten times over. */
    STRIDE(); SPIN(b,1); SPIN(c,2); SPIN(d,3);
    STRIDE(); SPIN(b,3); SPIN(c,2); SPIN(d,1);
    STRIDE(); SPIN(b,1); SPIN(c,2); SPIN(d,3);
    STRIDE(); SPIN(b,3); SPIN(c,2); SPIN(d,1);

    /* BURNED AT THE SHRINE: 512 bits of heap collapse to 64 and are gone.
       Rotations differ per lane so no two lanes can cancel each other. */
    for (j_ = 0; j_ < 8; j_++) {
        uint64_t t1 = ((uint64_t)a[j_] << 32) | (uint64_t)b[j_];
        uint64_t t2 = ((uint64_t)c[j_] << 32) | (uint64_t)d[j_];
        f += ROTL64(t1, (5  * j_ + 1) & 63);
        f ^= ROTL64(t2, (11 * j_ + 7) & 63);
    }

    /* the running shape carried forward into the next heap's racing;
       non-commutative, so heap order matters */
    return ROTL64(chain ^ f, 27) + (f >> 7) + 0x9e3779b97f4a7c15ULL;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t chain;
    size_t rem, ctr;

    /* REGIME 1 -- too few marks to make a heap: never leave the riverbank. */
    if (len < 32) {
        uint64_t x = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0xbf58476d1ce4e5b9ULL);
        uint64_t y = 0x94d049bb133111ebULL + (uint64_t)len;
        if (len >= 8) {
            size_t i = 0;
            uint64_t v;
            do {
                v = rd64(data + i);
                x ^= v;  x = ROTL64(x, 29) + 0x9e3779b97f4a7c15ULL;
                y += ROTL64(v ^ x, 17);
                i += 8;
            } while (i + 8 <= len);
            v = rd64(data + len - 8);      /* walk back over marks already laid */
            x ^= ROTL64(v, 31);
            y ^= ROTL64(v + x, 13);
        } else if (len > 0) {
            uint64_t v = (uint64_t)len;    /* every mark read; 8*7+8 bits fit exactly */
            size_t i;
            for (i = 0; i < len; i++) v = (v << 8) | (uint64_t)data[i];
            x ^= v;
            y ^= ROTL64(v, 19);
        }
        return owl(x ^ owl(y));            /* the dreamer inside */
    }

    chain = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0x9ddfea08eb382d69ULL);

    /* REGIME 2 -- enough for one heap only: pad to a single heap, length in the tag. */
    if (len < 128) {
        unsigned char buf[128];
        memset(buf, 0, sizeof buf);
        memcpy(buf, data, len);
        chain = absorb_heap(buf, chain, 0x165667b19e3779f9ULL ^ (uint64_t)len);
        return owl(chain ^ ((uint64_t)len << 32));
    }

    /* REGIME 3 -- many heaps: stride the mountain path, one shrine stone surviving. */
    rem = len;
    ctr = 0;
    {
        const unsigned char *p = data;
        while (rem >= 128) {
            chain = absorb_heap(p, chain,
                        (uint64_t)ctr * 0x9e3779b97f4a7c15ULL + 0x2545f4914f6cdd1dULL);
            p   += 128;
            rem -= 128;
            ctr += 1;
        }
    }
    if (rem) {
        /* leftover that cannot fill a heap: re-walk the last 128 marks laid,
           rather than invent marks that were never there. No padding, no
           branch inside the hot loop. */
        chain = absorb_heap(data + len - 128, chain,
                    (uint64_t)rem * 0xff51afd7ed558ccdULL ^ 0x9e3779b97f4a7c15ULL);
    }
    return owl(chain ^ ((uint64_t)len * 0xc4ceb9fe1a85ec53ULL));
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 5.0**

Stated before any measurement, against the FNV-1a reference in the prompt as `dp`. Reasoning: FNV-1a is latency-bound at xor(1c) + imul(3–5c) **per byte** ≈ 4 cycles/byte. One 128-byte heap is 4 column-rounds whose critical path is ~16 cycles each plus spins and fold ≈ 80–90 cycles ⇒ ~0.65 cycles/byte, with two independent squares riding in the same vector registers. That is ~6× analytically; I shade to 5.0 for benchmark overhead and because the `len < 32` regime can only reach ~2×, so a size-mixed harness will pull the mean down. Avalanche I predict near-ideal (score ≈ 0.5 / bias ≈ 0): 2 ChaCha double-rounds give full 16-word diffusion in-block and `fmix64` guarantees output avalanche regardless.

Falsifiable failure modes I'd accept as a loss: (a) if the harness is dominated by `len ≤ 32`, speedup lands near 1.5–2×; (b) if `-march=native` has no AVX2, the two squares cost two SSE2 ops each and the win shrinks toward 3–4×; (c) if gcc refuses to vectorize the lane loops at all, this degenerates to ~145 scalar ops / 128 B ≈ 1.1 c/B and still beats FNV, but only ~3×.

## MEASUREMENT

**Not measured by me.** `hash_bench` and `hash_contract` were not reachable in this session — no tools were available at all, so I ran nothing. I will not dress an estimate up as a result: the numbers above are an op-count/latency model, not data. The pipeline's compile-and-measure pass is the first observation, and if it reports under ~3× or a non-ideal avalanche score, the prediction above is simply wrong and should be recorded as wrong.

What I was able to check by hand, exhaustively, since I could not check it by tool:
- **Coverage** — every byte is read in all three regimes. `len<8`: explicit per-byte loop. `8≤len<32`: 8-byte chunks cover `0..8⌊(len-? )/8⌋`, the overlapping `rd64(data+len-8)` covers the rest (worst case `len=31`: loop covers 0–23, tail covers 23–30). `32≤len<128`: zero-padded, all of `len` copied. `len≥128`: whole heaps plus an overlapping final heap. No byte is ever skipped — which is the native's point about walking *every* mark.
- **No undefined shifts** — all `ROTL64` amounts are `(5j+1)&63 ∈ {1,6,11,16,21,26,31,36}` and `(11j+7)&63 ∈ {7,18,29,40,51,62,9,20}`; none is 0 or 64. `ROTL32` amounts are the literals 16/12/8/7.
- **No restrict violation** — `restrict` is confined to `absorb_heap`'s parameter; `kernel` deliberately does *not* alias `data` through a second restrict-qualified local, which is why the hot loop uses a scoped `p` and the overlapping tail uses `data` outside that scope.
- **Fixed points / ossification** — a per-heap counter enters row `d`, so a run of identical blocks yields different contributions, and the chain update `ROTL64(chain^f,27) + (f>>7) + K` is non-commutative, so permuting heaps changes the digest. These are exactly the weaknesses a bare multiply-xor accumulator has, and they are closed by the native's "never let it ossify" line rather than by me bolting something on.
- **`SPIN` inverse** — `SPIN(v,3)` undoes `SPIN(v,1)`, `SPIN(v,2)` is self-inverse; the diagonalize/undiagonalize pairing matches ChaCha's.

## VERDICT

The native's description decodes, object for object, into a real and validated design: **ChaCha-core ARX over a 4×4 square (two squares per heap), Merkle–Damgård chaining through a single counter-fed value, murmur3 `fmix64` as the seal** — with one deliberate departure from the validated original, the round budget cut from 10 double-rounds to 2, licensed by his own "throw the method away if a corner survives untouched" test plus the owl's finalizer. Three assumptions fall: per-byte mixing (the whole heap is laid down in silence first), single-accumulator state (32 lanes), and multiply-per-byte (the stride is add/rot/xor only; multiplication survives *only* in the owl, once per call). The in-order *dependency* falls too; the in-order *coverage* does not, and I'd rather say that plainly than claim a break I didn't get.

Risks I named above, and how each is discharged rather than shipped:
- *"Overhead if small"* — guarded, not hoped away: `len < 32` never builds a square at all, and `32 ≤ len < 128` needs only one padded heap instead of the 128-byte minimum. Both are size checks with a simpler fallback path, as the metaphor's own "heaps of even count" demands.
- *"Only helps if vectorized"* — degrades to ~3× scalar rather than to a loss, because even unvectorized the per-byte work is lower than a serial multiply chain. No intrinsics and no `#ifdef` ladder, so there is no ISA on which it fails to compile.
- *Thread parallelism* — deliberately **not** shipped. The metaphor's unit of work is 128 bytes and its chain is explicitly serial; adding OpenMP would be me overruling the native for a speedup the benchmark sizes almost certainly can't pay for. Dropped rather than guarded.

Where I expect to be judged wrong, if I am: the round budget. Two double-rounds is the aggressive call here, and it is the one knob whose failure would show up directly in the avalanche score rather than in throughput. If the measured avalanche is off-ideal, the fix the native himself prescribes is to go back to the riverbank and add one double-stride — not to touch anything else.