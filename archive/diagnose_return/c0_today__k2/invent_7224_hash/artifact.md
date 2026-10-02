# MAPPING

### SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."

| World object | Problem object |
|---|---|
| mark | one input byte `data[i]` |
| pile in its given order | the buffer `data[0..len)` |
| servant with the inflated house on a stick (keisaku) | the load phase: *no arithmetic at all* while reading |
| swat holding my mouth shut | the running state is **not** updated when a byte is read — silence = no accumulate-on-read |
| flat stone at the riverbank | a 4×4 lane grid; a byte's **position** in the grid (lane j, word offset) carries its order |
| "the order never gets spoken aloud and lost" | order is encoded *structurally* by placement + an injected `len`, not *temporally* by the sequence of folds |

**Assumption broken:** "each byte must be mixed into the running state before the next byte is read." And derivatively — because order now lives in *where* a byte lands, not *when* it is folded — "the whole buffer must be read once, start to end, in order": bytes may be read in 4 interleaved orders, and the tail may be re-read (overlapping) at zero semantic cost.

### SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."

| World object | Problem object |
|---|---|
| heap of even count / deer's fixed stride | a fixed 32-byte stripe (4 lanes × 8 bytes) |
| uphill twist | `rotl64` |
| downhill fold | `+` and `^` |
| doubling-back that eats its own trail | the add→rot→xor cross of a SipRound: carries move information up, xors fold it back |
| the one fixed thing: the final stone at the shrine carried into the next heap's racing | the chained state persists across stripes |
| "a calculation grown too satisfied with itself is a pillar that brings the roof down" | do **not** add rounds blindly; no all-zero fixed point; minimum rounds that pass the avalanche test |

**Assumption broken:** "mixing one byte requires a multiplication" — the entire compression is add/rotate/xor, multiplication-free. Also "more mixing rounds always means better mixing."

### SEED 3 — "The owl calls the final folded shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned at the shrine."

| World object | Problem object |
|---|---|
| owl taking the pen | the finalizer |
| dreamer inside the dreamer | nested avalanche: `owl(owl(h))` |
| sealed to one fixed size, same as any token | `uint64_t` return regardless of `len` |
| intermediate heaps burned the moment the next swallows them | no retained buffer; message words are overwritten, state is compressed |
| the butterfly who cannot recall being a man | the digest is not invertible back to the bytes |

**Assumption broken:** "the state is a single accumulator updated in place, one value" (it is a 16-word grid that only collapses to one value at the owl), plus "more rounds = better."

# CHOSEN SEED

**SEED 1.** It is the only one of the three that touches the preferred assumption — being plain about it: no seed breaks "read once, start to end, in order" *as its own subject*; SEED 1 breaks it **derivatively but necessarily**, because once order is fixed by placement rather than by the sequence of accumulations, in-order single-pass reading stops being required. It is also the most literal (silence-during-load ↔ no accumulate-on-read is a one-to-one correspondence) and the most distant from FNV-1a, whose entire identity is "mix each byte before reading the next."

The other two seeds are not discarded — they are the *mechanism* SEED 1 licenses, and the native tells one continuous story: silence during the load (S1), then heaps racing the deer's stride with the 90° square-turn (S2), then the owl (S3).

# ASSUMPTION BROKEN

- **Primary:** each byte must be mixed into the running state before the next byte is read. → Bytes are *placed* into 4 parallel lanes; nothing is mixed on read.
- **Consequent:** whole buffer read once, start to end, in order. → Four interleaved reading orders; the final stripe is re-read overlapping the already-read tail (`data+len-32`), so some bytes are read twice and out of order.
- **Also:** single accumulator (→ 4×4 grid), multiplication required (→ ARX only in compression), more rounds is better (→ 1 round per 8 bytes, 3 at the end, and no more).

**Where the mechanism lands on validated ground (step 4):** I did not invent a mixer. "Uphill twist, downhill fold, doubling-back that eats its own trail" with the explicit `rotl(·,32)` *is* **SipRound**; one round per 8-byte word plus three finishing rounds *is* **SipHash-1-3**, the default hasher in Rust's std, deployed at enormous scale. "Searching in squares, turning ninety degrees" *is* **ChaCha's column→diagonal lane rotation**. The owl *is* **murmur3's `fmix64`**. The construction is a 4-lane-parallel SipHash-1-3 with a ChaCha lane-turn and an fmix64 feed-out: three validated primitives, composed the way the native composes them.

**Regime recognition, in the metaphor's own words (step 5):** *"I gather the marks into small heaps of even count"* — if there are not enough marks to make even one heap, there is no racing, and the marks go straight from the stone to the owl. That is the runtime branch: `len < 32` takes a short path (overlapping first/last fixed-position loads + one owl, no grid, no rounds — the XXH3-style short ladder); `len >= 32` takes the racing path. Second regime pair: SIMD availability — AVX2 grid when present, bit-identical scalar grid fallback otherwise. No thread parallelism: the deer's stride is 32 bytes and the shrine stone chains, so OpenMP would be pure overhead at any plausible benchmark size — I am not shipping it.

**Risk I must guard rather than hand-wave:** the nested dreamer (6 multiplies) is overhead on very short inputs. Guarded — the second owl is only applied when `len >= 16`.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ================= riverbank: placement-only reads =================
   The servant's swat: during the load nothing is computed. A mark's
   order is carried by WHERE it lands (lane j, word offset), never by
   WHEN it is folded.                                                */
static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ================= the owl: murmur3 fmix64 ========================= */
static inline uint64_t owl(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

/* 16 distinct nothing-up-my-sleeve stones: BLAKE2b IV ++ SHA-512 K[0..7].
   None zero -> the all-zero grid is not a fixed point ("never let it ossify"). */
static const uint64_t SHRINE[16] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, /* row A */
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL, /* row B */
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, /* row C */
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL  /* row D */
};

/* ===== the shrine stone: fold the 4x4 square along its four diagonals.
   Every one of the 16 words appears exactly once -- no corner survives
   untouched -- then the dreamer inside the dreamer seals one token.   */
static inline uint64_t shrine_fold(const uint64_t *S, size_t len) {
    uint64_t d0 = S[0] ^ S[ 5] ^ S[10] ^ S[15];
    uint64_t d1 = S[1] ^ S[ 6] ^ S[11] ^ S[12];
    uint64_t d2 = S[2] ^ S[ 7] ^ S[ 8] ^ S[13];
    uint64_t d3 = S[3] ^ S[ 4] ^ S[ 9] ^ S[14];
    uint64_t h  = d0 + rotl64(d1, 13);
    h ^= rotl64(d2, 29);
    h += rotl64(d3, 47);
    h ^= (uint64_t)len;
    return owl(owl(h));
}

/* ---- scalar racing path (4 heaps, one SipRound each per 8-byte word) ---- */
#define SROUND() do { int j_; for (j_ = 0; j_ < 4; j_++) {                               \
    A[j_] += B[j_]; B[j_] = rotl64(B[j_],13); B[j_] ^= A[j_]; A[j_] = rotl64(A[j_],32);  \
    C[j_] += D[j_]; D[j_] = rotl64(D[j_],16); D[j_] ^= C[j_];                            \
    A[j_] += D[j_]; D[j_] = rotl64(D[j_],21); D[j_] ^= A[j_];                            \
    C[j_] += B[j_]; B[j_] = rotl64(B[j_],17); B[j_] ^= C[j_]; C[j_] = rotl64(C[j_],32);  \
} } while (0)

/* the 90-degree turn: lanes slide, so each heap's next race is run with a
   neighbour's shape -- cross-lane diffusion for the price of a shuffle. */
#define STURN() do { uint64_t t_;                                 \
    t_=B[0]; B[0]=B[1]; B[1]=B[2]; B[2]=B[3]; B[3]=t_;            \
    t_=C[0]; C[0]=C[2]; C[2]=t_; t_=C[1]; C[1]=C[3]; C[3]=t_;     \
    t_=D[3]; D[3]=D[2]; D[2]=D[1]; D[1]=D[0]; D[0]=t_;            \
} while (0)

#define SSTRIPE(q) do { int j_; uint64_t m_[4];                   \
    for (j_=0;j_<4;j_++) m_[j_] = ld64((q) + 8*j_);               \
    for (j_=0;j_<4;j_++) D[j_] ^= m_[j_];                         \
    SROUND();                                                      \
    for (j_=0;j_<4;j_++) A[j_] ^= m_[j_];                         \
} while (0)

#if defined(__AVX2__)
#define VROT(x,r)  _mm256_or_si256(_mm256_slli_epi64((x),(r)), _mm256_srli_epi64((x), 64-(r)))
#define VROT32(x)  _mm256_shuffle_epi32((x), 0xB1)
#define VROUND() do {                                                                   \
    vA = _mm256_add_epi64(vA,vB); vB = VROT(vB,13); vB = _mm256_xor_si256(vB,vA); vA = VROT32(vA); \
    vC = _mm256_add_epi64(vC,vD); vD = VROT(vD,16); vD = _mm256_xor_si256(vD,vC);        \
    vA = _mm256_add_epi64(vA,vD); vD = VROT(vD,21); vD = _mm256_xor_si256(vD,vA);        \
    vC = _mm256_add_epi64(vC,vB); vB = VROT(vB,17); vB = _mm256_xor_si256(vB,vC); vC = VROT32(vC); \
} while (0)
#define VTURN() do {                                       \
    vB = _mm256_permute4x64_epi64(vB, 0x39);  /* <<1 */     \
    vC = _mm256_permute4x64_epi64(vC, 0x4E);  /* <<2 */     \
    vD = _mm256_permute4x64_epi64(vD, 0x93);  /* <<3 */     \
} while (0)
#define VSTRIPE(q) do {                                                   \
    __m256i m_ = _mm256_loadu_si256((const __m256i *)(const void *)(q));  \
    vD = _mm256_xor_si256(vD, m_);                                        \
    VROUND();                                                             \
    vA = _mm256_xor_si256(vA, m_);                                        \
} while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * const p0 = data;

    /* ===== regime test, in the native's own terms: are there enough marks
       to make even one heap of even count? If not, no racing happens --
       the marks go from the stone straight to the owl. (Validated XXH3-style
       first/last overlapping ladder; every byte read at least once.)      */
    if (len < 32) {
        uint64_t a, b, x;
        if (len >= 16) {
            a = ld64(p0)            ^ rotl64(ld64(p0 + 8), 23);
            b = ld64(p0 + len - 16) ^ rotl64(ld64(p0 + len - 8), 41);
        } else if (len >= 8) {
            a = ld64(p0);
            b = ld64(p0 + len - 8);
        } else if (len >= 4) {
            a = (uint64_t)ld32(p0);
            b = (uint64_t)ld32(p0 + len - 4);
        } else if (len) {
            a = ((uint64_t)p0[0] << 16) | ((uint64_t)p0[len >> 1] << 8) | (uint64_t)p0[len - 1];
            b = 0;
        } else {
            a = 0; b = 0;
        }
        x = (a + 0x6a09e667f3bcc908ULL)
          ^ rotl64(b + 0xbb67ae8584caa73bULL, 32)
          ^ ((uint64_t)len * 0x9e3779b97f4a7c15ULL);
        x = owl(x);
        /* guard: the nested dreamer's extra multiplies are pure overhead on
           a handful of marks -- only pay for them once there are >= 16.    */
        if (len >= 16) x = owl(x ^ (uint64_t)len);
        return x;
    }

    /* ===== the racing path: 4 heaps abreast, 32-byte deer stride ===== */
    {
        uint64_t S[16];
        size_t rem = len;
        const unsigned char *p = p0;

#if defined(__AVX2__)
        __m256i vA = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 0));
        __m256i vB = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 4));
        __m256i vC = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 8));
        __m256i vD = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 12));
        int r_;

        while (rem >= 64) { VSTRIPE(p); VSTRIPE(p + 32); VTURN(); p += 64; rem -= 64; }
        if (rem >= 32)    { VSTRIPE(p); VTURN(); p += 32; rem -= 32; }
        /* the tail heap is read overlapping the marks already read: legal
           only because order lives in placement, not in reading sequence. */
        if (rem)          { VSTRIPE(p0 + len - 32); VTURN(); }

        vD = _mm256_xor_si256(vD, _mm256_set1_epi64x((long long)(uint64_t)len));
        vC = _mm256_xor_si256(vC, _mm256_set1_epi64x((long long)0xffULL));
        for (r_ = 0; r_ < 3; r_++) { VROUND(); VTURN(); }

        _mm256_storeu_si256((__m256i *)(void *)(S +  0), vA);
        _mm256_storeu_si256((__m256i *)(void *)(S +  4), vB);
        _mm256_storeu_si256((__m256i *)(void *)(S +  8), vC);
        _mm256_storeu_si256((__m256i *)(void *)(S + 12), vD);
#else
        uint64_t A[4], B[4], C[4], D[4];
        int j, r_;
        for (j = 0; j < 4; j++) { A[j] = SHRINE[j]; B[j] = SHRINE[4+j]; C[j] = SHRINE[8+j]; D[j] = SHRINE[12+j]; }

        while (rem >= 64) { SSTRIPE(p); SSTRIPE(p + 32); STURN(); p += 64; rem -= 64; }
        if (rem >= 32)    { SSTRIPE(p); STURN(); p += 32; rem -= 32; }
        if (rem)          { SSTRIPE(p0 + len - 32); STURN(); }

        for (j = 0; j < 4; j++) { D[j] ^= (uint64_t)len; C[j] ^= 0xffULL; }
        for (r_ = 0; r_ < 3; r_++) { SROUND(); STURN(); }

        for (j = 0; j < 4; j++) { S[j] = A[j]; S[4+j] = B[j]; S[8+j] = C[j]; S[12+j] = D[j]; }
#endif
        return shrine_fold(S, len);
    }
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 14**

(Stated before any measurement.) Reasoning: FNV-1a is a hard serial dependency of one `imul` (3c) + `xor` per byte ≈ 4–5 cycles/byte. The racing path consumes 32 bytes per vectorized SipRound; I count ≈26 vector uops per round, so ≈8–9 cycles/32 B ≈ 0.27 cycles/byte throughput-bound (critical path ≈6 cycles/round, so not latency-bound) — roughly 16–18× on bulk data, derated for finalization and for whatever fraction of the benchmark sits at small sizes. If the scalar fallback is taken, ≈2.2 ops/byte ≈ 0.55 c/byte ≈ 9×.

Avalanche prediction: ≈0.50 flip rate with small bias (expect a near-top score). Every input bit reaches all 16 grid words through ≥1 SipRound plus ≥3 finishing rounds with lane turns, each word enters `shrine_fold` exactly once on a distinct diagonal with a distinct rotation, and the output passes through `fmix64` twice — `fmix64` alone has measured bias under ~0.4%. A single-bit input change cannot cancel in the fold (its two contributions land at different bit offsets).

Falsifiers I would accept: speedup < 4× ⇒ my uop accounting is wrong or the benchmark is dominated by sub-32-byte inputs where the owl's latency floor (~18 cycles) is the whole cost; avalanche < 0.45 or a visibly biased bit ⇒ the 1-round-per-word compression plus lane-turn is not enough and the native's own test ("if even a corner survives untouched I throw the whole method away") has failed, in which case the honest fix is a second round per stripe, not a prettier fold.

# MEASUREMENT

**Not performed in this session — no tools were available.** `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all unreachable, so there is no throughput number and no avalanche score to report, and I will not fabricate one. The four allotted improvement rounds were therefore not spent; the artifact above is the *unmeasured first iteration*, hardened only by the static reasoning above and by using pre-validated primitives rather than by evidence.

What I verified by hand instead, so the pipeline's measurement is the only open question:
- Bounds: every load is in range for its branch (`len>=16`, `>=8`, `>=4`, `1..3`, `0` dereferences nothing; racing path requires `len>=32` for the overlapping `p0+len-32`).
- Lane-permutation immediates: `0x39`, `0x4E`, `0x93` are left-rotations by 1/2/3 under `result[i]=src[(imm>>2i)&3]`, and `STURN()` reproduces them exactly, so the AVX2 and scalar paths are bit-identical.
- `0xB1 = _MM_SHUFFLE(2,3,0,1)` swaps the 32-bit halves of each 64-bit element = `rotl64(x,32)`.
- `shrine_fold` touches indices 0–15 exactly once (checked each diagonal).
- No all-zero fixed point: all 16 initial stones are nonzero and distinct.

# VERDICT

The native's story translates with unusual fidelity, and the translation is *not* a novelty: holding silence during the load (SEED 1) is literally a 4-lane positional grid with no accumulate-on-read; the deer's uphill-twist/downhill-fold/trail-eating stride (SEED 2) is literally **SipRound**; searching in squares by turning ninety degrees (the drifting man) is literally **ChaCha's lane rotation**; the owl sealing one fixed-size token with the intermediate heaps burned (SEED 3) is literally a **`fmix64` feed-out over a discarded state**. Taking the metaphor seriously delivered 4-lane-parallel **SipHash-1-3 + ChaCha turn + fmix64** — three deployed, validated designs — which is exactly the outcome step 4 asks for, and it breaks the targeted assumption honestly: nothing is mixed on read, the buffer is traversed in four interleaved orders, and the tail is deliberately re-read out of order.

Stated conditions under which this is worse than FNV-1a, and what I did about each:
1. **Very short inputs.** The grid's fixed cost (3 finishing rounds + 2 owls) dwarfs the payload below ~32 bytes. **Guarded:** a `len < 32` branch skips the grid entirely, and the second owl is further gated to `len >= 16`.
2. **Pathologically tiny inputs (`len <= 2`).** FNV-1a there is ~5 cycles; my floor is the single `fmix64`, ~12 cycles. I cannot fix this without destroying avalanche (FNV-1a on one byte has essentially none), so I state it plainly rather than guard it away: at `len <= 2` this kernel is expected to be ~2× *slower* than the baseline, and that is the price of the avalanche.
3. **No AVX2.** **Guarded:** a bit-identical scalar path, still ~9× by op count.
4. **Thread parallelism.** Deliberately omitted: the metaphor's unit of work is a 32-byte stride and the shrine stone chains, so OpenMP cannot be justified at plausible benchmark sizes. I did not ship a mechanism I could not defend.

The one thing I cannot claim is the thing that matters most: **this has not been measured.** Until `hash_bench` runs, "14×" and "0.50 avalanche" are predictions from uop counting and primitive provenance, not results — and the native's own discipline applies to me too: if a corner of the output survives a flipped input bit, this goes back to the riverbank.

---

*Note on two session-start instructions I could not honor: the `brainny` CLI could not be probed (no shell/tool access in this session), so the ambient capture/recall/sync skills were skipped. Separately, the **claude.ai PubMed** connector needs authorization before its tools can be used — that has to be done from your claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session); it's unavailable until then. Neither affected this task.*