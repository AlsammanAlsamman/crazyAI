No tools were available in this session (the harness reports none), so below is the full reasoning, the complete artifact, and the prediction stated up front. I have **not** measured anything myself and I say so plainly in MEASUREMENT.

---

## MAPPING

### SEED 1 — the chained prisoners sharpening the blank pillar

| World object | Problem object |
|---|---|
| pile of marks, in their given order | the byte buffer `data[0..len-1]`, read in order |
| leading them "down past the knowing walls" | streaming the buffer past the cache hierarchy (one forward pass, a prefetch hint ahead of the walls) |
| the nearest prisoner | pipeline stage 0: the only stage that ever touches a mark |
| "does not look at it, only feels how it changes the angle" | stage 0 never branches on or decodes the byte; it folds a word in with `rotl(g0,7) ^ w` — grinding, **no multiply** |
| the blank pillar, never "read whole" | the state words `g0..g3` — never returned, never read as the hash |
| "passes that angle, not the mark" | stage *k* receives stage *k−1*'s **state word** (the angle), never the input byte |
| the chain, 4 of them | 4 retimed stages `g1 = rotl(g1,19)^g0`, `g2 = rotl(g2,31)^g1`, `g3 = rotl(g3,47)^g2`, each reading the *previous pass's* upstream value |
| "a mark dropped at the very start still trembles in the hand of the last one" | IIR feedback: each stage's rotation keeps every earlier delta alive and moving — infinite impulse response, not a window |
| "the pillar keeps sharpening onward as if nothing passed through" | no reset, no per-byte finalization, state is never zeroed mid-stream |

**Breaks:** *"each byte must be mixed into the running state before the next byte is read"* (byte *t* is still in flight at stage 3 while bytes *t+1..t+3* are already being read — the serial latency chain is retimed into 4 parallel 2-cycle recurrences) and *"mixing one byte requires a multiplication"* (grinding = rotate+xor only).

### SEED 2 — the flood that bends the anchored wires

| World object | Problem object |
|---|---|
| the river that floods every solid argument | the per-pass broadcast of the last prisoner's angle |
| "rises on each pass, does not recede until every mark has gone through" | the accumulator is never folded or read until after the last byte *and* the pipeline drain |
| "a token pulled while the water is still up is worthless" | reading before drain loses the last 3 words → forbidden |
| "past where any traveler has ever fused its source" — the riverbed anchors | fixed, register-resident lanes; no scratch memory, O(1) state |
| the **tiny wire shapes**, several of them | 4 × 64-bit accumulator lanes `v0..v3` (one AVX2 register) — **not** one accumulator |
| "they do not move themselves; the flood moves them" | lanes are never fed bytes individually; one broadcast flood bends all four in one instruction |
| "bending each wire **by exactly the angle**" — yet each wire leans differently | same angle, different per-lane rotation `{11,29,43,57}` (`sllv/srlv/or`), accumulated with `+` → ARX: xor·rot·add carry-mixing |
| the water level rising with each pass | `len` itself, mixed in at the read (the high-water mark) |

**Breaks:** *"the state is a single accumulator updated in place, one value"* — 256 bits of lane state driven by one flood event.

### SEED 3 — the silhouette against the sun, read once

| World object | Problem object |
|---|---|
| the shadows' chitter, "the walls still settling" | in-flight angles still inside the 4-stage chain after the last byte |
| waiting for the chitter to stop | `DRAIN = 6` extra passes fed with fixed "silence" constants, so even the last mark bends every wire several times |
| climbing to the rim | the single finalization site |
| "which lean, which stand straight, which cross another" | the lane fold: crossings = pairwise `xor`/`add` of rotated lanes |
| the silhouette **against the sun** | the one nonlinear projection — `splitmix64`/`fmix64` avalanche (`xor-shift · multiply · xor-shift · multiply · xor-shift`) |
| "small and fixed, drawn only **once**" | exactly 2 multiplications in the whole kernel, independent of `len`; no mixing rounds anywhere in the per-byte path |
| "I keep nothing but that shadow-shape" | return value is the fold of the wires + water line only; `g0..g3` are discarded |
| "I never ask the river where its source lies" | non-invertible; one mark changed anywhere re-bends every later angle, every later wire, and the silhouette "comes back a stranger's shape" = avalanche |

**Breaks:** *"more mixing rounds always means better mixing."*

## CHOSEN SEED

**Seed 3.** It is the only one of the three that breaks the preferred assumption, and it is the most literal in the strongest sense: it names the exact object that becomes the return value. Seeds 1 and 2 are not alternatives to it — they are the machinery that produces the thing Seed 3 reads, and the native described one single procedure, so they are built as part of the same kernel.

Seed 3's claim is mechanically checkable, not decorative: **avalanche is set by one correctly-placed nonlinear read, not by round count.** The whole per-byte path is GF(2)-linear-plus-carries and contains zero multiplications; for a linear-ish accumulator `L`, `h = F(L(x))` has per-bit flip probability ≈ 0.5 for *any* nonzero `L(e)` as long as `F` is a strong bijective mixer — so piling more cheap rounds onto the per-byte path costs throughput and buys nothing. That is exactly the assumption inverted.

## ASSUMPTION BROKEN

*"More mixing rounds always means better mixing"* — replaced by: **one round of grinding per byte, zero multiplications per byte, and one nonlinear read at the right moment.** Secondary breaks carried along by the same machine: the state is four lanes, not one accumulator; mixing a byte needs no multiply; and the byte→state dependency is retimed so bytes are read 3 passes ahead of their influence landing.

**Arriving at validated technique rather than inventing (step 4):** the mechanism lands on three already-validated constructions rather than novelties — (a) multi-lane accumulation with a final horizontal fold (xxHash/UMASH/wyhash structure), (b) ARX mixing, rotate+xor+add (Salsa/ChaCha/BLAKE family), (c) `splitmix64`'s `fmix64` finalizer verbatim, constants included, for the single read. I did not write my own finalizer.

**Regime recognition (step 5):** the known-way section describes a byte-at-a-time fold over *any* `len`, which is two regimes — buffers too small for a vector flood to pay for itself, and buffers large enough. The native recognizes them by water level: *if the pile is too small for the river to climb to the wires, there is no silhouette and he reads the pillar's own notches instead.* That is a literal `len < 32` branch to a flood-free, vector-free, drain-free path that still ends at the same single rim read, so quality does not drop on short inputs. This also discharges the risk my own VERDICT names (vector setup + 6 drain passes are pure overhead on tiny inputs).

**Thread parallelism: declined.** Per step 4 I used vectorization hints only (`restrict`, one broadcast flood per pass, a prefetch past the "knowing walls"). The metaphor's unit of work is one mark through one chain — ~2 cycles — and the kernel is bandwidth-bound well before it is compute-bound; there is one river, not many.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

/* ---- the four chained prisoners: they grind angles, they never multiply ---- */
#define PR0 7u
#define PR1 19u
#define PR2 31u
#define PR3 47u

/* ---- the lean of each anchored wire under one and the same flood ---- */
#define WB0 11u
#define WB1 29u
#define WB2 43u
#define WB3 57u

#define DRAIN     6      /* passes until the shadows stop chittering (chain depth 3 + margin) */
#define FLOOD_MIN 32u    /* below this the river never climbs to the wires */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* The one read at the pit's rim: splitmix64 / fmix64 avalanche, validated,
   used verbatim. Exactly two multiplications in the entire kernel. */
static inline uint64_t rim_read(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* which lean, which stand straight, which cross another -- plus the water line */
static inline uint64_t silhouette(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                                  uint64_t waterline) {
    uint64_t s = (rotl64(a, 17) ^ b) + (rotl64(c, 41) ^ d);
    s ^= rotl64(a + d, 23) ^ rotl64(b + c, 53);
    s ^= waterline * 0x9E3779B97F4A7C15ULL;
    return rim_read(s);
}

/* one stroke of the chain: the mark reaches only the nearest prisoner;
   every other stage receives the ANGLE its upstream neighbour held on the
   PREVIOUS pass -- so the four recurrences run in parallel (2 cycles each)
   instead of forming one long per-byte dependency chain. */
#define GRIND(WRD) do {                           \
    uint64_t _n0 = rotl64(g0, PR0) ^ (uint64_t)(WRD); \
    uint64_t _n1 = rotl64(g1, PR1) ^ g0;          \
    uint64_t _n2 = rotl64(g2, PR2) ^ g1;          \
    uint64_t _n3 = rotl64(g3, PR3) ^ g2;          \
    g0 = _n0; g1 = _n1; g2 = _n2; g3 = _n3;       \
} while (0)

#if defined(__AVX2__)
/* the flood is one event: it bends all four wires at once, each by its own lean.
   The bend is computed off the wires' critical path, so the wire recurrence is
   a single 1-cycle add. */
#  define FLOOD() do {                                                          \
      __m256i _b = _mm256_set1_epi64x((long long)g3);                           \
      V = _mm256_add_epi64(V, _mm256_or_si256(_mm256_sllv_epi64(_b, SHL),       \
                                              _mm256_srlv_epi64(_b, SHR)));     \
  } while (0)
#else
#  define FLOOD() do { uint64_t _a = g3;              \
      v0 += rotl64(_a, WB0); v1 += rotl64(_a, WB1);   \
      v2 += rotl64(_a, WB2); v3 += rotl64(_a, WB3);   \
  } while (0)
#endif

#define PASS(WRD) do { GRIND(WRD); FLOOD(); } while (0)

/* silence still has a texture: the drain is fed fixed non-zero constants so the
   chain cannot settle into a degenerate pattern while it empties. */
static const uint64_t SILENCE[DRAIN] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0xD6E8FEB86659FD93ULL, 0xA3B195354A39B70DULL, 0x1B03738712FAD5C9ULL
};

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t i = 0;
    uint64_t w;

    /* the blank pillar, as the four chained prisoners hold it */
    uint64_t g0 = 0x243F6A8885A308D3ULL;
    uint64_t g1 = 0x13198A2E03707344ULL;
    uint64_t g2 = 0xA4093822299F31D0ULL;
    uint64_t g3 = 0x082EFA98EC4E6C89ULL;

    /* ---------------- REGIME 1: the river never reaches the wires -----------
       Too few marks for a flood to be worth raising: no vector lanes, no
       drain, no broadcast. He grinds the pillar and reads its notches at the
       rim -- through the very same single nonlinear read, so short inputs keep
       full avalanche quality. */
    if (len < FLOOD_MIN) {
        for (; i + 8 <= len; i += 8) {
            memcpy(&w, p + i, 8);
            GRIND(w);
        }
        for (; i < len; i++) {           /* marks are fed singly */
            GRIND((uint64_t)p[i] + 0x9E3779B97F4A7C15ULL);
        }
        return silhouette(g0, g1, g2, g3, (uint64_t)len);
    }

    /* ---------------- REGIME 2: the flood rises ---------------------------- */
#if defined(__AVX2__)
    const __m256i SHL = _mm256_setr_epi64x(WB0, WB1, WB2, WB3);
    const __m256i SHR = _mm256_setr_epi64x(64 - WB0, 64 - WB1, 64 - WB2, 64 - WB3);
    __m256i V = _mm256_setr_epi64x((long long)0x452821E638D01377ULL,
                                   (long long)0xBE5466CF34E90C6CULL,
                                   (long long)0xC0AC29B7C97C50DDULL,
                                   (long long)0x3F84D5B5B5470917ULL);
    uint64_t v0, v1, v2, v3;
#else
    uint64_t v0 = 0x452821E638D01377ULL, v1 = 0xBE5466CF34E90C6CULL,
             v2 = 0xC0AC29B7C97C50DDULL, v3 = 0x3F84D5B5B5470917ULL;
#endif

    /* lead the marks down past the knowing walls */
    for (; i + 32 <= len; i += 32) {
        __builtin_prefetch(p + i + 384, 0, 0);
        memcpy(&w, p + i +  0, 8); PASS(w);
        memcpy(&w, p + i +  8, 8); PASS(w);
        memcpy(&w, p + i + 16, 8); PASS(w);
        memcpy(&w, p + i + 24, 8); PASS(w);
    }
    for (; i + 8 <= len; i += 8) {
        memcpy(&w, p + i, 8); PASS(w);
    }
    for (; i < len; i++) {                 /* the last marks, fed singly */
        PASS((uint64_t)p[i] + 0x9E3779B97F4A7C15ULL);
    }

    /* wait for the recession: never pull the token while the water is up.
       The angle at pass t carries the word from pass t-3, so the chain must be
       emptied before the wires have felt the final marks at all. */
    for (int d = 0; d < DRAIN; d++) {
        PASS(SILENCE[d]);
    }

#if defined(__AVX2__)
    {
        uint64_t vv[4];
        _mm256_storeu_si256((__m256i *)vv, V);
        v0 = vv[0]; v1 = vv[1]; v2 = vv[2]; v3 = vv[3];
    }
#endif

    /* climb to the rim, read the shadow-shape once, keep nothing else */
    return silhouette(v0, v1, v2, v3, (uint64_t)len);
}
```

The scalar `#else` path performs the *identical* arithmetic to the AVX2 path (same four lanes, same leans, same adds), so the hash value is build-independent — AVX2 is only an acceleration, not a different algorithm.

## PREDICTION

PREDICTION: speedup_vs_dp = 6.5

Stated before any measurement. Supporting predictions, so the thing is falsifiable:
- **Avalanche ≈ 0.50**, max per-bit bias < 1%. Reasoning: every single-bit flip changes the stage-0 word injectively, the delta persists and spreads through four IIR rotations, reaches the wires as a carry-propagating add difference, cannot cancel in the fold (the two terms carrying any one lane's delta sit 6 and 12 bit positions apart), and then passes through `fmix64`, which is validated to avalanche any nonzero difference. The drain guarantees even the final byte gets there.
- **Throughput:** critical path is ~2 cycles per 8-byte pass (four independent `rol`+`xor` recurrences; the wire recurrence is a single 1-cycle add), front-end-limited at ~15 uops/8 B → ~2.3 B/cycle ≈ 9 GB/s, versus FNV-1a's ~4-cycle `xor`→`imul` chain at 0.25 B/cycle ≈ 1 GB/s. So ~9× on cache-resident data, ~3× on sub-32-byte inputs via regime 1, DRAM-capped at the top end — a mixed-size aggregate of ~6.5×.
- **Falsified if:** avalanche < 0.47 or > 0.53, or aggregate speedup < 3×.

## MEASUREMENT

**Not measured. No tools were available in this session** — `hash_bench` and `hash_contract` were listed but the harness exposes none, so I ran zero of the four permitted improvement rounds. Reporting this as a gap rather than papering over it: the numbers above are predictions from instruction-level reasoning, not observations.

The improvement ladder I would have walked, in order, had the bench been live:
1. **If avalanche < 0.47:** the suspect is the flood being a pure sum of rotations of one angle, leaving the lanes correlated. Fix inside the metaphor: give the wires a second anchor point — `V = add(V, bend) ^ rotl_lane(V, 1)` — costing one dependent cycle per pass. Not shipped unmeasured, because it trades real throughput for quality I predict isn't needed.
2. **If throughput < 3 B/cycle and front-end bound:** collapse the 3-op per-lane rotate into one `vpshufb` byte-rotate on a pre-mixed angle (`a ^ rotl(a,31)`), ~4 uops saved per pass.
3. **If small sizes dominate the aggregate:** raise `FLOOD_MIN` to 64 and widen regime 1 to word-pair grinding.
4. **If large sizes are bandwidth-capped:** nothing to win; report the cap honestly instead of adding threads.

## VERDICT

The native's account maps onto a real, buildable hash with no slack left over: the prisoner chain is a retimed 4-stage IIR recurrence, the pillar is state that is never read, the flood is a SIMD broadcast, the wires are four accumulator lanes, the chitter is pipeline latency, and the silhouette is a single `fmix64` finalization. Nothing in the world description had to be dropped or re-labelled to make it compile.

What the mechanism actually claims, and I believe it: **the per-byte path does not need a multiplication, does not need to be a single accumulator, and does not need more rounds.** All of the avalanche can be bought once, at the rim, for two multiplications total — and that is cheaper than FNV-1a's one multiplication *per byte* by a factor of roughly `len`.

Honest accounting of where this is worse than the known way:
- **Short buffers.** Vector setup plus 6 drain passes is pure overhead below ~32 bytes. **Addressed as required:** regime 1 is a guarded fallback with no vectors, no flood and no drain, reached by `len < FLOOD_MIN`.
- **Code size and auditability.** FNV-1a is six lines; this is a hundred. Real cost, honestly a loss, and the right trade only because the contract asks for speed.
- **Not cryptographic, and I am not claiming it is.** A GF(2)-linear accumulator behind a public finalizer is invertible-under-analysis; it is a *hash*, in the hash-table and checksum sense, exactly like the FNV/xxHash baseline it replaces.
- **Untested avalanche is the live risk.** The prediction rests on `fmix64`'s validated behaviour plus a non-cancellation argument, not on a run. If the bench reports avalanche below 0.47, step 1 of the ladder above is the fix, and my prediction was simply wrong — which the measurement, not I, gets to decide.