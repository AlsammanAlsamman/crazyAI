## MAPPING

| World object (native's words) | Computational object | Why it is literal |
|---|---|---|
| the pile of marks, in their given order | `data[0..len-1]`, read in order | bytes, order-dependent |
| the desk's numbered groove | the byte offset / `len` itself | the groove is *numbered*: the length is known before he starts |
| ship's cycle-light, same starlight color no matter the hour | fixed nothing-up-my-sleeve init constants (π digits), no seed, no time dependence | deterministic, hourless |
| the bone-colored **die-stone** | a multi-**faced** state: 8 × 64-bit lanes (`A` = 4 lanes, `B` = 4 lanes) | a *die* is by definition a body with several faces that **turns**; not one number |
| "press each mark against it, never lifting it off the desk" | state is never re-initialized between blocks | no reset |
| "drop the next mark's weight into the **same seated place**" | `A = A + W` — the group of marks is **added** into the one fixed rate-face | the injection site is fixed; the stone moves under it |
| "let the stone's own memory of the last press **bend how deep this one goes**" | **addition, not xor** — carry propagation means the current contents change how far this weight travels | xor would be "a clean face"; add is literally a depth bent by what's already there |
| "**turn it a quarter** through the chalk-bank groove" | rotation: `rotl64(·,32)` / `rotl64(·,24)` **plus** a cyclic lane permutation `_mm256_permute4x64_epi64` | a quarter turn is a rotation of a faced body — not a multiply |
| "one fold for every mark, centuries the same as a handful" | fixed-size state, one streaming pass, O(1) memory | a sponge, not a tree |
| "no two folding-paths that started differently ever walk the same last step" | invertible round ⇒ zero entropy loss; differential trails never re-merge | strict avalanche |
| "wooden **numbered keeps** at the desk's edge" | the squeeze: rotate-and-XOR fold of the 8 lanes into one `uint64_t` | a separate, fixed reading apparatus at the edge |
| "groove-dust, intermediate turns, chalk residue — swept off and thrown away" | **capacity discarded**: 512 bits of state, 64 bits emitted | only a truncation of the final state leaves |
| "a handful of marks" vs "centuries of marks" | runtime regime test `len < 128` → small four-faced stone; else wide eight-faced stone | the groove is numbered, so he *sees* which pile he has before pressing |

**Which silent assumption each SEED breaks**

- **SEED 1** (never resets, every fold carries the callus of all prior folds) → breaks *"more mixing rounds always means better mixing"*: because nothing washes clean, one cheap fold per block suffices and the mixing budget belongs at the **end**, not per byte. It does **not** break the single-accumulator assumption — FNV already never resets.
- **SEED 2** (pressed into the stone's **already-turned** position, not a clean face) → breaks **"the state is a single accumulator updated in place, one value"** *and* **"mixing one byte requires a multiplication"**. A die-stone that *turns* is a multi-faced body; a quarter turn is a rotation + lane permutation; the byte lands on a different face each time because the stone moved, not because the arithmetic changed.
- **SEED 3** (only the final seated number leaves; residue discarded) → also breaks **"the state is a single accumulator updated in place, one value"**: if only part of the state is ever read, the state must be *wider* than the output. That is exactly a sponge capacity.

## CHOSEN SEED

**SEED 2** — "Each mark's weight is pressed into the stone's already-turned position rather than onto a clean face."

Two seeds (2 and 3) break the preferred assumption; SEED 2 is the more literal and the more different from the known way, because it dictates the *whole* inner loop (turning faced body + add-injection + no multiply), and SEED 3 falls out of it for free as the output rule (I implement both).

## ASSUMPTION BROKEN

**"the state is a single accumulator updated in place, one value."**

Also broken, as a consequence: *"mixing one byte requires a multiplication"* (there is **not one multiply instruction** in this kernel — only add, rotate, xor, and lane permutation), and *"each byte must be mixed into the running state before the next byte is read"* (32 marks are seated in one press).

**Step 4 — let the mechanism arrive at a validated technique, don't invent.** Followed literally, "never-reset faced stone + fixed injection site + quarter turn between injections + only a narrow final reading leaves" **is the sponge construction with an ARX permutation** — Keccak/Ascon/Xoodoo/BLAKE2's family. I did not invent a new primitive:

- **Small pile (`len < 128`)**: the four-faced stone with quarter turns is *exactly* **SipHash-1-3** (state `v0..v3`, word pressed into `v3`, quarter turns = the SipRound rotations, only `v0^v1^v2^v3` leaves the desk). Validated: Rust's default hasher, SMHasher-clean, built for short keys. I use it verbatim with a fixed key.
- **Large pile (`len ≥ 128`)**: the same mechanism widened to an eight-faced stone, with the quarter turn expressed as AVX2 `shuffle_epi32` (rot 32), `shuffle_epi8` (rot 24), and `permute4x64` (the lane turn) — a BLAKE2b/Xoodoo-shaped ARX sponge with rate 256 / capacity 256.

**Step 5 — regime recognition inside the metaphor.** The desk's groove is *numbered*: the native sees the pile length before pressing. `len < 128` ⇒ the small four-faced stone (fast for a handful, where the wide stone's fixed ten finishing turns would dominate); otherwise the wide stone. This is also the guard demanded by step 4 for my own stated risk ("the wide stone loses on short piles").

**No thread parallelism, deliberately.** The metaphor forbids it: *one* stone, *one* desk, each fold carrying the callus of the one before. Splitting across threads means several stones and breaks mechanism #2. SIMD is allowed because it is one stone with more faces turning together. So: vectorization only, per the instruction's default.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ============================================================
   "THE TURNING STONE"
   A never-reset, multi-faced die-stone. Each group of marks is
   ADDED (pressed, depth bent by what is already seated) into one
   fixed face; then the stone takes a quarter turn (rotate + turn
   the faces round). Nothing is ever washed clean. At the end the
   stone keeps turning with nothing dropped in, and only its final
   seated number -- read against the wooden numbered keeps -- ever
   leaves the desk. No multiplication anywhere.
   ============================================================ */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> ((64 - r) & 63));
}

/* ---------- the numbered groove says: a handful of marks ----------
   The small four-faced stone. This is SipHash-1-3 verbatim
   (fixed key): v3 is the seated face, SipRound is the quarter
   turn, only v0^v1^v2^v3 leaves the desk.                        */

#define SIPROUND                                                     \
    do {                                                             \
        v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);\
        v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;                     \
        v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;                     \
        v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);\
    } while (0)

static uint64_t stone_small(const unsigned char *p, size_t len) {
    const uint64_t k0 = 0x736f6d6570736575ULL;
    const uint64_t k1 = 0x646f72616e646f6dULL;
    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;
    size_t blocks = len & ~(size_t)7;
    size_t i, rem;
    uint64_t m, t;

    for (i = 0; i < blocks; i += 8) {
        memcpy(&m, p + i, 8);
        v3 ^= m;            /* press into the seated face   */
        SIPROUND;           /* quarter turn                 */
        v0 ^= m;
    }
    rem = len - blocks;
    t = 0;
    if (rem) memcpy(&t, p + blocks, rem);   /* little-endian host */
    m = t | ((uint64_t)(len & 0xff) << 56); /* the groove's number */
    v3 ^= m; SIPROUND; v0 ^= m;

    v2 ^= 0xff;
    SIPROUND; SIPROUND; SIPROUND;           /* the wooden keeps    */
    return v0 ^ v1 ^ v2 ^ v3;
}

/* ---------- the numbered groove says: centuries of marks ---------- */

#if defined(__AVX2__)

#define VROT(x, r)  _mm256_or_si256(_mm256_slli_epi64((x), (r)),      \
                                    _mm256_srli_epi64((x), 64 - (r)))
/* the faces turn by one: new[0]=old[3], new[1]=old[0], ...          */
#define TURN_FACES(x) _mm256_permute4x64_epi64((x), _MM_SHUFFLE(2,1,0,3))

/* one fold: press 32 marks into the seated face, then quarter turn.
   rot32 and rot24 are single-cycle shuffles on x86.                 */
#define FOLD_FAST(W)                                                   \
    do {                                                               \
        A = _mm256_add_epi64(A, (W));                                  \
        B = _mm256_xor_si256(B, A);                                    \
        B = _mm256_shuffle_epi32(B, _MM_SHUFFLE(2,3,0,1)); /* rotl 32 */\
        A = _mm256_add_epi64(A, B);                                    \
        A = _mm256_shuffle_epi8(A, R24);                   /* rotl 24 */\
        B = TURN_FACES(B);                                             \
    } while (0)

/* a finishing turn: the stone turns with nothing dropped in          */
#define FOLD_ROUND(r1, r2)                                             \
    do {                                                               \
        B = _mm256_xor_si256(B, A);                                    \
        B = VROT(B, (r1));                                             \
        A = _mm256_add_epi64(A, B);                                    \
        A = VROT(A, (r2));                                             \
        B = TURN_FACES(B);                                             \
    } while (0)

static uint64_t stone_wide(const unsigned char *p, size_t len) {
    const __m256i R24 = _mm256_setr_epi8(
        5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12,
        5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12);
    /* starlight colour: digits of pi, the same no matter the hour */
    __m256i A = _mm256_setr_epi64x((long long)0x243f6a8885a308d3ULL,
                                   (long long)0x13198a2e03707344ULL,
                                   (long long)0xa4093822299f31d0ULL,
                                   (long long)0x082efa98ec4e6c89ULL);
    __m256i B = _mm256_setr_epi64x((long long)0x452821e638d01377ULL,
                                   (long long)0xbe5466cf34e90c6cULL,
                                   (long long)0xc0ac29b7c97c50ddULL,
                                   (long long)0x9216d5d98979fb1bULL);
    size_t n = len & ~(size_t)31;
    size_t i;
    uint64_t t[8];

    for (i = 0; i < n; i += 32) {
        __m256i W = _mm256_loadu_si256((const __m256i *)(p + i));
        FOLD_FAST(W);
    }
    {   /* the last, partial handful, swept level with chalk */
        unsigned char tail[32];
        __m256i W;
        memset(tail, 0, sizeof tail);
        if (len - n) memcpy(tail, p + n, len - n);
        tail[len - n] = 0x80;                 /* len-n is 0..31 */
        W = _mm256_loadu_si256((const __m256i *)tail);
        FOLD_FAST(W);
    }

    /* the groove's own number goes in, then the stone keeps turning */
    A = _mm256_xor_si256(A, _mm256_set1_epi64x((long long)len));
    FOLD_ROUND(32, 24); FOLD_ROUND(16, 37); FOLD_ROUND(53, 19);
    FOLD_ROUND(32, 24); FOLD_ROUND(11, 43); FOLD_ROUND(29, 47);
    FOLD_ROUND(32, 24); FOLD_ROUND(23, 59); FOLD_ROUND(41, 13);
    FOLD_ROUND(32, 24);

    /* read the final seated position against the wooden keeps;
       everything else is swept off the desk and thrown away      */
    _mm256_storeu_si256((__m256i *)t, A);
    _mm256_storeu_si256((__m256i *)(t + 4), B);
    return t[0]              ^ rotl64(t[1],  8) ^ rotl64(t[2], 16)
         ^ rotl64(t[3], 24)  ^ rotl64(t[4], 32) ^ rotl64(t[5], 40)
         ^ rotl64(t[6], 48)  ^ rotl64(t[7], 56);
}

#else  /* no AVX2: the same eight-faced stone, turned by hand */

#define SFOLD(w0,w1,w2,w3,r1,r2)                                       \
    do {                                                               \
        a0 += (w0); a1 += (w1); a2 += (w2); a3 += (w3);                \
        b0 ^= a0; b1 ^= a1; b2 ^= a2; b3 ^= a3;                        \
        b0 = rotl64(b0,(r1)); b1 = rotl64(b1,(r1));                    \
        b2 = rotl64(b2,(r1)); b3 = rotl64(b3,(r1));                    \
        a0 += b0; a1 += b1; a2 += b2; a3 += b3;                        \
        a0 = rotl64(a0,(r2)); a1 = rotl64(a1,(r2));                    \
        a2 = rotl64(a2,(r2)); a3 = rotl64(a3,(r2));                    \
        { uint64_t tt = b3; b3 = b2; b2 = b1; b1 = b0; b0 = tt; }      \
    } while (0)

static uint64_t stone_wide(const unsigned char *p, size_t len) {
    uint64_t a0 = 0x243f6a8885a308d3ULL, a1 = 0x13198a2e03707344ULL;
    uint64_t a2 = 0xa4093822299f31d0ULL, a3 = 0x082efa98ec4e6c89ULL;
    uint64_t b0 = 0x452821e638d01377ULL, b1 = 0xbe5466cf34e90c6cULL;
    uint64_t b2 = 0xc0ac29b7c97c50ddULL, b3 = 0x9216d5d98979fb1bULL;
    size_t n = len & ~(size_t)31, i;
    uint64_t w[4];

    for (i = 0; i < n; i += 32) {
        memcpy(w, p + i, 32);
        SFOLD(w[0], w[1], w[2], w[3], 32, 24);
    }
    {
        unsigned char tail[32];
        memset(tail, 0, sizeof tail);
        if (len - n) memcpy(tail, p + n, len - n);
        tail[len - n] = 0x80;
        memcpy(w, tail, 32);
        SFOLD(w[0], w[1], w[2], w[3], 32, 24);
    }
    a0 ^= (uint64_t)len; a1 ^= (uint64_t)len;
    a2 ^= (uint64_t)len; a3 ^= (uint64_t)len;
    SFOLD(0,0,0,0,32,24); SFOLD(0,0,0,0,16,37); SFOLD(0,0,0,0,53,19);
    SFOLD(0,0,0,0,32,24); SFOLD(0,0,0,0,11,43); SFOLD(0,0,0,0,29,47);
    SFOLD(0,0,0,0,32,24); SFOLD(0,0,0,0,23,59); SFOLD(0,0,0,0,41,13);
    SFOLD(0,0,0,0,32,24);

    return a0 ^ rotl64(a1,  8) ^ rotl64(a2, 16) ^ rotl64(a3, 24)
         ^ rotl64(b0, 32) ^ rotl64(b1, 40) ^ rotl64(b2, 48) ^ rotl64(b3, 56);
}
#endif

/* ---------- the numbered groove: which pile is this? ---------- */

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *__restrict p = data;
    if (len < 128) return stone_small(p, len);   /* a handful of marks */
    return stone_wide(p, len);                   /* centuries of marks */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Stated before any measurement (no tools are available in this session; the pipeline will measure it).

Reasoning behind the number, so it can be falsified cleanly:

- Reference FNV-1a is a serial dependent chain of `imul` (3-cycle latency) + `xor` per **byte** ⇒ ≈ 4 cycles/byte.
- The wide stone's critical path per fold is `add → xor → shuffle → add → shuffle` ≈ 5 cycles for **32 bytes** ⇒ ≈ 0.16 cycles/byte when resident in L1/L2, i.e. ~25× on paper. Discounting for memory bandwidth on large buffers, the fixed 11-fold finisher, and the small-pile path (SipHash-1-3 ≈ 1.0–1.5 cycles/byte, only ~3–4× FNV), I predict a measured **≈12×** overall, with **>20×** if the benchmark is L1/L2-resident and **≈4×** if buffers are under 128 bytes.
- Avalanche: I predict a **near-ideal score (bit-flip probability ≈ 0.50)** on both paths. Small path: SipHash-1-3 is SMHasher-clean, so this is an inherited guarantee, not a hope. Wide path: every step is invertible (no entropy loss ever), the face-turn gives full cross-lane coverage every 4 folds, and the last block still sees 11 folds (~2.75 full cross-lane passes) with non-byte-aligned rotations before it is read.

## MEASUREMENT

**Not measured — no tools were available in this session.** I will not invent numbers. The prediction above is on the record before the fact; the falsification criteria are explicit:

- If measured `speedup_vs_dp` lands in **8–20** and avalanche ≈ 0.5, the prediction holds.
- If speedup is **< 4**, my cycle accounting for the fold's critical path is wrong (most likely cause: the benchmark is DRAM-bandwidth bound, or the buffers are short enough that the `len < 128` path dominates).
- If avalanche is **materially below 0.5** while speed is high, the failure is in the wide path's finisher, and the honest fix is more finishing turns (they are a fixed cost, so this trades almost no large-buffer throughput) — *not* bolting a multiply finalizer on, which would re-import the very assumption I am breaking.

## VERDICT

The core of this kernel **is** the native's mechanism, not a multiply-accumulator with mechanism-flavoured commentary. Concretely, and checkable by reading the code:

1. **One stone, never reset** — `A`/`B` are initialised once and carry every prior fold's callus to the end. No per-block re-seeding, no independent lanes hashed separately and combined.
2. **Pressed into an already-turned position** — the injection site is *fixed* (`A = A + W`); what changes between marks is that the stone has **turned** (`shuffle_epi32` / `shuffle_epi8` = the quarter turn in bits, `permute4x64_epi64` = the faces turning round). The press is **addition**, so carries from earlier marks literally bend how deep this one goes; xor would have been the "clean face" the native explicitly rejects. There is **zero** multiplication in the kernel.
3. **Only the final seated number leaves** — 512 bits of state, 64 bits emitted; 448 bits of capacity are swept off the desk. That discarded capacity is what makes the narrow output well-behaved, and it is the structural break with "one accumulator, one value."

Where this could still lose, and what I did about it, per step 4:

- **Risk: the wide stone loses on short piles**, because its eleven finishing turns are a fixed cost. **Addressed, not hand-waved**: the numbered groove is read at runtime (`len < 128`) and short piles go to the small four-faced stone — SipHash-1-3, a validated real-world instance of this exact mechanism, purpose-built for short keys. I did not ship a kernel whose stated risk is unguarded.
- **Risk: inventing an unvalidated primitive.** Addressed by letting the metaphor land on the sponge/ARX family rather than a novelty: the small path is SipHash verbatim; the wide path's round is BLAKE2b/Xoodoo-shaped (add, byte-aligned rotate, lane permute) with a 1:1 rate:capacity ratio and 10 blank finishing turns — conservative by sponge standards.
- **Risk: threads.** Rejected on the metaphor's own terms — several stones would break the callus chain of mechanism #1 — and on the instruction's default of SIMD-before-threads. No OpenMP is used.
- **Honest caveat:** the kernel assumes a little-endian host (true for the stated `gcc -march=native` x86 target) and non-AVX2 builds take a hand-turned scalar path that will be perhaps 3–5× slower than the vector one. Both facts are visible in the code, not buried.

*(Note: the `claude.ai` and `PubMed` MCP connectors in this environment are unauthorized and no tools were callable in this session; nothing above depended on them.)*