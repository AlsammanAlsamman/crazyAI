# MAPPING

The three seeds describe one machine. Here is each seed's world→problem table, taken literally.

## SEED 1 — counting-bars + eight strings under one bridge = the mixing engine

| World object | Problem object |
|---|---|
| the counting-bars, "the bars alone hold true count" | the fixed-width 64-bit machine word (mod 2⁶⁴ arithmetic) — and `len`, the true count, known *before* the first pluck |
| eight violin strings **strung over the landscape** | 8 independent 64-bit accumulator lanes `s0..s7`, striped across the whole buffer: lane *l* takes byte *l* of every 8-byte block |
| "I pluck the string nearest the mark's own weight" | the lane a mark lands in is fixed by which feather weighed it (feather index ≡ lane index ≡ `i mod 8`) — a pure index, no branch |
| "never softly, **always the same force**" | one single constant-cost operation per byte (`+=`), branch-free, data-independent control flow |
| "the strings are bound under **one bridge**, a shudder in one bends the pitch of all the rest **before it settles**" | the **carry chain of the 64-bit adder**: a disturbance in one byte-string propagates up through all the strings above it and settles. That is why accumulation is `+` and not `^` |
| the shudder travelling all the way *around* the instrument | one rotation per lane per 64-byte super-block, carrying the top strings' shudder back down to the bottom strings |
| "I catch them mid-tremor and carry that tremor forward" | lanes are never reset or flushed; the state is live and carried across the whole buffer |
| **Assumption broken** | "the state is a single accumulator updated in place, one value" — and "each byte must be mixed into the running state before the next byte is read" (8 independent dependency chains; bytes *i+1…i+7* are mixed without waiting for byte *i*) |

## SEED 2 — the feather's ink-weight, fresh and never repeated = strict order

| World object | Problem object |
|---|---|
| the inkwell that never runs dry | a 16 KiB L1-resident constant table, filled once from a splitmix64 stream; never consumed, never depleted |
| "the feather does not record the mark's **shape**, only its **weight**" | the byte value is never arithmetically operated on — it is only an *index*. `byte → random 64-bit weight` = **table lookup instead of multiply** |
| "pigment sinks differently for every stroke" | 256 unrelated random weights per feather; neighbouring byte values give totally unrelated weights (no arithmetic locality) |
| **a fresh child-feather, no two marks ever share a feather** | the feather for byte *i* is selected so that no two positions in the buffer share an effective weight function: table index = `(block + lane) mod 8` (a Latin square — within a 64-byte super-block every (lane, table) pair is unique), and successive super-blocks are separated by a per-lane rotation, so rotate∘add is non-commutative and position is never forgotten |
| "fixes strict order into the running count" | provably order-sensitive: each lane's update is a composition of bijections (add-weight, rotate), so no two distinct orderings collapse |
| **Assumption broken** | "mixing one byte requires a multiplication" — zero multiplications in the body |

## SEED 3 — the storm's single pass, then the pickers

| World object | Problem object |
|---|---|
| "Only when the last mark has been plucked do I **wait — truly wait, hands off**" | *all* strong mixing is deferred: nothing expensive happens per byte, and the finalizer is reached only after the last byte |
| the storm passes **once over the strings** | exactly **one** finalization round (Murmur3 `fmix64`), applied once, total — not per byte, not per block |
| the storm channels "the sky's dissolving pigment" | the multiply/xor-shift cascade `h^=h>>33; h*=C1; h^=h>>33; h*=C2; h^=h>>33` — the only 2 multiplications in the entire kernel, *O(1)* of them |
| "dissolves it down into **eight small notes, no more**" | the 8 lanes are folded into exactly 8 bytes = the 64-bit token |
| the pickers take apart every discarded feather and used drop | no scratch buffer, no retained intermediates: every per-mark weight dies in a register |
| "so nothing of the original order survives in the world above" | the finalizer is one-way; the token is not invertible back to the lane pattern |
| **Assumption broken** | **"more mixing rounds always means better mixing"** — the body performs the *minimum conceivable* mixing per byte (one add of a random word), and there is exactly ONE strong round in the whole kernel, at the very end |

---

# CHOSEN SEED

**SEED 3** — the storm's single pass. It is the one seed of the three that breaks the preferred assumption, and it is the architectural decision that makes the kernel fast: the native does not mix as he goes, he *accumulates* as he goes and mixes once.

I state plainly that SEED 3 cannot be implemented alone — the storm must have strings to read and weighed feathers to have produced them — so SEEDs 1 and 2 supply the body (8 lanes under one carry-bridge; fresh-feather ink-weights). But the *choice* that distinguishes this kernel from FNV-1a/xxHash is SEED 3's: zero mixing rounds in the loop, one at the end.

# ASSUMPTION BROKEN

Primary: **"more mixing rounds always means better mixing."** The body has no mixing round at all — one table-load and one `add` per byte. One terminal round suffices because of SEED 2: each byte's contribution is already a *full-width random 64-bit word*, so there is nothing left for per-byte rounds to spread; they would only re-stir already-uniform ink.

Secondary, carried along by the mapping: multiplication-per-byte (SEED 2), and single-accumulator / strict serial dependency (SEED 1).

**Step 4 check — this mechanism lands on a validated known technique, not an invention.** Table-lookup-per-byte accumulation with per-position tables *is* **simple tabulation hashing** (Zobrist 1970; Pătraşcu–Thorup), a real, analysed, provably 3-independent, multiplication-free hash family. The terminal storm is **Murmur3 `fmix64`/splitmix64's finalizer**, a validated avalanche mixer. I let the native's feathers and storm arrive at those two rather than inventing new constants or a new finalizer. The only thing not off the shelf is the metaphor's own wiring (8-lane Latin-square feather assignment + carry-bridge + per-super-block rotation), which is what gives strict order and ILP.

**Regime recognition, in-metaphor (step 5).** "The bars alone hold true count" — the native reads the count *before* plucking. A pile too short to span eight strings is plucked on a **single** string: stringing eight strings and summoning a storm to fold them costs more than five marks are worth. So `len < 32` → single-lane path (this is also the explicit guard for the risk my own verdict names). `len ≥ 64` → the 64-byte super-block loop. `32 ≤ len < 64` and the remainder → the 8-byte block loop, then the ≤7 last marks.

**No thread parallelism.** The metaphor's unit of work is one mark — a few cycles — and "the storm waits for *all* the strings"; threading would also violate "never skipping, never doubling back." Thread spawn (µs) dwarfs any benchmark buffer that fits in cache. I use cache-layout/ILP hints only (64-byte-aligned tables, 8 independent chains, 64-byte unroll). SIMD gather (`vpgatherqq`) was considered and rejected: its throughput (~5–8 cycles per 8 elements) is no better than 8 scalar L1 loads on 3 load ports, and it costs the carry-bridge.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* ================= the inkwell that never runs dry ================= */
static inline uint64_t ink_drop(uint64_t *s) {          /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ===== eight feathers x 256 ink-weights: 16 KiB, L1-resident =====
   the feather records not the mark's shape but its weight: a byte is
   never multiplied, only used as an index.  (= simple tabulation /
   Zobrist hashing)                                                  */
static uint64_t FEATHER[8][256] __attribute__((aligned(64)));
static int feathers_inked = 0;

static void ink_the_feathers(void) {
    uint64_t s = 0x243F6A8885A308D3ULL;                 /* pi */
    for (int f = 0; f < 8; ++f)
        for (int v = 0; v < 256; ++v)
            FEATHER[f][v] = ink_drop(&s);
    feathers_inked = 1;
}
#if defined(__GNUC__)
__attribute__((constructor)) static void ink_before_main(void) { ink_the_feathers(); }
#endif

/* ===== the storm: ONE pass over the final tremor, no more =====
   Murmur3 fmix64 -- the only two multiplications in the kernel.     */
static inline uint64_t storm_pass(uint64_t h) {
    h ^= h >> 33;  h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;  h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}

/* one 8-byte block = eight plucks, one per string, same force each.
   feather index (J+l)&7 is a Latin square: inside a 64-byte
   super-block no two marks ever share a feather.                    */
#define BLOCK(J)                                                      \
    do {                                                              \
        const unsigned char *q = p + 8 * (J);                         \
        s0 += FEATHER[((J) + 0) & 7][q[0]];                           \
        s1 += FEATHER[((J) + 1) & 7][q[1]];                           \
        s2 += FEATHER[((J) + 2) & 7][q[2]];                           \
        s3 += FEATHER[((J) + 3) & 7][q[3]];                           \
        s4 += FEATHER[((J) + 4) & 7][q[4]];                           \
        s5 += FEATHER[((J) + 5) & 7][q[5]];                           \
        s6 += FEATHER[((J) + 6) & 7][q[6]];                           \
        s7 += FEATHER[((J) + 7) & 7][q[7]];                           \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    if (!feathers_inked) ink_the_feathers();            /* belt + braces */

    const unsigned char *p = data;

    /* ---- the bars report the count before the first pluck: a short
       pile does not get eight strings strung across the landscape ---- */
    if (len < 32) {
        uint64_t h = 0x9E3779B97F4A7C15ULL + (uint64_t)len;
        for (size_t i = 0; i < len; ++i)
            h = ROTL64(h, 13) + FEATHER[i & 7][p[i]];   /* one string */
        return storm_pass(h);                           /* one storm   */
    }

    /* ---- eight strings, never let go still ---- */
    uint64_t s0 = 0x243F6A8885A308D3ULL + (uint64_t)len;
    uint64_t s1 = 0x13198A2E03707344ULL + (uint64_t)len;
    uint64_t s2 = 0xA4093822299F31D0ULL + (uint64_t)len;
    uint64_t s3 = 0x082EFA98EC4E6C89ULL + (uint64_t)len;
    uint64_t s4 = 0x452821E638D01377ULL + (uint64_t)len;
    uint64_t s5 = 0xBE5466CF34E90C6CULL + (uint64_t)len;
    uint64_t s6 = 0xC0AC29B7C97C50DDULL + (uint64_t)len;
    uint64_t s7 = 0x3F84D5B5B5470917ULL + (uint64_t)len;

    size_t n = len;

    /* ---- the long landscape: 64 marks a breath, never doubling back ---- */
    while (n >= 64) {
        BLOCK(0); BLOCK(1); BLOCK(2); BLOCK(3);
        BLOCK(4); BLOCK(5); BLOCK(6); BLOCK(7);
        /* the shudder finishes its circuit of the bridge: the top
           strings' tremor comes back around onto the bottom ones.
           all amounts odd -> full 64-step orbit per lane.            */
        s0 = ROTL64(s0,  7); s1 = ROTL64(s1, 13);
        s2 = ROTL64(s2, 19); s3 = ROTL64(s3, 29);
        s4 = ROTL64(s4, 37); s5 = ROTL64(s5, 43);
        s6 = ROTL64(s6, 53); s7 = ROTL64(s7, 59);
        p += 64; n -= 64;
    }

    /* ---- whole blocks left (at most 7) ---- */
    size_t j = 0;
    while (n >= 8) {
        s0 += FEATHER[(j + 0) & 7][p[0]];
        s1 += FEATHER[(j + 1) & 7][p[1]];
        s2 += FEATHER[(j + 2) & 7][p[2]];
        s3 += FEATHER[(j + 3) & 7][p[3]];
        s4 += FEATHER[(j + 4) & 7][p[4]];
        s5 += FEATHER[(j + 5) & 7][p[5]];
        s6 += FEATHER[(j + 6) & 7][p[6]];
        s7 += FEATHER[(j + 7) & 7][p[7]];
        p += 8; n -= 8; ++j;
    }

    /* ---- the last marks (at most 7), each still its own feather ---- */
    if (n) {
        uint64_t t = 0;
        for (size_t k = 0; k < n; ++k)
            t = ROTL64(t, 9) + FEATHER[k & 7][p[k]];
        s7 += t;
    }

    /* ---- the storm reads the whole shivering pattern and dissolves
       it into eight small notes, no more ---- */
    uint64_t a = s0 + ROTL64(s1, 17);
    uint64_t b = s2 + ROTL64(s3, 23);
    uint64_t c = s4 + ROTL64(s5, 31);
    uint64_t d = s6 + ROTL64(s7, 41);
    a ^= ROTL64(b, 13);
    c ^= ROTL64(d, 47);
    uint64_t h = a + ROTL64(c, 29) + (uint64_t)len;
    return storm_pass(h);
}
```

(Byte→lane assignment is little-endian-order by construction; on a big-endian host the token values differ but every property below holds unchanged.)

# PREDICTION

**PREDICTION: speedup_vs_dp = 5.0**

Stated before any measurement, with the cycle budget it comes from:

- *Reference (FNV-1a, "dp"):* `h ^= b; h *= prime` is a **serial latency chain** — 1 cycle xor + 3 cycle `imul` = **≈4 cycles/byte**, independent of ILP.
- *This kernel, long regime:* per byte = 1 `movzx` (load port) + 1 `add rX, [table + idx*8]` (load port + ALU) ⇒ 2 load µops + 1 ALU µop per byte, plus 8 rotates per 64 bytes. On 3 load ports that is **≈0.67–0.8 cycles/byte**, and the 8 lanes mean *no* latency chain is on the critical path. ⇒ **5–6× on large buffers.**
- *Short regime (len<32):* ~2 cycles/byte serial (rotate+add) + ~12 cycles for the single storm ⇒ still ~1.5–2× faster than FNV-1a at len=8–31, never slower. A benchmark averaging small and large sizes should land near 5×, hence the point prediction rather than the 6× large-buffer figure.
- *Avalanche:* I predict **near-ideal (flip ratio ≈ 0.500, score at or near the tool's maximum)**, and the claim is stronger than statistical hope: flipping one input bit changes that byte's weight by a nonzero difference of two independent random table words; each lane update (`add weight`, `rotate`) is a **bijection**, so the lane's final value provably differs; the fold reaches `h` from every lane through adds/xors of rotations such that a single changed lane always changes `h`; `fmix64` then avalanches that difference to ≈32 flipped output bits. Risk of a *weak* avalanche is therefore not "unlikely" but structurally excluded, for every input length including `len=0,1`.

# MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were not available in this session — there are no tools in this environment at all, so I ran nothing. I will not dress up the analysis above as a measurement: the prediction of 5.0× and of near-ideal avalanche is **untested**, and the pipeline's numbers, not mine, decide it.

What I can report honestly is the static accounting that is checkable by inspection rather than by running:

| quantity | reference (FNV-1a) | this kernel |
|---|---|---|
| multiplications per byte | 1 | **0** |
| multiplications total | `len` | **2** (plus 2·`len` once, at table ink time, amortised over the process) |
| strong mixing rounds | `len` | **1** |
| independent dependency chains | 1 | **8** |
| critical-path latency per byte | ~4 cycles | ~0 (throughput-bound, ~0.7–0.8 c/B) |
| working set beyond the data | 0 | 16 KiB (half of a 32 KiB L1) |

Falsifiable places to look when it *is* measured: (1) if the reported speedup is ≪3×, the 16 KiB feather set is thrashing L1 against the streaming data — the fix is 4 feathers (8 KiB) at the cost of some order-distinctness; (2) if the avalanche score is not near maximal, the fold, not the body, is the suspect, since the body's guarantee is a bijection argument.

# VERDICT

The native's machine translates with no substitutions: *eight strings under one bridge* = 8 lanes under one carry chain; *the feather's ink-weight* = tabulation hashing; *the single storm* = one `fmix64`. Nothing was quietly replaced by the textbook method — in particular the loop contains **no multiply**, which is the exact opposite of FNV-1a/xxHash, and that is the native's doing, not mine.

The honest shape of the claim:

- **Where it should win:** buffers ≥ 64 bytes, by breaking the per-byte multiply latency chain into 8 multiplication-free chains. This is a known-good technique, so the win is credible rather than speculative.
- **Where it could lose, and how that is already guarded:** (a) *small inputs* — eight strings plus a storm-fold have fixed cost, so `len < 32` falls back to a single-string path with no 8-lane setup and no fold (guard present in the code, not merely noted); (b) *L1 pressure* — the 16 KiB feather set is real overhead the reference does not pay, and this is the one condition under which this kernel can genuinely lose: if the caller's hot working set already fills L1, the table loads become L2 hits (~14 cycles, pipelined) and the advantage shrinks toward ~2×. I cannot guard that with a size check because it depends on the *caller's* footprint, not on `len`, so I state it as a live risk rather than hiding it; (c) *thread parallelism* — deliberately not shipped, because the metaphor's unit of work is one mark and the storm must wait for every string.
- **What I would not claim:** that this beats xxHash3/wyhash on large buffers. Those read 8–32 bytes per multiply and reach ~0.1–0.2 c/B; a one-lookup-per-byte machine cannot go below ~0.33 c/B on three load ports. The native's design beats the *stated* reference comfortably and is multiplication-free with provable order-sensitivity and provable per-bit avalanche — it does not beat the modern wide-multiply hashes, and saying otherwise would be dishonest.