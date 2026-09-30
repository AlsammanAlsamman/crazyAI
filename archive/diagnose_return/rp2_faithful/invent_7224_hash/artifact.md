## MAPPING

**SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."**

| World object | Problem object |
|---|---|
| a mark | one input byte `data[i]` |
| the pile "in their given order" | the buffer, index order = semantic order |
| the flat stone at the riverbank | a 128-byte staging slab (`uint64_t m[16]`) filled by `memcpy` before any arithmetic |
| the servant's swat holding my mouth shut | the barrier: **no mixing op issues until the whole heap is laid down** — the 128-byte load is one unbroken block-load |
| "the order never gets spoken aloud and lost" | order is carried by *placement* (lane index + block counter added into `s[12]`), not by the sequence of arithmetic |

*Breaks:* "each byte must be mixed into the running state before the next byte is read" **and** "the whole buffer must be read once, start to end, in order" (order survives as position-keying, so the read is block-wise and reorderable, not byte-serial).

**SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."**

| World object | Problem object |
|---|---|
| musk deer's fixed stride | fixed 128-byte block stride (and 32-byte stride in the short-ground regime) |
| heaps of even count | 16 lanes × 64 bits = a 4×4 grid |
| uphill twist | `rotl64` |
| downhill fold | `x ^= y` / carry-propagating `x += y` |
| doubling-back that eats its own trail | `x ^= x >> k` (self-consuming xorshift) |
| "no heap keeps the shape it entered with" | the whole grid is permuted, not an accumulator updated in place |

*Breaks:* "mixing one byte requires a multiplication" and "the state is a single accumulator updated in place, one value". **No multiply appears anywhere in the artifact.**

**SEED 3 — "The owl calls the final shape inside the dreamer inside and seals it to one fixed size, while every intermediate heap is burned."**

| World object | Problem object |
|---|---|
| the owl taking the pen | the finalizer |
| dreamer inside the dreamer | nested fold: 16 → 8 → 4 → 2 → 1 lanes |
| "one fixed size regardless of how many marks" | `uint64_t` return independent of `len` |
| heaps burned the moment the next swallows them | staging slab is overwritten, nothing retained; no scratch array survives |
| the butterfly who cannot recall the man | one-wayness; also: zero backing store, everything in registers |

*Breaks:* "more mixing rounds always means better mixing" — the native pays rounds **once, at the shrine**, not per byte.

## CHOSEN SEED

**SEED 1.** It is the only one whose literal reading directly contradicts the byte-serial traversal, and its mapping is the most concrete (a physical slab you fill before you are allowed to speak = a block-load barrier). Seeds 2 and 3 are not alternatives — they are the same native's continuation of seed 1 (what happens to a heap once laid, and how it is sealed), so building seed 1 honestly requires building all three. I have not switched mechanisms anywhere.

## ASSUMPTION BROKEN

Primary: **"the whole buffer must be read once, start to end, in order."** Here order is stored as *position* (lane index + heap counter folded into lane 12), so the traversal unit is a 128-byte heap, not a byte, and within a heap there is no ordering constraint at all — 16 lanes absorb simultaneously.

Secondary, unavoidably: **"each byte must be mixed before the next is read"** (the swat), **"single accumulator"** (a 1024-bit grid), **"mixing requires a multiplication"** (zero multiplies), **"more rounds is always better"** (one double round per heap; rounds are spent once at the shrine).

## ARTIFACT

Which code implements which part of the native's mechanism:

- `RIVERBANK[16]` — the flat stone; sixteen distinct fixed places a mark can be laid.
- `memcpy(m, data + i, 128)` then the xor loop — **the servant's swat.** Every mark of the heap is laid before a single mixing instruction fires. This is the literal silence.
- `s[12] += blk++` — the order spoken by placement, not by voice. Two identical heaps in different positions diverge.
- `QR(...)` — the musk deer's stride: `rotl64` is the uphill twist, `^=`/`+=` the downhill fold, and `a ^= a>>29; c ^= c>>37;` is the **doubling-back that eats its own trail** (right-shift diffusion, the one direction rotations and carries do not supply). No multiply.
- `COLROUND` then `DIAGROUND` inside `TURN90` — **searching in squares, turning ninety degrees.** Column quarter-rounds, then the grid re-read on its diagonals: corners folded into the centre.
- one `TURN90` per heap, four at the end — the native's refusal to let the calculation ossify: rounds are minimal in the hot loop and paid once at the shrine.
- the `x/y/z` cascade — **the dreamer inside the dreamer**, 16→8→4→2→1.
- `owl()` and `uint64_t` return — sealed small, same size however many marks came.
- `len < 128` branch — **regime recognition inside the metaphor.** When the ground is too short for the deer's full stride it gathers a smaller square (4 lanes, 32-byte strides) and pays only three square-rounds at the shrine, so the short-ground traveller is not taxed with a mountain-sized finalization. This is the guarded fallback for the small-input regime named in my verdict.
- No OpenMP, no threads: the ops-per-byte and the dependency-chain length come out at the same ~0.23 cyc/byte, so the single grid is already simultaneously issue-bound and latency-bound; a second racing heap would add register pressure and buy nothing. Deliberately not shipped.

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the flat stone at the riverbank: sixteen fixed places for marks ---- */
static const uint64_t RIVERBANK[16] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL
};

/* uphill twist */
static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* the musk deer's stride: uphill twist, downhill fold, doubling-back that
   eats its own trail.  No multiplication anywhere. */
#define QR(a,b,c,d) do {                                     \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 32);           \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 24);           \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 16);           \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 63);           \
    (a) ^= (a) >> 29;                                        \
    (c) ^= (c) >> 37;                                        \
} while (0)

/* searching the square: straight on ... */
#define COLROUND(s) do {                                     \
    QR((s)[0],(s)[4],(s)[ 8],(s)[12]);                       \
    QR((s)[1],(s)[5],(s)[ 9],(s)[13]);                       \
    QR((s)[2],(s)[6],(s)[10],(s)[14]);                       \
    QR((s)[3],(s)[7],(s)[11],(s)[15]);                       \
} while (0)

/* ... then turned ninety degrees, corners folded into the centre */
#define DIAGROUND(s) do {                                    \
    QR((s)[0],(s)[5],(s)[10],(s)[15]);                       \
    QR((s)[1],(s)[6],(s)[11],(s)[12]);                       \
    QR((s)[2],(s)[7],(s)[ 8],(s)[13]);                       \
    QR((s)[3],(s)[4],(s)[ 9],(s)[14]);                       \
} while (0)

#define TURN90(s) do { COLROUND(s); DIAGROUND(s); } while (0)

/* the owl's seal */
static inline uint64_t owl(uint64_t h) {
    h ^= h >> 32;
    h += rotl64(h, 27);
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *restrict data, size_t len)
{
    if (len < 128) {
        /* ---- short ground: the deer gathers a smaller square ---- */
        uint64_t a = RIVERBANK[0] ^ (uint64_t)len;
        uint64_t b = RIVERBANK[1];
        uint64_t c = RIVERBANK[2];
        uint64_t d = RIVERBANK[3] ^ rotl64((uint64_t)len, 32);
        uint64_t t[4];
        uint64_t blk = 0;
        size_t i = 0, rem;

        while (len - i >= 32) {
            memcpy(t, data + i, 32);          /* silence until all four are laid */
            a ^= t[0]; b ^= t[1]; c ^= t[2]; d ^= t[3];
            c += blk++;                       /* order fixed by placement */
            QR(a, b, c, d);
            QR(a, c, d, b);                   /* the ninety-degree turn */
            i += 32;
        }

        rem = len - i;
        t[0] = t[1] = t[2] = t[3] = 0;
        if (rem) memcpy(t, data + i, rem);
        a ^= t[0]; b ^= t[1]; c ^= t[2]; d ^= t[3];
        b ^= (uint64_t)rem + 0x9E3779B97F4A7C15ULL;
        d ^= (uint64_t)len;
        c += blk;

        QR(a, b, c, d); QR(a, c, d, b);       /* three square-rounds at the shrine */
        QR(a, b, c, d); QR(a, c, d, b);
        QR(a, b, c, d); QR(a, c, d, b);

        return owl((a + rotl64(b, 21)) ^ (c + rotl64(d, 43)));
    } else {
        /* ---- the mountain path: heaps of 128 marks, one square per heap ---- */
        uint64_t s[16];
        uint64_t m[16];
        uint64_t blk = 0;
        size_t i = 0, rem;
        int j;

        for (j = 0; j < 16; j++) s[j] = RIVERBANK[j];
        s[0] ^= (uint64_t)len;
        s[7] ^= rotl64((uint64_t)len, 32);

        while (len - i >= 128) {
            memcpy(m, data + i, 128);         /* the swat: whole heap laid, nothing spoken */
            for (j = 0; j < 16; j++) s[j] ^= m[j];
            s[12] += blk++;                   /* the order, written as place */
            TURN90(s);                        /* the heap races; the shrine carries it forward */
            i += 128;
        }

        rem = len - i;
        memset(m, 0, sizeof m);
        if (rem) memcpy(m, data + i, rem);
        for (j = 0; j < 16; j++) s[j] ^= m[j];
        s[12] += blk;
        s[13] ^= (uint64_t)rem + 0x9E3779B97F4A7C15ULL;
        s[15] ^= (uint64_t)len;

        /* again and again, until one changed mark has smeared across every square */
        TURN90(s); TURN90(s); TURN90(s); TURN90(s);

        {   /* the owl calls the shape inside the dreamer inside: 16 -> 8 -> 4 -> 2 -> 1 */
            uint64_t x0 = s[0] ^ rotl64(s[ 8], 13);
            uint64_t x1 = s[1] ^ rotl64(s[ 9], 23);
            uint64_t x2 = s[2] ^ rotl64(s[10], 31);
            uint64_t x3 = s[3] ^ rotl64(s[11], 41);
            uint64_t x4 = s[4] ^ rotl64(s[12], 47);
            uint64_t x5 = s[5] ^ rotl64(s[13], 53);
            uint64_t x6 = s[6] ^ rotl64(s[14],  7);
            uint64_t x7 = s[7] ^ rotl64(s[15], 17);
            uint64_t y0 = x0 + rotl64(x4, 21);
            uint64_t y1 = x1 + rotl64(x5, 29);
            uint64_t y2 = x2 + rotl64(x6, 37);
            uint64_t y3 = x3 + rotl64(x7, 43);
            uint64_t z0 = y0 ^ rotl64(y2, 25);
            uint64_t z1 = y1 ^ rotl64(y3, 51);
            return owl(z0 + rotl64(z1, 33));
        }
    }
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12.0

Reasoning stated before any measurement: FNV‑1a is a strict serial chain of `xor` (1 cyc) + `imul` (3 cyc latency) per byte ≈ 4 cyc/byte ≈ 0.75 GB/s at 3 GHz. The heap kernel issues 8 quarter-rounds × 16 ops = 128 ops per 128 bytes = 1.0 op/byte, with four *independent* quarter-rounds per round, so it is bound at roughly max(1.0/4 issue, 28 cyc / 128 B latency) ≈ 0.23 cyc/byte ≈ 13 GB/s scalar — consistent with measured scalar ChaCha (~0.45 cyc/byte per double round on 64‑byte blocks; we get double the bytes per double round). That is ~16×, which I discount to **12.0** for loop/load overhead, the mandatory 4-double-round shrine cost amortising poorly below ~1 KB, and the possibility that the benchmark buffer is RAM-resident and caps us near 12–15 GB/s. If GCC's SLP vectorizer catches the column/diagonal rounds under `-march=native`, the compute cost drops ~4× and the result will be memory-bandwidth-bound instead, which would push the number higher, not lower.

Avalanche prediction: **≈ 0.4995–0.5005 flip rate, essentially ideal.** Every input bit passes through at least 5 double rounds (10 BLAKE2b-style rounds) of a 1024-bit grid plus the nested fold; bits in early heaps pass through far more. The native's own acceptance test — flip one mark, demand that no corner survives — is exactly the strict-avalanche criterion, and 10 rounds of column+diagonal quarter-rounds is roughly 2.5× the depth at which this permutation family reaches full diffusion.

## MEASUREMENT

**Not performed.** No tools were available in this session — `hash_bench` and `hash_contract` could not be invoked, so there are zero measured numbers here and I am not going to manufacture any. The prediction above is written blind, from an instruction-count and dependency-chain argument, and the pipeline's measurement is the first real evidence either way. I used zero of my four allotted improvement iterations because I could not observe anything to improve against; the artifact is the first draft of the mechanism, reasoned but unmeasured.

Falsifiable claims, so the measurement can cleanly kill them:
1. speedup ≥ 8× on any buffer ≥ 1 KB. If it comes back below 4×, my issue-rate model is wrong (most likely cause: GCC spilling the 16-lane grid badly rather than scalarising it into registers).
2. avalanche score within 1% of ideal at every tested size, *including* the smallest. If the short-ground branch scores worse than the mountain branch, three square-rounds on four lanes was too few and the fix is a fourth square-round there — not a change of mechanism.

## VERDICT

The native's procedure translates without any fudging. Every object had a referent: the swat is a block-load barrier, the deer's stride is the block size, the ninety-degree turn is the column→diagonal alternation, the burned heaps are the absence of scratch storage, the owl is the fold. Nothing in the description had to be discarded as decoration, and — the part I take as the real signal — the description told me to spend rounds *once at the shrine* rather than per byte, which is precisely the inversion that makes the thing fast. The textbook method pays a 3-cycle multiply per byte on a chain it cannot break; this pays one cheap permutation per 128 bytes on sixteen chains it deliberately keeps separate.

**The named risk and how it is addressed.** The mechanism's fixed finalization cost is a tax that a byte-at-a-time hash does not pay, so on very short inputs it can lose. I have guarded it the way step 4 requires: the `len < 128` branch is a genuine size check that drops to a 4-lane square with a 32-byte stride and a 3-round shrine, cutting the fixed cost by roughly 4×. **Residual, stated plainly rather than hidden:** below about 24 bytes the two are at parity, and at 8 bytes FNV‑1a is probably still ~1.5× faster in latency, because ~6 quarter-rounds cannot be made cheaper than 8 multiply-xor steps. I chose to bound that gap rather than close it by splicing FNV‑1a in underneath, since that would be shipping the textbook method wearing the native's clothes for part of the domain. If the benchmark turns out to be dominated by sub-32-byte inputs, this kernel will lose and I would rather that show up in the number than be papered over.

**What I deliberately did not build.** Two racing heaps (dual grid) and OpenMP striping are both licensed by the metaphor and both tempting. I dropped them because the arithmetic says the single grid is already simultaneously issue-bound and latency-bound at ~0.23 cyc/byte — a second grid doubles register pressure without reducing ops/byte, and threads cannot pay for their ~10 µs launch below ~1 MB. If measurement shows the kernel is latency-bound rather than issue-bound (i.e. GCC vectorised the rounds and throughput is well under 0.06 cyc/byte of issue), then a second racing heap is the correct next iteration and it is a ten-line change; that is the one place I would spend improvement budget.

---

*Note, unrelated to the task: the `claude.ai PubMed` MCP server is not authorized in this environment, and this session is non-interactive so the OAuth flow cannot be run here. If you need it, authorize it from your claude.ai connector settings (or via `claude mcp` / `/mcp` in an interactive session); until then those tools are unavailable.*