# MAPPING

| World object | Problem object | Assumption it touches |
|---|---|---|
| **SEED 1** — huge sheet | the byte buffer `data` | — |
| pile of marks, read in order, one at a time, never skipping/never looking ahead | the byte stream `data[0..len-1]`, traversed strictly left→right | reinforces "whole buffer read once, in order" |
| one fold per mark | one state-update step per byte | reinforces "each byte mixed before next is read" |
| fold's angle set by (mark, crease before it) | `state_i = f(byte_i, state_{i-1})` | reinforces "state is a single accumulator" — this *is* the FNV-1a pattern already given |
| **SEED 2** — four wire birds, beaks | four independent accumulator lanes `v1..v4` | breaks "single accumulator, one value" |
| press fold's corner against a bird's beak | update one lane with its own rotate/multiply rule | — |
| "all four beaks agree" → crease set | the four lanes are only reconciled/combined once, at merge time | — |
| refold until spiral/snail marks are "wrong on purpose, scrambled" | deliberate nonlinear mixing (rotate+multiply+xor) so distinct inputs diverge (avalanche, not collision) | avalanche requirement, not one of the 4 named assumptions |
| **SEED 3** — carry wad to water's edge, thin to one dense coin, discard scraps/mud | finalization: collapse internal state to the returned 64-bit value via a fixed, one-shot avalanche mix; intermediate lane states are never exposed | weakly touches "more rounds = better" (it's one thinning pass, not repeated), not one of the other named assumptions |

# CHOSEN SEED

Checking the target assumption first: **"each byte must be mixed into the running state before the next byte is read."** None of the three seeds breaks it — the passage's frame sentence ("read the pile of marks in order, one at a time, never skipping, never looking ahead") governs all of SEED 1, 2, and 3 equally; SEED 2's four birds validate *one* fold made from *one* mark, they don't let marks be read out of order or ahead of time; SEED 3 only acts after every mark is folded. **Stating this plainly: none of the seeds breaks assumption 1.**

Falling back to "most literal and most different from the known way": SEED 1 is literal but *is* the known way (FNV-1a is already `state = f(byte, prev_state)`). SEED 2 is equally literal (bird ↔ lane is a clean 1:1 mapping) and is structurally different from the single-accumulator FNV-1a example given. I choose **SEED 2**.

# ASSUMPTION BROKEN

"The state is a single accumulator updated in place, one value" — replaced by **four independent accumulator lanes**, combined only once at the end.

# ARTIFACT

Object mapping used in code: byte = `data[i]` (an 8-byte "word" when bulk-processed); state = four 64-bit lanes `v1..v4` (the four wire birds), each with its own rotate/multiply "beak" rule (`lane_round`); mixing = `acc += input*PRIME2; acc=rotl(acc,31); acc*=PRIME1` (the scrambling that keeps different piles from lining up); "all four beaks agree" = the merge step (`rotl64(v1,1)+rotl64(v2,7)+...` plus `merge_round`) done once after the bulk loop; "thin to one coin, discard the mud" = the fixed final avalanche (`xor-shift, *PRIME, xor-shift, *PRIME, xor-shift`) that collapses everything to the returned value, with no intermediate state surviving.

Per step 4, this is exactly xxHash64's published, validated design — I let SEED 2 arrive at that known technique rather than inventing a novel multi-lane scheme, since it already satisfies the broken assumption and is extensively benchmarked. Stated risk (VERDICT below): the finalization avalanche is fixed-cost work independent of `len`, so its relative overhead is largest for tiny inputs — I address this by keeping that cost O(1) (6 ALU ops) by construction rather than adding an untested small-input fallback branch that I have no way to validate in this tool-less session; I did not add OpenMP thread parallelism, since a single buffer's lane reduction is not embarrassingly parallel across threads without a tree-restructure I can't test here, and per the instructions the validated single-threaded technique is preferred over an invented threaded one.

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint32_t read32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

/* one bird's beak: how a single lane accepts a mark */
static inline uint64_t lane_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = lane_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const uint64_t seed = 0ULL;
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *const limit = end - 32;
        /* four wire birds */
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        /* each mark folded against its own bird, independently -
           no lane waits on another lane's multiply to finish */
        do {
            v1 = lane_round(v1, read64(p));      p += 8;
            v2 = lane_round(v2, read64(p));      p += 8;
            v3 = lane_round(v3, read64(p));      p += 8;
            v4 = lane_round(v4, read64(p));      p += 8;
        } while (p <= limit);

        /* "all four beaks agree": reconcile the lanes once */
        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = merge_round(h64, v1);
        h64 = merge_round(h64, v2);
        h64 = merge_round(h64, v3);
        h64 = merge_round(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = lane_round(0, read64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)read32(p) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    /* thin the wad to one dense coin - fixed O(1) cost, mud discarded */
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}
```

# PREDICTION

Single-accumulator FNV-1a serializes every multiply on the previous byte's result (dependency chain ≈ multiply latency × len). The four-lane version lets the CPU overlap the four independent multiplies per 32-byte block (latency-hidden, throughput-bound instead of latency-bound), and processes 8–32 bytes per "fold" instead of 1.

PREDICTION: speedup_vs_dp = 6.0

# MEASUREMENT

No tools were available in this session (explicitly stated in the environment note), so `hash_contract`/`hash_bench` could not be invoked here — the artifact above is handed to the external pipeline to compile (`gcc -O3 -march=native -fopenmp -lm`) and measure throughput + avalanche score against the FNV-1a baseline (`dp`). I have not iterated the "up to four times" improvement loop because there is no measured feedback to react to in this session; the single artifact above is my final, best-reasoned candidate rather than a guess dressed up as a measured result.

# VERDICT

The four-bird / four-lane mapping breaks the single-accumulator assumption by literal, direct correspondence, and it lands on xxHash64 — a validated, widely benchmarked technique — rather than an invented scheme, satisfying step 4's preference for known-good over novel-untested. Stated risk: the fixed 3-multiply/3-shift finalization ("thinning to one coin") is paid on every call regardless of `len`, so its relative overhead is largest on very small inputs (a handful of bytes); I address this by keeping that cost strictly O(1) (six ALU ops, no branching added) so it stays a fixed few nanoseconds rather than a scaling term — I deliberately did not bolt on an extra small-input fallback branch, since I have no way to validate such a branch without the measurement tools in this session, and an unverified branch is worse than a small, bounded, well-understood constant cost. No thread parallelism was added: the per-lane reduction is sequential across the whole buffer, the metaphor gives no natural large-grain parallel unit, and the validated real-world technique here (xxHash64) is itself single-threaded, so adding OpenMP would be exactly the kind of invented, untested mechanism step 4 warns against.