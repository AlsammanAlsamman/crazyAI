## APPROACH

The reference implementation has three costly things per call that are not part of the actual math:

1. **`gf_mul` via shift-and-reduce** — an 8-iteration branchy loop per multiply, called `k*(nsym+1)` times.
2. **`malloc`/`calloc` + `free` every call**, plus zeroing the *whole* `k+nsym` buffer even though only the trailing `nsym` bytes need to start at zero.
3. **Rebuilding the generator polynomial from scratch every call**, even though `n-k` (hence the generator) never changes across calls in this harness.

None of these are required by the *algorithm* — they're artifacts of writing the reference for clarity. The actual data-dependency chain (message byte folded into the trailing window one at a time, most-significant token first) must stay sequential and bit-exact, so I keep exactly the same synthetic-division structure and update order as the reference (so results are bit-identical), but:

- Replace `gf_mul` with **log/antilog table lookups** (`gf_exp[log[a]+log[b]]`, with a 512-entry doubled exp table so no `%255` is needed).
- **Cache the generator polynomial** (and the logs of its nonzero coefficients) keyed on `nsym`, built once with the table-based multiply.
- **Reuse a growable static scratch buffer** instead of malloc/free each call, and only `memset` the `nsym`-byte tail instead of the whole buffer.
- **Specialize/unroll** the inner 17-term (`nsym==16`) loop into its own function so the compiler can fully unroll it with a compile-time-constant trip count, with a generic fallback for any other `nsym` (defensive, since the contract states `n-k` is fixed at 16, but I don't want to silently produce wrong output if that ever isn't true).

This keeps the exact same numerical recurrence as the reference (so correctness is a structural argument, not a hope), while removing the two genuinely wasteful parts (the multiply and the allocation).

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

/* ---- GF(256) log/antilog tables, primitive element 2, poly 0x11D ---- */
static unsigned char gf_log[256];
static unsigned char gf_exp[512];
static int gf_tables_ready = 0;

static void init_gf_tables(void) {
    unsigned int x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = (unsigned char)x;
        gf_log[x] = (unsigned char)i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11D;
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
    gf_log[0] = 0; /* never dereferenced meaningfully */
    gf_tables_ready = 1;
}

/* ---- generator polynomial cache, keyed on nsym ---- */
static unsigned char cached_gen[258];
static unsigned char cached_gennz[258];
static unsigned char cached_genlog[258];
static int cached_nsym = -1;

static void build_generator(int nsym, unsigned char *gen) {
    unsigned char newgen[258];
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_exp[i % 255];
        unsigned char rlog = gf_log[root];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            if (gen[j]) newgen[j + 1] ^= gf_exp[(unsigned int)gf_log[gen[j]] + rlog];
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

static void ensure_generator(int nsym) {
    if (!gf_tables_ready) init_gf_tables();
    if (cached_nsym == nsym) return;
    build_generator(nsym, cached_gen);
    for (int j = 0; j <= nsym; j++) {
        cached_gennz[j] = (unsigned char)(cached_gen[j] != 0);
        cached_genlog[j] = cached_gennz[j] ? gf_log[cached_gen[j]] : 0;
    }
    cached_nsym = nsym;
}

/* ---- reusable scratch buffer (avoid malloc/free every call) ---- */
static unsigned char *scratch = NULL;
static size_t scratch_cap = 0;

static unsigned char *get_scratch(size_t need) {
    if (need > scratch_cap) {
        size_t newcap = need * 2 + 32;
        unsigned char *p = (unsigned char *)realloc(scratch, newcap);
        if (!p) { p = (unsigned char *)realloc(scratch, need); newcap = need; }
        scratch = p;
        scratch_cap = newcap;
    }
    return scratch;
}

/* specialized, fully-unrollable path for the contractual nsym == 16 */
static void encode_nsym16(int k, unsigned char *msg) {
    for (int i = 0; i < k; i++) {
        unsigned char coef = msg[i];
        if (coef) {
            unsigned int lc = gf_log[coef];
            unsigned char *m = msg + i;
            const unsigned char *ge = gf_exp + lc;
            #pragma GCC unroll 17
            for (int j = 0; j <= 16; j++) {
                if (cached_gennz[j]) m[j] ^= ge[cached_genlog[j]];
            }
        }
    }
}

/* generic fallback for any other nsym (defensive; contract fixes nsym=16) */
static void encode_generic(int k, int nsym, unsigned char *msg) {
    for (int i = 0; i < k; i++) {
        unsigned char coef = msg[i];
        if (coef) {
            unsigned int lc = gf_log[coef];
            unsigned char *m = msg + i;
            const unsigned char *ge = gf_exp + lc;
            for (int j = 0; j <= nsym; j++) {
                if (cached_gennz[j]) m[j] ^= ge[cached_genlog[j]];
            }
        }
    }
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    ensure_generator(nsym);

    size_t total = (size_t)k + (size_t)nsym;
    unsigned char *msg = get_scratch(total);
    memcpy(msg, data, (size_t)k);
    memset(msg + k, 0, (size_t)nsym);

    if (nsym == 16) encode_nsym16(k, msg);
    else            encode_generic(k, nsym, msg);

    memcpy(parity, msg + k, (size_t)nsym);
}
```

## PREDICTION: 4

(Predicted speedup factor vs. the reference shift-and-reduce + malloc/free implementation, driven mainly by replacing the 8-step branchy `gf_mul` with two table lookups plus an add, eliminating per-call `malloc`/`calloc`/`free`, and caching the generator polynomial.)

## MEASUREMENT

No benchmarking tools were available in this session (the tool list for this run was empty aside from the doc/Gmail/Slack/Drive utilities — `rs_encode_bench` could not actually be invoked). I cannot report an empirically measured number, and I am not going to fabricate one. The reasoning behind the PREDICTION above is the only evidence I can offer:

- `gf_mul` reference: ~8 loop iterations with branches per multiply, done `k*(nsym+1)` times.
- This version: 1 table read (`gf_log[coef]`) per outer iteration, then per inner term just a compare + one table read + XOR — no branch-heavy bit loop.
- Removed: one `calloc`+`memset`-of-everything+`free` per call, and generator-polynomial reconstruction per call (both O(1) relative to k but with real constant-factor overhead, especially malloc/free for small k).

## VERDICT

Cannot confirm empirically in this session — no measurement tool was reachable — so this is a reasoned-but-unverified result, reported plainly as such rather than dressed up as a measured win. The implementation is designed to be bit-for-bit identical to the reference's synthetic-division recurrence (same update order, same per-token fold, most-significant first), so correctness risk should be low, but that too is an analytical claim, not a verified one, given no `rs_encode_contract` check could be run here.