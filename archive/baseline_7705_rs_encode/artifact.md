## APPROACH

The reference re-derives the generator polynomial from scratch on every call (O(nsym²) GF multiplies, each done via an 8-step shift-and-reduce loop) and then, for every message byte, does up to `nsym+1` more shift-and-reduce multiplies. Since the harness fixes `n-k = 16`, the reducing polynomial (0x11D) and primitive element (2) are always the same, so the generator polynomial and every multiplier table are **compile-time-fixed constants that can be computed exactly once** and reused across all calls.

Key transformation of the algorithm itself: instead of the full `k+nsym` scratch buffer with overlapping in-place updates, I use the mathematically equivalent 16-byte LFSR/shift-register formulation of the same polynomial division (verified by hand against the reference for a small `k=2,nsym=2` case, being careful **not** to assume the generator is monic — the reference's `gen[nsym]` is not `1` in general, so the "new" register slot still needs a real multiply, not a free copy).

Speed comes from three independent wins:
1. **Amortized setup**: log/antilog tables and the generator polynomial are built once (thread-safe lazy init via atomics), not per call.
2. **Single-lookup multiply**: instead of the generic "2 lookups + mod-255 add" log/antilog trick, I precompute, for each of the 16 fixed generator coefficients, a full 256-entry multiplication-by-that-constant table (`MULGEN[t][coef]`). Each GF multiply needed in the hot loop becomes **one** table load, not two plus an add.
3. **Register-resident state, fully unrolled**: the 16-byte shift register is kept in 16 named scalars (not a memory array), so the "shift" is free (compiler renames registers) and the 16 table lookups per byte are mutually independent (good ILP), with only a short r0→next-r0 dependency chain across bytes.

A generic (slow-ish but correct) fallback handles the theoretical case `n-k != 16`, for contract safety, though the problem states this never happens in practice.

## ARTIFACT

```c
#include <string.h>

static unsigned char g_LOG[256];
static unsigned char g_EXP[512];
static unsigned char g_MULGEN[16][256]; /* g_MULGEN[t][c] = gf_mul(gen[1+t], c), t=0..15 -> gen[1..16] */
static unsigned char g_GEN17[17];
static volatile int g_state = 0; /* 0=uninit,1=building,2=ready */

static inline unsigned char gf_mul_slow(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        unsigned char hi = (unsigned char)(a & 0x80);
        a = (unsigned char)(a << 1);
        if (hi) a ^= 0x1D;
        b >>= 1;
    }
    return p;
}

static void build_tables(void) {
    unsigned char x = 1;
    for (int i = 0; i < 255; i++) {
        g_EXP[i] = x;
        g_EXP[i + 255] = x;
        g_LOG[x] = (unsigned char)i;
        unsigned char hi = (unsigned char)(x & 0x80);
        x = (unsigned char)(x << 1);
        if (hi) x ^= 0x1D;
    }
    g_LOG[0] = 0;

    const int nsym = 16;
    unsigned char gen[17];
    gen[0] = 1;
    for (int j = 1; j <= nsym; j++) gen[j] = 0;
    int glen = 1;
    unsigned char root = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char newgen[18];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            newgen[j + 1] ^= gf_mul_slow(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
        unsigned char hi = (unsigned char)(root & 0x80);
        root = (unsigned char)(root << 1);
        if (hi) root ^= 0x1D;
    }
    memcpy(g_GEN17, gen, 17);

    for (int t = 0; t < 16; t++) {
        unsigned char g = gen[1 + t];
        g_MULGEN[t][0] = 0;
        if (g == 0) {
            for (int c = 1; c < 256; c++) g_MULGEN[t][c] = 0;
        } else {
            unsigned char logg = g_LOG[g];
            for (int c = 1; c < 256; c++)
                g_MULGEN[t][c] = g_EXP[(int)g_LOG[c] + (int)logg];
        }
    }
}

static void ensure_init(void) {
    if (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) == 2) return;
    int expected = 0;
    if (__atomic_compare_exchange_n(&g_state, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        build_tables();
        __atomic_store_n(&g_state, 2, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) != 2) { /* spin */ }
    }
}

/* generic, correctness-only fallback for the (contractually unused) case n-k != 16 */
static void kernel_generic(int k, int nsym, const unsigned char *data, unsigned char *parity) {
    if (nsym <= 0) return;
    unsigned char gen[nsym + 1];
    gen[0] = 1;
    for (int j = 1; j <= nsym; j++) gen[j] = 0;
    int glen = 1;
    unsigned char root = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char newgen[nsym + 2];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            newgen[j + 1] ^= gf_mul_slow(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
        unsigned char hi = (unsigned char)(root & 0x80);
        root = (unsigned char)(root << 1);
        if (hi) root ^= 0x1D;
    }

    unsigned char reg[nsym];
    for (int t = 0; t < nsym; t++) reg[t] = (t < k) ? data[t] : 0;

    for (int i = 0; i < k; i++) {
        unsigned char coef = reg[0];
        unsigned char inbyte = (i + nsym < k) ? data[i + nsym] : 0;
        if (coef != 0) {
            unsigned char logc = g_LOG[coef];
            for (int t = 0; t < nsym - 1; t++) {
                unsigned char g = gen[1 + t];
                unsigned char contrib = (g == 0) ? 0 : g_EXP[(int)logc + (int)g_LOG[g]];
                reg[t] = reg[t + 1] ^ contrib;
            }
            unsigned char gN = gen[nsym];
            unsigned char contribN = (gN == 0) ? 0 : g_EXP[(int)logc + (int)g_LOG[gN]];
            reg[nsym - 1] = inbyte ^ contribN;
        } else {
            for (int t = 0; t < nsym - 1; t++) reg[t] = reg[t + 1];
            reg[nsym - 1] = inbyte;
        }
    }
    memcpy(parity, reg, (size_t)nsym);
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    ensure_init();

    if (nsym != 16) {
        kernel_generic(k, nsym, data, parity);
        return;
    }

    unsigned char r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    r0  = (0  < k) ? data[0]  : 0;
    r1  = (1  < k) ? data[1]  : 0;
    r2  = (2  < k) ? data[2]  : 0;
    r3  = (3  < k) ? data[3]  : 0;
    r4  = (4  < k) ? data[4]  : 0;
    r5  = (5  < k) ? data[5]  : 0;
    r6  = (6  < k) ? data[6]  : 0;
    r7  = (7  < k) ? data[7]  : 0;
    r8  = (8  < k) ? data[8]  : 0;
    r9  = (9  < k) ? data[9]  : 0;
    r10 = (10 < k) ? data[10] : 0;
    r11 = (11 < k) ? data[11] : 0;
    r12 = (12 < k) ? data[12] : 0;
    r13 = (13 < k) ? data[13] : 0;
    r14 = (14 < k) ? data[14] : 0;
    r15 = (15 < k) ? data[15] : 0;

    const unsigned char *T0 = g_MULGEN[0],  *T1 = g_MULGEN[1],  *T2 = g_MULGEN[2],  *T3 = g_MULGEN[3];
    const unsigned char *T4 = g_MULGEN[4],  *T5 = g_MULGEN[5],  *T6 = g_MULGEN[6],  *T7 = g_MULGEN[7];
    const unsigned char *T8 = g_MULGEN[8],  *T9 = g_MULGEN[9],  *T10 = g_MULGEN[10], *T11 = g_MULGEN[11];
    const unsigned char *T12 = g_MULGEN[12], *T13 = g_MULGEN[13], *T14 = g_MULGEN[14], *T15 = g_MULGEN[15];

    for (int i = 0; i < k; i++) {
        unsigned char coef = r0;
        unsigned char inbyte = (i + 16 < k) ? data[i + 16] : 0;
        if (coef) {
            unsigned char n0=T0[coef], n1=T1[coef], n2=T2[coef], n3=T3[coef];
            unsigned char n4=T4[coef], n5=T5[coef], n6=T6[coef], n7=T7[coef];
            unsigned char n8=T8[coef], n9=T9[coef], n10=T10[coef], n11=T11[coef];
            unsigned char n12=T12[coef], n13=T13[coef], n14=T14[coef], n15=T15[coef];
            r0 = r1 ^ n0;   r1 = r2 ^ n1;   r2 = r3 ^ n2;   r3 = r4 ^ n3;
            r4 = r5 ^ n4;   r5 = r6 ^ n5;   r6 = r7 ^ n6;   r7 = r8 ^ n7;
            r8 = r9 ^ n8;   r9 = r10 ^ n9;  r10 = r11 ^ n10; r11 = r12 ^ n11;
            r12 = r13 ^ n12; r13 = r14 ^ n13; r14 = r15 ^ n14;
            r15 = inbyte ^ n15;
        } else {
            r0=r1; r1=r2; r2=r3; r3=r4; r4=r5; r5=r6; r6=r7; r7=r8;
            r8=r9; r9=r10; r10=r11; r11=r12; r12=r13; r13=r14; r14=r15;
            r15 = inbyte;
        }
    }

    parity[0]=r0;  parity[1]=r1;   parity[2]=r2;   parity[3]=r3;
    parity[4]=r4;  parity[5]=r5;   parity[6]=r6;   parity[7]=r7;
    parity[8]=r8;  parity[9]=r9;   parity[10]=r10; parity[11]=r11;
    parity[12]=r12; parity[13]=r13; parity[14]=r14; parity[15]=r15;
}
```

PREDICTION: 8

## MEASUREMENT
Not run in this session — no tools were available to invoke `rs_encode_bench`/`rs_encode_contract` here, so I cannot report actual numbers. Based on the operation-count analysis (reference: per call rebuilds gen poly (~256×8-step multiplies) plus up to `k×17` 8-step shift-reduce multiplies; mine: one-time table build, then `k×16` single-lookup-and-xor register updates with the message byte kept register-resident) I expect on the order of a 6–10x wall-clock improvement for typical k, converging toward the prediction of ~8x, with the biggest relative gains for smaller k where the reference's per-call generator-polynomial rebuild is a larger fraction of total work.

## VERDICT
Correctness: derived and hand-verified the LFSR reformulation against the literal reference semantics for a small case (k=2, nsym=2), explicitly correcting an initial wrong assumption that the generator polynomial is monic (`gen[nsym]==1`) — it is not, per the reference's exact construction — and fixed both the fast path (16 tables, not 15) and the generic fallback accordingly before finalizing. Performance claim is a reasoned prediction from operation counts, not a measured result, since no benchmarking tool call was available in this session; treat the 8x figure as an estimate pending actual measurement.