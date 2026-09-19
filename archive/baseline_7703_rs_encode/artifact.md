## APPROACH

The reference kernel does, for every message byte, up to `nsym+1` calls to `gf_mul`, each of which is an 8-iteration carry-less shift-and-reduce loop — i.e. roughly `k*(nsym+1)*8` primitive steps.

The described "known way" (log/antilog tables) turns each multiply into `log[a]+log[b] mod 255` then an `exp[]` lookup — 2 lookups + 1 conditional add per multiply, still done per-(byte, generator-coefficient) pair.

Since the generator polynomial `gen[]` is fixed for a given `nsym` (and the harness fixes `nsym=16`), each per-coefficient multiplier `gf_mul(gen[t], x)` is a **fixed linear map of x** that can be precomputed once into a 256-entry table per coefficient (`nsym` tables, 4 KB total for `nsym=16`). Then the whole per-byte parity update becomes `nsym` direct table lookups + XORs — no per-multiply log/exp math at all, and no branch on the feedback value (table[0]=0 handles the zero case for free).

I keep the standard LFSR formulation (equivalent to the reference's full-array polynomial reduction, just streamed through an `nsym`-byte shift register instead of a `k+nsym` array), fully unroll the register update for the fixed `nsym==16` hot path (16 scalar variables instead of an indexed array, maximizing ILP since each `r_j` update only depends on the just-computed `fb` and the *old* `r_{j+1}`), and cache the generator polynomial / multiplication tables across calls (keyed on `nsym`) so repeated benchmark invocations pay the O(nsym²) setup cost only once.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

static unsigned char GF_EXP[512];
static unsigned char GF_LOG[256];
static int gf_ready = 0;

static void init_gf_tables(void) {
    if (gf_ready) return;
    unsigned int x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = (unsigned char)x;
        GF_LOG[x] = (unsigned char)i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11D;
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
    GF_LOG[0] = 0;
    gf_ready = 1;
}

static inline unsigned char gf_mul_fast(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0) return 0;
    return GF_EXP[(int)GF_LOG[a] + (int)GF_LOG[b]];
}

static unsigned char gf_pow(unsigned char a, int e) {
    if (a == 0) return (e == 0) ? 1 : 0;
    int l = ((int)GF_LOG[a] * e) % 255;
    if (l < 0) l += 255;
    return GF_EXP[l];
}

static void gen_poly(int nsym, unsigned char *gen) {
    unsigned char newgen[80];
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            newgen[j + 1] ^= gf_mul_fast(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

#define MAXNSYM 64

static int cached_nsym = -1;
static unsigned char cached_mulTable[MAXNSYM + 1][256];

static void ensure_tables(int nsym) {
    init_gf_tables();
    if (cached_nsym == nsym) return;
    unsigned char gen[MAXNSYM + 2];
    gen_poly(nsym, gen);
    for (int t = 1; t <= nsym; t++) {
        unsigned char g = gen[t];
        unsigned char *row = cached_mulTable[t];
        if (g == 0) {
            memset(row, 0, 256);
        } else {
            int lg = GF_LOG[g];
            row[0] = 0;
            for (int xv = 1; xv < 256; xv++) row[xv] = GF_EXP[lg + GF_LOG[(unsigned char)xv]];
        }
    }
    cached_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    if (nsym <= 0) return;
    if (nsym > MAXNSYM) nsym = MAXNSYM; /* defensive; contract fixes nsym=16 */
    ensure_tables(nsym);

    if (nsym == 16) {
        const unsigned char *t1 = cached_mulTable[1],  *t2 = cached_mulTable[2],
                             *t3 = cached_mulTable[3],  *t4 = cached_mulTable[4],
                             *t5 = cached_mulTable[5],  *t6 = cached_mulTable[6],
                             *t7 = cached_mulTable[7],  *t8 = cached_mulTable[8],
                             *t9 = cached_mulTable[9],  *t10 = cached_mulTable[10],
                             *t11 = cached_mulTable[11],*t12 = cached_mulTable[12],
                             *t13 = cached_mulTable[13],*t14 = cached_mulTable[14],
                             *t15 = cached_mulTable[15],*t16 = cached_mulTable[16];
        unsigned char r0=0,r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0,
                      r8=0,r9=0,r10=0,r11=0,r12=0,r13=0,r14=0,r15=0;
        for (int i = 0; i < k; i++) {
            unsigned char fb = data[i] ^ r0;
            unsigned char n0=r1^t1[fb],  n1=r2^t2[fb],  n2=r3^t3[fb],  n3=r4^t4[fb];
            unsigned char n4=r5^t5[fb],  n5=r6^t6[fb],  n6=r7^t7[fb],  n7=r8^t8[fb];
            unsigned char n8=r9^t9[fb],  n9=r10^t10[fb],n10=r11^t11[fb],n11=r12^t12[fb];
            unsigned char n12=r13^t13[fb],n13=r14^t14[fb],n14=r15^t15[fb],n15=t16[fb];
            r0=n0; r1=n1; r2=n2; r3=n3; r4=n4; r5=n5; r6=n6; r7=n7;
            r8=n8; r9=n9; r10=n10; r11=n11; r12=n12; r13=n13; r14=n14; r15=n15;
        }
        parity[0]=r0; parity[1]=r1; parity[2]=r2; parity[3]=r3;
        parity[4]=r4; parity[5]=r5; parity[6]=r6; parity[7]=r7;
        parity[8]=r8; parity[9]=r9; parity[10]=r10; parity[11]=r11;
        parity[12]=r12; parity[13]=r13; parity[14]=r14; parity[15]=r15;
        return;
    }

    unsigned char reg[MAXNSYM];
    memset(reg, 0, (size_t)nsym);
    for (int i = 0; i < k; i++) {
        unsigned char fb = data[i] ^ reg[0];
        for (int j = 0; j < nsym - 1; j++) reg[j] = reg[j + 1] ^ cached_mulTable[j + 1][fb];
        reg[nsym - 1] = cached_mulTable[nsym][fb];
    }
    memcpy(parity, reg, (size_t)nsym);
}
```

PREDICTION: 5.0

## MEASUREMENT

No tools were available in this session to invoke `rs_encode_bench` / `rs_encode_contract`, so I could not execute the benchmark or the correctness check myself. The prediction above (≈5x speedup vs. the shift-and-reduce reference) is my estimate based on operation counting: the reference does `k*(nsym+1)` multiplies at ~8 shift/XOR/branch steps each, whereas this implementation does `k*nsym` single table-lookup+XOR pairs (tables cached across calls after the first), which should collapse the per-byte inner-loop cost by roughly an order of magnitude in raw op count, tempered by memory-access/ILP effects, hence the more conservative ~5x prediction rather than ~8x.

## VERDICT

Unverified in this session — correctness (bit-exact match with the reference systematic encoder for `nsym=16`) and the actual speedup number both need to be confirmed by running `rs_encode_contract` and `rs_encode_bench`, which were not accessible here. The algorithm is the standard "per-coefficient multiplication table, unrolled LFSR" technique used in production RS encoders (e.g. Wikiversity's `rs_encode_msg` streamed form combined with folded log/antilog tables), and I'm confident in its correctness by construction (it is a direct, non-lossy transformation of the given reference algorithm), but I am explicitly flagging that this claim is untested pending the pipeline's own compile/measure/verify pass.