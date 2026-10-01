#include <stdint.h>
#include <stddef.h>

#if defined(__x86_64__)
  #define CUP_X86 1
  #include <immintrin.h>
  #if defined(__SSE4_2__)
    #define CUP_TGT
  #else
    #define CUP_TGT __attribute__((target("sse4.2")))
  #endif
#endif

/* ---------------- the seven organs ----------------
   The cup's last colour is carried through seven bendings. Each bending is
   invertible and each throws away half of what it received: a right xorshift
   discards the half shifted out and keeps only the bits that refuse to sit
   still (those differing from their shifted partner); a 64x64 multiply
   discards the high half of the 128-bit product. Seven, not more: the count
   was closed at the garden door, where output bits flip at exactly one half. */
static inline uint64_t organs(uint64_t x) {
    const uint64_t C = 0xbea225f9eb34556dULL;
    x ^= x >> 32;   /* organ 1 */
    x *= C;         /* organ 2 */
    x ^= x >> 29;   /* organ 3 */
    x *= C;         /* organ 4 */
    x ^= x >> 32;   /* organ 5 */
    x *= C;         /* organ 6 */
    x ^= x >> 29;   /* organ 7 */
    return x;
}

/* ------------- sickening the cup by hand -------------
   Reflected CRC-32C generator 0x82F63B78. Its unreflected form 0x11EDC6F41
   has 18 terms - an even count - hence it is divisible by (x+1): the wine is
   cut. That impurity is precisely what makes the cup remember every drop. */
static uint32_t cup_tab[256];
static int cup_ready = 0;

static void cup_sicken(void) {
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0x82F63B78u & (uint32_t)(-(int32_t)(c & 1u)));
        cup_tab[i] = c;
    }
    cup_ready = 1;
}

/* Software pour: same cup, same cut wine, poured drop by drop. */
static uint64_t pour_soft(const unsigned char *data, size_t len,
                          uint32_t a, uint32_t b) {
    const unsigned char *restrict p = data;
    if (!cup_ready) cup_sicken();
    for (size_t i = 0; i < len; i++) {
        uint32_t t = (uint32_t)p[i] ^ (a & 0xFFu);          /* mark meets sediment */
        uint32_t c = (b >> 8) ^ cup_tab[(b ^ t) & 0xFFu];   /* one pull            */
        a = b; b = c;                                       /* colour becomes seed */
    }
    return ((uint64_t)a << 32) | (uint64_t)b;
}

#if CUP_X86
/* Hardware pour. One unbroken serial chain: b_{n+1} = crc32c(b_n, w_n ^ b_{n-1}).
   The only loop-carried latency is the crc32 itself (3 cycles / 8 bytes); the
   sediment XOR and the loads hang off the chain, not in it. No multiply. */
CUP_TGT
static uint64_t pour_hw(const unsigned char *data, size_t len,
                        uint32_t a, uint32_t b) {
    const unsigned char *restrict p = data;
    size_t i = 0;

    #define PULL(W) do {                                                  \
        uint64_t s_ = (uint64_t)a; s_ ^= s_ << 32;   /* sediment colours */ \
        uint32_t c_ = (uint32_t)_mm_crc32_u64((uint64_t)b, (W) ^ s_);     \
        a = b; b = c_;                                                    \
    } while (0)

    /* the pile is deep: pour in eights, four pulls to a breath */
    for (; i + 32 <= len; i += 32) {
        uint64_t w0, w1, w2, w3;
        __builtin_memcpy(&w0, p + i,      8);
        __builtin_memcpy(&w1, p + i +  8, 8);
        __builtin_memcpy(&w2, p + i + 16, 8);
        __builtin_memcpy(&w3, p + i + 24, 8);
        PULL(w0); PULL(w1); PULL(w2); PULL(w3);
    }
    /* the pile thins */
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        __builtin_memcpy(&w, p + i, 8);
        PULL(w);
    }
    /* the last drops, one at a time - every mark poured, none twice */
    for (; i < len; i++) {
        uint32_t t = (uint32_t)p[i] ^ (a & 0xFFu);
        uint32_t c = (uint32_t)_mm_crc32_u8((unsigned int)b, (unsigned char)t);
        a = b; b = c;
    }
    #undef PULL
    return ((uint64_t)a << 32) | (uint64_t)b;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the cup is rinsed with the measure of the pile */
    uint32_t a = 0x9E3779B9u ^ (uint32_t)len;
    uint32_t b = 0x85EBCA6Bu ^ (uint32_t)(len >> 32)
                             ^ (uint32_t)(len << 7) ^ (uint32_t)(len >> 3);
    uint64_t cup;

#if defined(__SSE4_2__)
    cup = pour_hw(data, len, a, b);
#elif CUP_X86
    cup = __builtin_cpu_supports("sse4.2") ? pour_hw(data, len, a, b)
                                           : pour_soft(data, len, a, b);
#else
    cup = pour_soft(data, len, a, b);
#endif

    return organs(cup);   /* carry the last colour through seven organs */
}
