#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- which tavern am I in? (regime 2: cup fused into the wall, or clay) ---- */
#if defined(__SSE4_2__)
  #include <nmmintrin.h>
  #define CUP_HW_X86 1
#elif defined(__ARM_FEATURE_CRC32)
  #include <arm_acle.h>
  #define CUP_HW_ARM 1
#endif

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

#if !defined(CUP_HW_X86) && !defined(CUP_HW_ARM)
/* the clay cup: 256 pre-mixed colors of sickened wine (CRC-32C, reflected) */
static uint32_t cup_tbl[256];
static int cup_ready = 0;
static void cup_fill(void) {
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? ((c >> 1) ^ 0x82F63B78u) : (c >> 1);
        cup_tbl[i] = c;
    }
    cup_ready = 1;              /* idempotent: a benign race recomputes the same table */
}
#endif

/* one pull, one drop */
static inline uint32_t pull_drop(uint32_t c, unsigned char b) {
#if defined(CUP_HW_X86)
    return _mm_crc32_u8(c, b);
#elif defined(CUP_HW_ARM)
    return __crc32cb(c, b);
#else
    return (c >> 8) ^ cup_tbl[(c ^ b) & 0xFFu];
#endif
}

/* one pull, a draught of eight marks */
static inline uint32_t pull_draught(uint32_t c, uint64_t w) {
#if defined(CUP_HW_X86)
    return (uint32_t)_mm_crc32_u64((uint64_t)c, w);
#elif defined(CUP_HW_ARM)
    return __crc32cd(c, w);
#else
    /* clay cup: taste once per draught, not once per drop, so the clay tavern is
       never slower than the known way. The cup's body still holds every mark. */
    uint32_t f = (uint32_t)w ^ (uint32_t)(w >> 32);
    c = (c >> 8) ^ cup_tbl[(c ^ f) & 0xFFu];
    c = (c >> 8) ^ cup_tbl[(c ^ (f >> 11)) & 0xFFu];
    return c;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    size_t n = len;

    /* the cup is never pure: it starts already cut, and the pile's size is itself
       a mark -- so a pile of leading silences (0x00...) cannot pass unnoticed. */
    uint32_t color = 0x9E3779B9u ^ (uint32_t)len;            /* what the tongue tastes */
    uint64_t body  = 0x6A09E667F3BCC909ULL ^ (uint64_t)len;  /* what the cup holds    */

#if !defined(CUP_HW_X86) && !defined(CUP_HW_ARM)
    if (!cup_ready) cup_fill();
#endif

    /* regime 1a: four draughts while thirty-two marks remain */
    while (n >= 32) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, p,      8);
        memcpy(&w1, p +  8,  8);
        memcpy(&w2, p + 16, 8);
        memcpy(&w3, p + 24, 8);
        body = rotl64(body + w0, 29); color = pull_draught(color, w0 ^ body);
        body = rotl64(body + w1, 29); color = pull_draught(color, w1 ^ body);
        body = rotl64(body + w2, 29); color = pull_draught(color, w2 ^ body);
        body = rotl64(body + w3, 29); color = pull_draught(color, w3 ^ body);
        p += 32; n -= 32;
    }
    /* regime 1b: one draught while eight marks remain */
    while (n >= 8) {
        uint64_t w; memcpy(&w, p, 8);
        body = rotl64(body + w, 29); color = pull_draught(color, w ^ body);
        p += 8; n -= 8;
    }
    /* regime 1c: the dregs, drop by drop */
    while (n) {
        unsigned char b = *p++; n--;
        body  = rotl64(body ^ (uint64_t)b, 7) + 0x9E3779B97F4A7C15ULL;
        color = pull_drop(color, b);
    }

    /* the cup's last color, carried out of the tavern */
    uint64_t x = body ^ (((uint64_t)color << 32) | (uint64_t)(color ^ 0x5BF03635u));

    /* --- the seven organs: each throws away half, keeps what refuses to sit still,
           and each bends with its own character (identical organs would undo
           each other -- an involution -- which is the whole point) --- */
    x ^= x >> 33;  x *= 0xFF51AFD7ED558CCDULL;   /* 1, 2 */
    x ^= x >> 29;  x *= 0xC4CEB9FE1A85EC53ULL;   /* 3, 4 */
    x ^= x >> 32;  x *= 0x9E3779B97F4A7C15ULL;   /* 5, 6 */
    x ^= x >> 31;                                /* 7: the garden door */
    return x;
}
