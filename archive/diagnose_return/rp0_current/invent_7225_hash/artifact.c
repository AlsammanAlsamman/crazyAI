#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(__amd64__))
  #include <immintrin.h>
  #define CUP_HW 1
#else
  #define CUP_HW 0
#endif

/* ---- the cup that is never pure -------------------------------------- */
/* A pull is a CRC-32C step: an index into a fixed impurity.  No multiply. */

#if CUP_HW
  /* the tavern's own cup: hardware, 3-cycle latency, 1/cycle throughput */
  #define CUP8(c,b)  ((uint64_t)_mm_crc32_u8 ((unsigned int)(uint32_t)(c),(unsigned char)(b)))
  #define CUP32(c,v) ((uint64_t)_mm_crc32_u32((unsigned int)(uint32_t)(c),(uint32_t)(v)))
  #define CUP64(c,v) ((uint64_t)_mm_crc32_u64((uint64_t)(c),(uint64_t)(v)))
#else
  /* the cup I carry: 256 pre-poisoned colors, same polynomial */
  static uint32_t cup_tab[256];
  static int cup_ready = 0;
  static void cup_fill(void) {
      unsigned i; int k;
      for (i = 0; i < 256; ++i) {
          uint32_t c = (uint32_t)i;
          for (k = 0; k < 8; ++k)
              c = (c >> 1) ^ (0x82F63B78u & (uint32_t)(0u - (c & 1u)));
          cup_tab[i] = c;                      /* idempotent: benign race */
      }
      cup_ready = 1;
  }
  static inline uint64_t cup8_(uint64_t c, unsigned char b) {
      uint32_t x = (uint32_t)c;
      return (uint64_t)(cup_tab[(x ^ b) & 0xFFu] ^ (x >> 8));
  }
  static inline uint64_t cup32_(uint64_t c, uint32_t v) {
      int i; for (i = 0; i < 4; ++i) { c = cup8_(c, (unsigned char)v); v >>= 8; } return c;
  }
  static inline uint64_t cup64_(uint64_t c, uint64_t v) {
      int i; for (i = 0; i < 8; ++i) { c = cup8_(c, (unsigned char)v); v >>= 8; } return c;
  }
  #define CUP8(c,b)  cup8_ ((c),(unsigned char)(b))
  #define CUP32(c,v) cup32_((c),(uint32_t)(v))
  #define CUP64(c,v) cup64_((c),(uint64_t)(v))
#endif

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* ---- the seven organs -------------------------------------------------
   Each bending throws away half of what came before:
     a shift destroys the shifted-out half;
     a 64-bit multiply destroys the upper half of the 128-bit product.
   Four shifts + three multiplies = seven.  Constants are the validated
   splitmix64 pair plus one degski64 constant; none are invented here.   */
static inline uint64_t seven_organs(uint64_t x) {
    x ^= x >> 30;                      /* 1 */
    x *= 0xBF58476D1CE4E5B9ULL;        /* 2 */
    x ^= x >> 27;                      /* 3 */
    x *= 0x94D049BB133111EBULL;        /* 4 */
    x ^= x >> 31;                      /* 5 */
    x *= 0xD6E8FEB86659FD93ULL;        /* 6 */
    x ^= x >> 32;                      /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    size_t n = len;
    /* the cup is never empty: it starts stained with the weight of the pile */
    uint64_t color = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;

#if !CUP_HW
    if (!cup_ready) cup_fill();
#endif

    /* --- palm-sized pile: never touches the cup, straight to the organs --- */
    if (n <= 8) {
        if (n >= 4) {
            uint32_t a, b;
            memcpy(&a, p,         4);
            memcpy(&b, p + n - 4, 4);
            color ^= ((uint64_t)a << 32) ^ (uint64_t)b;
        } else if (n > 0) {
            color ^=  (uint64_t)p[0]
                   ^ ((uint64_t)p[n >> 1] << 19)
                   ^ ((uint64_t)p[n - 1]  << 41);
        }
        return seven_organs(color);
    }

    /* --- heavy pile: handfuls of thirty-two ---
       Four pulls chain through the cup (3 cycles each); the drops
       themselves are added to the cup off the critical path.          */
    while (n >= 32) {
        uint64_t w0, w1, w2, w3, c;
        memcpy(&w0, p +  0, 8);
        memcpy(&w1, p +  8, 8);
        memcpy(&w2, p + 16, 8);
        memcpy(&w3, p + 24, 8);
        c = CUP64(color, w0);
        c = CUP64(c,     w1);
        c = CUP64(c,     w2);
        c = CUP64(c,     w3);
        color = (rotl64(color, 37) ^ w0 ^ rotl64(w1, 13)
                                   ^ rotl64(w2, 29) ^ rotl64(w3, 47)) + c;
        p += 32; n -= 32;
    }

    /* --- handfuls of eight --- */
    while (n >= 8) {
        uint64_t w; memcpy(&w, p, 8);
        color = (rotl64(color, 37) ^ w) + CUP64(color, w);
        p += 8; n -= 8;
    }

    /* --- one handful of four --- */
    if (n >= 4) {
        uint32_t v; memcpy(&v, p, 4);
        color = (rotl64(color, 37) ^ (uint64_t)v) + CUP32(color, v);
        p += 4; n -= 4;
    }

    /* --- stragglers, drop by drop; no mark read twice --- */
    while (n) {
        uint64_t b = (uint64_t)(*p);
        color = (rotl64(color, 37) ^ b) + CUP8(color, b);
        ++p; --n;
    }

    return seven_organs(color);
}
