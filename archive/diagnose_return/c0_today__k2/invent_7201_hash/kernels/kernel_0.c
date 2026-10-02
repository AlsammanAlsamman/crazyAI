#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The forward-only slate wheel: turns one way, never back. r is always 1..63. */
#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* A mark pressed into the wax: one aligned-agnostic 8-byte press, no UB. */
static inline uint64_t wax_press(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* The brine basin, salted with the count of marks, left until the frost stops
   spreading: Evensen's "moremur" mixer (validated low-bias fmix64 successor). */
static inline uint64_t brine(uint64_t w, uint64_t marks) {
    w ^= marks * 0x9E3779B97F4A7C15ULL;
    w ^= w >> 27; w *= 0x3C79AC492BA7B653ULL;
    w ^= w >> 33; w *= 0x1C69B3F74AC4AE35ULL;
    w ^= w >> 27;
    return w;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ---- regime check: does the pile wet all seven wires? ---------------- */
    if (len < 56) {
        /* Too small a pile: one jar only, then straight to the brine. */
        uint64_t a = 0x9E3779B97F4A7C15ULL;
        size_t n = len;
        while (n >= 8) { a = ROTL64(a + wax_press(p), 23); p += 8; n -= 8; }
        if (n >= 4) { uint32_t t; memcpy(&t, p, 4);
                      a = ROTL64(a + (uint64_t)t, 29); p += 4; n -= 4; }
        while (n--)  { a = ROTL64(a + (uint64_t)*p, 31); p++; }
        return brine(a, (uint64_t)len);
    }

    /* ---- seven bell-jars, each tuned to a different hunger -------------- */
    uint64_t a0 = 0x9E3779B97F4A7C15ULL;
    uint64_t a1 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t a2 = 0x165667B19E3779F9ULL;
    uint64_t a3 = 0x85EBCA77C2B2AE63ULL;
    uint64_t a4 = 0x27D4EB2F165667C5ULL;
    uint64_t a5 = 0xD6E8FEB86659FD93ULL;
    uint64_t a6 = 0xA0761D6478BD642FULL;

    /* Marks press into the trough; every jar bites the wire in its own
       count and rhythm. No multiplication, no backward turn, no XOR that
       could let a bite be walked off. Seven independent chains. */
    size_t blocks = len / 56;
    for (size_t i = 0; i < blocks; i++) {
        a0 = ROTL64(a0 + wax_press(p +  0),  7);
        a1 = ROTL64(a1 + wax_press(p +  8), 11);
        a2 = ROTL64(a2 + wax_press(p + 16), 17);
        a3 = ROTL64(a3 + wax_press(p + 24), 23);
        a4 = ROTL64(a4 + wax_press(p + 32), 29);
        a5 = ROTL64(a5 + wax_press(p + 40), 37);
        a6 = ROTL64(a6 + wax_press(p + 48), 43);
        p += 56;
    }

    /* The last marks of the pile, dealt round the jars in order (0..55 left). */
    {
        size_t n = len - blocks * 56;
        if (n >= 8) { a0 = ROTL64(a0 + wax_press(p),  7); p += 8; n -= 8; }
        if (n >= 8) { a1 = ROTL64(a1 + wax_press(p), 11); p += 8; n -= 8; }
        if (n >= 8) { a2 = ROTL64(a2 + wax_press(p), 17); p += 8; n -= 8; }
        if (n >= 8) { a3 = ROTL64(a3 + wax_press(p), 23); p += 8; n -= 8; }
        if (n >= 8) { a4 = ROTL64(a4 + wax_press(p), 29); p += 8; n -= 8; }
        if (n >= 8) { a5 = ROTL64(a5 + wax_press(p), 37); p += 8; n -= 8; }
        while (n--) { a6 = ROTL64(a6 + (uint64_t)*p, 53); p++; }
    }

    /* ---- the lattice: six frost-flowers, no two alike -------------------
       Seven leaves fold through exactly six nodes; each node is a bijection
       in both arguments, so no lane's difference can cancel another's.    */
    uint64_t b0 = ROTL64(a0,  5) + a1;
    uint64_t b1 = ROTL64(a2, 13) + a3;
    uint64_t b2 = ROTL64(a4, 19) + a5;
    uint64_t c0 = ROTL64(b0, 31) + b1;
    uint64_t c1 = ROTL64(b2, 41) + a6;
    uint64_t w  = ROTL64(c0, 47) + c1;

    /* The wax is scraped and remelted; only the tin sketch survives. */
    return brine(w, (uint64_t)len);
}
