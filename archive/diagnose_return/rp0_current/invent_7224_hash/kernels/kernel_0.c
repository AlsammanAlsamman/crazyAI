#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* ------------------------------------------------------------------ *
 * seal64 -- splitmix64 / xxHash-class 64-bit avalanche finalizer.
 * Validated, multiply-based.  Used ONLY by the short regime: the
 * "small pile spoken straight into the single shrine stone".
 * ------------------------------------------------------------------ */
static inline uint64_t seal64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* ------------------------------------------------------------------ *
 * grow -- one "musk deer stride" layer over the 4x4 square.
 * Exactly BLAKE2b's G function (fixed strides 32,24,16,63: uphill
 * twist = rotate, downhill fold = xor, doubling-back that eats its
 * own trail = the d^=a / b^=c feedback), applied to all FOUR columns
 * at once.  The i-loop is the lane loop: 4 independent columns, unit
 * stride, add/xor/shift only -- written this way so -O3 -march=native
 * turns it into one vector op per scalar op.  No multiplication.
 * ------------------------------------------------------------------ */
static inline void grow(uint64_t *restrict s) {
    for (int i = 0; i < 4; i++) {
        uint64_t a = s[0 + i], b = s[4 + i], c = s[8 + i], d = s[12 + i];
        a += b; d ^= a; d = ROTL64(d, 32);
        c += d; b ^= c; b = ROTL64(b, 24);
        a += b; d ^= a; d = ROTL64(d, 16);
        c += d; b ^= c; b = ROTL64(b, 63);
        s[0 + i] = a; s[4 + i] = b; s[8 + i] = c; s[12 + i] = d;
    }
}

/* ------------------------------------------------------------------ *
 * turn90 -- "turning the shape ninety degrees": ShiftRows.  Rows 1,2,3
 * rotate left by 1,2,3 lanes, so the next grow() mixes along the other
 * axis.  grow+turn+grow reaches all 16 words from any one word, which
 * is the native's test ("if even a corner survives untouched, throw
 * the whole method away").
 * ------------------------------------------------------------------ */
static inline void turn90(uint64_t *restrict s) {
    uint64_t t;
    t = s[4];  s[4] = s[5];  s[5] = s[6];   s[6] = s[7];   s[7] = t;
    t = s[8];  s[8] = s[10]; s[10] = t;
    t = s[9];  s[9] = s[11]; s[11] = t;
    t = s[15]; s[15] = s[14]; s[14] = s[13]; s[13] = s[12]; s[12] = t;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* ===== regime test ============================================= *
     * "the flat stone is only worth walking to if there is a pile."
     * A small pile is spoken straight into the single shrine stone:
     * fall back to the simple known accumulator path, two lanes deep
     * so the multiply chain is not the critical path.
     * =============================================================== */
    if (len < 64) {
        uint64_t h0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
        uint64_t h1 = 0xC2B2AE3D27D4EB4FULL + (uint64_t)len;
        size_t i = 0;
        while (i + 16 <= len) {
            uint64_t w0, w1;
            memcpy(&w0, p + i, 8);
            memcpy(&w1, p + i + 8, 8);
            h0 = seal64(h0 ^ w0);
            h1 = seal64(h1 ^ w1);
            i += 16;
        }
        while (i + 8 <= len) {
            uint64_t w0;
            memcpy(&w0, p + i, 8);
            h0 = seal64(h0 ^ w0);
            i += 8;
        }
        if (i < len) {                       /* safe zero-padded tail */
            unsigned char t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t w0;
            memcpy(t, p + i, len - i);
            memcpy(&w0, t, 8);
            h0 = seal64(h0 ^ w0 ^ ((uint64_t)(len - i) << 56));
        }
        return seal64(h0 ^ ROTL64(h1, 32));
    }

    /* ===== the riverbank: 4x4 square, 128-byte state =============== */
    uint64_t s[16];
    s[0]  = 0x6A09E667F3BCC908ULL; s[1]  = 0xBB67AE8584CAA73BULL;
    s[2]  = 0x3C6EF372FE94F82BULL; s[3]  = 0xA54FF53A5F1D36F1ULL;
    s[4]  = 0x510E527FADE682D1ULL; s[5]  = 0x9B05688C2B3E6C1FULL;
    s[6]  = 0x1F83D9ABFB41BD6BULL; s[7]  = 0x5BE0CD19137E2179ULL;
    s[8]  = 0xCBBB9D5DC1059ED8ULL; s[9]  = 0x629A292A367CD507ULL;
    s[10] = 0x9159015A3070DD17ULL; s[11] = 0x152FECD8F70E5939ULL;
    s[12] = 0x67332667FFC00B31ULL; s[13] = 0x8EB44A8768581511ULL;
    s[14] = 0xDB0C2E0D64F98FA7ULL; s[15] = 0x47B5481DBEFA4FA4ULL;

    /* the count is laid down with the marks, before any heap is folded */
    s[8] ^= (uint64_t)len;

    const size_t nblk = len >> 6;
    for (size_t b = 0; b < nblk; b++) {
        /* --- silence: the 64 marks are only PLACED, never mixed.
         *     Order is held by lane position, so all 64 bytes are
         *     fetched at once and their read order is free. --- */
        uint64_t m[8];
        memcpy(m, p + (b << 6), 64);
        __builtin_prefetch(p + (b << 6) + 512, 0, 0);
        for (int i = 0; i < 8; i++) s[i] ^= m[i];

        /* --- the race: 2 double-rounds.  Held to the minimum that
         *     passes the native's corner test; not one layer more. --- */
        grow(s); turn90(s);
        grow(s); turn90(s);
        grow(s); turn90(s);
        grow(s); turn90(s);
    }

    /* the last, short heap: zero-padded, one 0x01 mark laid to close it */
    {
        const size_t r = len & 63;
        unsigned char t[64];
        uint64_t m[8];
        memset(t, 0, 64);
        if (r) memcpy(t, p + (nblk << 6), r);
        t[r] = 0x01;                          /* r <= 63, always in range */
        memcpy(m, t, 64);
        for (int i = 0; i < 8; i++) s[i] ^= m[i];
    }

    /* ===== the owl's seal: 12 layers, then the corners folded in ==== *
     * Constant cost, amortised away on any real buffer, and it is what
     * guarantees a bit flipped in the FINAL heap still smears across
     * every square.  12 layers = 6 BLAKE2b-equivalent rounds, well past
     * the ~2 rounds at which dependency completes.
     * =============================================================== */
    for (int k = 0; k < 12; k++) { grow(s); turn90(s); }

    uint64_t h = s[0] ^ ROTL64(s[5], 17) ^ ROTL64(s[10], 34) ^ ROTL64(s[15], 51);
    h ^= ROTL64(s[3], 11) ^ ROTL64(s[6], 27) ^ ROTL64(s[9], 43) ^ ROTL64(s[12], 59);
    h ^= h >> 32;
    h  = ROTL64(h, 29);
    h ^= h >> 31;
    return h;               /* the dreamer inside: one token, fixed size,
                               no path back to the marks */
}
