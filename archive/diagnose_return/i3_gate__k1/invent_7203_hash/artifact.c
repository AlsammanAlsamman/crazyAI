#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- eight odd anchors: where the wires are fixed in the riverbed ---- */
#define W1 0x9E3779B185EBCA87ULL
#define W2 0xC2B2AE3D27D4EB4FULL
#define W3 0x165667B19E3779F9ULL
#define W4 0x85EBCA77C2B2AE63ULL
#define W5 0x27D4EB2F165667C5ULL
#define W6 0xD6E8FEB86659FD93ULL
#define W7 0xA0761D6478BD642FULL
#define W8 0xE7037ED1A0B428DBULL

/* the angle the chained prisoner grinds into the pillar.  MUST be 0 < A < 64:
   that is exactly the condition under which no mark can cancel itself out. */
#define ANGLE 29

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* ONE STROKE.  The prisoner never looks at the mark; he only feels how it
   tilts what he was already sharpening, and hands on the angle, not the mark.
   Add-rotate-xor only: no multiplication anywhere in the streaming path.
   Flipping bit b of w moves this by +/-2^((b+ANGLE)%64) +/- 2^b, never 0. */
static inline uint64_t stroke(uint64_t acc, uint64_t w) {
    return rotl64(acc ^ w, ANGLE) + w;
}

/* THE SILHOUETTE.  Drawn once, only after the flood has drained and the
   shadows have stopped chittering.  Two multiply stages = "the walls have
   settled"; a third would add nothing, which is the whole point.  Bijective,
   so no upstream difference is ever destroyed here. */
static inline uint64_t silhouette(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    const unsigned char * const e = data + len;

    /* ---- the water level: regime check, in the metaphor's own terms ----
       A thin pile of marks never raises the flood as far as the wires, so
       there is nothing to bend and nothing to cross.  The native then reads
       the shadow of a single wire.  Guards the small-input overhead of the
       eight-lane setup and the eight-step fold. */
    if (len < 64) {
        uint64_t h = W5 ^ ((uint64_t)len * W1);
        while ((size_t)(e - p) >= 8) { h = stroke(h, ld64(p));        p += 8; }
        if    ((size_t)(e - p) >= 4) { h = stroke(h, ld32(p) ^ W6);   p += 4; }
        while (p < e)                { h = stroke(h, (uint64_t)(*p++) ^ W7); }
        return silhouette(h);                 /* read once, kept alone */
    }

    /* ---- eight wires, anchored, fixed in number regardless of len ---- */
    uint64_t l0=W1, l1=W2, l2=W3, l3=W4, l4=W5, l5=W6, l6=W7, l7=W8;

#if defined(__AVX2__)
    /* The flood moves the wires per PASS, not per mark: 64 bytes enter the
       riverbed at once and bend all eight wires by the same ground angle. */
    {
        __m256i a = _mm256_set_epi64x((long long)W4,(long long)W3,(long long)W2,(long long)W1);
        __m256i b = _mm256_set_epi64x((long long)W8,(long long)W7,(long long)W6,(long long)W5);
        while ((size_t)(e - p) >= 64) {
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(p));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
            __m256i t0 = _mm256_xor_si256(a, x0);
            __m256i t1 = _mm256_xor_si256(b, x1);
            a = _mm256_add_epi64(_mm256_or_si256(_mm256_slli_epi64(t0, ANGLE),
                                                 _mm256_srli_epi64(t0, 64 - ANGLE)), x0);
            b = _mm256_add_epi64(_mm256_or_si256(_mm256_slli_epi64(t1, ANGLE),
                                                 _mm256_srli_epi64(t1, 64 - ANGLE)), x1);
            p += 64;
        }
        uint64_t sa[4], sb[4];
        _mm256_storeu_si256((__m256i *)(void *)sa, a);
        _mm256_storeu_si256((__m256i *)(void *)sb, b);
        l0=sa[0]; l1=sa[1]; l2=sa[2]; l3=sa[3];
        l4=sb[0]; l5=sb[1]; l6=sb[2]; l7=sb[3];
    }
#else
    /* identical hash, eight independent chains, no SIMD required */
    while ((size_t)(e - p) >= 64) {
        l0 = stroke(l0, ld64(p     )); l1 = stroke(l1, ld64(p +  8));
        l2 = stroke(l2, ld64(p + 16)); l3 = stroke(l3, ld64(p + 24));
        l4 = stroke(l4, ld64(p + 32)); l5 = stroke(l5, ld64(p + 40));
        l6 = stroke(l6, ld64(p + 48)); l7 = stroke(l7, ld64(p + 56));
        p += 64;
    }
#endif

    /* the last marks of the pile, fed singly to the nearest prisoner */
    if ((size_t)(e - p) >= 8) { l0 = stroke(l0, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l1 = stroke(l1, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l2 = stroke(l2, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l3 = stroke(l3, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l4 = stroke(l4, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l5 = stroke(l5, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l6 = stroke(l6, ld64(p)); p += 8; }
    {
        uint64_t tw = 0; unsigned sh = 0;
        while (p < e) { tw |= (uint64_t)(*p++) << sh; sh += 8; }   /* never over-reads */
        l7 = stroke(l7, tw ^ ((uint64_t)sh * W6));
    }

    /* ---- THE SILHOUETTE: which lean, which stand straight, which cross
       another.  This is the only place the wires touch each other, and the
       only place a multiplication appears.  Every step is a bijection of v,
       so the nonzero difference guaranteed by stroke() survives to the end. */
    {
        uint64_t v = (uint64_t)len * W1;
        v = (v ^ rotl64(l0,  3)) * W2;
        v = (v ^ rotl64(l1, 11)) * W3;
        v = (v ^ rotl64(l2, 19)) * W4;
        v = (v ^ rotl64(l3, 27)) * W5;
        v = (v ^ rotl64(l4, 35)) * W6;
        v = (v ^ rotl64(l5, 43)) * W7;
        v = (v ^ rotl64(l6, 51)) * W8;
        v = (v ^ rotl64(l7, 59)) * W2;
        return silhouette(v);     /* read once; I keep nothing but the shadow */
    }
}
