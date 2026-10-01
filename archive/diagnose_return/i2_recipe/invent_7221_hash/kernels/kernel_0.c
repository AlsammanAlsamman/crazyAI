#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- *
 *  the sheet (the spiral of snail shells), the birds, the press     *
 * ---------------------------------------------------------------- */
#define ROTL64(x,r) (((x) << ((r) & 63)) | ((x) >> ((64 - (r)) & 63)))

#define SHELL_STEP  7                        /* one shell along a crease: rotl(.,7)+D   */
#define SPIRAL_D    0xD1B54A32D192ED03ULL    /* additive part of one shell step         */
#define SPIRAL_C    0x9E3779B97F4A7C15ULL    /* which shell a mark itself lies on       */
#define PRESS_K     0xBF58476D1CE4E5B9ULL    /* pressing a fold flat                    */
#define HALF_D      0x94D049BB133111EBULL    /* half a shell, for tightening            */
#define FAR_SHELL   0x452821E638D01377ULL    /* the farthest snail shell on the sheet   */

static const uint64_t BEAK[4] = {            /* the four wire birds, left edge          */
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL
};

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the sheet flat; set the four wire birds along its left edge,
       beaks toward the paper; stack the marks face down at the right hand, so
       only the top one is ever reachable and only in arrival order.          */
    uint64_t w0 = BEAK[0], w1 = BEAK[1], w2 = BEAK[2], w3 = BEAK[3];
    const unsigned char *mark = data;   /* top of the face-down pile */
    size_t remaining = len;

    /* step 2: before any mark is read, the seed crease — from the first bird's
       beak to the farthest snail shell. Input-independent, identical for every
       sheet, so two folders of the same pile begin identically.              */
    uint64_t A = BEAK[0];               /* where the last crease began */
    uint64_t B = FAR_SHELL;             /* where the last crease ended */

    while (remaining) {
        /* step 3: take the top mark off the pile; do not look at the one
           beneath it. This mark and the crease made last — those two, and
           nothing else, decide the next fold.                               */
        unsigned m = *mark;

        /* step 4: count the strokes in the mark; walk that many snail shells
           along the last crease from where the last crease ENDED; fold from
           that shell toward the point where the last crease BEGAN; press the
           fold flat. Where it stops is where the next one starts.
           (n shells = n applications of rotl(.,7)+SPIRAL_D, in closed form;
            the mark itself lies on its own shell (m+1)*SPIRAL_C.)           */
        unsigned n   = (unsigned)__builtin_popcount(m);
        uint64_t off = (uint64_t)n * SPIRAL_D + ((uint64_t)m + 1u) * SPIRAL_C;
        uint64_t P   = ROTL64(B, SHELL_STEP * n) + off;     /* the shell walked to */
        uint64_t E   = ROTL64(P ^ A, 23) * PRESS_K;         /* folded toward A, pressed */
        A = P;                                              /* next crease begins here */
        B = E;                                              /* ...and ends here        */

        /* step 5: carry the new fold's corner to each of the four beaks in
           turn. If all four meet the paper, the crease is set. If any beak
           misses (its lane lands on nothing), unfold THAT ONE CREASE ONLY —
           by inverting exactly those four touches — pull it a half-shell
           tighter, and touch again. Nothing earlier is ever unfolded.       */
        uint64_t corner = E ^ (E >> 29);
        unsigned tries  = 0;
        for (;;) {
            w0 += corner;
            w1 ^= corner;
            w2  = ROTL64(w2, 19) ^ corner;
            w3  = ROTL64(w3, 41) + corner;
            int all_four_meet = ((w0 != 0) & (w1 != 0) & (w2 != 0) & (w3 != 0));
            if (__builtin_expect(all_four_meet, 1) || tries >= 8) break;
            w0 -= corner;                                   /* unfold this touch only */
            w1 ^= corner;
            w2  = ROTL64(w2 ^ corner, 64 - 19);
            w3  = ROTL64(w3 - corner, 64 - 41);
            corner = ROTL64(corner, 3) + HALF_D;            /* a half shell tighter   */
            ++tries;
        }

        /* step 6: discard the mark into the mud at the tide line. It is never
           read a second time and is never kept anywhere it could be compared
           against later — the pointer only ever moves forward, single pass.  */
        ++mark;

        /* step 7: return to step 3 with the next top mark — once for every
           mark in the pile, one mark one crease, no crease without a mark.   */
        --remaining;
    }

    /* step 8: the marks are finished exactly when the right hand closes on
       nothing. No crease is added to tidy the wad and none is skipped because
       the sheet grew stiff: the loop above ran exactly len times, so
       creases == marks exactly. (Invariant, not an operation.)              */

    /* step 9: take the wad to the water's edge and press — thin, then
       thinner, then one hard dense corner no bigger than a coin — for one
       held breath, the same measure every time (fixed, data-independent).   */
    uint64_t t = w0 ^ ROTL64(w1, 17) ^ ROTL64(w2, 31) ^ ROTL64(w3, 47)
                    ^ ROTL64(A, 11) ^ B;                    /* thin  */
    t ^= t >> 33;  t *= 0xFF51AFD7ED558CCDULL;              /* thinner */
    t ^= t >> 29;  t *= 0xC4CEB9FE1A85EC53ULL;              /* one hard dense corner */
    t ^= t >> 32;                                           /* takes no more pressing */

    /* step 10: lift out the corner. That is the token — it holds its shape
       dry, so it is simply carried out as it stands.                        */
    uint64_t token = t;

    /* step 11: everything else goes into the mud where the tide-boundary
       closes — the birds, the last crease, the failed half-shell refolds,
       the wet remainder of the sheet. Nothing but the coin leaves.          */
    w0 = w1 = w2 = w3 = 0; A = B = 0; remaining = 0; mark = 0;

    /* step 12: to check a pile against a token later, fold it again from a
       fresh sheet and compare the two corners. This holds because the kernel
       is a pure function of (data,len): no statics, no clocks, no addresses,
       no retained state between calls.                                      */
    return token;
}
