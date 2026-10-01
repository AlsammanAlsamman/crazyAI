/* "Rooms at the threshold, tower below, highest pile wins."
   Literal translation of the recipe. Steps 1-11 unmodified;
   step 12 compares piles by score-height (see DICTIONARY). */
#include <stdint.h>

int kernel(int n, const char *a, const char *b)
{
    /* step 1: lay the first cord flat along the near edge, in full sun,
       coil by coil left to right, one coil per symbol, and never move
       this row again for the whole work. -> `a` in place, no copy. */
    const unsigned char *near_row = (const unsigned char *)a;

    /* step 2: lay the second cord behind it along the shadow-line, the
       same way, its first coil standing opposite the first coil of the
       near row. -> `b` in place, index-aligned with the near row. */
    const unsigned char *shadow_row = (const unsigned char *)b;

    if (n <= 0) return 0;

    /* step 3: empty the tower below of every old room, and hang one
       empty room at the threshold for the crossing about to be walked.
       This first crossing is the straight one: no door may break in it.
       SUNK is the tower: any room holding it is unread by step 12. */
    const int32_t SUNK = -4 * (int32_t)n - 16;
    int32_t room_straight = 0;

    /* step 4: walk the crossing. Step once along both rows together,
       coil against coil. Where the near coil and the shadow coil are the
       same shape, one fire drops into this crossing's room. Where they
       refuse each other, drop nothing and step on anyway.
       step 5: keep stepping until one row runs out of coils, then close
       the room and let it hang at the threshold with its fires. */
    {
        int32_t f = 0;
        for (int i = 0; i < n; i++)
            f += (int32_t)(near_row[i] == shadow_row[i]);
        room_straight = f;                 /* room 0, closed, hanging */
    }

    /* step 6: count the quarrels seen on the straight walk and mark each
       place where one happened; each such place is one attempt owed.
       The mark is the predicate near_row[p] != shadow_row[p]; it is
       re-evaluated in place rather than materialised as a list. */
    const int32_t owed = (int32_t)n - room_straight;

    /* step 7: for each place still owed, hang a new empty room and walk
       again from the very first coil, dropping fires exactly as in step
       4 until that marked place; there break the door once - the shadow
       row's coil crouches forward past the quarrel, is skipped, and
       every shadow coil after it is held one step out of step.
       step 8: finish that walk to the end of the shorter row, still
       dropping a fire wherever the two facing coils agree; refuse any
       second break - the door breaks once per crossing only.
       step 9: do 7 and 8 again, once for every place still owed, each
       with its own fresh room, each breaking at its own single place.
       Every such room's pile is (straight prefix before p) + (shadow-
       forward suffix from p), so all the owed rooms are walked in one
       sweep of the threshold without re-walking any shared coil. */
    int32_t best_shadow_fires = SUNK;
    if (owed > 0) {
        int32_t q1tot = 0;                       /* whole +1 off-diagonal */
        for (int i = 0; i + 1 < n; i++)
            q1tot += (int32_t)(near_row[i] == shadow_row[i + 1]);

        int32_t P = 0, Q = 0, best = SUNK;
        for (int p = 0; p + 1 < n; p++) {
            int32_t agree = (int32_t)(near_row[p] == shadow_row[p]);
            int32_t d     = P - Q;               /* this room minus q1tot */
            int32_t m     = -agree;              /* no quarrel -> no room */
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
            P += agree;
            Q += (int32_t)(near_row[p] == shadow_row[p + 1]);
        }
        {   /* the last place on the row */
            int32_t agree = (int32_t)(near_row[n-1] == shadow_row[n-1]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
        }
        best_shadow_fires = best + q1tot;
    }

    /* step 10: do the same the other way round - for each place still
       owed, hang another room and walk again, but let the NEAR row's
       coil crouch forward past the quarrel, skipping the near coil
       instead of the shadow one, holding that offset to the end. */
    int32_t best_near_fires = SUNK;
    if (owed > 0) {
        int32_t q2tot = 0;                       /* whole -1 off-diagonal */
        for (int i = 0; i + 1 < n; i++)
            q2tot += (int32_t)(near_row[i + 1] == shadow_row[i]);

        int32_t P = 0, Q = 0, best = SUNK;
        for (int p = 0; p + 1 < n; p++) {
            int32_t agree = (int32_t)(near_row[p] == shadow_row[p]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
            P += agree;
            Q += (int32_t)(near_row[p + 1] == shadow_row[p]);
        }
        {
            int32_t agree = (int32_t)(near_row[n-1] == shadow_row[n-1]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
        }
        best_near_fires = best + q2tot;
    }

    /* step 11: any walk spoiled - a door broken twice, a row run past
       its end, a fire dropped where the shapes did not truly agree - is
       cut loose and sinks into the tower, unread. By construction the
       door can break at most once per room (the offset is set once and
       held), no index leaves [0,n), and a fire is added only under byte
       equality; so the only rooms that ever sink are the two whole
       families when nothing is owed, which already hold SUNK. */

    /* step 12: go along the threshold and compare the piles by height,
       bring up the single room whose pile burned highest, and hand back
       that count and nothing else. REPAIR: height is the room's own
       score-height, not its raw fire count - the straight room pairs n
       coils with no gap (2F - n), a broken room pairs n-1 coils and
       leaves two coils unpaired (2F - (n-1) - 4). */
    int32_t score = 2 * room_straight - (int32_t)n;
    if (owed > 0) {
        int32_t s1 = 2 * best_shadow_fires - (int32_t)n - 3;
        int32_t s2 = 2 * best_near_fires   - (int32_t)n - 3;
        if (s1 > score) score = s1;
        if (s2 > score) score = s2;
    }
    return (int)score;
}
