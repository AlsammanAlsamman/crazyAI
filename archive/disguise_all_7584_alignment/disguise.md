# TRANSFORMED PROBLEM

Mia and Leo each own a long string of beads. Both strings have exactly the same number of beads. They want a single fair number that says "how alike are these two strings, bead for bead?"

The rule they've agreed on: hold the two strings side by side and go bead by bead. At each spot:
- if the two beads are the same color, that spot earns 1 point,
- if the two beads are different colors, that spot loses 1 point,
- but sometimes it's worth pretending one string has a bead "missing" right there — everything after that point on that string gets nudged over by one — and doing that costs 2 points.

To find the very best possible score (the fairest way of lining the two strings up, missing-beads and all), the two of them draw a big square grid on the ground: Mia's beads numbered along the top, Leo's beads numbered down the side, so every square in the grid stands for "Mia's bead number this, next to Leo's bead number that." They fill in the squares one at a time, starting from the corner and sweeping row by row, and the score written in any square is built only from the square directly above it, the square directly to its left, and the square diagonally above-left of it.

Here's the part that starts to feel strange the longer you do it: the only way they're allowed to notice "ah, there's a missing bead here" at some square is by having already filled in the square right before it. You're never allowed to just look at the two strings and spot a missing bead directly — you have to walk the whole grid, one square earlier, one square earlier, one square earlier, before the missing-bead idea is even allowed to occur to you at the square where it actually matters. Even if the missing bead is obvious to the eye from ten steps away, the rule insists it can only be "discovered" by first finishing the comparison sitting right next to it.

# SOLUTION 1

Do exactly what the rule says, no shortcuts. Draw the full grid, start at the top-left corner, and fill it in one square at a time, going left to right along each row, then dropping to the next row. For every square, look at the score already written in the square above it, the square to its left, and the square diagonally above-left, and pick whichever of "same-color bonus," "different-color penalty," or "pretend-a-bead-is-missing penalty" gives the best running total for that square. Never skip a square, never fill one out of order. When the whole grid is done, the square in the bottom-right corner holds the final answer. This is slow because every single square has to wait its turn, but it's guaranteed to match the answer anyone else would get doing it the same careful way.

# SOLUTION 2

Notice that a whole slanted line of squares running from the top-right toward the bottom-left never actually needs each other — each one only needs squares from the slanted line just before it, not squares sitting next to it on its own line. So instead of one person crawling row by row, line up a row of helpers, one per square on that slanted line, and let them all fill in their square at the same moment, since none of them are waiting on each other. Once that whole slanted line is done, the next slanted line of helpers steps in and does the same thing, using only what the previous line just finished. You still end up filling in the entire grid and you still get the identical final number in the bottom-right corner, but because whole slanted lines of squares get done together instead of one square at a time, the whole board gets finished much faster.

# SOLUTION 3

Skip the grid entirely at first. Just walk both strings side by side from the start, bead by bead, straight through, keeping a running tally of same-color bonuses and different-color penalties as you go — no missing-bead penalty allowed yet. Keep an eye out for a stretch where the beads keep disagreeing over and over in a row; that's the sign something has actually slipped out of step, not that it's just bad luck. When you spot such a stretch, stop, and try a direct test right there: nudge one string over by a single bead at that exact spot, pay the missing-bead cost once, and re-walk the rest of the strings straight through from that point onward with the new lineup. Compare that adjusted tally to the plain straight-through tally and keep whichever total is bigger. This way, a missing bead gets spotted by noticing the pattern of disagreement itself, not by having crawled through every earlier square first — you jump straight to the suspicious stretch and test your hunch on the spot.