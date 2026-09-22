# TRANSFORMED PROBLEM

In Sunny Hollow there's a neighborhood of tree-houses connected by rope bridges. Every bridge has a crossing time written on a little sign — some bridges take one minute, some take twenty, none ever take a negative amount of time (you can't finish crossing before you start, and standing still costs nothing). Bridges only work one way: a bridge from the Oak house to the Pine house doesn't mean there's any way back across that same rope.

Ricky Squirrel lives in the Oak house. He wants to know, for any other tree-house in the hollow, the fewest possible minutes it takes to get there by hopping across bridges, possibly through other tree-houses along the way.

The neighborhood has an old, trusted custom for figuring this out, and everyone follows it the same way:
- You keep a running scratch-guess of "fastest time so far" for every tree-house, starting with Oak at zero minutes and every other house marked "no idea yet."
- You're never allowed to trust a shortcut *through* a tree-house until that tree-house's own fastest time has been carved in stone as final — otherwise you might be building your guess on a number that could still get smaller.
- Before you carve any new house's time in stone, you have to check your whole scratch-list of not-yet-finished houses and find whichever one currently has the smallest guess — that's the only one allowed to be finalized next.
- Once a house's time IS carved in stone, you walk its outgoing bridges and see if they shave time off any of its neighbors' guesses.
- You repeat this, one tree-house at a time, in order from nearest to farthest.

One day Ricky's cousin, who lives clear on the other side of the hollow, asks a much smaller question: "I don't care about the whole neighborhood — I just want to know the fastest time from your house to the Chestnut house, way out past a dozen others I've never even heard of." Ricky starts explaining the custom, and halfway through realizes something odd: to answer his cousin's one simple question honestly, the custom says he has to first finalize the fastest time to every single closer tree-house in the whole hollow — including ones nowhere near Chestnut, in directions his cousin doesn't even care about — before he's "allowed" to be sure about Chestnut. That feels like a lot of running around the entire neighborhood just to answer a question about one house.

# SOLUTION 1

Ricky makes a scratch-list with every tree-house's name and a guessed time (Oak gets zero, everyone else gets "unknown, treat as forever"). Then he repeats a loop: he walks past every single not-yet-finalized house on his list, compares their current guessed times side by side, and picks whichever one has the smallest guess right now. He declares that house's time final, then crosses every bridge leading out of it to see if any neighbor's guess should be lowered. He does this over and over, re-scanning the entire remaining list each time, until every tree-house (including Chestnut) has a final time. It works, but every single round he re-checks the whole neighborhood's list from scratch, which gets slower the more tree-houses there are.

# SOLUTION 2

Ricky keeps the same scratch-list of guesses, but instead of re-scanning everyone every round, he keeps his not-yet-finalized houses arranged on a little sorted shelf, nearest-guess in front. Whenever he lowers a guess after crossing a bridge, he slides that house forward on the shelf to keep things in order. To pick the next house to finalize, he just grabs whatever's at the front of the shelf — no need to compare against everyone each time. He finalizes it, crosses its bridges, updates and re-shelves any neighbors whose guesses shrink, and keeps going until Chestnut (or everyone) is finalized. Same end result as Solution 1, much less wasted comparing.

# SOLUTION 3

Ricky's cousin only cares about Chestnut, so they team up. Ricky starts hopping outward from Oak the usual way, finalizing nearby houses first. At the same moment, the cousin starts at Chestnut and hops backward — following bridges in reverse, the way you'd retrace a route — finalizing the houses nearest to Chestnut first. They keep going, each only worrying about their own side, until their two growing bubbles of "finalized" houses touch somewhere in the middle. The instant that happens, they compare notes on the meeting spot and can immediately announce the fastest Oak-to-Chestnut time — without either of them ever having bothered to finalize the far-flung tree-houses way off in directions neither the start nor the destination cared about.