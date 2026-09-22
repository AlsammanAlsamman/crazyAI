# TRANSFORMED PROBLEM

Pip the squirrel lives in the Big Oak at the edge of the forest, and she wants to know, for every other tree in the forest, the fewest minutes it would take to scamper there from her home — using the rope bridges and springy branches that connect the trees.

Every bridge takes a certain number of minutes to cross (never a negative number — bridges don't give you time back), and some bridges only let you cross in one direction (a branch that droops one way, easy to run down but impossible to climb back up).

Pip keeps a big chalkboard with one line for every tree in the forest and her current best guess for how many minutes it would take to reach it. At the start, only the Big Oak shows "zero minutes" — every other tree just shows a giant question mark.

The way Pip has always done it: she is only allowed to trust a tree's number as final, and start counting minutes across bridges leading out of it, once that tree is fully "locked in." And a tree only gets locked in when Pip is sure no faster guess could ever come along. To be sure of that, every single round, she walks down the *entire* chalkboard, top to bottom, comparing every remaining question-marked or guessed tree against every other one, just to find whichever has the smallest number right now. Only then does she lock that tree in and update the guesses for the trees just across its bridges.

With ten trees, this is fine. But the forest has hundreds of trees, and every round Pip re-reads the whole chalkboard from scratch, even though most numbers haven't changed since the last round — just to find one smallest number. And even if she only actually cares about the fastest way to reach, say, the Crooked Pine on the far side of the forest, she still has to keep this whole locking-in process going across every tree, not just the ones on the way to the Crooked Pine.

# SOLUTION 1

Pip does it her usual way. She keeps the one big chalkboard listing every tree and her current best guess. Each round: she walks the entire board from the first line to the last, comparing every number as she goes, to find whichever unlocked tree has the smallest guess. She locks that tree in, crosses out its question mark, and walks the bridges leading out of it, updating the guesses written for the trees on the other side if she's found a quicker way to reach them. Then she starts the whole board-reading over again for the next round. She keeps doing this, one full read of the board per lock-in, until every tree is locked.

# SOLUTION 2

Pip sets up a long row of baskets on the ground in front of the Big Oak, one basket for each minute-mark: a "0 minutes" basket, a "1 minute" basket, a "2 minutes" basket, and so on, stretching out as far as she needs. Whenever she has a guess for how long it takes to reach some tree, she drops a little tag with that tree's name into the basket matching the guess. If she later finds a quicker way to that tree, she just pulls its tag out and drops it into an earlier basket instead.

To find the next tree to lock in, she no longer reads a whole board — she just walks along her row of baskets starting from the lowest minute-mark, and the moment she finds a basket with a tag in it, that's her tree. She locks it in, follows its bridges, updates a few other tags by moving them to earlier baskets if needed, and goes back to walking along the baskets from wherever she left off. She never has to compare every tree at once — just check baskets in order until she finds one that isn't empty.

# SOLUTION 3

Pip asks her friend Squeaky to hold onto a special narrow filing box instead of using the chalkboard. The box always keeps its cards arranged so the soonest guess is right at the front and the rest trail off toward the back, sorted least to most.

Whenever Pip has a new or better guess for some tree, she writes it on a card and hands it to Squeaky, who tucks it into the box in its correct spot — sliding it past just a few neighboring cards, not digging through the whole box. When Pip needs to know which tree to lock in next, she simply asks Squeaky for whatever card is sitting at the very front. She locks that tree in, walks its bridges, hands Squeaky any updated cards for the trees on the other side, and asks for the front card again. She never once has to look at the whole stack — only ever the front of it, and each update only nudges a small handful of nearby cards rather than touching everything.