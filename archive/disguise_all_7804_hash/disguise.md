# TRANSFORMED PROBLEM

Grandma Rosa runs a little bakery. On a long shelf she keeps a row of jars, each filled with a different color of sprinkles — jar 1, jar 2, jar 3, and so on, all the way down the shelf to the last jar. Before the big cake contest, she needs to make one small bowl of "signature swirl" — a single blended color that stands for every jar on the shelf put together.

Two things matter a lot to her:

- **It has to change completely if even one jar changes.** If someone swapped the sprinkles in just jar 7 for a different color, the final swirl in the bowl should come out looking totally different — not just a little different in one corner, but different all over, so nobody could tell which jar was swapped just by looking at the new swirl.
- **It has to be ready fast**, before the judges arrive.

Her usual way of doing it: take one bowl, walk to jar 1, scoop a spoonful in, give the bowl a hard twist-stir, walk to jar 2, scoop, twist-stir again, and so on — jar after jar, always the very next one on the shelf, never skipping ahead or doubling back, until she's walked the whole shelf from the first jar to the last.

But here's the catch on contest morning: the shelf isn't just sitting neatly in one room anymore. Some jars are up on the high shelf in the kitchen, a few got moved down to the cellar for storage, and a cart of jars just arrived from the delivery truck and is sitting by the back door. Walking to every single jar in strict shelf order — first to last, one at a time, no exceptions — means crisscrossing the whole bakery, up and down the cellar stairs, back and forth to the door, over and over. It's slow, and it feels like she's making the job harder than it needs to be just to respect an order that nobody actually needs respected — the judges only care about the final swirl, not which room she visited first.

Rosa needs a way to fold every jar's sprinkles into one final swirl — one that goes wild with color change if any single jar is different — without wasting the whole morning marching the shelf start to finish in a fixed line.

---

# SOLUTION 1

Ignore the inconvenience and just do it the old way. Carry one bowl. Go to jar 1 wherever it happens to be, scoop a spoonful in, twist-stir hard. Go find jar 2, scoop, twist-stir. Keep going, strictly in jar-number order, chasing down each next jar wherever it's been moved — kitchen, cellar, back door — until every jar from the first to the last has had its turn and the bowl has been twist-stirred once for each. Whatever color the bowl ends up is the signature swirl. It works, but Rosa spends most of her morning walking instead of baking.

# SOLUTION 2

Call in a couple of helpers. Split the jars into a few clusters — not necessarily in shelf order, just whatever's convenient: "you take everything in the cellar," "you take the cart by the door," "I'll take the kitchen shelf." Each helper gets their own small bowl and twist-stirs their cluster's jars into it in whatever order is handy for them, all at the same time, nobody waiting on anybody else. Once every helper has a finished mini-swirl, Rosa pours all the mini-swirls together into one final bowl and gives that a couple of hard twist-stirs to blend them into the one signature swirl. Because every jar still gets folded in and blended hard at some point, changing any single jar's sprinkles still turns the final swirl unrecognizable — but now three people are working at once instead of one person marching the whole bakery.

# SOLUTION 3

Send two people to start from opposite ends: one starts at jar 1 and heads toward the middle, the other starts at the very last jar and works backward toward the middle, each with their own bowl, twist-stirring as they go. They meet somewhere in the middle of the shelf, having covered every jar between them without either one having to trek across the whole bakery alone. When they meet, they pour their two bowls together and give it one last firm twist-stir to make the final signature swirl. The jars still all get folded in exactly once — just approached from both directions instead of a single long march front to back — and the result is just as sensitive to any single jar being swapped.