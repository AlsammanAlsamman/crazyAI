# TRANSFORMED PROBLEM

At Summer Camp, there is one Home Cabin and a scatter of other cabins spread through the woods, all connected by a tangle of trails. Every trail has a walking time nailed to a little sign at its start — "9 minutes," "14 minutes," "2 minutes" — and no trail sign ever says a negative number, because nobody walks backward in time.

The Camp Director wants to know, for every single cabin in the woods, the fastest possible number of minutes a runner could get there from Home, carrying a message — picking the best combination of trails the whole way. There's an old, trusted logbook from last year with the correct answer written down for each cabin, and whatever method is used this year has to land on those exact same numbers. The Director also cares about not wasting the counselors' morning doing this — whatever method gets used should take as little fuss and running-around as possible.

A few things about the woods make this interesting:
- A cabin's fastest time only really means something once you're sure no other combination of trails could beat it — so you don't want to declare a cabin "done" too early.
- The trails out of a cabin only matter for figuring out further cabins once that cabin's own time is trustworthy — no point sending a runner onward from a guess.
- To know the fastest time to any one particular cabin — even a cabin right at the edge of the woods — you seemingly have to work out the fastest times to a lot of other cabins along the way too, since you can't be sure which route wins without ruling the others out.