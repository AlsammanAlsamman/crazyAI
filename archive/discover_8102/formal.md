The essential structure here is a **conservation-gated rotary token**, and it resolves fairly cleanly into a formal statement, with one piece left over that doesn't reduce.

**The clean part:**

Let there be a hub with exactly *n* faces, in bijection with *n* arriving directions (no slack in that correspondence — this is stated as an axiom, not an outcome). Each direction *i* carries two flows relative to the hub: an outgoing quantity it owes (delivery) and an incoming quantity it is owed (removal). Define a balance *b_i* = (delivered so far) − (returned so far).

The rule: the hub does not advance while any *b_i* for the currently engaged direction is nonzero. It advances — rotates exactly one face — only at the moment *b_i* returns to zero, i.e., when outflow and inflow have been fully matched for that exchange. No other criterion (urgency, order of arrival, size of request) enters into when it turns.

Effect of advancing: authority — the capacity to be "the one who decides" — transfers to whatever occupies the newly-aligned face, exclusively and for exactly one increment of turning. Everything downstream treats this transfer as self-authorizing: there is no contest, no yielding, no negotiation between the previous holder and the next — subordinate processes simply orient to whichever face is currently lit.

So, stripped down: it's an *n*-way mutual-exclusion rotor where the *only* admission criterion for advancing the exclusive-access token is local reciprocity (flow balances to zero), and where possession of the token is unambiguous, non-contested, and automatically recognized by everything that must obey it. This is close in shape to a token-ring protocol, or a round-robin scheduler with a hard flow-control gate instead of a timer — the ordering is fixed by geometry (the faces), but the *timing* of each handoff is fixed by a settled ledger, not by a clock or a priority rule.

**The part that resists reduction:**

What the hub is *made of* — "carved reflection," and a borrowed "wholeness" from a priest-analog that "can be lent without diminishing" — isn't describing the process, it's describing a property of the substrate the process runs on: something non-rivalrous, transferable without depletion. That's a real structural claim (a conserved-but-shareable quantity, unlike the ash-ledger which is a rivalrous, zero-sum-per-direction thing that must balance), but it sits *alongside* the rotor mechanism rather than inside its formal rule. I can name it, but I can't fold it into the same equation as the balance-gated rotation without forcing a fit that isn't really there — so I'll report it as a second, looser ingredient rather than pretend the whole thing collapses into one clean symmetry.