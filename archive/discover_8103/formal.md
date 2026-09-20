**Structure, stated as generally as I can:**

There is one shared, globally-ordered collection (call it a heap: elements ranked by a scalar magnitude, "height"). Bound to it is a second structure — a registry of named entries whose members can be in a *protected* or *unprotected* state, and whose protection decays on its own over time (a conditional, expiring flag, not a permanent mark).

Two operations act on this pair, and they are asymmetric in a specific way:

1. **Insertion** — an agent spends one unit of a personal, non-renewable token to add a new element to the collection (at the top, by their own act) and to set the corresponding registry entry to *protected*.
2. **Removal** — an agent who has previously inserted may reclaim their own token back. But the *target* of the removal is not chosen by them and is not necessarily their own element: it is always the current maximum of the whole collection, whoever it belongs to. Ownership of the returned token and ownership of the affected entry are decoupled.

Three constraints complete it:
- Each agent may perform this deposit/withdrawal pair **at most once** — a strict per-agent transaction cap of exactly two crossings, never more.
- The token is conserved (what you put in is what you get back), but the *effect* on the shared structure is not conserved with it — it's redirected by a global rule (max-selection) rather than local intent.
- Removing/erasing an entry is only possible through this paid, rule-bound channel — never by unilateral, costless fiat from outside the structure.

**The net dynamic:** this is a negative-feedback (leveling) process on rank. Every withdrawal event, no matter who triggers it or why, preferentially strips protection from whatever is currently most prominent. Prominence is not rewarded with safety — it is the thing placed at risk every time *anyone* exits. Combined with the one-time-per-agent cap and the requirement that erasure be paid for out of something already invested (rather than asserted for free by an outside authority), the whole thing functions as a scarce, conservation-respecting, anti-accumulation exchange: growth is tolerated, but the tallest element is structurally the first thing sacrificed, and sacrifice always costs the sacrificer something they can never spend again.

**Where it resists clean reduction:** the coupling between the *physical* ordered structure (heap/height) and the *symbolic* one (name/registry/decaying light) is doing real work that a pure data-structure description flattens — there are really two linked state machines here, not one, and the decay-over-time of protection is a third, independent process layered on top of the insert/remove rule rather than derivable from it. I can name all three parts, but I can't collapse them into a single equation without losing the fact that they're separately motivated.