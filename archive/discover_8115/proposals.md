Looking at the abstract structure on its own terms (not trying to reverse-engineer the riddle's literal referent), two real structures actually match it closely enough to be worth naming. A third candidate (age-length keys in fisheries) came close but the "single-use" and "absorption" properties don't hold cleanly there, so I'm leaving it out rather than forcing it.

**1. Cuckoo hashing with a stash (computer science / data structures)**

This is close to an exact match, property for property:
- *Two independent coordinates → one cell*: an item's two hash values h₁(x), h₂(x) locate two candidate table slots; the item is deterministically bound to whichever the insertion algorithm settles on — not chosen, computed from the key.
- *Domain restriction*: cuckoo tables are used for a specific class of workloads (bulk static/dynamic keys with cheap re-hashing); other key types (e.g. small dense integers) route to direct-addressed arrays elsewhere in the same system.
- *Single-use consumption*: each key is inserted once; it isn't repeatedly re-evaluated against the hash functions.
- *Totality via absorption*: when an insertion would otherwise cycle forever (an adversarially crafted or unlucky key sequence), the structure doesn't reject it — it diverts it into a distinguished overflow "stash," a small side-table that absorbs exactly the cases that broke the normal rule.
- *Porous cells*: the defining mechanic of cuckoo hashing is that insertion cascades — an incoming item can evict a resident, which gets re-placed in its alternate slot, evicting the next, etc. Cells are continuously connected by these eviction chains, not sealed.

**How to check it**: implement cuckoo hashing with a stash (well-studied since Kirsch–Mitzenmacher–Wieder 2009), then run both random and adversarially-constructed key sequences designed to maximize collisions/eviction cycles. Measure stash occupancy vs. table load factor and compare against the published theoretical bounds (stash size needed to keep failure probability negligible). If observed stash usage tracks the theoretical curve, the structural mapping is doing real work, not just decoration.

**2. Cancer stage migration ("Will Rogers phenomenon") in oncology**

- *Two coordinates → discrete cell*: TNM staging bins a patient into a stage group from two accumulated, measured quantities (tumor extent T, nodal involvement N) — read off the patient's own biology, not chosen.
- *Domain restriction*: staging rules differ by cancer type; a given TNM table applies to one tumor class, others use separate staging systems entirely.
- *Absorbing override*: distant metastasis (M1) doesn't get computed from the T/N grid position at all — its presence unconditionally routes the case to Stage IV, a distinguished escape value outside the normal lookup, exactly analogous to a malformed/extreme input being redirected rather than fit into the ordinary grid.
- *Porous cells*: this is the strongest and most literal match. "Stage migration" is a real, named, well-documented effect (Feinstein et al. 1985) — as diagnostic sensitivity improves (better imaging), patients get reclassified into higher stages than before, artificially improving the apparent survival of *every* stage group without any treatment change. Patients genuinely move between the discrete bins after initial assignment.

**How to check it**: this is already empirically checked in the literature — compare stage-specific survival curves for the same cancer type before and after a diagnostic technology upgrade (e.g., pre/post PET-CT adoption) using a registry like SEER. The Will Rogers phenomenon predicts survival improves *within* every stage even with zero change in treatment, purely from reclassification traffic between bins. That prediction is falsifiable against existing registry data and has already been confirmed for several cancers.

Both of these are genuine, checkable structures — I'd flag the cuckoo-hashing match as the cleaner one-to-one fit, and the oncology one as the more consequential if the goal is real-world impact.