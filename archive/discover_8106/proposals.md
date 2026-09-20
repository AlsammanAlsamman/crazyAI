Looking at this stripped down, two of its more unusual clauses — "closure triggers trace-free deletion of a topologically-enclosed region" and "the substrate doesn't shrink because it was edgeless to begin with" — do correspond to real, checkable engineering problems. I'll give those two and flag what I rejected along the way.

## 1. Garbage-cycle collection in reference-counted memory managers

**The problem:** Plain reference counting can't reclaim *cyclic* garbage — objects that reference each other in a closed loop but are unreachable from any root. Real collectors (Bacon & Rajan's trial-deletion algorithm, Python's `gc` module, Java's cycle collectors) solve this by walking the reference graph looking for **subgraphs with no incoming edges from outside the subgraph** — i.e., topologically closed regions — and reclaiming everything inside in one shot.

**How the structure maps:**
- Edgeless lattice → the heap/object graph, which grows by allocation and has no fixed size ceiling.
- External, unscheduled event stream → the *mutator* (the running program) creates and rewires references at its own pace; the collector doesn't control or predict this.
- One marker class closing around a contiguous region of the other → the classic closure test: a set of objects with zero external in-edges (the "boundary" class = live/rooted references; the enclosed class = the cyclic garbage).
- Deletion with no trace, substrate not shrinking → freed cells go back to a free list and get overwritten by future allocations; there's no scar/tombstone marking "something used to be here," and total heap capacity is unaffected.
- One-to-one quantization → each detected closed cycle is collected as exactly one unit per detection pass, never partially.
- Fixed-category, non-repeating emission, agentless → finalizer/destructor invocation: always the same *kind* of callback, but each fires with a unique object identity, and it's the runtime that trips it, not the caller.

**How to check it:** Build (or instrument an existing) reference-counting collector with cycle detection. Verify: (a) after a cycle is collected, no metadata anywhere in the heap distinguishes that memory from cells that were never allocated (a memory-diff/zeroing test); (b) heap high-water-mark is unaffected by how many cycles get collected, only by live-object count; (c) the number of finalizer calls equals exactly the number of closed unreachable subgraphs found, never more or less.

## 2. Trace-free deletion in append-only / immutable ledgers (crypto-shredding for GDPR-style erasure)

**The problem:** Event-sourced systems, blockchains, and backup logs are deliberately append-only for auditability, but privacy law sometimes requires that a specific person's data become *actually unrecoverable* — not just marked deleted, since a "deleted" tombstone is itself evidence the data existed. The standard real technique is **crypto-shredding**: encrypt each subject's records with a unique key, and "delete" by destroying only the key. The ciphertext stays physically in place (log doesn't shrink or get rewritten) but becomes indistinguishable from noise.

**How the structure maps:**
- Edgeless lattice → the append-only log, unbounded by design.
- External stochastic trigger → deletion/erasure requests arrive from users or regulators at unpredictable times the system doesn't schedule.
- Closure around a contiguous region → the boundary of "all records under this subject's key" being identified and enclosed for erasure.
- Trace-free deletion, no shrink → key destruction leaves ciphertext blocks in the log (extent unchanged) but no longer decodable, and ideally no statistical test can distinguish shredded ciphertext from a block that never held real data — literally erasing "the evidence a state was ever there."
- One-to-one coupling → one erasure request destroys exactly one key, and only that one.
- Fixed-category, unique-token, agentless emission → an audit-log entry of type `erasure_event` with a fresh unique ID, written automatically by the system rather than by a human in the moment.

**How to check it:** Implement a toy event-sourced store with per-subject envelope encryption. After key destruction, run statistical/indistinguishability tests (e.g., compare byte-entropy and any classifier trained to detect "real ciphertext" vs. "shredded ciphertext" vs. "random padding") to confirm no distinguishing trace remains. Separately confirm log length and structure are unchanged by erasure count, and that the audit trail's erasure-event count matches the erasure-request count exactly, with no two requests destroying the same key twice.

## What I didn't force

I checked and set aside: Go's territory/capture rule and 2D percolation contour arguments (Peierls-type closed loops) — the geometry of "closure around a contiguous region" matches those very well, but neither has the *trace-free, history-erasing* deletion or the *unique non-repeating signal* clauses doing real work, so the mapping would only use half the structure. I also considered scar-free fetal wound healing in biology, but that's almost certainly the poem's own wordplay ("no seam, no scar") rather than an independent structural match — the biology doesn't have the quantized 1:1 external trigger or the unbounded-token signal, so I'm not including it.