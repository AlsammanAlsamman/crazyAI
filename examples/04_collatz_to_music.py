"""Example 4 - an unsolved problem, as an image, as music, read backwards, offline.

The Collatz orbits of 1..N become a structure (x = n, y = stopping time,
weight = log peak), the structure becomes an image description, the image a
score, the score is reversed and mapped all the way back. compare_structures
then reports exactly what survived the round trip - and what the "reversed
reading" is actually made of.
"""

import math

from crazyai.toolkit import native
from crazyai.toolkit.registry import build_toolkit


def main() -> None:
    tk = build_toolkit(seed=5)
    N = 64
    orb = tk.call("symbolic_collatz_orbits", {"n": N})
    print(f"backend: {native.backend()}  max stopping time up to {N}: {orb['max_stopping_time']} at n={orb['argmax']}")

    ln, pk = native.collatz(N)
    structure = {"nodes": [{"id": str(k), "x": k, "y": int(ln[k]), "weight": round(math.log(int(pk[k])) + 0.5, 3),
                            "group": int(ln[k]) % 4} for k in range(1, N + 1)]}

    img = tk.call("transform_structure_to_image_description", {"structure": structure})
    score = tk.call("transform_image_description_to_music", {"image": img})
    rev = tk.call("transform_reverse", {"score": score})
    img_back = tk.call("transform_music_to_image_description", {"score": rev})
    back = tk.call("transform_image_description_to_structure", {"image": img_back})

    print(f"image: {len(img['elements'])} circles; score: {len(score['notes'])} notes, "
          f"pitch range {min(n['pitch'] for n in score['notes']):.1f}-{max(n['pitch'] for n in score['notes']):.1f}")
    print("first four notes:", [(n['id'], round(n['pitch'], 1), n['duration']) for n in score['notes'][:4]])
    print("first four notes, reversed:", [(n['id'], round(n['pitch'], 1), n['duration']) for n in rev['notes'][:4]])

    cmp = tk.call("symbolic_compare_structures", {"a": structure, "b": back, "tolerance": 1e-3})
    print(f"\nround trip identical: {cmp['identical']}   order preserved: {cmp['order_preserved']}   "
          f"order reversed: {cmp['order_reversed']}")
    print(f"field differences: {len(cmp['field_differences'])}  (first: {cmp['field_differences'][:2]})")

    # what did the reversed reading do? x was mirrored: n -> N+1-n. So the 'reversed problem'
    # pairs each n with N+1-n. That is an artefact of the encoding, not a property of Collatz.
    fwd = tk.call("transform_music_to_image_description", {"score": score})
    fwd_back = tk.call("transform_image_description_to_structure", {"image": fwd})
    cmp2 = tk.call("symbolic_compare_structures", {"a": structure, "b": fwd_back, "tolerance": 1e-3})
    print(f"un-reversed round trip identical: {cmp2['identical']}")
    print("\nverdict: the reversed reading maps n to N+1-n; it is an encoding artefact, not an invariant.")


if __name__ == "__main__":
    main()
