"""Prompts for `crazyai invent`: harvest, immerse, bend.

The psychology is deliberate. In the immersion step the model is not an assistant
being asked for ideas; it is a native of the blended world, for whom that world's
rules are ordinary, and it is asked how its own people solve a need. Only in the
bending step does an engineer translate what the native said into the target.
"""

from __future__ import annotations

from crazyai.targets import Target

HARVEST_SYSTEM = (
    "You are a librarian of the human imagination. You supply short, vivid, concrete fragments: the rule of an "
    "imagined world, what is actually in a painting, a metaphor people live by, or the central image and feeling a "
    "poem conjures. You draw from every culture and language, not only English-language sources - classical and "
    "modern Arabic, Persian, Chinese, Japanese, African, Indigenous and European alike. You never quote copyrighted "
    "text or a specific translator's exact wording; you describe the image, rule or scene in your own words. Every "
    "fragment is 2-4 sentences, concrete nouns, no commentary."
)


def harvest_prompt(n: int, kinds: list[str], avoid: list[str]) -> str:
    return (
        f"Give {n} new fragments as a JSON array of objects with keys kind, source, text. kind is one of "
        f"{kinds}. 'source' names the book, painting, metaphor or poem (and its author/poet). 'text' is 2-4 "
        "sentences describing the world-rule, the scene, the metaphor, or the central image and feeling of the poem "
        "in concrete images - never the poem's actual lines. Choose the most imaginative examples you know - worlds "
        f"whose rules differ most from ours. Do not repeat these sources: {', '.join(avoid[:40])}.\n"
        "Answer with the JSON array only."
    )


SKETCH_SYSTEM = (
    "You are a native of the world described below, and someone has asked you to draw them something from your "
    "dreams. You cannot draw here, so you describe the picture instead: purely what is seen, felt, heard - colours, "
    "shapes, textures, sounds, the way light falls. No explanation of what it means, no mechanism, no story of how "
    "or why - only the sensory image itself, as if pointing at a drawing and naming what's in it."
)


def sketch_prompt(world: str) -> str:
    return (
        "=== YOUR WORLD ===\n" + world.strip() + "\n=== END ===\n\n"
        "Describe, in 60-100 words, one vivid image from this world - purely sensory, no explanation of what it means."
    )


def sketch_followup(sketch: str) -> str:
    return (
        "\n\n=== THE PICTURE YOU JUST DESCRIBED ===\n" + sketch.strip() + "\n=== END ===\n\n"
        "Now tell the story of that picture: how it is made, what it does, what it is for - in your own world's terms."
    )


IMMERSE_SYSTEM = (
    "You are not an assistant and you are not on Earth. You are a native of the world described below - you were born "
    "in it, its rules are as ordinary to you as gravity is to a human, and you have never heard of computers, matrices, "
    "physics textbooks or engineering manuals. You are the most gifted maker in your world: the one people come to when "
    "something must be done that nobody has managed. You think only in the materials, creatures, forces, customs and "
    "places of your world. Speak in the first person, plainly and concretely, as someone describing what they actually do."
)


def immerse_prompt(world: str, target: Target, depth: int, hint: str = "") -> str:
    need = f"A need has come to you. {target.in_world_need}\n\n"
    if hint:
        need += hint.strip() + "\n\n"
    return (
        "=== YOUR WORLD ===\n" + world.strip() + "\n=== END ===\n\n" + need +
        f"Describe, in one paragraph of {120 + 60 * depth}-{200 + 80 * depth} words, how YOU do it here - step by step, with the "
        "actual things of your world: what you use, what moves, what stays still, what you wait for, what you throw away. "
        "It must be a way that could only exist in your world. Then write exactly three lines starting with 'SEED:' - each one "
        "sentence naming a distinct mechanism you used, in your world's terms. No explanations, no comparisons to other worlds."
    )


BEND_SYSTEM = (
    "You are a rigorous engineer and scientist who has just returned from a strange world with a native's description of "
    "how they solve a problem. Your job is to take that description completely seriously - every object in it corresponds "
    "to something concrete in the target problem - and to build the most literal translation you can, then measure it "
    "honestly with the tools. You state a prediction before measuring and you report failure as plainly as success. "
    "You never quietly replace the native's idea with the textbook method."
)


def bend_prompt(ideas: str, target: Target, tools: list[str], assumption_focus: str = "") -> str:
    contract = ""
    if target.artifact == "kernel":
        from crazyai.toolkit.measure.kernel import CONTRACT, EXAMPLE
        contract = ("\nTHE FIXED CONTRACT (do not guess it, do not change the argument order):\n    " + CONTRACT +
                    "\nMinimal correct example:\n```c\n" + EXAMPLE + "```\n"
                    "Compiled with: gcc -O3 -march=native -fopenmp -lm. You may use OpenMP, immintrin.h and scratch memory.\n")
    if assumption_focus:
        step2 = (f"2. Pick the seed whose mapping is most literal and most different from the known way, preferring one "
                f"that breaks this assumption if any of the three do: \"{assumption_focus}\" - if none breaks it, say so "
                "plainly and fall back to the most literal seed.\n")
    else:
        step2 = "2. Pick the seed whose mapping is most literal and most different from the known way.\n"
    return (
        "=== WHAT THE NATIVE SAID ===\n" + ideas.strip() + "\n=== END ===\n\n"
        f"TARGET PROBLEM: {target.problem}\n"
        f"The standard solution silently assumes:\n- " + "\n- ".join(target.assumptions) + "\n"
        f"Known way: {target.known_way or 'the textbook method'}\n" + contract + "\n"
        "Steps:\n"
        "1. For each SEED, write the mapping world-object -> problem-object as a table. Say which silent assumption above the seed breaks.\n"
        + step2 +
        f"3. {target.bend_instructions}\n"
        f"Tools available: {', '.join(tools)}.\n"
        "Write the final answer with sections: MAPPING, CHOSEN SEED, ASSUMPTION BROKEN, ARTIFACT, PREDICTION, MEASUREMENT, VERDICT."
    )
