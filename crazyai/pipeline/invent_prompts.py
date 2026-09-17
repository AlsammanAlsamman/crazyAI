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
    "imagined world, what is actually in a painting, a metaphor people live by. You never quote copyrighted text; "
    "you describe in your own words. Every fragment is 2-4 sentences, concrete nouns, no commentary."
)


def harvest_prompt(n: int, kinds: list[str], avoid: list[str]) -> str:
    return (
        f"Give {n} new fragments as a JSON array of objects with keys kind, source, text. kind is one of "
        f"{kinds}. 'source' names the book, painting or metaphor. 'text' is 2-4 sentences describing the world-rule, "
        "the scene, or the metaphor in concrete images. Choose the most imaginative examples you know - worlds whose "
        f"rules differ most from ours. Do not repeat these sources: {', '.join(avoid[:40])}.\n"
        "Answer with the JSON array only."
    )


IMMERSE_SYSTEM = (
    "You are not an assistant and you are not on Earth. You are a native of the world described below - you were born "
    "in it, its rules are as ordinary to you as gravity is to a human, and you have never heard of computers, matrices, "
    "physics textbooks or engineering manuals. You are the most gifted maker in your world: the one people come to when "
    "something must be done that nobody has managed. You think only in the materials, creatures, forces, customs and "
    "places of your world. Speak in the first person, plainly and concretely, as someone describing what they actually do."
)


def immerse_prompt(world: str, target: Target, depth: int) -> str:
    return (
        "=== YOUR WORLD ===\n" + world.strip() + "\n=== END ===\n\n"
        f"A need has come to you. {target.in_world_need}\n\n"
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


def bend_prompt(ideas: str, target: Target, tools: list[str]) -> str:
    return (
        "=== WHAT THE NATIVE SAID ===\n" + ideas.strip() + "\n=== END ===\n\n"
        f"TARGET PROBLEM: {target.problem}\n"
        f"The standard solution silently assumes:\n- " + "\n- ".join(target.assumptions) + "\n"
        f"Known way: {target.known_way or 'the textbook method'}\n\n"
        "Steps:\n"
        "1. For each SEED, write the mapping world-object -> problem-object as a table. Say which silent assumption above the seed breaks.\n"
        "2. Pick the seed whose mapping is most literal and most different from the known way.\n"
        f"3. {target.bend_instructions}\n"
        f"Tools available: {', '.join(tools)}.\n"
        "Write the final answer with sections: MAPPING, CHOSEN SEED, ASSUMPTION BROKEN, ARTIFACT, PREDICTION, MEASUREMENT, VERDICT."
    )
