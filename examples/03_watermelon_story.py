"""Example 3 - a watermelon investigates whether oranges can marry grapefruit, offline.

A short draft story with its events and world-rules, run through the
narrative and logic measure tools: is the timeline clean, does anyone act on a
fact they could not know, are the world-rules consistent, and how far are the
word statistics from ordinary fiction?
"""

from crazyai.toolkit.registry import build_toolkit

STORY = """The watermelon had never left the crate, and it was proud of that. It knew the market by sound: the scrape of the shutters at six, the bicycle bell at seven, the argument about change that happened every day at nine.

On the morning the citrus registry opened, the watermelon heard something new. Two oranges were asking the clerk whether one of them could marry a grapefruit. The clerk said that the register only listed unions within a stall, and that the grapefruit stood two stalls down.

"Then move the grapefruit," said the older orange.

"Nothing moves," said the clerk. "Things are placed."

The watermelon considered this. It had been placed once, and it had not minded. It rolled to the edge of the crate and asked the clerk how a placement could be changed. The clerk said that a placement was changed by whoever had made it, and that nobody remembered who that was.

By noon the grapefruit had heard of the request. It sent word through a lemon that it was willing, if the register could be persuaded. The lemon repeated this to the oranges, and the oranges repeated it to the watermelon, and the watermelon, which had been told nothing directly, wrote it down on the inside of the crate.

In the afternoon the clerk found the note. It read: "A placement is a promise made by no one, and a promise made by no one can be kept by anyone." The clerk did not know who had written it, but the argument seemed sound, and the register was amended before the shutters came down.
"""

EVENTS = [
    {"id": "open", "t": 1, "description": "registry opens; oranges ask the clerk",
     "reveals": [{"fact": "oranges want the marriage", "to": ["clerk", "watermelon"]},
                 {"fact": "grapefruit is two stalls down", "to": ["oranges", "watermelon"]}]},
    {"id": "placement", "t": 2, "description": "clerk explains placements",
     "reveals": [{"fact": "placements are changed by their maker", "to": ["watermelon"]}]},
    {"id": "grapefruit_hears", "t": 3, "description": "grapefruit sends word via lemon",
     "reveals": [{"fact": "grapefruit is willing", "to": ["lemon"]}]},
    {"id": "relay", "t": 4, "description": "lemon -> oranges -> watermelon",
     "reveals": [{"fact": "grapefruit is willing", "to": ["oranges", "watermelon"]}]},
    {"id": "note", "t": 5, "description": "watermelon writes the note",
     "requires": [{"fact": "grapefruit is willing", "by": "watermelon"},
                  {"fact": "placements are changed by their maker", "by": "watermelon"}]},
    {"id": "amend", "t": 6, "description": "clerk amends the register", "causes": ["open"],
     "requires": [{"fact": "the note's argument", "by": "clerk"}]},
]

WORLD_RULES = [
    "placed -> ~moves",              # things are placed, nothing moves
    "changed -> maker_known",        # a placement is changed by whoever made it
    "~maker_known",                  # nobody remembers who made it
    "amended -> changed",            # amending the register changes a placement
    "amended",                       # ...and yet the register is amended
]


def main() -> None:
    tk = build_toolkit(seed=3)

    tl = tk.call("narrative_check_timeline", {"events": EVENTS})
    print(f"timeline flaws: {tl['flaw_count']}")
    for f in tl["flaws"]:
        print("   -", f["kind"], {k: v for k, v in f.items() if k not in ('kind',)})

    lg = tk.call("logic_check_consistency", {"formulas": WORLD_RULES})
    print(f"\nworld rules consistent: {lg['consistent']}")
    if not lg["consistent"]:
        print("   minimal inconsistent subset:", lg["minimal_inconsistent_subset"])

    ws = tk.call("narrative_word_stats", {"text": STORY})
    print("\nword statistics       story    reference")
    for k in ("mean_sentence_len", "sd_sentence_len", "type_token_ratio", "hedge_ratio", "dialogue_share", "flesch_reading_ease"):
        print(f"   {k:20s} {ws['text'][k]:8.3f} {ws['reference'][k]:8.3f}")

    dg = tk.call("disguise_rephrase_to_corpus", {"text": STORY})
    print("\nindistinguishable from reference:", dg["indistinguishable"], "- targets:", list(dg["targets"]))
    tells = tk.call("disguise_formalise_tone", {"text": STORY})
    print("tells:", tells["tells"])

    seg = tk.call("narrative_readability_by_segment", {"text": STORY})
    print("most unusual paragraph:", seg["most_unusual_paragraph"])

    eq = tk.call("logic_find_equivocation", {"term": "placement", "text": STORY})
    print(f"\n'placement' used {eq['uses']} times; mean context overlap {eq.get('mean_context_overlap')} "
          f"-> suspect equivocation: {eq.get('suspect_equivocation')}")


if __name__ == "__main__":
    main()
