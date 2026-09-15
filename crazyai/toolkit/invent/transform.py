"""transform.* - cross-modal, reproducible encodings.

structure -> image description -> music score -> (reverse) -> image description -> structure

All four maps are deterministic and the forward/backward pairs are exact
inverses on the fields they carry, so `measure.symbolic.compare_structures`
can report precisely what a round trip preserved. A *structure* is
{"nodes": [{"id", "x", "y", "weight", "group"}]}; an *image description* is a
list of drawable elements; a *score* is a list of notes.
"""

from __future__ import annotations

from typing import Any

from crazyai.toolkit.registry import tool

_PALETTE = ["#4B0082", "#8A2BE2", "#FF7F50", "#FFBF00", "#00CED1", "#2E8B57", "#DC143C", "#708090"]
_PITCH_LO, _PITCH_HI = 36, 96  # MIDI range used for y


def _norm_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, n in enumerate(nodes):
        out.append({
            "id": str(n.get("id", i)), "x": float(n.get("x", i)), "y": float(n.get("y", 0.0)),
            "weight": float(n.get("weight", 1.0)), "group": int(n.get("group", 0)),
        })
    return out


@tool("transform", "invent")
def structure_to_image_description(structure: dict) -> dict:
    """Map a structure {nodes:[{id,x,y,weight,group}]} to a precise, renderable image description.

    Args:
        structure: The structure to encode.
    """
    nodes = _norm_nodes(structure.get("nodes", []))
    if not nodes:
        return {"error": "structure has no nodes"}
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    elements = [{
        "kind": "circle", "id": n["id"], "cx": n["x"], "cy": n["y"], "r": n["weight"],
        "fill": _PALETTE[n["group"] % len(_PALETTE)], "group": n["group"],
    } for n in nodes]
    return {
        "canvas": {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)},
        "elements": elements,
        "legend": "cx,cy = node position; r = weight; fill = group colour",
    }


@tool("transform", "invent")
def image_description_to_music(image: dict) -> dict:
    """Map an image description to a score: pitch from cy, duration from r, velocity from cx, instrument from group.

    Args:
        image: Output of transform_structure_to_image_description.
    """
    els = [e for e in image.get("elements", []) if e.get("kind") == "circle"]
    if not els:
        return {"error": "no circle elements"}
    c = image.get("canvas") or {}
    y0, y1 = c.get("y_min", min(e["cy"] for e in els)), c.get("y_max", max(e["cy"] for e in els))
    x0, x1 = c.get("x_min", min(e["cx"] for e in els)), c.get("x_max", max(e["cx"] for e in els))
    notes = []
    for e in sorted(els, key=lambda e: (e["cx"], e["id"])):
        py = 0.5 if y1 == y0 else (e["cy"] - y0) / (y1 - y0)
        px = 0.5 if x1 == x0 else (e["cx"] - x0) / (x1 - x0)
        notes.append({
            "id": e["id"], "pitch": round(_PITCH_LO + py * (_PITCH_HI - _PITCH_LO), 4),
            "duration": e["r"], "velocity": round(40 + px * 87, 4), "instrument": e.get("group", 0),
            "_cx": e["cx"], "_cy": e["cy"],  # carried so the map is invertible
        })
    return {"tempo": 120, "notes": notes, "canvas": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}}


@tool("transform", "invent")
def reverse(score: dict) -> dict:
    """Reverse a score in time (last note first). Everything else is unchanged.

    Args:
        score: Output of transform_image_description_to_music.
    """
    notes = list(score.get("notes", []))
    return {**score, "notes": notes[::-1], "reversed": not score.get("reversed", False)}


@tool("transform", "invent")
def music_to_image_description(score: dict) -> dict:
    """Map a score back to an image description (inverse of image_description_to_music).

    Args:
        score: A score, possibly reversed.
    """
    c = score.get("canvas") or {}
    notes = score.get("notes", [])
    if not notes:
        return {"error": "empty score"}
    y0, y1 = c.get("y_min", 0.0), c.get("y_max", 1.0)
    x0, x1 = c.get("x_min", 0.0), c.get("x_max", 1.0)
    elements = []
    for i, n in enumerate(notes):
        py = (n["pitch"] - _PITCH_LO) / (_PITCH_HI - _PITCH_LO)
        px = (n["velocity"] - 40) / 87
        cy = y0 + py * (y1 - y0)
        cx = x0 + px * (x1 - x0)
        if score.get("reversed"):
            # time has been reversed: positions in time now run backwards, so x is mirrored
            cx = x0 + x1 - cx
        elements.append({"kind": "circle", "id": n["id"], "cx": round(cx, 6), "cy": round(cy, 6),
                         "r": n["duration"], "fill": _PALETTE[n.get("instrument", 0) % len(_PALETTE)],
                         "group": n.get("instrument", 0), "order": i})
    return {"canvas": c, "elements": elements}


@tool("transform", "invent")
def image_description_to_structure(image: dict) -> dict:
    """Map an image description back to a structure (inverse of structure_to_image_description).

    Args:
        image: An image description.
    """
    nodes = [{"id": e["id"], "x": e["cx"], "y": e["cy"], "weight": e["r"], "group": e.get("group", 0)}
             for e in image.get("elements", []) if e.get("kind") == "circle"]
    return {"nodes": nodes}
