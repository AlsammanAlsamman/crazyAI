from crazyai.diagnose import immersion_metrics


def test_native_text_scores_immersed():
    world = "Glass herons carry lanterns over the salt marsh at dusk."
    m = immersion_metrics("I send my glass herons with lanterns across the marsh, and we wait for dusk.", world)
    assert m["tech_rate"] == 0 and m["meta_rate"] == 0
    assert m["first_person"] > 0 and m["world_rate"] > 0.5


def test_bend_back_is_detected():
    text = "I walk the marsh with my herons. Metaphorically, the loop iterates over the array in the cpu cache."
    m = immersion_metrics(text)
    assert m["meta_rate"] > 0 and m["tech_rate"] > 0 and m["drift"] > 0
