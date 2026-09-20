import pytest

from animation import ANIMATIONS
from semantics import LABEL_CONCEPTS, SemanticMatcher, SemanticProfile
from vision import COCO_LABELS, Detection


def det(label, conf=0.9):
    return Detection(label, conf, (0, 0, 10, 10))


@pytest.fixture
def matcher():
    return SemanticMatcher({n: a.semantics for n, a in ANIMATIONS.items()})


def test_label_concepts_only_reference_real_coco_labels():
    assert set(LABEL_CONCEPTS) <= set(COCO_LABELS)


def test_affinity_zero_for_unknown_label():
    assert ANIMATIONS["spider"].semantics.affinity("toaster") == 0.0


def test_affinity_in_unit_interval():
    for anim in ANIMATIONS.values():
        for label in COCO_LABELS:
            assert 0.0 <= anim.semantics.affinity(label) <= 1.0


def test_pick_empty_scene_falls_back_to_idle(matcher):
    chosen, _ = matcher.pick([])
    assert chosen == "idle"


def test_pick_plant_chooses_flower(matcher):
    chosen, _ = matcher.pick([det("potted_plant")])
    assert chosen == "flower"


def test_pick_animal_chooses_spider(matcher):
    chosen, _ = matcher.pick([det("cat")])
    assert chosen == "spider"


def test_pick_person_chooses_greet(matcher):
    chosen, _ = matcher.pick([det("person")])
    assert chosen == "greet"


def test_pick_ball_chooses_wave(matcher):
    chosen, _ = matcher.pick([det("sports_ball")])
    assert chosen == "wave"


def test_pick_laptop_chooses_scan(matcher):
    chosen, _ = matcher.pick([det("laptop")])
    assert chosen == "scan"


def test_backward_direction_expected_labels_for_flower():
    labels = [l for l, _ in ANIMATIONS["flower"].semantics.expected_labels()[:3]]
    assert "potted_plant" in labels and "vase" in labels


def test_hysteresis_keeps_current_on_marginal_change(matcher):
    matcher.pick([det("potted_plant")])
    # a faint cat should not immediately override a strong plant
    chosen, _ = matcher.pick([det("potted_plant", 0.6), det("cat", 0.55)])
    assert chosen == "flower"


def test_confidence_weighting_prefers_stronger_detection(matcher):
    chosen, _ = matcher.pick([det("cat", 0.95), det("potted_plant", 0.3)])
    assert chosen == "spider"
