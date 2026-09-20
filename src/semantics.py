"""Bidirectional semantic matching between detected object labels and animations.

Detected labels are expanded into concept words; each animation declares the
concepts it embodies. Matching runs both ways: labels -> animation (which
animation best fits the scene) and animation -> labels (which objects would
summon a given animation), and the final score combines the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vision import Detection

# What each COCO label *means*, as concepts the animations can also speak.
LABEL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "person": ("human", "social", "face", "attention"),
    "cat": ("animal", "creature", "predator", "small", "playful"),
    "dog": ("animal", "creature", "friendly", "playful"),
    "bird": ("animal", "creature", "small", "flight", "nature"),
    "horse": ("animal", "creature", "large"),
    "sheep": ("animal", "creature"),
    "cow": ("animal", "creature", "large"),
    "elephant": ("animal", "creature", "large"),
    "bear": ("animal", "creature", "predator", "large", "threat"),
    "zebra": ("animal", "creature"),
    "giraffe": ("animal", "creature", "large"),
    "teddy_bear": ("toy", "playful", "soft", "creature"),
    "potted_plant": ("plant", "nature", "flower", "growth", "calm"),
    "vase": ("flower", "plant", "decor", "calm"),
    "broccoli": ("plant", "food", "nature"),
    "banana": ("food", "plant"),
    "apple": ("food", "plant"),
    "orange": ("food", "plant"),
    "carrot": ("food", "plant"),
    "bottle": ("object", "handheld", "drink"),
    "cup": ("object", "handheld", "drink"),
    "wine_glass": ("object", "handheld", "drink"),
    "bowl": ("object", "food"),
    "knife": ("object", "handheld", "sharp", "threat"),
    "scissors": ("object", "handheld", "sharp", "threat"),
    "fork": ("object", "handheld", "sharp"),
    "cell_phone": ("device", "handheld", "screen", "attention"),
    "laptop": ("device", "screen", "work"),
    "tv": ("device", "screen"),
    "keyboard": ("device", "work"),
    "mouse": ("device", "work", "small"),
    "remote": ("device", "handheld"),
    "book": ("object", "paper", "calm", "work"),
    "clock": ("object", "time"),
    "chair": ("furniture", "static"),
    "couch": ("furniture", "static", "calm"),
    "bed": ("furniture", "static", "calm", "rest"),
    "dining_table": ("furniture", "static"),
    "backpack": ("bag", "travel"),
    "handbag": ("bag", "travel"),
    "suitcase": ("bag", "travel", "large"),
    "umbrella": ("object", "weather", "flower", "open"),
    "sports_ball": ("toy", "playful", "round", "motion"),
    "frisbee": ("toy", "playful", "round", "flight", "motion"),
    "kite": ("toy", "flight", "wind", "motion"),
    "skateboard": ("toy", "motion"),
    "bicycle": ("vehicle", "motion"),
    "car": ("vehicle", "motion", "large"),
    "motorcycle": ("vehicle", "motion"),
    "bus": ("vehicle", "motion", "large"),
    "truck": ("vehicle", "motion", "large"),
    "train": ("vehicle", "motion", "large"),
    "airplane": ("vehicle", "flight", "large"),
    "boat": ("vehicle", "water"),
    "pizza": ("food", "round"),
    "donut": ("food", "round"),
    "cake": ("food", "celebration"),
    "tie": ("clothing", "formal", "human"),
}


@dataclass(frozen=True)
class SemanticProfile:
    """The concepts an animation embodies, with weights in (0, 1]."""

    animation: str
    concepts: dict[str, float]

    def expected_labels(self) -> list[tuple[str, float]]:
        """Animation -> labels: which objects would evoke this animation, strongest first."""
        scored = [(label, self.affinity(label)) for label in LABEL_CONCEPTS]
        return sorted([s for s in scored if s[1] > 0], key=lambda s: s[1], reverse=True)

    def affinity(self, label: str) -> float:
        """Label -> animation: fraction of this animation's concept weight the label touches."""
        label_concepts = LABEL_CONCEPTS.get(label, ())
        total = sum(self.concepts.values())
        if not label_concepts or total <= 0:
            return 0.0
        hit = sum(w for c, w in self.concepts.items() if c in label_concepts)
        return hit / total


@dataclass
class SemanticMatcher:
    profiles: dict[str, SemanticProfile]
    fallback: str = "idle"
    switch_margin: float = 0.15  # new winner must beat the current by this much (hysteresis)
    current: str | None = field(default=None, init=False)

    def score(self, detections: list[Detection]) -> dict[str, float]:
        """Both directions combined: forward = confidence-weighted label->animation affinity;
        backward = how much of the animation's expected-label mass the scene actually contains."""
        if not detections:
            return {name: 0.0 for name in self.profiles}
        seen = {d.label: max(d.confidence, *(x.confidence for x in detections if x.label == d.label)) for d in detections}
        scores = {}
        for name, profile in self.profiles.items():
            forward = sum(conf * profile.affinity(label) for label, conf in seen.items())
            expected = profile.expected_labels()
            total_expected = sum(a for _, a in expected) or 1.0
            backward = sum(a for label, a in expected if label in seen) / total_expected
            scores[name] = 0.7 * forward + 0.3 * backward
        return scores

    def pick(self, detections: list[Detection]) -> tuple[str, dict[str, float]]:
        scores = self.score(detections)
        best = max(scores, key=scores.get)
        if scores[best] <= 0.0:
            self.current = self.fallback
            return self.fallback, scores
        if self.current in scores and self.current != best and scores[best] - scores[self.current] < self.switch_margin:
            return self.current, scores
        self.current = best
        return best, scores
