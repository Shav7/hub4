import pytest

from animation import ANIMATIONS, MACROS, MICROS
from choreography import MOODS, Choreographer
from poses import LIMB_NAMES
from semantics import SemanticMatcher


def test_every_mood_references_real_animations_of_right_scale():
    for mood in MOODS.values():
        assert all(ANIMATIONS[m].scale == "macro" for m in mood.macros)
        assert all(ANIMATIONS[m].scale == "micro" for m in mood.micros)
        assert mood.entry is None or (ANIMATIONS[mood.entry].scale == "macro" and not ANIMATIONS[mood.entry].loop)
        assert set(mood.rest_pose) == set(LIMB_NAMES)


def test_moods_cover_every_semantic_animation():
    semantic = {n for n, a in ANIMATIONS.items() if a.semantics.concepts or n == "idle"}
    assert semantic <= set(MOODS)


def test_micro_animations_are_small():
    for name in MICROS:
        anim = ANIMATIONS[name]
        for step in range(50):
            assert all(abs(a) <= 20 for a in anim.angles_at(anim.period_s * step / 50).values()), name


def test_micro_offsets_stay_within_limits():
    for mood in MOODS.values():
        for name in mood.micros:
            anim = ANIMATIONS[name]
            for step in range(30):
                for limb, a in anim.angles_at(anim.period_s * step / 30).items():
                    assert -95 <= a + mood.rest_pose[limb] <= 95, (mood.name, name)


def test_entry_gesture_plays_first_after_mood_change():
    ch = Choreographer(seed=1)
    ch.set_mood("spider")
    assert ch.next_clip().name == "crouch_snap"


def test_no_entry_for_idle_start():
    ch = Choreographer(seed=1)
    clip = ch.next_clip()
    assert clip.name in MOODS["idle"].macros


def test_sequence_mixes_macro_and_micro_and_avoids_macro_repeats():
    ch = Choreographer(seed=3)
    ch.set_mood("greet")
    clips = [ch.next_clip() for _ in range(60)]
    scales = {c.scale for c in clips}
    assert scales == {"macro", "micro"}
    macros = [c.name for c in clips if c.scale == "macro"][1:]  # skip entry
    assert all(a != b for a, b in zip(macros, macros[1:]))
    assert len(set(macros)) >= 2


def test_micro_clips_carry_rest_pose_offset_and_macros_do_not():
    ch = Choreographer(seed=5)
    ch.set_mood("spider")
    clips = [ch.next_clip() for _ in range(40)]
    assert all((c.offset == MOODS["spider"].rest_pose) if c.scale == "micro" else c.offset is None for c in clips)


def test_clip_speed_scales_with_mood_tempo():
    fast = Choreographer(seed=7); fast.set_mood("spider")
    slow = Choreographer(seed=7); slow.set_mood("rest")
    f = [fast.next_clip().speed for _ in range(20)]
    s = [slow.next_clip().speed for _ in range(20)]
    assert min(f) > max(s)


def test_clip_durations_reasonable():
    ch = Choreographer(seed=11)
    for mood in MOODS:
        ch.set_mood(mood)
        for _ in range(30):
            c = ch.next_clip()
            assert 0.3 <= c.duration_s <= 15, (mood, c.name, c.duration_s)


def test_unknown_mood_raises():
    with pytest.raises(KeyError):
        Choreographer().set_mood("disco")


def test_semantic_matcher_only_returns_mood_names():
    matcher = SemanticMatcher({n: a.semantics for n, a in ANIMATIONS.items()})
    from vision import Detection
    for label in ("person", "cat", "potted_plant", "laptop", "sports_ball", "couch", "toaster"):
        chosen, _ = matcher.pick([Detection(label, 0.9, (0, 0, 5, 5))])
        assert chosen in MOODS, (label, chosen)
