import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from modules.fall_detection import FallTracker, is_horizontal, torso_angle


def make_kpts(shoulders, hips):
    k = np.zeros((17, 3))
    for i in (5, 6):
        k[i] = [shoulders[0], shoulders[1], 1.0]
    for i in (11, 12):
        k[i] = [hips[0], hips[1], 1.0]
    return k


def test_upright_person_angle_is_near_zero():
    k = make_kpts((100, 100), (100, 200))
    assert torso_angle(k) < 5


def test_lying_person_angle_is_near_90():
    k = make_kpts((100, 300), (200, 300))
    assert torso_angle(k) > 85


def test_horizontal_check_uses_angle():
    assert is_horizontal(make_kpts((100, 300), (200, 300)), (0, 0, 200, 100))[0]
    assert not is_horizontal(make_kpts((100, 100), (100, 200)), (0, 0, 100, 300))[0]


def test_aspect_ratio_fallback_when_keypoints_hidden():
    hidden = np.zeros((17, 3))
    assert is_horizontal(hidden, (0, 0, 300, 100))[0]       # wide box
    assert not is_horizontal(hidden, (0, 0, 100, 300))[0]   # tall box


def test_fall_needs_hold_time():
    t = FallTracker(hold_seconds=1.5)
    assert not t.update(1, True, 300, 100, 0.0)["fallen"]
    assert not t.update(1, True, 300, 100, 1.0)["fallen"]
    s = t.update(1, True, 300, 100, 1.6)
    assert s["fallen"] and s["new_fall"]
    s = t.update(1, True, 300, 100, 2.0)
    assert s["fallen"] and not s["new_fall"]    # alert fires only once


def test_brief_bend_is_not_a_fall():
    t = FallTracker(hold_seconds=1.5)
    t.update(1, True, 300, 100, 0.0)
    t.update(1, True, 300, 100, 0.8)
    assert not t.update(1, False, 200, 200, 1.0)["fallen"]  # stood back up


def test_can_fall_again_after_getting_up():
    t = FallTracker(hold_seconds=1.5)
    t.update(1, True, 300, 100, 0.0)
    assert t.update(1, True, 300, 100, 1.6)["new_fall"]
    t.update(1, False, 200, 200, 2.5)
    t.update(1, True, 300, 100, 3.0)
    assert t.update(1, True, 300, 100, 4.6)["new_fall"]


def test_sudden_drop_is_detected():
    t = FallTracker(hold_seconds=1.5)
    t.update(1, False, 100, 200, 0.0)
    t.update(1, False, 100, 200, 0.3)
    assert t.update(1, True, 250, 120, 0.6)["sudden"]