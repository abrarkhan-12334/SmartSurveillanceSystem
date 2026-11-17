import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.weapon_detection import StreakCounter


def test_needs_n_consecutive_detections():
    s = StreakCounter(3)
    assert not s.update(True)
    assert not s.update(True)
    assert s.update(True)


def test_gap_resets_the_streak():
    s = StreakCounter(3)
    s.update(True)
    s.update(True)
    s.update(False)           # missed a frame
    assert not s.update(True)
    assert not s.update(True)
    assert s.update(True)


def test_single_flicker_never_confirms():
    s = StreakCounter(3)
    for positive in (True, False, True, False, True):
        assert not s.update(positive)