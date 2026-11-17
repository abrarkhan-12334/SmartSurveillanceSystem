import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from modules.restricted_zone import RestrictedZone, parse_points

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
RIGHT_HALF = parse_points("0.5,0; 1,0; 1,1; 0.5,1")


def person(box):
    return {"label": "person", "conf": 0.9, "box": box}


def test_person_inside_zone_is_detected():
    zone = RestrictedZone(RIGHT_HALF)
    assert len(zone.check(FRAME, [person((400, 100, 500, 300))])) == 1


def test_person_outside_zone_is_ignored():
    zone = RestrictedZone(RIGHT_HALF)
    assert zone.check(FRAME, [person((50, 100, 150, 300))]) == []


def test_non_person_objects_are_ignored():
    zone = RestrictedZone(RIGHT_HALF)
    car = {"label": "car", "conf": 0.9, "box": (400, 100, 500, 300)}
    assert zone.check(FRAME, [car]) == []


def test_invalid_zone_text_raises():
    with pytest.raises(ValueError):
        parse_points("0.1,0.1; 0.5,0.5")      # only 2 points
    with pytest.raises(ValueError):
        parse_points("2,0; 0,0; 0,1")         # value out of range