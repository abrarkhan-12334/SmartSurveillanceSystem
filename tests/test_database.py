"""Tests for the surveillance event database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from modules.event_manager import EventManager
from modules import database


def test_trigger_saves_event_and_snapshot():
    em = EventManager("test_cam")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    event_id = em.trigger("fall", frame, 0.91, "test fall")
    assert event_id is not None
    events = database.get_events(event_type="fall", camera_id="test_cam")
    assert events[0]["id"] == event_id
    assert Path(events[0]["snapshot_path"]).exists()


def test_cooldown_blocks_duplicate_alerts():
    em = EventManager("test_cam2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert em.trigger("weapon", frame, 0.8) is not None
    assert em.trigger("weapon", frame, 0.8) is None  # blocked by cooldown