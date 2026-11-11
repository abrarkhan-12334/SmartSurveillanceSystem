"""Event creation and handling for surveillance detections."""
import time
from collections import deque
from datetime import datetime

import cv2

from config import SNAPSHOT_DIR, ALERT_COOLDOWN
from modules import database


class EventManager:
    """Single entry point for every detector: events.trigger(...)"""

    def __init__(self, camera_id="cam1"):
        self.camera_id = camera_id
        self._last_fired = {}
        self.recent_alerts = deque(maxlen=20)  # shown live on the dashboard
        database.init_db()

    def trigger(self, event_type, frame, confidence=None, details=""):
        """Returns the event id if an alert was raised, None if on cooldown."""
        now = time.time()
        key = (self.camera_id, event_type)
        if now - self._last_fired.get(key, 0) < ALERT_COOLDOWN:
            return None
        self._last_fired[key] = now

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOT_DIR / f"{event_type}_{self.camera_id}_{stamp}.jpg"
        cv2.imwrite(str(path), frame)

        event_id = database.add_event(
            event_type, self.camera_id, confidence, path, details
        )
        self.recent_alerts.appendleft({
            "id": event_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "confidence": confidence,
            "snapshot": str(path),
        })
        return event_id