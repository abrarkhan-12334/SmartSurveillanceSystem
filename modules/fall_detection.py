import math
import time
from collections import deque

import cv2
from ultralytics import YOLO

import config

# COCO keypoint indices used here
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def torso_angle(kpts, min_conf=0.3):
    """Angle of the shoulder-to-hip line from vertical: 0 = upright, 90 = lying.
    Returns None if the shoulders or hips are not visible."""
    sh = [kpts[i] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] >= min_conf]
    hp = [kpts[i] for i in (L_HIP, R_HIP) if kpts[i][2] >= min_conf]
    if not sh or not hp:
        return None
    sx = sum(p[0] for p in sh) / len(sh)
    sy = sum(p[1] for p in sh) / len(sh)
    hx = sum(p[0] for p in hp) / len(hp)
    hy = sum(p[1] for p in hp) / len(hp)
    dx, dy = abs(hx - sx), abs(hy - sy)
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy))


def is_horizontal(kpts, box):
    """Returns (horizontal: bool, angle: float or None)."""
    angle = torso_angle(kpts)
    if angle is not None:
        return angle >= config.FALL_ANGLE, angle
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (h > 0 and w / h >= config.FALL_ASPECT), None


class FallTracker:
    """Per-person memory: how long they've been down, and whether they dropped fast."""

    def __init__(self, hold_seconds=None, history_seconds=1.0):
        self.hold = config.FALL_HOLD_SECONDS if hold_seconds is None else hold_seconds
        self.history_seconds = history_seconds
        self.tracks = {}

    def update(self, tid, horizontal, cy, h, now):
        t = self.tracks.setdefault(tid, {
            "hist": deque(), "since": None, "alerted": False,
            "drop_at": None, "last_seen": now,
        })
        t["last_seen"] = now

        hist = t["hist"]
        hist.append((now, cy, h))
        while hist and now - hist[0][0] > self.history_seconds:
            hist.popleft()

        ref_h = max(x[2] for x in hist)
        if len(hist) > 1 and (cy - hist[0][1]) > config.FALL_DROP_RATIO * ref_h:
            t["drop_at"] = now
        sudden = t["drop_at"] is not None and now - t["drop_at"] <= 3.0

        fallen = new_fall = False
        if horizontal:
            if t["since"] is None:
                t["since"] = now
            if now - t["since"] >= self.hold:
                fallen = True
                if not t["alerted"]:
                    t["alerted"] = True
                    new_fall = True
        else:
            t["since"] = None
            t["alerted"] = False

        return {"fallen": fallen, "new_fall": new_fall, "sudden": sudden}

    def prune(self, now, max_age=5.0):
        for tid in [k for k, v in self.tracks.items() if now - v["last_seen"] > max_age]:
            del self.tracks[tid]


class FallDetector:
    def __init__(self, model_path=None, conf=0.4):
        self.model = YOLO(model_path or config.POSE_MODEL)
        self.conf = conf
        self.tracker = FallTracker()

    def reset(self):
        """Call when the video source restarts, so old tracks don't carry over."""
        self.tracker.tracks.clear()
        self.model.predictor = None

    def detect(self, frame):
        result = self.model.track(frame, persist=True, conf=self.conf,
                                  verbose=False)[0]
        now = time.time()
        people = []
        if (result.boxes is None or len(result.boxes) == 0
                or result.keypoints is None):
            self.tracker.prune(now)
            return people

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int).tolist()
        else:
            ids = list(range(len(boxes)))  # fallback if tracking is unavailable

        for tid, box, conf, kp in zip(ids, boxes, confs, kpts):
            x1, y1, x2, y2 = map(int, box)
            horizontal, angle = is_horizontal(kp, (x1, y1, x2, y2))
            state = self.tracker.update(
                tid, horizontal, (y1 + y2) / 2, max(1, y2 - y1), now)
            people.append({
                "id": tid, "box": (x1, y1, x2, y2), "conf": float(conf),
                "angle": angle, "horizontal": horizontal, **state,
            })

        self.tracker.prune(now)
        return people

    @staticmethod
    def draw(frame, people):
        for p in people:
            x1, y1, x2, y2 = p["box"]
            if p["fallen"]:
                color, label = (0, 0, 255), f"FALL DETECTED (id {p['id']})"
            elif p["horizontal"]:
                color, label = (0, 165, 255), f"person {p['id']} down..."
            else:
                color, label = (0, 255, 0), f"person {p['id']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if p["fallen"] else 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame