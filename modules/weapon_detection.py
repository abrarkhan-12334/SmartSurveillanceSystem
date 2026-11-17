from pathlib import Path

import cv2
from ultralytics import YOLO

import config


class StreakCounter:
    """Counts consecutive positive results. Confirms only after N in a row."""

    def __init__(self, needed):
        self.needed = needed
        self.count = 0

    def update(self, positive):
        self.count = self.count + 1 if positive else 0
        return self.count >= self.needed


class WeaponDetector:
    def __init__(self, model_path=None, conf=None):
        path = Path(model_path or config.WEAPON_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"Weapon model not found: {path}")
        self.model = YOLO(str(path))
        self.conf = conf if conf is not None else config.WEAPON_CONF
        self.streak = StreakCounter(config.WEAPON_CONFIRM_FRAMES)

    def reset(self):
        self.streak.count = 0

    def detect(self, frame):
        """Returns (detections, confirmed). Call once per processed frame."""
        result = self.model(frame, conf=self.conf, imgsz=config.WEAPON_IMGSZ,
                            verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": self.model.names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })
        confirmed = self.streak.update(len(detections) > 0)
        return detections, confirmed

    @staticmethod
    def draw(frame, detections, confirmed):
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color = (0, 0, 255) if confirmed else (0, 165, 255)
            tag = "WEAPON" if confirmed else "possible weapon"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f'{tag}: {d["label"]} {d["conf"]:.2f}',
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame