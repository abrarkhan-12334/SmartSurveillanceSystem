"""Restricted-zone monitoring."""
import cv2
import numpy as np


def parse_points(text):
    """'0.5,0.1; 0.9,0.1; ...' -> [(0.5, 0.1), ...]. Raises ValueError if invalid."""
    points = []
    for pair in text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        x, y = float(x), float(y)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Values must be between 0 and 1")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("A zone needs at least 3 points")
    return points


class RestrictedZone:
    def __init__(self, points_norm, name="Restricted Zone"):
        self.points_norm = points_norm
        self.name = name

    def _polygon(self, shape):
        h, w = shape[:2]
        return np.array([[int(x * w), int(y * h)] for x, y in self.points_norm],
                        dtype=np.int32)

    def check(self, frame, detections):
        """Return the list of person detections whose feet are inside the zone."""
        poly = self._polygon(frame.shape)
        intruders = []
        for d in detections:
            if d["label"] != "person":
                continue
            x1, y1, x2, y2 = d["box"]
            feet = (float((x1 + x2) / 2), float(y2))  # bottom-center of the box
            if cv2.pointPolygonTest(poly, feet, False) >= 0:
                intruders.append(d)
        return intruders

    def draw(self, frame, intruders):
        poly = self._polygon(frame.shape)
        color = (0, 0, 255) if intruders else (0, 215, 255)  # red / yellow (BGR)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [poly], True, color, 2)

        label = f"{self.name}: INTRUSION" if intruders else self.name
        x, y = poly[0]
        cv2.putText(frame, label, (int(x), max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for d in intruders:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return frame