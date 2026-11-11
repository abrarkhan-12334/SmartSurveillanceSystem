import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": self.model.names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })
        return detections

    @staticmethod
    def draw(frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{d["label"]} {d["conf"]:.2f}', (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame