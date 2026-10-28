import cv2
import time
from ultralytics import YOLO


def main():
    print("Loading AI model...")

    # Load a lightweight object detection model
    model = YOLO("yolo11n.pt")

    # Open the laptop webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Smart Surveillance System started!")
    print("Press Q to exit.")

    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            # Reduce the processing load
            frame = cv2.resize(frame, (640, 480))

            # Detect objects using the CPU
            results = model.predict(
                source=frame,
                imgsz=416,
                conf=0.40,
                device="cpu",
                verbose=False
            )

            output = results[0].plot()

            # Calculate approximate FPS
            current_time = time.time()
            elapsed = current_time - previous_time
            fps = 1 / elapsed if elapsed > 0 else 0
            previous_time = current_time

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Smart Surveillance System", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Surveillance system stopped.")


if __name__ == "__main__":
    main()