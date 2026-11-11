import time
import cv2
import streamlit as st

import config
from modules.camera import Camera
from modules.object_detection import ObjectDetector
from modules.event_manager import EventManager
from modules.restricted_zone import RestrictedZone, parse_points

st.set_page_config(page_title="Smart Surveillance System", layout="wide")
st.title("Smart Surveillance System")


@st.cache_resource
def load_detector():
    return ObjectDetector("yolo11n.pt")


@st.cache_resource
def load_events():
    return EventManager("cam1")


# ---------- Sidebar ----------
source = st.sidebar.text_input("Source (0 = webcam, or video path / RTSP URL)", "0")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("---")
zone_on = st.sidebar.checkbox("Enable restricted zone", value=True)
zone_text = st.sidebar.text_area("Zone points (x,y; x,y; ...)", config.DEFAULT_ZONE)
zone = None
if zone_on:
    try:
        zone = RestrictedZone(parse_points(zone_text))
    except ValueError as e:
        st.sidebar.error(f"Invalid zone: {e}")

st.sidebar.markdown("---")
run = st.sidebar.toggle("Start camera")
test_alert = st.sidebar.button("Send test alert")

st.sidebar.subheader("Recent alerts")
alerts_box = st.sidebar.empty()
fps_box = st.sidebar.empty()

frame_box = st.empty()


def render_alerts(events):
    if not events.recent_alerts:
        alerts_box.caption("No alerts yet.")
        return
    lines = [f"**{a['time']}**  {a['type']}  (id {a['id']})"
             for a in list(events.recent_alerts)[:8]]
    alerts_box.markdown("\n\n".join(lines))


# ---------- Main loop ----------
events = load_events()
render_alerts(events)

if run:
    src = int(source) if source.isdigit() else source
    detector = load_detector()
    detector.conf = conf
    cam = Camera(src)
    fire_test = test_alert
    try:
        prev = time.time()
        while True:
            frame = cam.read()
            if frame is None:
                continue

            dets = detector.detect(frame)
            frame = detector.draw(frame, dets)

            # --- Restricted area monitoring ---
            if zone:
                intruders = zone.check(frame, dets)
                frame = zone.draw(frame, intruders)
                if intruders:
                    best = max(d["conf"] for d in intruders)
                    if events.trigger("restricted_area", frame, best,
                                      f"{len(intruders)} person(s) in zone"):
                        render_alerts(events)

            if fire_test:
                events.trigger("anomaly", frame, 0.99, "manual test")
                fire_test = False
                render_alerts(events)

            now = time.time()
            fps_box.metric("FPS", f"{1 / (now - prev):.1f}")
            prev = now
            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)
    finally:
        cam.release()
else:
    st.info("Turn on 'Start camera' in the sidebar.")