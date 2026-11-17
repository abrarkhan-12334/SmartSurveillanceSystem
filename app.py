import time
import cv2
import streamlit as st

import config
from modules.camera import Camera
from modules.object_detection import ObjectDetector
from modules.event_manager import EventManager
from modules.restricted_zone import RestrictedZone, parse_points
from modules.fall_detection import FallDetector
from modules.weapon_detection import WeaponDetector

st.set_page_config(page_title="Smart Surveillance System", layout="wide")
st.title("Smart Surveillance System")


@st.cache_resource
def load_detector():
    return ObjectDetector("yolo11n.pt")


@st.cache_resource
def load_fall_detector():
    return FallDetector()


@st.cache_resource
def load_weapon_detector():
    return WeaponDetector()


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
fall_on = st.sidebar.checkbox("Enable fall detection", value=True)

weapon_available = config.WEAPON_MODEL.exists()
weapon_on = st.sidebar.checkbox("Enable weapon detection", value=weapon_available,
                                disabled=not weapon_available)
if not weapon_available:
    st.sidebar.caption("models/weapon.pt not found. Train it first (Phase 5).")

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
    fall_detector = load_fall_detector() if fall_on else None
    if fall_detector:
        fall_detector.reset()
    weapon_detector = load_weapon_detector() if (weapon_on and weapon_available) else None
    if weapon_detector:
        weapon_detector.reset()

    cam = Camera(src)
    fire_test = test_alert
    frame_i = 0
    dets, people, weapons = [], [], []
    try:
        prev = time.time()
        while True:
            frame = cam.read()
            if frame is None:
                continue

            frame_i += 1
            processed = (frame_i - 1) % config.PROCESS_EVERY_N == 0
            weapon_confirmed = False

            # Detect on the clean frame first. On skipped frames, reuse old results.
            if processed:
                dets = detector.detect(frame)
                people = fall_detector.detect(frame) if fall_detector else []
                if weapon_detector:
                    weapons, weapon_confirmed = weapon_detector.detect(frame)

            # Draw objects (people are drawn by the fall detector when it is on)
            to_draw = [d for d in dets if d["label"] != "person"] if fall_detector else dets
            frame = detector.draw(frame, to_draw)

            # --- Restricted area monitoring ---
            if zone:
                intruders = zone.check(frame, dets)
                frame = zone.draw(frame, intruders)
                if intruders and processed:
                    best = max(d["conf"] for d in intruders)
                    if events.trigger("restricted_area", frame, best,
                                      f"{len(intruders)} person(s) in zone"):
                        render_alerts(events)

            # --- Fall detection ---
            if fall_detector:
                frame = fall_detector.draw(frame, people)
                new_falls = [p for p in people if p["new_fall"]] if processed else []
                if new_falls:
                    p = max(new_falls, key=lambda x: x["conf"])
                    note = "sudden drop" if p["sudden"] else "slow descent"
                    angle = f"{p['angle']:.0f} deg" if p["angle"] is not None else "n/a"
                    if events.trigger("fall", frame, p["conf"],
                                      f"person {p['id']}, torso angle {angle}, {note}"):
                        render_alerts(events)

            # --- Weapon detection ---
            if weapon_detector:
                shown_confirmed = weapon_detector.streak.count >= config.WEAPON_CONFIRM_FRAMES
                frame = weapon_detector.draw(frame, weapons, shown_confirmed)
                if weapon_confirmed:
                    best = max(weapons, key=lambda d: d["conf"])
                    if events.trigger("weapon", frame, best["conf"],
                                      f"{best['label']} detected"):
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