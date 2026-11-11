from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "events.db"
SNAPSHOT_DIR = BASE_DIR / "snapshots"

# Seconds to wait before the same event type from the same camera can fire again.
# Without this, one fallen person would create hundreds of alerts per minute.
ALERT_COOLDOWN = 10

EVENT_TYPES = [
    "weapon", "fall", "fight", "fire", "smoke",
    "restricted_area", "anomaly",
]

for d in (DB_PATH.parent, SNAPSHOT_DIR):
    d.mkdir(parents=True, exist_ok=True)
    # Zone corners as x,y between 0 and 1 (fractions of frame width/height).
# Default: a rectangle covering the right half of the frame.
DEFAULT_ZONE = "0.5,0.1; 0.95,0.1; 0.95,0.95; 0.5,0.95"
# ---- Fall detection ----
POSE_MODEL = "yolo11n-pose.pt"   # downloads automatically on first run (needs internet)
FALL_ANGLE = 60            # torso angle (degrees from vertical) counted as "lying"
FALL_ASPECT = 1.2          # fallback: box width/height ratio if body points are hidden
FALL_HOLD_SECONDS = 1.5    # how long a person must stay down to count as a fall
FALL_DROP_RATIO = 0.4      # sudden drop = body center falls more than 40% of body height in 1s