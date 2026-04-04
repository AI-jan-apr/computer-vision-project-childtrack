# كل الأرقام والإعدادات في مكان واحد
# Time Window، Distance، Threshold، وغيرها
# ذكر جميع القيم اللي ممكن نحتاج نغيرها بسهولة في المستقبل

#detect.py prameters:

# ---------------------------
# YOLO Detection Settings
# ---------------------------
YOLO_MODEL_PATH = "models/yolo/best.pt"
YOLO_CONFIDENCE = 0.3

# ---------------------------
# Input Source
# ---------------------------
#for camera input: 0
#for video input: "data/videos/test.mp4"
VIDEO_SOURCE = "data/videos/test.mp4"

# ---------------------------
# Display Settings
# ---------------------------
WINDOW_NAME = "CCMS - Detection Test"

# ---------------------------
# Drawing Colors (BGR)
# ---------------------------
ADULT_COLOR = (255, 100, 0)   # Blue-ish
CHILD_COLOR = (0, 200, 0)     # Green
UNKNOWN_COLOR = (200, 200, 200)

# ---------------------------
# Text / UI
# ---------------------------
FONT_SCALE = 0.55
FONT_THICKNESS = 2
BOX_THICKNESS = 2
CENTROID_RADIUS = 4

# ---------------------------
# Later Stages (prepare from now)
# ---------------------------
ENTRY_TIME_WINDOW = 4            # seconds
DISTANCE_THRESHOLD = 120         # pixels (initial guess)
EXIT_CONFIRM_FRAMES = 10

# ---------------------------
# Output Paths (for later)
# ---------------------------
OUTPUT_RESULTS_DIR = "outputs/results/"
OUTPUT_ALERTS_DIR = "outputs/alerts/"
OUTPUT_LOGS_DIR = "outputs/logs/"

#======================================================================================

#track.py parameters:

TRACK_MAX_AGE      = 60 #30     # frames to keep a lost track before deleting it
TRACK_MIN_HITS     = 3      # frames a person must appear before getting a stable ID
TRACK_IOU_THRESH   = 0.3    # IoU threshold for matching between frames

# ──────────────────────────────────────
# Direction Settings
# ──────────────────────────────────────
ENTRY_DIRECTION    = 'left' # 'right' or 'left' — depends on your camera setup
MOVEMENT_THRESHOLD = 3      # pixels — movement below this = stationary

YOLO_MODEL_PATH = YOLO_MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\child_track\computer-vision-project-childtrack\yolo\best (2).pt"


