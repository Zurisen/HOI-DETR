"""
configs.py
----------
Visualisation constants for the Co-DETR hand-object interaction demo.

These rarely need editing. Kept separate so demo.py stays short and the
numbers can be tuned without touching any logic.

Convention: values ending in _REF are tuned at scale=1.0 (≈ 1080p) and are
multiplied by the per-image scale at draw time (see helpers.compute_style).
"""

# ──────────────────────────────────────────────
# Colors (BGR)
# ──────────────────────────────────────────────
COL_HAND      = (32,  50, 220)
COL_FIRST     = (10, 194, 255)
COL_SECOND    = (210, 180,  20)
CLASS_COLORS  = [COL_HAND, COL_FIRST, COL_SECOND]

# Link colors: white halo + class-colored core. Core = source class.
COL_LINE_HF   = COL_HAND        # H -> F link core
COL_LINE_FS   = COL_FIRST       # F -> S link core
COL_LINE_HALO = (255, 255, 255) # shared halo color

# ──────────────────────────────────────────────
# Resolution scaling
# scale = (max(H, W) / REF_EDGE) ** SCALE_EXPONENT, clamped to [MIN, MAX]
# Longer edge (not diagonal) is used so panoramas don't get an unfair
# thickness boost over squares of similar visible size.
# ──────────────────────────────────────────────
REF_EDGE = 1200.0     # was 1920.0 — makes every image score higher on the scale curve
SCALE_EXPONENT  = 0.85
SCALE_MIN = 0.4
SCALE_MAX       = 5.0

# ──────────────────────────────────────────────
# Reference sizes (all in 1080p-reference pixels)
# ──────────────────────────────────────────────
# Boxes
BOX_THICKNESS_REF = 8.0
DOT_RADIUS_REF    = 7
BOX_FILL_ALPHA       = 0.15



# Links
LINK_HALO_REF     = 10
LINK_CORE_REF     = 8

# Detection label (class name + score)
DET_FONT_SCALE_REF   = 0.58
DET_FONT_THICK_REF   = 1.15
DET_PAD_X_REF        = 4.5
DET_PAD_Y_REF        = 1.5

# Link probability badge
LNK_FONT_SCALE_REF   = 0.54
LNK_FONT_THICK_REF   = 1.15
LNK_PAD_X_REF        = 4.5
LNK_PAD_Y_REF        = 3.0

# Background translucency (1.0 = fully opaque)
LABEL_BG_ALPHA       = 0.75
BADGE_BG_ALPHA       = 0.80

# ──────────────────────────────────────────────
# Smart-hiding thresholds (1080p-reference pixels; scaled at draw time)
# ──────────────────────────────────────────────
MIN_BOX_SIDE_FOR_LABEL = 60   # shorter side below this -> no box label
MIN_LINK_LEN_FOR_BADGE = 85   # link length below this -> no badge

# ──────────────────────────────────────────────
# Coincident first/second-object detection
# ──────────────────────────────────────────────
COINCIDENT_IOU = 0.85   # IoU above which a F/S pair is treated as coincident

# ──────────────────────────────────────────────
# Classes (match your trained model's head)
# ──────────────────────────────────────────────
CLASS_NAMES = ['hand', 'firstobject', 'secondobject']
