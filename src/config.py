"""Configuration settings for the hand gesture classification application."""

from pathlib import Path


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Model file paths
GESTURE_MODEL_PATH = MODELS_DIR / "gesture_model.tflite"
HISTORY_MODEL_PATH = MODELS_DIR / "history_model.tflite"

# Gesture classification labels
GESTURE_LABELS = [
    "thumbsup",
    "thumbsdown",
    "pointing",
    "victory",
    "fist",
    "palm",
    "ILoveU",
    "ok",
]

# History (motion) classification labels
HISTORY_LABELS = ["still", "clockwise", "anticlockwise", "move"]

# Model parameters
# Length of gesture history for motion classification
HISTORY_LENGTH = 16
# Input image shape (width, height)
IMAGE_SHAPE = (1280, 720)
# Maximum number of hands to detect
MAX_HANDS = 2

# Bounding box settings
PADDING = 20
BOXCOLOR = (255, 255, 0)
CORNER_LENGTH = 50
