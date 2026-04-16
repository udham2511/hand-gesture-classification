"""Real-time hand gesture and motion classification system."""

import mediapipe as mp
import cv2

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

import numpy as np
from collections import deque, Counter

from src import config
from src.processor import normalize_gesture_landmarks, normalize_history_landmarks
from src.classifier import TFLiteClassifier
from src.visualizer import draw_landmarks, draw_info_text, draw_history_points


# Load pre-trained gesture and motion classifiers
gesture_ai = TFLiteClassifier(config.GESTURE_MODEL_PATH)
history_ai = TFLiteClassifier(config.HISTORY_MODEL_PATH)

# Motion history tracking for each hand (up to MAX_HANDS)
histories = {
    0: deque(maxlen=config.HISTORY_LENGTH),
    1: deque(maxlen=config.HISTORY_LENGTH),
}

# Store motion classification results for voting
result_history = {
    0: deque(maxlen=config.HISTORY_LENGTH),
    1: deque(maxlen=config.HISTORY_LENGTH),
}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, config.IMAGE_SHAPE[0])
cap.set(4, config.IMAGE_SHAPE[1])

# Configure MediaPipe hand landmark detector
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"./models/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=config.MAX_HANDS,
)

with vision.HandLandmarker.create_from_options(options) as recognizer:
    while cap.isOpened():
        # Read video frame
        ret, frame = cap.read()

        if not ret:
            continue

        # Mirror frame horizontally
        cv2.flip(frame, 1, frame)

        # Convert frame to MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # Detect hand landmarks
        result = recognizer.detect_for_video(
            mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        if result.hand_landmarks:
            active_indices = set()

            for i, hand_landmark in enumerate(result.hand_landmarks):
                if i >= config.MAX_HANDS:
                    break

                # Draw hand skeleton
                draw_landmarks(frame, hand_landmark)

                active_indices.add(i)

                # Convert landmark coordinates to pixel positions
                landmarks = np.array(
                    [
                        [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                        for landmark in hand_landmark
                    ]
                )

                # Classify gesture
                norm_gest = normalize_gesture_landmarks(landmarks)

                gesture_id = gesture_ai.predict(norm_gest)
                gesture_label = config.GESTURE_LABELS[gesture_id]

                label = gesture_label

                # Classify motion if gesture is pointing
                if gesture_id == 2:
                    # Track finger tip position
                    histories[i].append(landmarks[8].astype(int))

                    history_label = "Scanning..."

                    # Classify motion when buffer is full
                    if len(histories[i]) == config.HISTORY_LENGTH:
                        norm_hist = normalize_history_landmarks(
                            np.array(histories[i]), frame.shape
                        )

                        history_id = history_ai.predict(norm_hist)
                        result_history[i].append(history_id)

                        # Use majority voting for stable classification
                        most_common_history_id = Counter(
                            result_history[i]
                        ).most_common()[0][0]

                        history_label = config.HISTORY_LABELS[most_common_history_id]

                    label = history_label

                # Calculate bounding box
                x_coords = landmarks[:, 0].astype(int)
                y_coords = landmarks[:, 1].astype(int)

                bbox = (
                    min(x_coords) - config.PADDING,
                    min(y_coords) - config.PADDING,
                    max(x_coords) + config.PADDING,
                    max(y_coords) + config.PADDING,
                )

                # Draw bounding box and label
                draw_info_text(
                    frame,
                    bbox,
                    label,
                    config.CORNER_LENGTH,
                    config.BOXCOLOR,
                    gesture_id,
                    result.handedness[i][0].display_name[0],
                )

                # Draw motion trail for pointing gesture
                if gesture_id == 2:
                    draw_history_points(frame, histories[i])

            # Clear history for inactive hands
            for i in range(config.MAX_HANDS):
                if i not in active_indices:
                    histories[i].clear()

        else:
            # Clear all histories when no hands detected
            for h in histories:
                histories[h].clear()

        # Display frame
        cv2.imshow("Hand Gestuer Classifier", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
