"""Training data collector for gesture and motion classification models."""

import numpy as np
import cv2
import mediapipe as mp

import os
from collections import deque

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from src import config
from src.processor import normalize_gesture_landmarks, normalize_history_landmarks
from src.visualizer import draw_landmarks, draw_history_points


# Create data directories
os.makedirs(config.DATA_DIR / "gesture", exist_ok=True)
os.makedirs(config.DATA_DIR / "history", exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, config.IMAGE_SHAPE[0])
cap.set(4, config.IMAGE_SHAPE[1])

# Display menu
print("\n--- Data Collector ---")
print("1. Collect Static GESTURES (Single Frames)")
print("2. Collect Movement HISTORY (Sequences of 16 Frames)")

mode = input("Select Mode (1 or 2): ")

# Set labels and directory based on mode
labels = config.GESTURE_LABELS if mode == "1" else config.HISTORY_LABELS
mode_name = "gesture" if mode == "1" else "history"

print(f"\nLabels: {labels}")
print("Press 's' to start recording a label. Press 'q' to quit.")

# Configure MediaPipe hand landmark detector
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"./models/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=config.MAX_HANDS,
)

# Motion history tracking for each hand
histories = {
    0: deque(maxlen=config.HISTORY_LENGTH),
    1: deque(maxlen=config.HISTORY_LENGTH),
}

# Collection variables
DATASET = []
STARTSAVING = False
TOTALDATAPOINTS = 5000

sampleCount = 0
labelCount = 0

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

        # Stop saving when all labels completed
        if len(labels) <= labelCount:
            STARTSAVING = False

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

                # Track finger tip for motion history mode
                if mode == "2":
                    histories[i].append(landmarks[8])

                if STARTSAVING:
                    if mode == "2":
                        # Collect motion history samples
                        if len(histories[i]) == config.HISTORY_LENGTH:
                            norm_hist = normalize_gesture_landmarks(histories[i])

                            DATASET.append(norm_hist)

                            sampleCount += 1

                            # Save and move to next label when quota reached
                            if sampleCount >= TOTALDATAPOINTS:
                                np.save(
                                    os.path.join(
                                        config.DATA_DIR, mode_name, labels[labelCount]
                                    ),
                                    DATASET,
                                )

                                STARTSAVING = False

                                labelCount += 1
                                sampleCount = 0

                                DATASET.clear()

                                for key in histories.keys():
                                    histories[key].clear()

                    else:
                        # Collect static gesture samples
                        norm_gest = normalize_gesture_landmarks(landmarks)

                        DATASET.append(norm_gest)

                        sampleCount += 1

                        # Save and move to next label when quota reached
                        if sampleCount >= TOTALDATAPOINTS:
                            np.save(
                                os.path.join(
                                    config.DATA_DIR, mode_name, labels[labelCount]
                                ),
                                DATASET,
                            )

                            STARTSAVING = False

                            labelCount += 1
                            sampleCount = 0

                            DATASET.clear()

                # Draw motion trail for history mode
                if mode == "2":
                    draw_history_points(frame, histories[i])

            # Clear history for inactive hands
            for i in range(config.MAX_HANDS):
                if i not in active_indices:
                    histories[i].clear()

        elif mode == "2":
            # Clear all histories when no hands detected
            for h in histories:
                histories[h].clear()

        # Calculate label text width for UI panel
        width = max(
            cv2.getTextSize(
                labels[min(labelCount, len(labels) - 1)],
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                2,
            )[0][0],
            cv2.getTextSize(
                str(sampleCount).zfill(len(str(TOTALDATAPOINTS))),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                2,
            )[0][0],
        )

        # Draw current label text
        cv2.putText(
            frame, "Label:", (50, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
        )

        cv2.putText(
            frame,
            labels[min(labelCount, len(labels) - 1)],
            (160, 65),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        # Draw sample count text
        cv2.putText(
            frame, "Count:", (50, 105), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            str(sampleCount if labelCount != len(labels) else TOTALDATAPOINTS).zfill(
                len(str(TOTALDATAPOINTS))
            ),
            (160, 105),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Draw UI panel background
        cv2.rectangle(frame, (30, 20), (180 + width, 130), (255, 0, 0), 3)

        cv2.imshow("Data Collector", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            STARTSAVING = not STARTSAVING


# Cleanup resources
cap.release()
cv2.destroyAllWindows()
