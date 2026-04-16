"""Hand visualization utilities for gesture recognition system."""

import mediapipe as mp
import cv2
import numpy as np

# MediaPipe drawing utilities
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils


def draw_landmarks(frame, hand_landmark):
    """Draw hand landmarks and connections on frame.

    Args:
        frame: Input video frame
        hand_landmark: Hand landmark positions
    """
    # Red joints, green connections
    mp_drawing.draw_landmarks(
        frame,
        hand_landmark,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec((0, 0, 255), -1, 6),
        mp_drawing.DrawingSpec((0, 255, 0), 3, -1),
    )


def draw_info_text(frame, bbox, label, corner_length, color, gesture_id, hand_label):
    """Draw bounding box corners and classification label.

    Args:
        frame: Input video frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        label: Classification label text
        corner_length: Length of corner lines
        color: Color in BGR format
        gesture_id: Gesture identifier
        hand_label: Hand side ('L' or 'R')
    """
    x1, y1, x2, y2 = bbox

    # Bottom-right corner
    cv2.line(frame, (x2 - corner_length, y2), (x2, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 3)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 3)

    # Top-right corner (skip if both hands detected)
    if gesture_id != 2 or (gesture_id == 2 and hand_label == "L"):
        cv2.line(frame, (x2 - corner_length, y1), (x2, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 3)

    # Top-left corner (skip if both hands detected)
    if gesture_id != 2 or (gesture_id == 2 and hand_label == "R"):
        cv2.line(frame, (x1 + corner_length, y1), (x1, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 3)

    # Calculate label background dimensions
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)

    # Draw label background rectangle
    cv2.rectangle(
        frame,
        (x1, y1 - (height + baseline) - 20),
        (x1 + width + 60, y1 - 10),
        color,
        -1,
    )

    # Draw label text in white
    cv2.putText(
        frame,
        label,
        (x1 + 30, y1 - baseline // 2 - 15),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def draw_history_points(frame, history_deque):
    """Draw finger tip trail over past frames.

    Args:
        frame: Input video frame
        history_deque: Deque of historical finger tip positions
    """
    # Draw points with increasing size for recent points
    for i, point in enumerate(history_deque):
        if point[0] != 0:  # Skip null points (0,0)
            cv2.circle(frame, list(map(int, point)), 1 + int(i / 2), (157, 255, 157), 2)
