"""Utility functions for processing hand landmarks in gesture classification."""

import numpy as np


def normalize_gesture_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes landmarks relative to the wrist (index 0).
    Used for static gesture classification.
    """
    landmarks_copy = landmarks.copy()

    base_point = landmarks_copy[
        0
    ]  # Use the wrist landmark (index 0) as the reference point
    landmarks_copy = (
        landmarks_copy - base_point
    ).flatten()  # Center all landmarks around the wrist and flatten to 1D

    max_val = np.abs(
        landmarks_copy
    ).max()  # Find the maximum absolute value for scaling

    if max_val == 0:  # Handle edge case where all normalized values are zero
        return landmarks_copy

    return landmarks_copy / max_val  # Normalize to the range [-1, 1]


def normalize_history_landmarks(landmarks: np.ndarray, frame_shape) -> np.ndarray:
    """
    Normalizes landmarks based on frame dimensions.
    Used for history/movement classification.
    """
    landmarks_copy = landmarks.copy()
    base_point = landmarks_copy[0]  # Use the wrist landmark as the reference point

    height, width = frame_shape[:2]

    scale_factors = np.array(
        [width, height]
    )  # Create scaling factors for x and y coordinates

    landmarks_copy = (
        landmarks_copy - base_point
    ) / scale_factors  # Normalize relative to wrist and scale by frame dimensions

    return landmarks_copy.flatten()  # Flatten the 2D landmarks to a 1D array
