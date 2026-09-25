"""
Single source of truth: convert MediaPipe hand landmarks
→ 126-D vector (2 hands × 21 landmarks × 3 axes)
"""
import numpy as np
import mediapipe as mp

N_LANDMARKS = 21
N_DIMS    = 3          # x,y,z
MAX_HANDS = 2

def landmarks_to_vector(hand_landmarks_list):
    """
    hand_landmarks_list: list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    returns: np.array shape=(126,)  always!
    """
    vec = np.zeros((MAX_HANDS, N_LANDMARKS, N_DIMS), dtype=np.float32)

    for idx, lands in enumerate(hand_landmarks_list):
        if idx >= MAX_HANDS:
            break
        for j, lm in enumerate(lands.landmark):
            vec[idx, j] = [lm.x, lm.y, lm.z]

    return vec.flatten()          # shape = (126,)
    