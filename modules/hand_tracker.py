import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os

# Connections between hand landmarks (same as classic mediapipe HAND_CONNECTIONS)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17),          # Palm
]

# Path to the downloaded mediapipe hand landmarker model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')


class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._results = None
        self._frame_ts = 0  # monotonically increasing timestamp in ms

    def find_hands(self, img, draw=True):
        """
        Processes an OpenCV BGR frame, runs hand landmark detection,
        draws the skeleton (leaving the trail on screen), and returns
        the annotated frame.
        """
        self._frame_ts += 33  # ~30 fps tick

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._results = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if draw and self._results and self._results.hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in self._results.hand_landmarks:
                # Convert normalized coords to pixel coordinates
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # Draw connections (white lines)
                for start_idx, end_idx in _HAND_CONNECTIONS:
                    cv2.line(img, pts[start_idx], pts[end_idx], (255, 255, 255), 2)

                # Draw landmark dots (green circles)
                for pt in pts:
                    cv2.circle(img, pt, 5, (0, 255, 0), cv2.FILLED)

        return img

    def get_landmarks(self, img):
        """
        Returns a list of dicts: [{'hand_idx': int, 'landmarks': [(id, cx, cy, z), ...]}]
        """
        results = []
        if not self._results or not self._results.hand_landmarks:
            return results

        h, w, _ = img.shape
        for hand_idx, hand_landmarks in enumerate(self._results.hand_landmarks):
            hand_info = {'hand_idx': hand_idx, 'landmarks': []}
            for lm_id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_info['landmarks'].append((lm_id, cx, cy, lm.z))
            results.append(hand_info)
        return results
