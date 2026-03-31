import math
import os
import pickle

# ── SVM model loading ────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_handsign.pkl')

_svm_bundle = None
if os.path.exists(_MODEL_PATH):
    try:
        with open(_MODEL_PATH, 'rb') as f:
            _svm_bundle = pickle.load(f)
        print(f"[SequenceMatcher] SVM model loaded from {_MODEL_PATH}")
    except Exception as e:
        print(f"[SequenceMatcher] WARNING: Could not load SVM model: {e}")
else:
    print("[SequenceMatcher] No SVM model found — using heuristic fallback.")
    print("                  Run tools/collect_data.py then tools/train_svm.py to train.")

# ── Jutsu combinations ───────────────────────────────────────────────────────
# Note: "Snake" sign = "Serpent" in our label system
JUTSU_COMBOS = {
    # Fire Style: Fireball Jutsu — Snake, Ram, Monkey, Boar, Horse, Tiger
    ("Serpent", "Ram", "Monkey", "Boar", "Horse", "Tiger"): "Fireball Jutsu",

    # Chidori / Lightning Blade — Ox, Hare, Monkey
    ("Ox", "Hare", "Monkey"): "Chidori",

    # Summoning Jutsu — Boar, Dog, Bird, Monkey, Ram
    ("Boar", "Dog", "Bird", "Monkey", "Ram"): "Kuchiyose no Jutsu",
}

# ── Landmark normalization (mirrors collect_data.py) ─────────────────────────
_NUM_LM = 21

def _normalize_hand(landmarks):
    """Translate to wrist-origin, scale to unit max-distance. Returns flat list length 63."""
    if not landmarks:
        return [0.0] * (_NUM_LM * 3)
    wrist_x, wrist_y, wrist_z = landmarks[0][1], landmarks[0][2], landmarks[0][3]
    translated = [(lm[1] - wrist_x, lm[2] - wrist_y, lm[3] - wrist_z)
                  for lm in landmarks]
    max_dist = max((x**2 + y**2) ** 0.5 for x, y, z in translated) or 1.0
    flat = []
    for x, y, z in translated:
        flat += [x / max_dist, y / max_dist, z / max_dist]
    return flat


def _heuristic_sign(landmarks_list):
    """Simple distance-based fallback used when no SVM model exists."""
    if len(landmarks_list) < 2:
        return None
    hand1 = landmarks_list[0]['landmarks']
    hand2 = landmarks_list[1]['landmarks']
    dist = math.hypot(hand2[8][1] - hand1[8][1], hand2[8][2] - hand1[8][2])
    if dist < 50:
        return "Serpent"
    elif dist < 150:
        return "Tiger"
    return None


class SequenceMatcher:
    def __init__(self, debounce_frames=15, svm_confidence=0.6):
        self.current_sequence = []
        self.last_sign = None
        self.frames_since_last_sign = 0
        self.debounce_frames = debounce_frames
        self.svm_confidence = svm_confidence  # min probability to accept SVM prediction

    # ── Gesture detection ──────────────────────────────────────────────────
    def detect_gesture(self, landmarks_list):
        """
        Classifies the current hand pose into a sign name.
        Uses the SVM model if available, otherwise falls back to heuristics.
        """
        if not landmarks_list:
            return None

        if _svm_bundle is not None:
            return self._svm_predict(landmarks_list)
        else:
            return _heuristic_sign(landmarks_list)

    def _svm_predict(self, landmarks_list):
        scaler  = _svm_bundle['scaler']
        clf     = _svm_bundle['clf']
        labels  = _svm_bundle['labels']

        # Build the same 126-feature vector as collect_data.py
        hand0 = _normalize_hand(landmarks_list[0]['landmarks'] if len(landmarks_list) >= 1 else None)
        hand1 = _normalize_hand(landmarks_list[1]['landmarks'] if len(landmarks_list) >= 2 else None)
        features = [hand0 + hand1]

        scaled = scaler.transform(features)
        proba  = clf.predict_proba(scaled)[0]
        idx    = proba.argmax()

        if proba[idx] >= self.svm_confidence:
            return labels[idx] if idx < len(labels) else None
        return None  # uncertain — don't record

    # ── Sequence buffering ─────────────────────────────────────────────────
    def add_sign(self, sign):
        if sign is None:
            self.frames_since_last_sign += 1
            if self.frames_since_last_sign > self.debounce_frames * 3:
                self.current_sequence = []
            return

        if sign != self.last_sign or self.frames_since_last_sign > self.debounce_frames:
            self.current_sequence.append(sign)
            self.last_sign = sign
            self.frames_since_last_sign = 0
            if len(self.current_sequence) > 15:
                self.current_sequence.pop(0)
        else:
            self.frames_since_last_sign = 0

    def check_jutsu(self):
        for combo, jutsu_name in JUTSU_COMBOS.items():
            combo_len = len(combo)
            if (len(self.current_sequence) >= combo_len and
                    tuple(self.current_sequence[-combo_len:]) == combo):
                self.current_sequence = []
                return jutsu_name
        return None

    def evaluate(self, landmarks_list):
        """Main per-frame call. Returns (current_sign, matched_jutsu_or_None)."""
        sign = self.detect_gesture(landmarks_list)
        self.add_sign(sign)
        return sign, self.check_jutsu()
