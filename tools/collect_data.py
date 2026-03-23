"""
collect_data.py — Interactive hand sign data recorder
======================================================
Run this script to record hand landmark data for each jutsu sign and save
it to data/handsigns.csv for SVM training.

Controls while the window is open:
  [0–9] / [a–z]  → select a label slot (defined in SIGN_LABELS below)
  [SPACE]         → start 5-second countdown then capture landmarks
  [s]             → save CSV and exit
  [q]             → quit without saving
"""

import cv2
import csv
import os
import sys
import time

# Make sure modules folder is importable from tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.hand_tracker import HandTracker

# ── Configure your sign labels here ──────────────────────────────────────────
SIGN_LABELS = [
    "Tiger",      # 0
    "Serpent",    # 1
    "Ram",        # 2
    "Monkey",     # 3
    "Dragon",     # 4
    "Rat",        # 5
    "Bird",       # 6
    "Ox",         # 7
    "Boar",       # 8
    "Horse",      # 9
    "Dog",        # a
    "Hare",       # b
]
# Map keyboard keys → label index
KEY_MAP = {ord(str(i)): i for i in range(10)}          # 0–9
KEY_MAP.update({ord(chr(ord('a') + i)): 10 + i         # a–z
                for i in range(max(0, len(SIGN_LABELS) - 10))})

# Output CSV path
_OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')

# Number of hands × 21 landmarks × 3 coords (x, y, z) = 126 features per row
# (if only 1 hand visible, the second hand's columns are filled with 0)
_NUM_HANDS = 2
_NUM_LM    = 21
_COLS      = [f"h{h}_lm{i}_{c}"
              for h in range(_NUM_HANDS)
              for i in range(_NUM_LM)
              for c in ('x', 'y', 'z')] + ['label', 'label_name']


def normalize_hand(landmarks):
    """
    Translates landmarks so wrist (id=0) is the origin, then scales so the
    max distance from the wrist is 1.0.  Returns a flat list of (x, y, z)*21.
    """
    if not landmarks:
        return [0.0] * (_NUM_LM * 3)

    wrist_x, wrist_y, wrist_z = (landmarks[0][1], landmarks[0][2], landmarks[0][3])

    translated = [(lm[1] - wrist_x, lm[2] - wrist_y, lm[3] - wrist_z)
                  for lm in landmarks]

    max_dist = max((x**2 + y**2) ** 0.5 for x, y, z in translated) or 1.0

    flat = []
    for x, y, z in translated:
        flat += [x / max_dist, y / max_dist, z / max_dist]
    return flat


def main():
    os.makedirs(os.path.dirname(_OUT_CSV), exist_ok=True)

    # Load existing data so we can append
    rows = []
    if os.path.exists(_OUT_CSV):
        with open(_OUT_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"[collect] Loaded {len(rows)} existing samples from {_OUT_CSV}")

    tracker = HandTracker(max_num_hands=_NUM_HANDS)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_label = 0
    captured_this_session = 0

    print("\n=================================================")
    print(" Naruto Jutsu — Hand Sign Data Collector")
    print("=================================================")
    for k, v in KEY_MAP.items():
        if v < len(SIGN_LABELS):
            key_char = str(v) if v < 10 else chr(ord('a') + v - 10)
            print(f"  [{key_char}] {SIGN_LABELS[v]}")
    print("  [SPACE] 5s countdown → Capture   [s] Save & exit   [q] Quit")
    print("=================================================\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[collect] Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        landmarks_list = tracker.get_landmarks(frame)

        label_name = SIGN_LABELS[current_label] if current_label < len(SIGN_LABELS) else "?"
        h, w, _ = frame.shape

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"Label [{current_label}]: {label_name}  |  "
                           f"Captured this session: {captured_this_session}  |  "
                           f"Total: {len(rows)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # Hands detected indicator
        hand_count = len(landmarks_list)
        status_color = (0, 255, 0) if hand_count == 2 else (0, 165, 255) if hand_count == 1 else (0, 0, 255)
        cv2.putText(frame, f"Hands: {hand_count}/2", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Hand Sign Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[collect] Quit without saving.")
            break

        elif key == ord('s'):
            # Save to CSV
            with open(_OUT_CSV, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=_COLS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[collect] Saved {len(rows)} samples → {_OUT_CSV}")
            break

        elif key == ord(' '):
            # ── 5-second countdown before capture ──────────────────────────
            countdown_start = time.time()
            COUNTDOWN_SEC = 5
            captured = False

            while True:
                ok2, cframe = cap.read()
                if not ok2:
                    break
                cframe = cv2.flip(cframe, 1)
                cframe = tracker.find_hands(cframe, draw=True)
                clm = tracker.get_landmarks(cframe)

                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SEC - int(elapsed)

                ch, cw, _ = cframe.shape

                # Dark overlay banner at top
                cv2.rectangle(cframe, (0, 0), (cw, 45), (0, 0, 0), -1)
                cv2.putText(cframe,
                            f"Label [{current_label}]: {label_name}  |  "
                            f"Captured: {captured_this_session}  |  Total: {len(rows)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

                if remaining > 0:
                    # Big countdown number in centre
                    num_str = str(remaining)
                    scale = 8.0
                    thickness = 12
                    font = cv2.FONT_HERSHEY_DUPLEX
                    (tw, th), _ = cv2.getTextSize(num_str, font, scale, thickness)
                    tx, ty = (cw - tw) // 2, (ch + th) // 2
                    # Shadow
                    cv2.putText(cframe, num_str, (tx + 4, ty + 4),
                                font, scale, (0, 0, 0), thickness + 4)
                    # Coloured number (yellow)
                    cv2.putText(cframe, num_str, (tx, ty),
                                font, scale, (0, 220, 255), thickness)
                    # Instruction strip
                    cv2.putText(cframe, "Hold your pose!",
                                (10, ch - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 2)
                else:
                    # Flash green and capture
                    flash = cframe.copy()
                    flash[:] = (0, 200, 0)
                    cv2.addWeighted(flash, 0.3, cframe, 0.7, 0, cframe)
                    cv2.putText(cframe, "CAPTURED!",
                                (cw // 2 - 160, ch // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 2.5,
                                (255, 255, 255), 5)
                    cv2.imshow("Hand Sign Collector", cframe)
                    cv2.waitKey(400)  # hold flash 400 ms
                    captured = True
                    break

                cv2.imshow("Hand Sign Collector", cframe)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if not captured:
                continue

            # Use the landmarks from the last countdown frame
            if not clm:
                print("[collect] No hands detected at capture time — skipped.")
                continue

            hand0 = normalize_hand(clm[0]['landmarks'] if len(clm) >= 1 else None)
            hand1 = normalize_hand(clm[1]['landmarks'] if len(clm) >= 2 else None)

            row = {}
            for h_idx, flat in enumerate([hand0, hand1]):
                for lm_i in range(_NUM_LM):
                    base = lm_i * 3
                    row[f"h{h_idx}_lm{lm_i}_x"] = flat[base]
                    row[f"h{h_idx}_lm{lm_i}_y"] = flat[base + 1]
                    row[f"h{h_idx}_lm{lm_i}_z"] = flat[base + 2]
            row['label']      = current_label
            row['label_name'] = label_name

            rows.append(row)
            captured_this_session += 1
            print(f"[collect] Captured — label={label_name}, "
                  f"total={len(rows)}, this session={captured_this_session}")

        elif key in KEY_MAP and KEY_MAP[key] < len(SIGN_LABELS):
            current_label = KEY_MAP[key]
            print(f"[collect] Label changed → [{current_label}] {SIGN_LABELS[current_label]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
