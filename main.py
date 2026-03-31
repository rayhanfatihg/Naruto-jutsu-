import cv2
import time
import os
import pygame
from modules.hand_tracker import HandTracker
from modules.sequence_matcher import SequenceMatcher
from modules.voice_listener import VoiceListener
from modules.vfx_renderer import VFXRenderer

# ── Sound setup ───────────────────────────────────────────────────────────────
pygame.mixer.init()

_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'sounds')
_HANDSIGN_SOUNDS = [
    pygame.mixer.Sound(os.path.join(_SOUNDS_DIR, 'handsign.wav')),
    pygame.mixer.Sound(os.path.join(_SOUNDS_DIR, 'handsign_2.wav')),
]
_sound_index = 0  # cycles 0 → 1 → 2 → 0 → ...


def play_next_handsign_sound():
    """Play the next sound in the rotation and advance the index."""
    global _sound_index
    _HANDSIGN_SOUNDS[_sound_index].play()
    _sound_index = (_sound_index + 1) % len(_HANDSIGN_SOUNDS)


def main():
    global _sound_index

    print("Initializing Naruto Jutsu Experience...")
    tracker  = HandTracker(max_num_hands=2)
    matcher  = SequenceMatcher(debounce_frames=15)
    listener = VoiceListener(model_name="base")
    renderer = VFXRenderer()

    listener.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera loaded. Press 'q' to quit.")

    last_sign = None       # track previous sign so we only trigger on NEW signs
    last_sign_time = 0.0   # timestamp of last accepted sign
    SIGN_COOLDOWN = 0.5    # seconds to wait before accepting the next sign

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Flip image for a mirror effect
        img = cv2.flip(img, 1)

        # 1. Hand Tracking & Drawing Landmarks
        img = tracker.find_hands(img, draw=True)
        landmarks = tracker.get_landmarks(img)

        # 2. Sequence Matching
        current_sign, matched_jutsu = matcher.evaluate(landmarks)

        # 3. Play sound on every NEW hand sign detected (with cooldown)
        now = time.time()
        if current_sign and current_sign != last_sign and (now - last_sign_time) >= SIGN_COOLDOWN:
            play_next_handsign_sound()
            print(f"[sound] Sign: {current_sign} → "
                  f"handsign_{(_sound_index - 1) % len(_HANDSIGN_SOUNDS) + 1}.wav played")
            last_sign_time = now
        last_sign = current_sign

        # 4. Voice Listening
        phrase = listener.get_latest_phrase()
        if phrase:
            print(f"User said: '{phrase}'")
            if "fire" in phrase or "katon" in phrase or "fireball" in phrase:
                matched_jutsu = "Fireball Jutsu"
            elif "chidori" in phrase or "lightning" in phrase:
                matched_jutsu = "Chidori"
            elif "kuchiyose" in phrase or "summoning" in phrase:
                matched_jutsu = "Kuchiyose no Jutsu"

        # 5. Trigger VFX if jutsu completed
        if matched_jutsu:
            print(f"*** JUTSU ACTIVATED: {matched_jutsu} ***")
            renderer.trigger_jutsu(matched_jutsu)

        # 6. Render VFX overlay
        img = renderer.render(img)

        # Debug UI Overlays
        if current_sign:
            cv2.putText(img, f"Sign: {current_sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if matcher.current_sequence:
            seq_str = " -> ".join(matcher.current_sequence)
            cv2.putText(img, f"Seq: {seq_str}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Naruto Jutsu Tracker", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    listener.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
