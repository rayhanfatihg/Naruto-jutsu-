import cv2
from modules.hand_tracker import HandTracker
from modules.sequence_matcher import SequenceMatcher
from modules.voice_listener import VoiceListener
from modules.vfx_renderer import VFXRenderer

def main():
    print("Initializing Naruto Jutsu Experience...")
    # Initialize components
    tracker = HandTracker(max_num_hands=2)
    matcher = SequenceMatcher(debounce_frames=15)
    listener = VoiceListener(model_name="base")
    renderer = VFXRenderer()

    # Start voice listening thread
    listener.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera loaded. Press 'q' to quit.")

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

        # 3. Voice Listening
        phrase = listener.get_latest_phrase()
        if phrase:
            print(f"User said: '{phrase}'")
            # Heuristic checks for spoken jutsus
            if "fire" in phrase or "katon" in phrase:
                matched_jutsu = "Fireball Jutsu"
            elif "chidori" in phrase or "lightning" in phrase:
                matched_jutsu = "Chidori"

        # 4. Trigger VFX if conditions met
        if matched_jutsu:
            print(f"*** JUTSU ACTIVATED: {matched_jutsu} ***")
            renderer.trigger_jutsu(matched_jutsu)

        # 5. Render VFX overlay
        img = renderer.render(img)

        # Debug UI Overlays for signs
        if current_sign:
            cv2.putText(img, f"Sign: {current_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if hasattr(matcher, 'current_sequence') and matcher.current_sequence:
            seq_str = " -> ".join(matcher.current_sequence)
            cv2.putText(img, f"Seq: {seq_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Naruto Jutsu Tracker", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    listener.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
