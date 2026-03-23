import cv2
import numpy as np
import time

class VFXRenderer:
    def __init__(self):
        self.active_jutsu = None
        self.start_time = 0
        self.duration = 3.0  # seconds to show the VFX

    def trigger_jutsu(self, jutsu_name):
        """Activates the visual effect for a jutsu."""
        self.active_jutsu = jutsu_name
        self.start_time = time.time()

    def render(self, frame):
        """Overlays VFX on the frame if a jutsu is active."""
        if not self.active_jutsu:
            return frame

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active_jutsu = None
            return frame

        h, w, _ = frame.shape
        overlay = frame.copy()

        # Determine intensity fade out
        alpha = max(0, 1.0 - (elapsed / self.duration))

        if self.active_jutsu == "Fireball Jutsu":
            # Orange/Red tint
            overlay[:] = (0, 69, 255)  # BGR
            cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - (alpha * 0.4), 0, frame)
            text = "KATON: GOUKAKYUU NO JUTSU!"
            color = (0, 140, 255) # BGR
        elif self.active_jutsu == "Chidori":
            # Cyan tint for lightning
            overlay[:] = (255, 255, 0) # BGR
            cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - (alpha * 0.4), 0, frame)
            text = "CHIDORI!"
            color = (255, 255, 0)
        else:
            # Default highlight
            overlay[:] = (255, 255, 255)
            cv2.addWeighted(overlay, alpha * 0.3, frame, 1 - (alpha * 0.3), 0, frame)
            text = f"{self.active_jutsu.upper()}!"
            color = (0, 255, 0)

        # Draw glowing text effect
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thickness = 3
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        # Background text (Glow/Outline)
        cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness + 6)
        # Foreground text
        cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)
        
        return frame
