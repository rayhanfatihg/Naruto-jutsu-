import cv2
import time

# ── Jutsu visual config ───────────────────────────────────────────────────────
_JUTSU_STYLES = {
    "Fireball Jutsu": {
        "tint":   (0, 50, 220),    # deep orange-red (BGR)
        "color":  (0, 140, 255),   # orange text
        "label":  "KATON: GOUKAKYUU NO JUTSU!",
        "alpha":  0.45,
    },
    "Chidori": {
        "tint":   (220, 220, 0),   # electric cyan-yellow (BGR)
        "color":  (255, 255, 0),   # cyan text
        "label":  "CHIDORI!",
        "alpha":  0.45,
    },
    "Kuchiyose no Jutsu": {
        "tint":   (140, 0, 140),   # purple smoke (BGR)
        "color":  (255, 80, 255),  # purple text
        "label":  "KUCHIYOSE NO JUTSU!",
        "alpha":  0.35,
    },
}
_DEFAULT_STYLE = {
    "tint":  (255, 255, 255),
    "color": (0, 255, 0),
    "alpha": 0.3,
}


class VFXRenderer:
    def __init__(self):
        self.active_jutsu = None
        self.start_time   = 0
        self.duration     = 4.0   # seconds to display VFX

    def trigger_jutsu(self, jutsu_name):
        """Activates the visual effect for a jutsu."""
        self.active_jutsu = jutsu_name
        self.start_time   = time.time()

    def render(self, frame):
        """Overlays VFX on the frame if a jutsu is active."""
        if not self.active_jutsu:
            return frame

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active_jutsu = None
            return frame

        h, w, _ = frame.shape
        alpha_fade = max(0.0, 1.0 - (elapsed / self.duration))

        style = _JUTSU_STYLES.get(self.active_jutsu, _DEFAULT_STYLE)
        text  = style.get("label", f"{self.active_jutsu.upper()}!")

        # ── Colour tint overlay ────────────────────────────────────────────
        overlay = frame.copy()
        overlay[:] = style["tint"]
        tint_strength = style["alpha"] * alpha_fade
        cv2.addWeighted(overlay, tint_strength, frame, 1 - tint_strength, 0, frame)

        # ── Jutsu name ribbon at bottom ────────────────────────────────────
        font      = cv2.FONT_HERSHEY_DUPLEX
        scale     = 1.3
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = h - 30

        # Dark ribbon background
        pad = 14
        cv2.rectangle(frame,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + pad),
                      (0, 0, 0), -1)

        # Outline glow
        cv2.putText(frame, text, (tx, ty), font, scale,
                    (255, 255, 255), thickness + 6)
        # Coloured foreground text
        cv2.putText(frame, text, (tx, ty), font, scale,
                    style["color"], thickness)

        # ── Sequence hint at top-center (fades out after 1 s) ─────────────
        if elapsed < 1.0:
            hint = f"[ {self.active_jutsu} ]"
            cv2.putText(frame, hint, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        style["color"], 2)

        return frame
