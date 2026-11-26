"""Torso cursor control with MediaPipe Pose.

The cursor follows the shoulders' midpoint relative to a neutral posture
captured when tracking starts (sit naturally, facing the screen).

    Lean left / right            -> cursor left / right
    Sit up / sink down           -> cursor up / down
    Hold still (~1.2 s)          -> left click (dwell); move away to re-arm
    Shrug (shoulders up, ~0.3 s) -> right click
    Lean towards the camera      -> scroll down; lean back -> scroll up
                                    (the cursor holds still while scrolling)

Offsets are divided by the neutral shoulder width, so it behaves the same
whether you sit close to or far from the camera. Gains live in
calibration.json under "torso".
"""
import time

import calibration_manager
from control_common import CursorController, dist, scroll

NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
MIN_VISIBILITY = 0.5
SHRUG_HOLD = 0.3
PRE_SHRUG_LOOKBACK = 0.3


def _cfg():
    return calibration_manager.config.get("torso", calibration_manager.DEFAULT_CONFIG["torso"])


class TorsoController(CursorController):
    name = "torso"

    def __init__(self, get_screen_size):
        super().__init__(get_screen_size, smoothing=0.2, deadzone=0.004)
        self._reset_gestures()

    def _reset_gestures(self):
        self.dwell_anchor = None
        self.dwell_start = 0.0
        self.dwell_armed = False      # only arms after the cursor has moved
        self.shrug_since = None
        self.shrug_armed = True
        self.shrug_pos = None

    def load(self):
        import mediapipe as mp
        self.mp = mp
        # model_complexity=1: the "full" model ships inside the mediapipe wheel.
        # 0 (lite) and 2 (heavy) are downloaded on first use, which fails in the
        # packaged binary and on offline machines.
        self.model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def recenter(self):
        super().recenter()
        self._reset_gestures()

    def info(self):
        data = super().info()
        if self.dwell_armed and self.dwell_anchor is not None and self.status == "tracking":
            data["dwell"] = round(min(1.0, (time.time() - self.dwell_start) / _cfg()["dwell_time"]), 2)
        return data

    # ------------------------------------------------------------------------
    @staticmethod
    def measure(lm, aspect):
        def p(i):
            return (lm[i].x * aspect, lm[i].y)
        ls, rs, nose = p(L_SHOULDER), p(R_SHOULDER), p(NOSE)
        width = max(dist(ls, rs), 1e-6)
        mx, my = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        return {
            "mx": mx,
            "my": my,
            "width": width,
            # neck height over shoulder width: drops when you shrug
            "neck": (my - nose[1]) / width,
        }

    def process(self, frame, rgb, aspect):
        now = time.time()
        res = self.model.process(rgb)
        pose = getattr(res, "pose_landmarks", None)
        lm = pose.landmark if pose else None
        if (lm is None or lm[L_SHOULDER].visibility < MIN_VISIBILITY
                or lm[R_SHOULDER].visibility < MIN_VISIBILITY):
            self.status = "lost" if self.neutral.ready else "centering"
            self._reset_gestures()
            return "move"

        s = self.measure(lm, aspect)
        self._draw(frame, lm)

        if not self.neutral.ready:
            self.status = "centering"
            self.neutral.add(s)
            return "centering"
        self.status = "tracking"
        cfg = _cfg()
        n = self.neutral

        # --- lean in / out -> scroll -----------------------------------------
        # Checked before the shrug: leaning changes the shoulder width, a shrug
        # does not, and leaning in also shortens the apparent neck.
        scale = s["width"] / n["width"]
        lean = cfg["lean_scroll"]
        if scale > 1 + lean or scale < 1 - lean:
            self.dwell_armed = False
            excess = (scale - 1 - lean) if scale > 1 else (scale - 1 + lean)
            # leaning in (excess > 0) scrolls down = negative wheel delta
            amount = max(-120, min(120, -int(excess * cfg["scroll_speed"])))
            if amount:
                scroll(amount)
                return "scroll"
            return "scroll_ready"
        # --- shrug -> right click ----------------------------------------------
        neck_ratio = s["neck"] / max(n["neck"], 1e-6)
        if neck_ratio < 0.88:
            # Shoulders rising also moves the cursor up; hold it, and click at
            # the position from just before the shrug began.
            if self.shrug_since is None:
                self.shrug_since = now
                self.shrug_pos = self.history.at(now - PRE_SHRUG_LOOKBACK)
            if (neck_ratio < cfg["shrug_ratio"] and self.shrug_armed
                    and now - self.shrug_since >= SHRUG_HOLD):
                self.shrug_armed = False
                self.dwell_armed = False
                if self.click(now, "right", self.shrug_pos):
                    return "right_click"
            return "shrug"
        self.shrug_since = None
        self.shrug_armed = True

        if abs(scale - 1) > lean / 2:
            # Leaning towards/away from the camera also shifts the shoulders
            # vertically; freeze in the transition band so entering a scroll
            # does not drag the cursor.
            return "move"

        # --- cursor ---------------------------------------------------------
        dx = (s["mx"] - n["mx"]) / n["width"]
        dy = (s["my"] - n["my"]) / n["width"]
        self.move(0.5 + cfg["gain_x"] * dx, 0.5 + cfg["gain_y"] * dy, now)

        # --- dwell -> left click -------------------------------------------
        pos = (self.x, self.y)
        if self.dwell_anchor is None or dist(pos, self.dwell_anchor) > cfg["dwell_radius"]:
            if self.dwell_anchor is not None:
                self.dwell_armed = cfg["dwell_enabled"] >= 0.5
            self.dwell_anchor = pos
            self.dwell_start = now
        elif self.dwell_armed and now - self.dwell_start >= cfg["dwell_time"]:
            self.dwell_armed = False
            if self.click(now, "left"):
                return "left_click"
        return "move"

    def _draw(self, frame, lm):
        import cv2
        h, w = frame.shape[:2]
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in (NOSE, L_SHOULDER, R_SHOULDER)]
        cv2.line(frame, pts[1], pts[2], (0, 255, 0), 3)
        for pt in pts:
            cv2.circle(frame, pt, 6, (0, 255, 255), -1)
