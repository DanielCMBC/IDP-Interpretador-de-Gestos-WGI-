"""Eye / head-gaze cursor control with MediaPipe Face Mesh (iris landmarks on).

Pure eye tracking from a webcam is too noisy to point with, so the cursor
follows head orientation plus a small iris term.

    Look around                    -> move the cursor
    Wink LEFT eye                  -> left click
    Wink LEFT eye twice, quickly   -> double click
    Hold LEFT eye shut (~0.6 s)    -> press and drag; move your head to drag,
                                      open the eye to drop
    Wink RIGHT eye                 -> right click
    Hold mouth open, then nod      -> scroll (head down scrolls down)
    Normal blinks (both eyes)      -> ignored
    Laughing / coughing / sneezing -> cursor holds still, no clicks

Head orientation: the rigid part of the face (upper face outline, forehead,
nose bridge -- ~33 landmarks that expressions barely move) captured while
centering is the reference; every frame the best-fit rotation from that
reference to the current points (Kabsch) gives where the face points. Fitting
many points averages landmark noise away, and cheeks/jaw/mouth/nose tip --
which move when you laugh or talk -- are not used.

Eye openness is measured against each eye's *own recent open level* (learned
continuously), not a one-off snapshot: a snapshot taken while squinting made
every later wink look half as deep as it was.

"Left"/"right" are the user's own eyes. Settings live in calibration.json
under "eyes". Set WGI_TRACE=<file.csv> to log per-frame signals (numbers only).
"""
import collections
import math
import os
import time

import numpy as np

import calibration_manager
from control_common import CursorController, OneEuroFilter, dist, scroll

NOSE_TIP = 1
# Rigid face landmarks for head orientation (Face Mesh indices).
RIGID_POINTS = (
    # upper face outline: forehead to temples / ears (the jaw part is left out)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    109, 67, 103, 54, 21, 162, 127, 234,
    # forehead
    151, 9, 108, 337, 69, 299,
    # nose bridge (bone and cartilage; not the tip, which scrunches)
    168, 6, 197, 195, 122, 351, 196, 419, 193, 417,
)
EYE_A = dict(corners=(33, 133), lids=((160, 144), (158, 153)), top=159, bottom=145, iris=468)
EYE_B = dict(corners=(362, 263), lids=((385, 380), (387, 373)), top=386, bottom=374, iris=473)
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308

# Clicks land where the cursor was this long before the eye started closing:
# the closing lid drags the face landmarks a little for a few frames before
# the eye counts as closed, which would otherwise nudge every click.
PRE_WINK_LOOKBACK = 0.2
# Head speed (for the sneeze/cough guard) is measured over this window, not
# frame to frame: a real jerk keeps moving for 0.1 s, landmark noise does not.
JERK_WINDOW = 0.1
# An eye counts as closed below blink_ratio and as open again only above
# blink_ratio + EYE_HYSTERESIS, so a partial wink does not flicker in and out.
EYE_HYSTERESIS = 0.12
# Open-eye level = median of the eye's openness over the last
# OPEN_LEVEL_FRAMES frames in which neither eye was closing (the median, so a
# single glitch frame cannot inflate it).
OPEN_LEVEL_FRAMES = 120
OPEN_LEVEL_PERCENTILE = 50
WINK_MIN_FRAMES = 2


def _cfg():
    return calibration_manager.config.get("eyes", calibration_manager.DEFAULT_CONFIG["eyes"])


class _Wink:
    """Tracks one eye's closure from start to reopening."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.since = None       # when the wink started
        self.pos = None         # cursor position just before it started
        self.frames = 0
        self.both_frames = 0    # frames where the other eye was closed too

    @property
    def active(self):
        return self.since is not None

    def start(self, now, pos):
        self.since, self.pos, self.frames, self.both_frames = now, pos, 1, 0

    def looks_like_blink(self):
        # Mostly both-eyes-closed: a normal blink that happened to start
        # with one eye a frame early.
        return self.frames > 0 and self.both_frames * 2 > self.frames


class _OpenLevel:
    """Running estimate of how open one eye is when the user is not winking."""

    def __init__(self):
        self.samples = collections.deque(maxlen=OPEN_LEVEL_FRAMES)
        self.value = None

    def reset(self, initial=None):
        self.samples.clear()
        self.value = initial

    def add(self, ear):
        self.samples.append(ear)
        if len(self.samples) >= 15:
            self.value = float(np.percentile(self.samples, OPEN_LEVEL_PERCENTILE))


def kabsch_forward(ref, pts):
    """Best-fit rotation taking the centred reference points onto the centred
    current points; returns where the face's forward axis (towards the
    camera, -z) now points. x grows as the user turns right (mirrored image),
    y as they look down; both are ~sin(angle)."""
    p = pts - pts.mean(axis=0)
    hmat = ref.T @ p
    u, _, vt = np.linalg.svd(hmat)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag((1.0, 1.0, d)) @ u.T
    fwd = r @ np.array((0.0, 0.0, -1.0))
    return float(fwd[0]), float(fwd[1])


class EyeController(CursorController):
    name = "eyes"

    def __init__(self, get_screen_size):
        cfg = _cfg()
        super().__init__(get_screen_size,
                         smoother=OneEuroFilter(cfg["smooth_min_cutoff"], cfg["smooth_beta"],
                                                d_cutoff=0.5),
                         deadband_px=cfg["deadband_px"])
        self.left = _Wink()
        self.right = _Wink()
        self.open_l = _OpenLevel()
        self.open_r = _OpenLevel()
        self.l_closed = self.r_closed = False
        self.dragging = False
        self.pending_click = None     # (time, pos) of a left wink awaiting a possible 2nd wink
        self.scroll_anchor = None
        self.mouth_open_since = None
        self.hold_until = 0.0         # cursor/clicks frozen until then (laugh, cough, sneeze)
        self.heads = []               # recent (t, head_x, head_y), for jerk detection
        self.ref_samples = []
        self.ref = None               # neutral rigid points, centred
        self.trace = None

    def load(self):
        import mediapipe as mp
        self.mp = mp
        self.model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # adds the 10 iris landmarks
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        path = os.environ.get("WGI_TRACE")
        if path:
            self.trace = open(path, "a", buffering=1)
            self.trace.write("t,head_x,head_y,ear_l,ear_r,rl,rr,mar,target_x,target_y,"
                             "cursor_x,cursor_y,speed,disturbed,action\n")

    def close(self):
        super().close()
        if self.trace:
            self.trace.close()
            self.trace = None

    def _reset_gestures(self):
        self.mouse_up()
        self.dragging = False
        self.left.reset()
        self.right.reset()
        self.l_closed = self.r_closed = False
        self.pending_click = None
        self.scroll_anchor = None
        self.mouth_open_since = None
        self.heads = []

    def recenter(self):
        super().recenter()
        self._reset_gestures()
        self.hold_until = 0.0
        self.ref_samples = []
        self.ref = None

    def info(self):
        data = super().info()
        if self.status == "tracking" and time.time() < self.hold_until:
            data["status"] = "paused"
        return data

    # ------------------------------------------------------------------------
    @staticmethod
    def measure(lm, aspect):
        """Signals from one face, plus the rigid points as an (N, 3) array.
        x (and z, which MediaPipe scales like x) are rescaled by the frame
        aspect so geometry is not squashed.

        The frame is mirrored before it reaches MediaPipe, so -- like a mirror
        -- the user's LEFT eye is the one on the image's left. Eyes are
        assigned by position rather than by landmark index so this cannot
        silently swap.
        """
        def p(i):
            return (lm[i].x * aspect, lm[i].y)

        def center(eye):
            c1, c2 = eye["corners"]
            return ((p(c1)[0] + p(c2)[0]) / 2, (p(c1)[1] + p(c2)[1]) / 2)

        # Eye openness = vertical lid gap / distance between the two eyes.
        # (Not the usual "EAR", which divides by each eye's own width: turning
        # the head foreshortens the two eyes' widths differently, so one open
        # eye read ~30% "more closed" than the other -- a phantom wink. Both
        # eyes share the inter-eye distance, so head turns cancel out.)
        interocular = max(dist(center(EYE_A), center(EYE_B)), 1e-6)

        def ear(eye):
            pairs = (*eye["lids"], (eye["top"], eye["bottom"]))
            return sum(dist(p(a), p(b)) for a, b in pairs) / len(pairs) / interocular

        def iris(eye):
            xs = sorted((p(eye["corners"][0])[0], p(eye["corners"][1])[0]))
            ix, iy = p(eye["iris"])
            top, bottom = p(eye["top"])[1], p(eye["bottom"])[1]
            rx = (ix - xs[0]) / max(xs[1] - xs[0], 1e-6)
            ry = (iy - top) / max(bottom - top, 1e-6)
            return rx, ry

        user_left, user_right = ((EYE_A, EYE_B) if center(EYE_A)[0] < center(EYE_B)[0]
                                 else (EYE_B, EYE_A))
        lx, ly = iris(user_left)
        rx, ry = iris(user_right)
        pts = np.array([(lm[i].x * aspect, lm[i].y, lm[i].z * aspect) for i in RIGID_POINTS])
        return {
            "iris_lx": lx, "iris_ly": ly,
            "iris_rx": rx, "iris_ry": ry,
            "ear_l": ear(user_left),
            "ear_r": ear(user_right),
            "mar": dist(p(MOUTH_TOP), p(MOUTH_BOTTOM)) / max(dist(p(MOUTH_LEFT), p(MOUTH_RIGHT)), 1e-6),
        }, pts

    def head(self, pts):
        return kabsch_forward(self.ref, pts)

    def _target(self, hx, hy, s, n, cfg):
        """Unfiltered cursor target in screen fractions."""
        # Iris term from the open eye(s) only: a closed eye's iris landmarks
        # are guesses. (During a drag the left eye is shut.)
        ix = [s[f"iris_{e}x"] - n[f"iris_{e}x"] for e, c in (("l", self.l_closed), ("r", self.r_closed)) if not c]
        iy = [s[f"iris_{e}y"] - n[f"iris_{e}y"] for e, c in (("l", self.l_closed), ("r", self.r_closed)) if not c]
        iris_x = sum(ix) / len(ix) if ix else 0.0
        iris_y = sum(iy) / len(iy) if iy else 0.0
        x = 0.5 + cfg["head_gain_x"] * hx + cfg["iris_gain_x"] * iris_x
        y = 0.5 + cfg["head_gain_y"] * hy + cfg["iris_gain_y"] * iris_y
        return x, y

    # ------------------------------------------------------------------------
    def process(self, frame, rgb, aspect):
        now = time.time()
        res = self.model.process(rgb)
        faces = getattr(res, "multi_face_landmarks", None)
        if not faces or len(faces[0].landmark) < 478:
            self.status = "lost" if self.neutral.ready else "centering"
            # Losing the face mid-drag must not leave the button held down.
            dropped = self.dragging
            self._reset_gestures()
            return "release_drag" if dropped else "move"

        lm = faces[0].landmark
        s, pts = self.measure(lm, aspect)
        self._draw(frame, lm)
        return self.step(s, pts, now)

    def step(self, s, pts, now):
        """Everything after landmark extraction (separate so recorded sessions
        can be replayed through it)."""
        if not self.neutral.ready:
            self.status = "centering"
            if pts is not None:
                self.ref_samples.append(pts - pts.mean(axis=0))
            if self.neutral.add(s):
                if self.ref_samples:
                    self.ref = np.mean(self.ref_samples, axis=0)
                self.ref_samples = []
                self.open_l.reset(self.neutral["ear_l"])
                self.open_r.reset(self.neutral["ear_r"])
            return "centering"
        self.status = "tracking"
        cfg = _cfg()
        n = self.neutral
        self.smoother.min_cutoff = cfg["smooth_min_cutoff"]
        self.smoother.beta = cfg["smooth_beta"]
        self.deadband_px = cfg["deadband_px"]

        hx, hy = self.head(pts)

        # Openness of each eye relative to its own recent open level.
        rl = s["ear_l"] / max(self.open_l.value or n["ear_l"], 1e-6)
        rr = s["ear_r"] / max(self.open_r.value or n["ear_r"], 1e-6)
        shut = cfg["blink_ratio"]
        self.l_closed = rl < (shut + EYE_HYSTERESIS if self.l_closed else shut)
        self.r_closed = rr < (shut + EYE_HYSTERESIS if self.r_closed else shut)
        resting = not (self.l_closed or self.r_closed or self.left.active
                       or self.right.active or self.dragging)
        if resting:
            self.open_l.add(s["ear_l"])
            self.open_r.add(s["ear_r"])

        tx, ty = self._target(hx, hy, s, n, cfg)

        # --- disturbance: laugh / cough / sneeze -------------------------------
        # A head jerk far faster than deliberate pointing, or both eyes
        # squeezed while the mouth is open (laughing), freezes everything for a
        # moment. Afterwards the cursor simply follows the head again: the
        # mapping is absolute, so nothing drifts.
        self.heads.append((now, hx, hy))
        while len(self.heads) > 2 and now - self.heads[1][0] >= JERK_WINDOW:
            self.heads.pop(0)
        pt, phx, phy = self.heads[0]
        speed = math.hypot(hx - phx, hy - phy) / max(now - pt, JERK_WINDOW) if now > pt else 0.0
        squint = rl < cfg["squint_ratio"] and rr < cfg["squint_ratio"]
        mouth_open = s["mar"] > max(cfg["mouth_open_ratio"], n["mar"] + 0.15)
        if speed > cfg["jerk_speed"] or (squint and mouth_open):
            self.hold_until = now + cfg["disturbance_hold"]
        disturbed = now < self.hold_until

        action = self._gestures(s, n, cfg, rl, rr, hy, tx, ty, now, disturbed, squint, mouth_open)
        if self.trace:
            self.trace.write(f"{now:.3f},{hx:.5f},{hy:.5f},{s['ear_l']:.4f},{s['ear_r']:.4f},"
                             f"{rl:.3f},{rr:.3f},{s['mar']:.3f},{tx:.4f},{ty:.4f},"
                             f"{self.x:.4f},{self.y:.4f},{speed:.2f},{int(disturbed)},{action}\n")
        return action

    def _gestures(self, s, n, cfg, rl, rr, hy, tx, ty, now, disturbed, squint, mouth_open):
        l_closed, r_closed = self.l_closed, self.r_closed
        ratio = cfg["wink_ratio"]
        # A wink: this eye closed AND clearly more closed than the other one
        # (a blink closes both about equally).
        l_wink = l_closed and rl < ratio * rr
        r_wink = r_closed and rr < ratio * rl

        # --- dragging (left eye held shut) ------------------------------------
        if self.dragging:
            if not l_closed:
                self.dragging = False
                self.left.reset()
                self.mouse_up()
                return "release_drag"
            if not r_closed and not disturbed:   # blink / sneeze mid-drag: pause
                self.move(tx, ty, now)
            return "drag"

        # --- a single left wink fires once the double-click window has passed --
        # (even during a disturbance: the wink was finished before it began)
        if (self.pending_click is not None and not self.left.active
                and now - self.pending_click[0] > cfg["double_window"]):
            pos = self.pending_click[1]
            self.pending_click = None
            if self.click(now, "left", pos):
                return "left_click"

        if disturbed:
            # A sneeze closes both eyes, often unevenly: drop any wink in
            # progress rather than let it become a click.
            self.left.reset()
            self.right.reset()
            self.scroll_anchor = None
            self.mouth_open_since = None
            return "paused"

        # --- left eye ----------------------------------------------------------
        if self.left.active:
            if l_closed:
                self.left.frames += 1
                # Blink-like only if the other eye is about as closed. Recorded
                # winks: the other eye squints to 0.30-0.46 (under blink_ratio)
                # while the winking eye is at ~0.07-0.25 -- still a clear wink.
                if not rl < ratio * rr:
                    self.left.both_frames += 1
                if now - self.left.since >= cfg["drag_hold"] and not self.left.looks_like_blink():
                    self.dragging = True
                    self.pending_click = None
                    self.mouse_down(self.left.pos)
                    return "drag"
                return "wink_left"          # cursor holds still while deciding
            held = now - self.left.since
            wink_pos, frames, blink = self.left.pos, self.left.frames, self.left.looks_like_blink()
            self.left.reset()
            if held >= cfg["wink_min"] and frames >= WINK_MIN_FRAMES and not blink:
                if self.pending_click is not None:
                    pos = self.pending_click[1]
                    self.pending_click = None
                    if self.click(now, "left", pos, clicks=2):
                        return "double_click"
                else:
                    self.pending_click = (now, wink_pos)
        elif l_wink and not self.right.active:
            self.left.start(now, self.history.at(now - PRE_WINK_LOOKBACK))
            return "wink_left"

        # --- right eye ---------------------------------------------------------
        if self.right.active:
            if r_closed:
                self.right.frames += 1
                if not rr < ratio * rl:
                    self.right.both_frames += 1
                return "wink_right"
            held = now - self.right.since
            wink_pos, frames, blink = self.right.pos, self.right.frames, self.right.looks_like_blink()
            self.right.reset()
            if (cfg["wink_min"] <= held <= cfg["wink_max"] and frames >= WINK_MIN_FRAMES
                    and not blink):
                if self.click(now, "right", wink_pos):
                    return "right_click"
        elif r_wink and not self.left.active:
            self.right.start(now, self.history.at(now - PRE_WINK_LOOKBACK))
            return "wink_right"

        # --- normal blink: hold the cursor still ------------------------------
        if l_closed or r_closed:
            return "eyes_closed"

        # --- mouth-open scroll -----------------------------------------------
        # The mouth must stay open for mouth_hold with the eyes not squinting:
        # talking and coughing open it only briefly, laughing squints.
        if mouth_open and not squint:
            if self.mouth_open_since is None:
                self.mouth_open_since = now
            if now - self.mouth_open_since >= cfg["mouth_hold"]:
                if self.scroll_anchor is None:
                    self.scroll_anchor = hy
                off = hy - self.scroll_anchor
                dead = cfg["scroll_deadzone"]
                if abs(off) > dead:
                    # head down (hy grows) -> scroll down (negative wheel delta)
                    amount = -int((abs(off) - dead) * cfg["scroll_speed"]) * (1 if off > 0 else -1)
                    amount = max(-120, min(120, amount))
                    if amount:
                        scroll(amount)
                        return "scroll"
                return "scroll_ready"
            return "move"       # opening: hold the cursor until it is clear
        self.mouth_open_since = None
        self.scroll_anchor = None

        # --- cursor -----------------------------------------------------------
        self.move(tx, ty, now)
        return "move"

    def _draw(self, frame, lm):
        import cv2
        h, w = frame.shape[:2]
        for i in (NOSE_TIP, EYE_A["iris"], EYE_B["iris"], *RIGID_POINTS):
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 2, (0, 255, 255), -1)
