"""Labelled synthetic benchmark for the eye-gesture classifier.

Same idea as benchmark_hands.py: episodes with known labels, generated from a
rotatable 3D face whose parameters were measured on real WGI_TRACE sessions:

  * eye-openness jitter: 0.019 per frame median (measured), head-signal
    jitter 0.004 per frame (measured)
  * while one eye winks, the other squints to 0.30-0.46 of its open level
    (measured on recorded right winks)
  * real winks last ~0.1-0.4 s; the eye's outer corner sits ~2.5 cm behind the
    inner one, so turning the head foreshortens the two eyes differently --
    the cause of the "phantom winks" seen in recordings

Stages (each reproduces one experiment, on the current code):

  E1  classic eye-aspect-ratio (lid gap / that eye's width), shallow
      thresholds (0.65 closed, 0.75 wink ratio), 70th-percentile open level,
      "other eye closed too" = blink
  E2  + openness measured against the distance between the eyes (head-turn proof)
  E3  + deeper thresholds (0.5 / 0.7) and a median open level
  E4  + blink check relative to the other eye (current)

Labels: left (wink left), double (two left winks), right (wink right),
drag (hold left eye shut while turning), scroll (mouth open + nod), none
(blinks, slow blinks, head turns, looking down, laughing, talking).

Run:  python evaluation/benchmark_eyes.py [--episodes 60] [--seed 7]
"""
import argparse
import json
import math
import os
import random
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "back-end"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

EVENTS = []
_fake = types.ModuleType("pyautogui")
_fake.FAILSAFE = False
_fake.PAUSE = 0
_fake.size = lambda: (1920, 1080)
_fake.moveTo = lambda *a, **k: None
_fake.click = lambda *a, **k: EVENTS.append(
    "right" if k.get("button") == "right" else ("double" if k.get("clicks", 1) == 2 else "left"))
_fake.doubleClick = lambda *a, **k: EVENTS.append("double")
_fake.mouseDown = lambda *a, **k: EVENTS.append("down")
_fake.mouseUp = lambda *a, **k: EVENTS.append("up")
_fake.scroll = lambda *a, **k: EVENTS.append("scroll")
sys.modules["pyautogui"] = _fake

import numpy as np                            # noqa: E402
import calibration_manager                    # noqa: E402
import control_common                         # noqa: E402
import eye_control                            # noqa: E402
from benchmark_hands import CLASSES, metrics, predict   # noqa: E402

FPS = 30
RIGID_NOISE = 0.0009      # gives the measured head-signal jitter
LID_NOISE = 0.0002        # gives the measured openness jitter
EYE_GAP = 0.009

BASE = {
    234: (-0.075, 0.0, 0.05), 127: (-0.07, -0.03, 0.04),
    454: (0.075, 0.0, 0.05), 356: (0.07, -0.03, 0.04),
    10: (0.0, -0.09, 0.0), 151: (0.0, -0.06, -0.02),
    168: (0.0, -0.02, -0.04), 6: (0.0, 0.0, -0.045), 197: (0.0, 0.012, -0.052), 195: (0.0, 0.022, -0.058),
    1: (0.0, 0.03, -0.07), 4: (0.0, 0.025, -0.068), 5: (0.0, 0.02, -0.065),
    # outer eye corners sit further back than the inner ones (curved face)
    33: (-0.045, -0.01, 0.005), 133: (-0.015, -0.01, -0.025),
    362: (0.015, -0.01, -0.025), 263: (0.045, -0.01, 0.005),
    468: (-0.03, -0.01, -0.03), 473: (0.03, -0.01, -0.03),
    78: (-0.025, 0.06, -0.04), 308: (0.025, 0.06, -0.04),
}
for _k, _i in enumerate(eye_control.RIGID_POINTS):
    if _i not in BASE:
        _x = -0.07 + 0.14 * ((_k * 37) % 29) / 28
        _y = -0.09 + 0.11 * ((_k * 17) % 23) / 22
        BASE[_i] = (_x, _y, 0.06 * (_x / 0.075) ** 2 - 0.03 - 0.02 * (1 - abs(_x) / 0.075))
LIDS_A = ((160, 144, -0.035), (158, 153, -0.025), (159, 145, -0.03))     # user's left eye
LIDS_B = ((385, 380, 0.025), (387, 373, 0.035), (386, 374, 0.03))
LID_IDS = {i for a, b, _ in LIDS_A + LIDS_B for i in (a, b)}


class LM:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z):
        self.x, self.y, self.z, self.visibility = x, y, z, 1.0


def face(s, rnd):
    """s: dict(yaw, pitch, left, right, mouth) -> fake Face Mesh result."""
    pts = dict(BASE)
    for a, b, x in LIDS_A:
        pts[a] = (x, -0.01 - EYE_GAP * s["left"] / 2, -0.028)
        pts[b] = (x, -0.01 + EYE_GAP * s["left"] / 2, -0.028)
    for a, b, x in LIDS_B:
        pts[a] = (x, -0.01 - EYE_GAP * s["right"] / 2, -0.028)
        pts[b] = (x, -0.01 + EYE_GAP * s["right"] / 2, -0.028)
    pts[13] = (0.0, 0.06 - s["mouth"] / 2, -0.045)
    pts[14] = (0.0, 0.06 + s["mouth"] / 2, -0.045)
    th, ph = math.radians(s["yaw"]), math.radians(s["pitch"])
    lm = [LM(0.5, 0.5, 0.0) for _ in range(478)]
    for i, (X, Y, Z) in pts.items():
        X, Z = X * math.cos(th) - Z * math.sin(th), X * math.sin(th) + Z * math.cos(th)
        Y, Z = Y * math.cos(ph) - Z * math.sin(ph), Y * math.sin(ph) + Z * math.cos(ph)
        n = LID_NOISE if i in LID_IDS else RIGID_NOISE
        lm[i] = LM(0.5 + X + rnd.gauss(0, n), 0.5 + Y + rnd.gauss(0, n), Z + rnd.gauss(0, n))
    return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=lm)])


def episode(label, rnd):
    st = {"yaw": 0.0, "pitch": 0.0, "left": 1.0, "right": 1.0, "mouth": 0.0}
    seq = []

    def emit(n, **target):
        start = dict(st)
        for k in range(n):
            t = (k + 1) / max(n, 1)
            for key, v in target.items():
                st[key] = start[key] + (v - start[key]) * t
            seq.append(dict(st))

    emit(int(1.0 * FPS))                                         # centering + settle
    emit(int(0.3 * FPS), yaw=rnd.uniform(-20, 20), pitch=rnd.uniform(-8, 8))
    emit(int(0.3 * FPS))

    def wink(eye, other, dur):
        closed, squint = rnd.uniform(0.03, 0.3), rnd.uniform(0.35, 0.65)
        emit(2, **{eye: closed, other: squint})
        emit(max(1, int(dur * FPS)))
        emit(2, **{eye: 1.0, other: 1.0})

    if label == "left":
        wink("left", "right", rnd.uniform(0.1, 0.4))
    elif label == "double":
        wink("left", "right", rnd.uniform(0.1, 0.2))
        emit(int(rnd.uniform(0.1, 0.22) * FPS))
        wink("left", "right", rnd.uniform(0.1, 0.2))
    elif label == "right":
        wink("right", "left", rnd.uniform(0.1, 0.4))
    elif label == "drag":
        emit(2, left=rnd.uniform(0.03, 0.25), right=rnd.uniform(0.4, 0.7))
        emit(int(rnd.uniform(0.6, 1.0) * FPS))
        emit(int(0.6 * FPS), yaw=st["yaw"] + rnd.choice((-1, 1)) * rnd.uniform(5, 12))
        emit(2, left=1.0, right=1.0)
    elif label == "scroll":
        emit(3, mouth=rnd.uniform(0.03, 0.04))
        emit(int(0.5 * FPS))
        emit(int(0.4 * FPS), pitch=st["pitch"] + rnd.choice((-1, 1)) * rnd.uniform(8, 12))
        emit(int(0.3 * FPS))
        emit(3, mouth=0.0)
    else:
        kind = rnd.choice(["blink", "slow_blink", "turn", "look_down", "laugh", "talk"])
        if kind in ("blink", "slow_blink"):
            dur = rnd.uniform(0.08, 0.2) if kind == "blink" else rnd.uniform(0.3, 0.5)
            lead = rnd.choice(("left", "right"))
            other = "right" if lead == "left" else "left"
            emit(1, **{lead: rnd.uniform(0.05, 0.3)})                 # one eye a frame early
            emit(1, **{other: rnd.uniform(0.03, 0.2)})
            emit(max(1, int(dur * FPS)))
            emit(2, left=1.0, right=1.0)
        elif kind == "turn":
            y = rnd.choice((-1, 1)) * rnd.uniform(15, 28)
            emit(int(0.6 * FPS), yaw=y)
            emit(int(0.5 * FPS))
            emit(int(0.6 * FPS), yaw=0.0)
        elif kind == "look_down":
            base = rnd.uniform(0.55, 0.7)
            emit(int(0.4 * FPS), pitch=st["pitch"] + 8, left=base + rnd.uniform(-0.08, 0.08),
                 right=base + rnd.uniform(-0.08, 0.08))
            emit(int(0.8 * FPS))
            emit(int(0.4 * FPS), pitch=st["pitch"] - 8, left=1.0, right=1.0)
        elif kind == "laugh":
            emit(3, left=0.6, right=0.6, mouth=0.035)
            for k in range(int(1.0 * FPS)):
                emit(1, pitch=st["pitch"] + 3 * math.sin(k / 3))
            emit(3, left=1.0, right=1.0, mouth=0.0)
        else:
            for _ in range(3):
                emit(3, mouth=rnd.uniform(0.02, 0.03))
                emit(int(0.15 * FPS))
                emit(3, mouth=0.0)
    emit(int(0.8 * FPS), left=1.0, right=1.0, mouth=0.0)
    return seq


STAGES = [
    ("E1", "Classic eye-aspect-ratio, shallow thresholds, 'other eye closed' = blink",
     dict(width=True, relative=False, percentile=70, blink_ratio=0.65, wink_ratio=0.75)),
    ("E2", "+ openness against the distance between the eyes (head-turn proof)",
     dict(width=False, relative=False, percentile=70, blink_ratio=0.65, wink_ratio=0.75)),
    ("E3", "+ deeper thresholds (0.5 / 0.7), median open level",
     dict(width=False, relative=False, percentile=50, blink_ratio=0.5, wink_ratio=0.7)),
    ("E4", "+ blink check relative to the other eye (current)",
     dict(width=False, relative=True, percentile=50, blink_ratio=0.5, wink_ratio=0.7)),
]


class FakeModel:
    def __init__(self):
        self.next = None

    def process(self, rgb):
        return self.next

    def close(self):
        pass


def run_stage(cfg, episodes, seed):
    saved = (eye_control.OPENNESS_BY_EYE_WIDTH, eye_control.RELATIVE_BLINK_CHECK,
             eye_control.OPEN_LEVEL_PERCENTILE, dict(calibration_manager.config["eyes"]))
    clock = [1000.0]
    eye_control.time = types.SimpleNamespace(time=lambda: clock[0])
    control_common.time = types.SimpleNamespace(time=lambda: clock[0])
    try:
        eye_control.OPENNESS_BY_EYE_WIDTH = cfg["width"]
        eye_control.RELATIVE_BLINK_CHECK = cfg["relative"]
        eye_control.OPEN_LEVEL_PERCENTILE = cfg["percentile"]
        calibration_manager.config["eyes"].update(blink_ratio=cfg["blink_ratio"], wink_ratio=cfg["wink_ratio"])
        rnd = random.Random(seed)
        canvas = np.zeros((48, 64, 3), np.uint8)
        out = []
        for label, seq in episodes:
            c = eye_control.EyeController(lambda: (1920, 1080))
            c.model = FakeModel()
            EVENTS.clear()
            for s in seq:
                c.model.next = face(s, rnd)
                c.process(canvas, None, 1.0)
                clock[0] += 1.0 / FPS
            c.close()
            out.append((label, predict(list(EVENTS))))
        return out
    finally:
        (eye_control.OPENNESS_BY_EYE_WIDTH, eye_control.RELATIVE_BLINK_CHECK,
         eye_control.OPEN_LEVEL_PERCENTILE) = saved[:3]
        calibration_manager.config["eyes"] = saved[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60, help="episodes per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results_eyes.json"))
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    episodes = [(c, episode(c, rnd)) for c in CLASSES for _ in range(args.episodes)]
    report = {"episodes_per_class": args.episodes, "seed": args.seed, "stages": []}
    print(f"{'stage':5} {'accuracy':>8} {'precision':>9} {'recall':>7} {'F1':>6} {'false-trigger':>13}  description")
    for sid, desc, cfg in STAGES:
        m = metrics(run_stage(cfg, episodes, args.seed))
        report["stages"].append({"id": sid, "description": desc, **m})
        print(f"{sid:5} {m['accuracy']:8.3f} {m['macro']['precision']:9.3f} {m['macro']['recall']:7.3f} "
              f"{m['macro']['f1']:6.3f} {m['false_trigger_rate']:13.3f}  {desc}")
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
