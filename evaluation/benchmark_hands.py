"""Labelled synthetic benchmark for the hand-gesture classifiers.

Why synthetic: precision / recall need ground truth (what the user *meant*),
and live recordings only contain what the system *did*. So every episode here
is generated with a known label, using parameters measured from real recorded
sessions (WGI_TRACE logs):

  * landmark noise: 0.95 mm per axis per frame (measured: 0.85 mm index tip,
    1.02 mm thumb tip, on 7,060 still frames)
  * finger coupling: a middle-finger pinch drags the ring to ~0.04-0.16 and the
    index to ~0.13-0.22 of their open distance; an index pinch drags the middle
    to ~0.4-0.6 (measured)
  * hand drift while tapping: up to ~40 px of cursor travel (measured: taps
    2-42 px, real drags 86-580 px)
  * for the original finger-bend design: the user's bends reach only 57-95 deg
    at the index joint, resting hand at ~128 deg index / ~146 deg middle
    (measured)

Each stage reproduces one experiment of the project on the current code:

  V0  original finger-bend angles (bend index = click, bend middle = right
      click, bend both = double, peace sign + move = scroll, hold bend = drag)
  S1  pinch, fingertip-to-fingertip, fixed release, hold-to-drag, fist guard
  S2  + flesh contact (thumb pad to finger pad)
  S3  + drag by moving the pinched hand instead of holding
  S4  + release relative to the deepest point, re-arm, clear-winner 1.1
  S5  + no fist guard, no drag/scroll from the release jolt, scroll threshold
      0.06  (= current code)

Pinch stages share one gesture vocabulary (double click = two quick thumb+index
taps). Episode labels: left, double, right, drag, scroll, none (negatives:
idle movement, relaxed half-curl, near miss, fist, fast sweep).

Run:  python evaluation/benchmark_hands.py [--episodes 60] [--seed 7]
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

# --------------------------------------------------------------------------
# A fake mouse that records what the classifier fires.
# --------------------------------------------------------------------------
EVENTS = []
_fake = types.ModuleType("pyautogui")
_fake.FAILSAFE = False
_fake.PAUSE = 0
_fake.size = lambda: (1920, 1080)
_fake.moveTo = lambda *a, **k: None
_fake.click = lambda *a, **k: EVENTS.append("right" if k.get("button") == "right" else "left")
_fake.doubleClick = lambda *a, **k: EVENTS.append("double")
_fake.mouseDown = lambda *a, **k: EVENTS.append("down")
_fake.mouseUp = lambda *a, **k: EVENTS.append("up")
_fake.scroll = lambda *a, **k: EVENTS.append("scroll")
sys.modules["pyautogui"] = _fake

import calibration_manager                     # noqa: E402
import cursor_movement                         # noqa: E402
import pinch_control                           # noqa: E402
from control_common import OneEuroFilter       # noqa: E402

FPS = 30
NOISE = 0.00095          # m, per axis per frame (measured)
CLASSES = ["left", "double", "right", "drag", "scroll", "none"]


class LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


# --------------------------------------------------------------------------
# Hand model (MediaPipe world coordinates, metres; fingers point up = -y)
# --------------------------------------------------------------------------
MCP_X = {"index": -0.025, "middle": -0.005, "ring": 0.015, "pinky": 0.032}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
THUMB = [(-0.03, -0.02, -0.01), (-0.045, -0.04, -0.02), (-0.05, -0.055, -0.03), (-0.05, -0.07, -0.04)]


def _lerp(a, b, t):
    return tuple(p + (q - p) * t for p, q in zip(a, b))


def pinch_hand(closure, pad=False, gap=0.0):
    """closure: finger -> 0 (open) .. 1 (touching the thumb)."""
    w = {0: (0.0, 0.0, 0.0)}
    for i, p in enumerate(THUMB, start=1):
        w[i] = p
    tt = THUMB[-1]
    for f, x in MCP_X.items():
        m = BASE[f]
        c = closure.get(f, 0.0)
        mcp, pip = (x, -0.09, 0.0), (x, -0.125, 0.0)
        dip_o, tip_o = (x, -0.15, 0.0), (x, -0.17, 0.0)
        g = gap * c
        if pad and c > 0:
            dip_t = (tt[0] + 0.012 + g, tt[1] - 0.006, tt[2] + 0.004)
            tip_t = (dip_t[0] + 0.018, dip_t[1] - 0.012, dip_t[2] + 0.012)
        else:
            tip_t = (tt[0] + 0.006 + g, tt[1] - 0.004, tt[2] + 0.003)
            dip_t = (tip_t[0] + 0.004, tip_t[1] - 0.016, tip_t[2] + 0.012)
        w[m], w[m + 1] = mcp, pip
        w[m + 2], w[m + 3] = _lerp(dip_o, dip_t, c), _lerp(tip_o, tip_t, c)
    return w


def bend_hand(flex):
    """flex: finger -> flexion at the middle joint in degrees (0 = straight)."""
    w = {0: (0.0, 0.0, 0.0)}
    for i, p in enumerate(THUMB, start=1):
        w[i] = p
    for f, x in MCP_X.items():
        m = BASE[f]
        b = math.radians(flex.get(f, 0.0))
        d = (0.0, -math.cos(b), math.sin(b))
        pip = (x, -0.125, 0.0)
        w[m], w[m + 1] = (x, -0.09, 0.0), pip
        w[m + 2] = (x, pip[1] + 0.025 * d[1], 0.025 * d[2])
        w[m + 3] = (x, pip[1] + 0.045 * d[1], 0.045 * d[2])
    return w


def frame(w, pos, rnd, scale):
    world = [LM(*(v * scale + rnd.gauss(0, NOISE) for v in w[i])) for i in range(21)]
    img = [LM(pos[0] + w[i][0] * 2.5 * scale, pos[1] + w[i][1] * 2.5 * scale, 0.0) for i in range(21)]
    return world, img


# --------------------------------------------------------------------------
# Episodes: list of (world-pose dict, palm position) per frame
# --------------------------------------------------------------------------
def ramp(n):
    return [(k + 1) / n for k in range(n)]


def pinch_episode(label, rnd):
    pos0 = (rnd.uniform(0.35, 0.65), rnd.uniform(0.4, 0.6))
    pad = rnd.random() < 0.5
    gap = rnd.uniform(0.0, 0.006)
    seq = []

    def hold(closure, n, start, end=None):
        end = end or start
        for k in range(n):
            t = (k + 1) / max(n, 1)
            seq.append((dict(closure), _lerp(start, end, t)))

    def tap(f, dur, start, drift, couple):
        end = (start[0] + drift[0], start[1] + drift[1])
        for c in ramp(3):
            hold(dict({f: c}, **{g: c * v for g, v in couple.items()}), 1, start)
        hold(dict({f: 1.0}, **couple), max(1, int(dur * FPS)), start, end)
        for c in reversed(ramp(3)):
            hold(dict({f: c}, **{g: c * v for g, v in couple.items()}), 1, end)
        return end

    def drift(maxd):
        a = rnd.uniform(0, 2 * math.pi)
        r = rnd.uniform(0, maxd)
        return (r * math.cos(a), r * math.sin(a))

    idx_couple = lambda: {"middle": rnd.uniform(0.2, 0.55), "ring": rnd.uniform(0.1, 0.35)}
    mid_couple = lambda: {"index": rnd.uniform(0.3, 0.75), "ring": rnd.uniform(0.5, 0.9)}

    hold({}, int(0.7 * FPS), pos0)                                 # open hand first
    p = pos0
    if label == "left":
        p = tap("index", rnd.uniform(0.08, 1.2), p, drift(0.010), idx_couple())
    elif label == "double":
        p = tap("index", rnd.uniform(0.08, 0.2), p, drift(0.004), idx_couple())
        hold({}, int(rnd.uniform(0.08, 0.18) * FPS), p)
        p = tap("index", rnd.uniform(0.08, 0.2), p, drift(0.004), idx_couple())
    elif label == "right":
        p = tap("middle", rnd.uniform(0.08, 0.6), p, (0.0, rnd.uniform(-0.025, 0.025)), mid_couple())
    elif label in ("drag", "scroll"):
        f = "index" if label == "drag" else "middle"
        couple = idx_couple() if label == "drag" else mid_couple()
        for c in ramp(3):
            hold(dict({f: c}, **{g: c * v for g, v in couple.items()}), 1, p)
        hold(dict({f: 1.0}, **couple), int(rnd.uniform(0.1, 0.4) * FPS), p)
        if label == "drag":
            a = rnd.uniform(0, 2 * math.pi); r = rnd.uniform(0.05, 0.2)
            end = (p[0] + r * math.cos(a), p[1] + r * math.sin(a))
        else:
            end = (p[0], p[1] + rnd.choice((-1, 1)) * rnd.uniform(0.08, 0.2))
        hold(dict({f: 1.0}, **couple), int(rnd.uniform(0.3, 0.8) * FPS), p, end)
        hold(dict({f: 1.0}, **couple), int(0.2 * FPS), end)
        for c in reversed(ramp(3)):
            hold(dict({f: c}, **{g: c * v for g, v in couple.items()}), 1, end)
        p = end
    else:                                                           # negatives
        kind = rnd.choice(["idle", "curl", "near", "fist", "sweep"])
        if kind == "idle":
            end = (p[0] + rnd.uniform(-0.15, 0.15), p[1] + rnd.uniform(-0.1, 0.1))
            hold({}, int(1.2 * FPS), p, end); p = end
        elif kind == "curl":
            c = rnd.uniform(0.3, 0.55)
            hold({f: c * k for f, k in (("index", 1), ("middle", 1), ("ring", 1), ("pinky", 1))}, int(1.0 * FPS), p)
        elif kind == "near":
            c = rnd.uniform(0.4, 0.55)
            for k in ramp(4):
                hold({"index": c * k}, 1, p)
            hold({"index": c}, int(0.4 * FPS), p)
        elif kind == "fist":
            c = rnd.uniform(0.9, 1.0)
            for k in ramp(4):
                hold({f: c * k for f in ("index", "middle", "ring", "pinky")}, 1, p)
            hold({f: c for f in ("index", "middle", "ring", "pinky")}, int(0.6 * FPS), p)
            for k in reversed(ramp(4)):
                hold({f: c * k for f in ("index", "middle", "ring", "pinky")}, 1, p)
        else:
            end = (p[0] + rnd.choice((-1, 1)) * 0.2, p[1])
            hold({f: 0.3 for f in ("index", "middle", "ring", "pinky")}, int(0.4 * FPS), p, end); p = end
    hold({}, int(0.7 * FPS), p)
    return [(pinch_hand(c, pad, gap), q) for c, q in seq]


REST = None


def bend_episode(label, rnd):
    """Finger-bend gestures for the original design, with the user's measured
    resting posture and reachable bend depth."""
    rest = {"index": rnd.uniform(30, 70), "middle": rnd.uniform(20, 50),
            "ring": rnd.uniform(20, 60), "pinky": rnd.uniform(20, 60)}
    pos0 = (rnd.uniform(0.35, 0.65), rnd.uniform(0.4, 0.6))
    seq = []

    def hold(flex, n, start, end=None):
        end = end or start
        for k in range(n):
            seq.append((dict(flex), _lerp(start, end, (k + 1) / max(n, 1))))

    def bend(target, n_hold, start, end=None):
        for k in ramp(4):
            hold({f: rest[f] + (target.get(f, rest[f]) - rest[f]) * k for f in rest}, 1, start)
        hold(dict(rest, **target), n_hold, start, end)
        e = end or start
        for k in reversed(ramp(4)):
            hold({f: rest[f] + (target.get(f, rest[f]) - rest[f]) * k for f in rest}, 1, e)
        return e

    hold(rest, int(0.7 * FPS), pos0)
    p = pos0
    reach = lambda: rnd.uniform(85, 140)       # measured: index joint 57-95 deg => flexion 85-123
    if label == "left":
        p = bend({"index": reach()}, int(rnd.uniform(0.15, 0.6) * FPS), p)
    elif label == "double":
        r = reach()
        p = bend({"index": r, "middle": r + rnd.uniform(-10, 10)}, int(rnd.uniform(0.2, 0.5) * FPS), p)
    elif label == "right":
        p = bend({"middle": reach(), "index": rest["index"] + rnd.uniform(10, 40)}, int(rnd.uniform(0.15, 0.6) * FPS), p)
    elif label == "drag":
        a = rnd.uniform(0, 2 * math.pi); r = rnd.uniform(0.05, 0.2)
        end = (p[0] + r * math.cos(a), p[1] + r * math.sin(a))
        p = bend({"index": reach()}, int(rnd.uniform(1.0, 1.6) * FPS), p, end)
    elif label == "scroll":
        end = (p[0], p[1] + rnd.choice((-1, 1)) * rnd.uniform(0.08, 0.2))
        p = bend({"index": 5, "middle": 5, "ring": 150, "pinky": 150}, int(0.8 * FPS), p, end)
    else:
        kind = rnd.choice(["idle", "curl", "fist", "sweep"])
        if kind == "idle":
            end = (p[0] + rnd.uniform(-0.15, 0.15), p[1] + rnd.uniform(-0.1, 0.1))
            hold(rest, int(1.2 * FPS), p, end); p = end
        elif kind == "curl":
            p = bend({f: rest[f] + rnd.uniform(20, 45) for f in rest}, int(1.0 * FPS), p)
        elif kind == "fist":
            p = bend({f: 150 for f in rest}, int(0.6 * FPS), p)
        else:
            end = (p[0] + rnd.choice((-1, 1)) * 0.2, p[1])
            hold(rest, int(0.4 * FPS), p, end); p = end
    hold(rest, int(0.7 * FPS), p)
    return [(bend_hand(c), q) for c, q in seq]


# --------------------------------------------------------------------------
# Classifier stages
# --------------------------------------------------------------------------
def tip_contact(geom, chain):
    return pinch_control._dist(geom[pinch_control.THUMB_PAD[-1]], geom[chain[-1]])


class StagePinch(pinch_control.PinchGestures):
    """Current pinch logic, with the two decisions that changed over the
    experiments switchable: hold-to-drag vs move-to-drag, and whether a drag or
    scroll may start during the release jolt."""
    hold_drag = False
    jolt_guard = True

    def _hold(self, x, y, hand_y, now):
        if self.contact is None:
            self._move(x, y, now)
            return "move"
        releasing = self.jolt_guard and self.pending != self.contact
        if self.contact == "index":
            if not self.dragging and not releasing:
                if self.hold_drag:
                    start = now - self.since >= 0.5
                else:
                    w, _ = self.get_screen_size()
                    start = math.dist(self._px((x, y)), self._px(self.touch_xy)) > pinch_control.DRAG_START * w
                if start:
                    self.dragging = True
                    self.last_tap = None
                    pinch_control.perform_mouse_action("mouseDown", *self._px(self.pos))
            if self.dragging:
                self._move(x, y, now)
                return "drag"
            return "holding_click"
        if self.contact == "middle" and (
                self.scrolling or (not releasing and abs(hand_y - self.touch_hand_y) > pinch_control.SCROLL_START)):
            if not self.scrolling:
                self.scrolling = True
                self.scroll_prev_y = hand_y
            amount = int(-(hand_y - self.scroll_prev_y) * pinch_control.SCROLL_GAIN)
            if amount:
                pinch_control.wheel(amount)
            self.scroll_prev_y = hand_y
            return "scroll"
        return "pinch"


STAGES = [
    ("S1", "Pinch v1: fingertip-to-fingertip, fixed release, hold-to-drag, fist guard",
     dict(tip=True, TOUCH_RATIO=0.35, RELEASE_DELTA=10.0, RELEASE_RATIO=0.5, hold_drag=True,
          OTHER_FINGER_MIN=0.3, CLEAR_WINNER=1.25, SCROLL_START=0.03, jolt_guard=False)),
    ("S2", "+ flesh contact (thumb pad to finger pad)",
     dict(tip=False, TOUCH_RATIO=0.4, RELEASE_DELTA=10.0, RELEASE_RATIO=0.55, hold_drag=True,
          OTHER_FINGER_MIN=0.3, CLEAR_WINNER=1.25, SCROLL_START=0.03, jolt_guard=False)),
    ("S3", "+ drag by moving the pinched hand",
     dict(tip=False, TOUCH_RATIO=0.4, RELEASE_DELTA=10.0, RELEASE_RATIO=0.55, hold_drag=False,
          OTHER_FINGER_MIN=0.3, CLEAR_WINNER=1.25, SCROLL_START=0.03, jolt_guard=False)),
    ("S4", "+ release relative to deepest point, re-arm, clear-winner 1.1",
     dict(tip=False, TOUCH_RATIO=0.4, RELEASE_DELTA=0.12, RELEASE_RATIO=0.55, hold_drag=False,
          OTHER_FINGER_MIN=0.3, CLEAR_WINNER=1.1, SCROLL_START=0.03, jolt_guard=False)),
    ("S5", "+ no fist guard, no drag/scroll from the release jolt, scroll threshold 0.06 (current)",
     dict(tip=False, TOUCH_RATIO=0.4, RELEASE_DELTA=0.12, RELEASE_RATIO=0.55, hold_drag=False,
          OTHER_FINGER_MIN=0.0, CLEAR_WINNER=1.1, SCROLL_START=0.06, jolt_guard=True)),
]


def predict(events):
    if "down" in events:
        return "drag"
    if "double" in events or events.count("left") >= 2:
        return "double"
    if events.count("scroll") >= 2:
        return "scroll"
    if "right" in events:
        return "right"
    if "left" in events:
        return "left"
    return "none"


def run_pinch_stage(cfg, episodes, seed):
    saved = {k: getattr(pinch_control, k) for k in cfg if k.isupper()}
    saved_contact = pinch_control.contact_distance
    try:
        for k, v in cfg.items():
            if k.isupper():
                setattr(pinch_control, k, v)
        if cfg["tip"]:
            pinch_control.contact_distance = tip_contact
        StagePinch.hold_drag = cfg["hold_drag"]
        StagePinch.jolt_guard = cfg["jolt_guard"]
        rnd = random.Random(seed)
        out = []
        t = 1000.0
        for label, frames in episodes:
            p = StagePinch(lambda: (1920, 1080))
            f = OneEuroFilter(0.8, 2.0, 1.0)
            scale = rnd.uniform(0.9, 1.1)
            # the classifier learns the open hand first, as it does live
            warm = [(pinch_hand({}), frames[0][1])] * 20
            EVENTS.clear()
            for w, pos in warm + frames:
                world, img = frame(w, pos, rnd, scale)
                geom = [(l.x, l.y, l.z) for l in world]
                palm = cursor_movement.get_palm_center(types.SimpleNamespace(landmark=img))
                x, y = f.update(*cursor_movement.get_target_position(palm), t)
                p.update(geom, x, y, img[0].y, t)
                t += 1.0 / FPS
            p.reset()
            out.append((label, predict(list(EVENTS))))
        return out
    finally:
        for k, v in saved.items():
            setattr(pinch_control, k, v)
        pinch_control.contact_distance = saved_contact


def run_bend_baseline(episodes, seed):
    import numpy as np
    import wgi
    clock = [1000.0]
    wgi.time = types.SimpleNamespace(time=lambda: clock[0])
    import calculo_distancia as util
    rnd = random.Random(seed)
    canvas = np.zeros((48, 64, 3), np.uint8)
    out = []
    for label, frames in episodes:
        wgi.reset_interaction_state()
        cursor_movement.prev_x = cursor_movement.prev_y = 0.5
        scale = rnd.uniform(0.9, 1.1)
        EVENTS.clear()
        for w, pos in frames:
            world, img = frame(w, pos, rnd, scale)
            lms = [(l.x, l.y) for l in img]
            geom = util.to_geometry_space(lms, [(l.x, l.y, l.z) for l in world], 1.0)
            processed = types.SimpleNamespace(multi_hand_landmarks=[types.SimpleNamespace(landmark=img)])
            wgi.detect_gesture(canvas, lms, geom, processed)
            clock[0] += 1.0 / FPS
        out.append((label, predict(list(EVENTS))))
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(results):
    n = len(results)
    acc = sum(1 for t, p in results if t == p) / n
    per = {}
    for c in CLASSES:
        tp = sum(1 for t, p in results if t == c and p == c)
        fp = sum(1 for t, p in results if t != c and p == c)
        fn = sum(1 for t, p in results if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    gestures = [c for c in CLASSES if c != "none"]
    macro = {k: sum(per[c][k] for c in gestures) / len(gestures) for k in ("precision", "recall", "f1")}
    false_trigger = 1.0 - per["none"]["recall"]
    return {"accuracy": acc, "macro": macro, "per_class": per, "false_trigger_rate": false_trigger,
            "confusion": {t: {p: sum(1 for a, b in results if a == t and b == p) for p in CLASSES} for t in CLASSES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60, help="episodes per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results_hands.json"))
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    pinch_eps = [(c, pinch_episode(c, rnd)) for c in CLASSES for _ in range(args.episodes)]
    rnd = random.Random(args.seed)
    bend_eps = [(c, bend_episode(c, rnd)) for c in CLASSES for _ in range(args.episodes)]

    report = {"episodes_per_class": args.episodes, "seed": args.seed, "stages": []}
    rows = [("V0", "Original finger-bend angles (calibration defaults)", run_bend_baseline(bend_eps, args.seed))]
    for sid, desc, cfg in STAGES:
        rows.append((sid, desc, run_pinch_stage(cfg, pinch_eps, args.seed)))

    print(f"{'stage':5} {'accuracy':>8} {'precision':>9} {'recall':>7} {'F1':>6} {'false-trigger':>13}  description")
    for sid, desc, res in rows:
        m = metrics(res)
        report["stages"].append({"id": sid, "description": desc, **m})
        print(f"{sid:5} {m['accuracy']:8.3f} {m['macro']['precision']:9.3f} {m['macro']['recall']:7.3f} "
              f"{m['macro']['f1']:6.3f} {m['false_trigger_rate']:13.3f}  {desc}")
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
