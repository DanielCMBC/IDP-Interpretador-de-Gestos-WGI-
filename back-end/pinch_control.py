"""Hand clicks by touching a finger to the thumb.

    Thumb + index, tap                -> left click
    Thumb + index, tap twice quickly  -> double click
    Thumb + index, touch and MOVE     -> drag; open the fingers to drop
    Thumb + middle, tap               -> right click
    Thumb + middle, touch and move    -> scroll (hand up = scroll up)

A tap is a touch without moving the hand, however long it lasts: recorded taps
moved the hand 2-42 px, recorded drags 86-580 px. (Deciding by hold time turned
slow taps into accidental drags.)

Contact is the closest approach between the thumb's last segment (its pad) and
the finger's last two segments (middle joint to tip) -- not tip to tip, so
pressing the thumb against the flesh of the finger counts as touching.

No fixed distances. That contact distance is divided by the hand's own size
(wrist to middle knuckle, in MediaPipe's 3D world coordinates), so it does not
matter how far the hand is from the camera. Then each finger's *open* distance
is learned continuously from the user's own hand while it is not pinching, and
"touching" means closing to a fraction of that.

One finger at a time. Fingers are mechanically coupled -- an index tap pulls the
middle finger along, a ring tap pulls the middle -- so the touching finger is
the closest one, and only when it is clearly closer than the runner-up. A fist
(all fingers equally close) does nothing. Once a pinch starts it stays on that
finger until the hand opens, so a drag or scroll cannot jump fingers.

A click lands where the cursor was just before the finger touched (the hand
shifts while pinching), or where it was when the finger started closing if
that is close by -- never somewhere the cursor merely passed through earlier.
"""
import collections
import math
import platform
import statistics

import pyautogui

from control_common import PositionHistory, scroll as wheel

THUMB_PAD = (3, 4)                       # thumb's last joint -> tip
FINGERS = (("index", (6, 7, 8)), ("middle", (10, 11, 12)), ("ring", (14, 15, 16)))
SAMPLES = 5                              # points sampled along each segment

# Ratios are contact distance / that finger's learned open distance.
TOUCH_RATIO = 0.4        # touching below this
# Released: a tap ends once the finger rises RELEASE_DELTA above the closest it
# got (fingers often hover half-open after a tap instead of springing back),
# within [TOUCH_RATIO, RELEASE_RATIO]. A drag or scroll -- a deliberate hold,
# whose grip loosens a little as the hand moves -- only ends above RELEASE_RATIO.
RELEASE_DELTA = 0.12
RELEASE_RATIO = 0.55
APPROACH_RATIO = 0.7     # the finger counts as "starting to close" below this
# The closest finger only wins if the runner-up is at least this much further.
# Recorded: a coupled neighbour sits at 1.2-1.5x the touching finger -- and so
# does a fist, which is told apart by OTHER_FINGER_MIN below instead.
CLEAR_WINNER = 1.1
# Fist guard: if another finger is touching too (below this), nothing starts.
# Disabled (0): a deep middle-finger press pulls the ring finger to 0.04-0.16 and
# the index to 0.13-0.22 (shared tendons) -- indistinguishable from a 'fist',
# and the guard was cancelling 6 of 12 recorded right clicks. The relative
# CLEAR_WINNER test still decides which single finger is touching.
OTHER_FINGER_MIN = 0.0
# Open distance = 75th percentile of recent non-pinching frames.
REST_FRAMES = 90
REST_MIN_FRAMES = 15
REST_PERCENTILE = 75
CONFIRM_FRAMES = 2           # a new contact must hold this many frames
RELEASE_CONFIRM_FRAMES = 3   # letting go must hold longer: a 1-frame flicker must not end a drag
# Moving the hand this far while touching turns a tap into a drag (index) or a
# scroll (middle). Fractions of the screen width / of the camera frame height.
DRAG_START = 0.035           # ~70 px on a 1920 px screen
SCROLL_START = 0.06          # recorded right-click taps drifted the wrist up to ~0.05
# Click position: the cursor this long before the touch registered; the
# "started closing" position is used instead only if it is this close to it.
TOUCH_LOOKBACK = 0.1
ANCHOR_MAX_JUMP = 0.016      # ~30 px
DOUBLE_RADIUS = 0.03         # second tap this close to the first -> double click
SCROLL_GAIN = 3500           # wheel units per frame-height of hand travel


def _os_double_click_time():
    """The OS's own double-click interval: a second click on the same pixel
    within it is a double click."""
    if platform.system() == "Windows":
        try:
            import ctypes
            return ctypes.windll.user32.GetDoubleClickTime() / 1000.0
        except Exception:
            pass
    return 0.5


DOUBLE_WINDOW = _os_double_click_time()


def _dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def _samples(points):
    """Evenly spaced points along a polyline of landmarks."""
    out = []
    for a, b in zip(points, points[1:]):
        for k in range(SAMPLES):
            f = k / SAMPLES
            out.append(tuple(p + (q - p) * f for p, q in zip(a, b)))
    out.append(points[-1])
    return out


def contact_distance(geom, chain):
    """Closest approach between the thumb pad and a finger's last two segments."""
    thumb = _samples([geom[i] for i in THUMB_PAD])
    finger = _samples([geom[i] for i in chain])
    return min(_dist(a, b) for a in thumb for b in finger)


class PinchGestures:
    def __init__(self, get_screen_size, cursor=None):
        """cursor: an optional control_common.CursorPump that moves the OS
        cursor on its own fast loop; without one the cursor is moved here."""
        self.get_screen_size = get_screen_size
        self.cursor = cursor
        self._follow = False
        self.rest = {name: collections.deque(maxlen=REST_FRAMES) for name, _ in FINGERS}
        self.history = PositionHistory()
        self.last = {}            # last measurement, for tracing
        self.last_tap = None      # (time, pos) of the last left click, for double clicks
        self._ratios = {}
        self._reset_contact()

    def _reset_contact(self):
        self.pending = None
        self.pending_frames = 0
        self.contact = None           # finger currently pinched (confirmed)
        self.pinch_min = None         # deepest ratio reached during the current pinch
        self.cancel_tap = False       # another finger closed too (a fist): end without clicking
        self.since = None
        self.pos = None               # where a click / drag will land
        self.touch_xy = None          # live hand-cursor position when the touch registered
        self.touch_hand_y = None
        self.approach = {name: None for name, _ in FINGERS}   # (t, pos) when closing began
        self.dragging = False
        self.scrolling = False
        self.scroll_prev_y = None
        # After a tap ends at ratio L, the same finger must close RELEASE_DELTA
        # deeper than L (or open fully first) to start a new tap: quick repeated
        # presses that never fully open still count separately, and tracking
        # noise around L cannot make extra clicks.
        self.rearm = {name: None for name, _ in FINGERS}

    def reset(self):
        """Hand lost or tracking stopped: never leave the button held."""
        if self.dragging:
            pyautogui.mouseUp()
        self._reset_contact()

    # ------------------------------------------------------------------------
    def learning_progress(self):
        return min(1.0, min(len(d) for d in self.rest.values()) / REST_MIN_FRAMES)

    def open_distance(self, name):
        return float(statistics.quantiles(self.rest[name], n=100)[REST_PERCENTILE - 1]) \
            if len(self.rest[name]) >= 2 else None

    def _px(self, pos):
        w, h = self.get_screen_size()
        return int(pos[0] * (w - 1)), int(pos[1] * (h - 1))

    def _move(self, x, y, now):
        self.history.push(x, y, now)
        if self.cursor is not None:
            self._follow = True          # the pump moves the OS cursor
        else:
            pyautogui.moveTo(*self._px((x, y)))

    # ------------------------------------------------------------------------
    def _touching_finger(self, ratios):
        """Which single finger is touching the thumb this frame, if any."""
        if self.contact is not None:
            r = ratios[self.contact]
            self.pinch_min = r if self.pinch_min is None else min(self.pinch_min, r)
            if self.dragging or self.scrolling:
                release = RELEASE_RATIO
            else:
                release = min(self.pinch_min + RELEASE_DELTA, RELEASE_RATIO)
            if not (self.dragging or self.scrolling) and any(
                    v < OTHER_FINGER_MIN for n, v in ratios.items() if n != self.contact):
                self.cancel_tap = True                 # closing into a fist: not a tap
                return None
            if r < release:
                return self.contact                    # sticky until the finger lets go
            return None
        for name, r in ratios.items():
            if r >= TOUCH_RATIO:
                self.rearm[name] = None          # finger opened fully: re-armed
        order = sorted(ratios, key=ratios.get)
        best, second = order[0], order[1]
        rearm = self.rearm[best]
        if (ratios[best] < TOUCH_RATIO
                and (rearm is None or ratios[best] < rearm - RELEASE_DELTA)
                and ratios[second] >= max(ratios[best] * CLEAR_WINNER, OTHER_FINGER_MIN)):
            return best
        return None

    def update(self, geom, x, y, hand_y, now):
        """Runs one frame; lets the cursor pump follow the hand only on frames
        where the cursor is meant to move (never during a tap)."""
        self._follow = False
        action = self._update(geom, x, y, hand_y, now)
        if self.cursor is not None:
            self.cursor.set_following(self._follow)
        return action

    def _update(self, geom, x, y, hand_y, now):
        """geom: 21 hand landmarks in geometry space (3D world when available).
        x, y: the cursor position for this frame (screen fractions).
        hand_y: the hand's vertical position in the image (for scrolling).
        Returns the action name for /state."""
        scale = _dist(geom[0], geom[9]) or 1e-6
        d = {name: contact_distance(geom, chain) / scale for name, chain in FINGERS}

        if self.learning_progress() < 1.0:
            for name in d:
                self.rest[name].append(d[name])
            self._move(x, y, now)
            self.last = {"d": d, "contact": ""}
            return "learning"

        ratios = {name: d[name] / max(self.open_distance(name), 1e-6) for name in d}
        self._ratios = ratios
        if min(ratios.values()) >= APPROACH_RATIO:
            for name in d:                 # hand open: keep learning its open distances
                self.rest[name].append(d[name])
        for name, r in ratios.items():     # remember where each finger began closing
            if r >= APPROACH_RATIO:
                self.approach[name] = None
            elif self.approach[name] is None:
                self.approach[name] = (now, (self.history.at(now) or (x, y)))

        raw = self._touching_finger(ratios)
        if raw == self.pending:
            self.pending_frames += 1
        else:
            self.pending, self.pending_frames = raw, 1
        action = None
        needed = RELEASE_CONFIRM_FRAMES if raw is None else CONFIRM_FRAMES
        if self.pending != self.contact and self.pending_frames >= needed:
            action = self._transition(self.contact, self.pending, now, x, y, hand_y)
            self.contact = self.pending
        self.last = {"d": d, "contact": self.contact or ""}
        if action:
            return action
        return self._hold(x, y, hand_y, now)

    def _transition(self, old, new, now, x, y, hand_y):
        self.pinch_min = None
        if new is None:
            self.since = None
            self.scroll_prev_y = None
            if old is not None and not (self.dragging or self.scrolling):
                self.rearm[old] = self._ratios.get(old)
            if self.dragging:
                self.dragging = False
                pyautogui.mouseUp()
                return "release_drag"
            if self.scrolling:
                self.scrolling = False
                return None
            if self.cancel_tap:
                self.cancel_tap = False
                return None
            if old == "index":
                prev = self.last_tap
                if (prev is not None and now - prev[0] <= DOUBLE_WINDOW
                        and math.dist(prev[1], self.pos) <= DOUBLE_RADIUS):
                    # Second click on the very same pixel, inside the OS's
                    # double-click time: the OS turns the pair into a double click.
                    pyautogui.click(*self._px(prev[1]))
                    self.last_tap = None
                    return "double_click"
                pyautogui.click(*self._px(self.pos))
                self.last_tap = (now, self.pos)
                return "left_click"
            if old == "middle":
                pyautogui.click(*self._px(self.pos), button="right")
                return "right_click"
            return None           # ring and others: no action

        # A finger just touched (contact only ever goes none <-> one finger).
        self.since = now
        self.touch_xy = (x, y)
        self.touch_hand_y = hand_y
        before = self.history.at(now - TOUCH_LOOKBACK) or (x, y)
        start = self.approach.get(new)
        if start is not None and math.dist(start[1], before) <= ANCHOR_MAX_JUMP:
            self.pos = start[1]
        else:
            self.pos = before
        return None

    def _hold(self, x, y, hand_y, now):
        if self.contact is None:
            self._move(x, y, now)
            return "move"
        # A drag or scroll may only START while the finger is still clearly
        # touching -- not while a release is being confirmed: opening the hand
        # jolts the wrist, which used to turn the end of a tap into a drag/scroll.
        releasing = self.pending != self.contact
        if self.contact == "index":
            if not self.dragging and not releasing:
                w, _ = self.get_screen_size()
                moved_px = math.dist(self._px((x, y)), self._px(self.touch_xy))
                if moved_px > DRAG_START * w:
                    self.dragging = True
                    self.last_tap = None
                    pyautogui.mouseDown(*self._px(self.pos))
            if self.dragging:
                self._move(x, y, now)
                return "drag"
            return "holding_click"
        if self.contact == "middle" and (
                self.scrolling or (not releasing and abs(hand_y - self.touch_hand_y) > SCROLL_START)):
            if not self.scrolling:
                self.scrolling = True
                self.scroll_prev_y = hand_y
            amount = int(-(hand_y - self.scroll_prev_y) * SCROLL_GAIN)   # hand up -> scroll up
            if amount:
                wheel(amount)
            self.scroll_prev_y = hand_y
            return "scroll"
        return "pinch"            # cursor holds still while the tap is decided
