"""Helpers shared by the hand, eye and torso controllers.

Every mouse action in every mode goes through perform_mouse_action(), which
falls back to xdotool on Linux/X11 when PyAutoGUI fails and prints the
macOS/Linux permission hints once.

The eye and torso controllers work the same way: measure a few signals from
the landmarks, capture a *neutral* value for each when tracking starts (or on
/recenter), and map the offset from neutral onto the screen.
"""
import math
import os
import platform
import shutil
import statistics
import subprocess
import time

import pyautogui

# How many valid frames to average when capturing the neutral pose (~1s @ 30fps).
NEUTRAL_FRAMES = 30

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Mouse output with platform fallbacks
# ---------------------------------------------------------------------------
_mouse_action_error_logged = False
_display_server_warning_logged = False


def get_linux_display_server():
    if platform.system() != "Linux":
        return None

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("x11", "wayland"):
        return session_type
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def warn_linux_display_server_once():
    global _display_server_warning_logged

    if platform.system() != "Linux" or _display_server_warning_logged:
        return

    display_server = get_linux_display_server()
    if display_server == "wayland":
        print("Aviso: Wayland detectado. A camera pode funcionar, mas o Wayland costuma bloquear controle global do cursor.")
        print("Para cliques/movimento do WGI, use uma sessao X11/Xorg ou configure uma ferramenta de injecao permitida pelo compositor.")
    elif display_server == "unknown":
        print("Aviso: Nenhum DISPLAY/WAYLAND_DISPLAY detectado. O backend pode abrir a camera, mas nao deve controlar o cursor.")

    _display_server_warning_logged = True


def _run_xdotool(args):
    xdotool = shutil.which("xdotool")
    if not xdotool:
        return False

    try:
        subprocess.run([xdotool] + args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _perform_x11_mouse_action(action_name, *args, **kwargs):
    if platform.system() != "Linux" or get_linux_display_server() != "x11":
        return False

    # Positioned actions (click(x, y), mouseDown(x, y)) move first.
    if action_name in ("click", "doubleClick", "mouseDown", "mouseUp") and len(args) >= 2:
        if not _run_xdotool(["mousemove", str(int(args[0])), str(int(args[1]))]):
            return False
    if action_name == "moveTo" and len(args) >= 2:
        return _run_xdotool(["mousemove", str(int(args[0])), str(int(args[1]))])
    if action_name == "click":
        button = "3" if kwargs.get("button", "left") == "right" else "1"
        repeat = str(int(kwargs.get("clicks", 1)))
        return _run_xdotool(["click", "--repeat", repeat, "--delay", "80", button])
    if action_name == "doubleClick":
        return _run_xdotool(["click", "--repeat", "2", "--delay", "80", "1"])
    if action_name == "mouseDown":
        return _run_xdotool(["mousedown", "1"])
    if action_name == "mouseUp":
        return _run_xdotool(["mouseup", "1"])
    if action_name == "scroll" and args:
        scroll_amount = int(args[0])
        button = "4" if scroll_amount > 0 else "5"
        repeat = min(abs(scroll_amount), 10)
        if repeat == 0:
            return True
        return _run_xdotool(["click", "--repeat", str(repeat), button])

    return False


def perform_mouse_action(action_name, *args, **kwargs):
    """pyautogui.<action_name>(*args, **kwargs), falling back to xdotool on
    Linux/X11. Never raises: a failed mouse action must not kill tracking."""
    global _mouse_action_error_logged

    try:
        getattr(pyautogui, action_name)(*args, **kwargs)
        return True
    except Exception as exc:
        if _perform_x11_mouse_action(action_name, *args, **kwargs):
            return True

        if not _mouse_action_error_logged:
            print(f"Aviso: PyAutoGUI nao conseguiu executar '{action_name}': {exc}")
            print("No macOS, permita Camera e Accessibility para o app/Terminal nas configuracoes de privacidade.")
            if platform.system() == "Linux":
                display_server = get_linux_display_server()
                if display_server == "x11":
                    print("No Linux X11, instale 'xdotool' para fallback de clique/movimento caso o PyAutoGUI falhe.")
                elif display_server == "wayland":
                    print("No Linux Wayland, controle global do cursor geralmente e bloqueado. Use sessao X11/Xorg para suporte completo.")
            _mouse_action_error_logged = True
        return False


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def scroll(amount):
    """pyautogui.scroll() is raw wheel delta on Windows (120 = one notch) but
    whole clicks on macOS/Linux. The hand scroll uses Windows-sized numbers, so
    scale them down elsewhere instead of scrolling ~40x too far."""
    if amount == 0:
        return
    if not _IS_WINDOWS:
        amount = int(math.copysign(max(1, abs(amount) // 40), amount))
    perform_mouse_action("scroll", amount)


class NeutralPose:
    """Collects NEUTRAL_FRAMES samples of a dict of signals and stores the median."""

    def __init__(self, frames=NEUTRAL_FRAMES):
        self.frames = frames
        self.reset()

    def reset(self):
        self.samples = []
        self.values = None

    @property
    def ready(self):
        return self.values is not None

    def progress(self):
        return 1.0 if self.ready else len(self.samples) / self.frames

    def add(self, signals):
        if self.ready:
            return True
        self.samples.append(signals)
        if len(self.samples) >= self.frames:
            self.values = {k: statistics.median(s[k] for s in self.samples)
                           for k in self.samples[0]}
            self.samples = []
            return True
        return False

    def __getitem__(self, key):
        return self.values[key]


class Smoother:
    """Exponential smoothing with a small deadzone so a still head or body does
    not make the cursor tremble."""

    def __init__(self, alpha=0.25, deadzone=0.004):
        self.alpha = alpha
        self.deadzone = deadzone
        self.x = self.y = None

    def reset(self):
        self.x = self.y = None

    def update(self, x, y, t=None):
        if self.x is None:
            self.x, self.y = x, y
        else:
            nx = self.x + (x - self.x) * self.alpha
            ny = self.y + (y - self.y) * self.alpha
            if math.hypot(nx - self.x, ny - self.y) > self.deadzone:
                self.x, self.y = nx, ny
        return self.x, self.y


class OneEuroFilter:
    """One Euro filter (Casiez et al., 2012) on a 2D point.

    Its cutoff frequency rises with speed: while the point is nearly still it
    smooths hard (kills landmark jitter), and when it moves fast it smooths
    little (no lag). A fixed EMA can only trade one of those for the other.

    min_cutoff (Hz): smoothing at rest -- lower = steadier, but laggier.
    beta: how quickly smoothing relaxes with speed -- higher = less lag on
          fast moves. Units here are screen fractions per second.
    """

    def __init__(self, min_cutoff=0.6, beta=1.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x = self.y = None
        self.dx = self.dy = 0.0
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, y, t=None):
        t = time.time() if t is None else t
        if self.x is None or self.t is None:
            self.x, self.y, self.t = x, y, t
            return self.x, self.y
        dt = max(t - self.t, 1e-3)
        self.t = t
        # smoothed speed
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx += a_d * ((x - self.x) / dt - self.dx)
        self.dy += a_d * ((y - self.y) / dt - self.dy)
        speed = math.hypot(self.dx, self.dy)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return self.x, self.y


class CursorPump:
    """Moves the OS cursor from its own thread at a fixed rate (default 120 Hz),
    decoupled from the camera's ~30 fps, through a One Euro filter.

    The camera thread only posts the latest target (set_target). The filtered
    position keeps tracking the target at all times (position()), but the OS
    cursor is only moved while following -- callers hold it still during a
    tap and let it go again afterwards. freeze() stops the filtered position
    itself (the hand-mode "freeze" gesture)."""

    def __init__(self, get_screen_size, rate=120, min_cutoff=0.8, beta=2.0, d_cutoff=1.0):
        import threading
        self.get_screen_size = get_screen_size
        self.interval = 1.0 / rate
        self.filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.lock = threading.Lock()
        self.target = None
        self.pos = None
        self.following = True
        self.frozen = False
        self._last_px = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="wgi-cursor")
        self.thread.start()

    def set_target(self, x, y):
        with self.lock:
            self.target = (clamp01(x), clamp01(y))
            if self.pos is None:
                self.pos = self.target

    def set_following(self, following):
        self.following = bool(following)

    def freeze(self, frozen):
        self.frozen = bool(frozen)

    def position(self):
        with self.lock:
            return self.pos if self.pos is not None else (0.5, 0.5)

    def stop(self):
        self._stop.set()

    def _run(self):
        # Windows wakes sleeping threads on a ~15.6 ms timer by default, which
        # caps this loop at ~64 Hz; ask for 1 ms timer precision while hand
        # tracking runs (what games/media players do) and give it back after.
        winmm = None
        if _IS_WINDOWS:
            try:
                import ctypes
                winmm = ctypes.windll.winmm
                winmm.timeBeginPeriod(1)
            except Exception:
                winmm = None
        try:
            self._loop()
        finally:
            if winmm is not None:
                winmm.timeEndPeriod(1)

    def _loop(self):
        while not self._stop.wait(self.interval):
            with self.lock:
                if self.target is None:
                    continue
                if not self.frozen:
                    self.pos = self.filter.update(self.target[0], self.target[1], time.time())
                pos, following = self.pos, self.following
            if not following:
                continue      # cursor held (tap): leave it exactly where it is
            w, h = self.get_screen_size()
            px = (int(round(clamp01(pos[0]) * (w - 1))), int(round(clamp01(pos[1]) * (h - 1))))
            if px != self._last_px:
                self._last_px = px
                try:
                    perform_mouse_action("moveTo", *px)
                except Exception:
                    pass


class CursorController:
    """Base for the non-hand controllers: owns the neutral pose, smoothing,
    click cooldown and the tracker status reported through /state.

    Subclasses implement load() (build the MediaPipe model; runs on a worker
    thread, so the import cost never blocks the HTTP server) and
    process(frame, rgb, aspect) -> action string.
    """

    CLICK_COOLDOWN = 0.6

    def __init__(self, get_screen_size, smoothing=0.25, deadzone=0.004,
                 smoother=None, min_pixel_step=0, deadband_px=0):
        self.get_screen_size = get_screen_size
        self.neutral = NeutralPose()
        self.smoother = smoother if smoother is not None else Smoother(smoothing, deadzone)
        # Skip OS cursor moves smaller than this many pixels: sub-pixel
        # residue of the filter otherwise shows up as a 1-2 px shimmer.
        self.min_pixel_step = min_pixel_step
        # Backlash deadband: the cursor only moves once the smoothed target is
        # more than this many pixels away, and then trails it by that amount.
        # Shake smaller than the band disappears entirely; real movement is
        # only offset by a few pixels, which nobody notices.
        self.deadband_px = deadband_px
        self._out = None
        self._last_px = None
        self.history = PositionHistory()
        self.model = None
        self.last_click = 0.0
        self.status = "centering"   # centering | tracking | lost
        self.x, self.y = 0.5, 0.5
        self.button_held = False

    # -- lifecycle -----------------------------------------------------------
    def load(self):
        raise NotImplementedError

    def close(self):
        self.mouse_up()     # never leave the OS button stuck down
        if self.model is not None:
            try:
                self.model.close()   # frees the native TFLite graph
            except Exception:
                pass
            self.model = None

    def recenter(self):
        self.mouse_up()
        self.neutral.reset()
        self.smoother.reset()
        self.history.reset()
        self._last_px = None
        self._out = None
        self.status = "centering"

    def info(self):
        return {"status": self.status, "progress": round(self.neutral.progress(), 2)}

    # -- output --------------------------------------------------------------
    def move(self, x, y, now):
        fx, fy = self.smoother.update(clamp01(x), clamp01(y), now)
        w, h = self.get_screen_size()
        tx, ty = clamp01(fx) * (w - 1), clamp01(fy) * (h - 1)
        if self._out is None or not self.deadband_px:
            self._out = (tx, ty)
        else:
            dx, dy = tx - self._out[0], ty - self._out[1]
            d = math.hypot(dx, dy)
            if d > self.deadband_px:
                k = (d - self.deadband_px) / d
                self._out = (self._out[0] + dx * k, self._out[1] + dy * k)
        self.x, self.y = self._out[0] / (w - 1), self._out[1] / (h - 1)
        self.history.push(self.x, self.y, now)
        px, py = int(round(self._out[0])), int(round(self._out[1]))
        if self._last_px == (px, py):
            return
        if (self._last_px is not None and self.min_pixel_step
                and abs(px - self._last_px[0]) < self.min_pixel_step
                and abs(py - self._last_px[1]) < self.min_pixel_step):
            return
        self._last_px = (px, py)
        perform_mouse_action("moveTo", px, py)

    def _jump_to(self, pos):
        if pos is not None:
            self.x, self.y = pos
            self.smoother.x, self.smoother.y = pos
            self._last_px = None
        w, h = self.get_screen_size()
        self._out = (self.x * (w - 1), self.y * (h - 1))
        return int(self.x * (w - 1)), int(self.y * (h - 1))

    def click(self, now, button="left", pos=None, clicks=1):
        """Clicks at `pos` (normalised) or the current cursor. Returns True if
        the click fired (False while cooling down)."""
        if now - self.last_click < self.CLICK_COOLDOWN:
            return False
        self.last_click = now
        px, py = self._jump_to(pos)
        perform_mouse_action("click", px, py, clicks=clicks, interval=0.05, button=button)
        return True

    def mouse_down(self, pos=None):
        if self.button_held:
            return
        px, py = self._jump_to(pos)
        perform_mouse_action("mouseDown", px, py, button="left")
        self.button_held = True

    def mouse_up(self):
        if self.button_held:
            self.button_held = False
            perform_mouse_action("mouseUp", button="left")


class PositionHistory:
    """Short history of cursor positions, so a click can land where the cursor
    was *before* the gesture itself (a blink, a shrug) started moving it."""

    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.items = []

    def reset(self):
        self.items = []

    def push(self, x, y, now=None):
        now = time.time() if now is None else now
        self.items.append((now, x, y))
        cutoff = now - self.seconds
        while self.items and self.items[0][0] < cutoff:
            self.items.pop(0)

    def at(self, t):
        """Latest position recorded at or before time t (oldest if none)."""
        best = None
        for item in self.items:
            if item[0] <= t:
                best = item
            else:
                break
        if best is None:
            best = self.items[0] if self.items else None
        return (best[1], best[2]) if best else None
