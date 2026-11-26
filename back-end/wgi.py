import os
import time
import threading
import math
import traceback
import platform
import pyautogui
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import calculo_distancia as util
import cursor_movement as cursor_movement
import left_click as left_click
import right_click as right_click
import double_click as double_click
import scroll as scroll
import calibration_manager as calibration_manager

# MediaPipe defaults are 0.5. 0.85 on *tracking* forces a full re-detection
# whenever confidence dips, which drops frames during fast motion; every dropped
# frame clears the gesture histories and restarts the 3-frame confirmation.
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
# Resolution is re-queried instead of frozen at import: the user can change
# display mode, dock a laptop, or move the browser to another monitor while the
# server runs. Cached briefly so it is not a syscall on every frame.
SCREEN_SIZE_TTL = 1.0
_screen_size_cache = (0.0, None)

def get_screen_size():
    global _screen_size_cache
    ts, size = _screen_size_cache
    now = time.time()
    if size is None or now - ts > SCREEN_SIZE_TTL:
        try:
            w, h = pyautogui.size()
            if w > 0 and h > 0:
                size = (int(w), int(h))
        except Exception:
            pass
        if size is None:
            size = (1920, 1080)
        _screen_size_cache = (now, size)
    return size

last_click_time = 0
click_cooldown = 0.7
click_confirmation_frames = 3
drag_hold_threshold = 0.8
is_dragging = False
last_left_click_detected_time = 0
left_click_start_time = 0
mouse_down_triggered = False
wizard_active = False
left_click_candidate_frames = 0
right_click_candidate_frames = 0
double_click_candidate_frames = 0
right_click_armed = True
double_click_armed = True

calibration_gesture = None
calibration_idx_history = []
calibration_mid_history = []
calibration_thumb_history = []

latest_frame = None          # full-resolution BGR frame (numpy array)
latest_frame_id = 0          # bumped every frame so streams know when to re-encode
camera_size = (0, 0)         # whatever the camera actually negotiated

# Requested capture size. Left unset by default so the camera keeps its own
# native mode; override to ask for something higher.
CAPTURE_WIDTH = int(os.environ.get("WGI_CAPTURE_WIDTH", "0"))
CAPTURE_HEIGHT = int(os.environ.get("WGI_CAPTURE_HEIGHT", "0"))

current_state = {
    "x": 0.5,
    "y": 0.5,
    "action": "move",
    "calibration": {"status": "idle", "gesture": None}
}
camera_ready_event = threading.Event()

def should_suppress_mouse_actions():
    return calibration_gesture is not None

def cancel_calibration():
    global calibration_gesture, calibration_idx_history, calibration_mid_history, calibration_thumb_history

    calibration_gesture = None
    calibration_idx_history = []
    calibration_mid_history = []
    calibration_thumb_history = []
    current_state["calibration"] = {"status": "idle", "gesture": None}

def reset_interaction_state():
    global is_dragging, last_left_click_detected_time, left_click_start_time, mouse_down_triggered
    global left_click_candidate_frames, right_click_candidate_frames, double_click_candidate_frames
    global right_click_armed, double_click_armed, last_click_time

    if is_dragging and mouse_down_triggered:
        pyautogui.mouseUp()

    is_dragging = False
    mouse_down_triggered = False
    # Cleared with the rest of the interaction state: a cooldown left over from
    # before a stop/start (or a calibration run) would silently swallow the
    # first right/double click of the new session.
    last_click_time = 0
    last_left_click_detected_time = 0
    left_click_start_time = 0
    left_click_candidate_frames = 0
    right_click_candidate_frames = 0
    double_click_candidate_frames = 0
    right_click_armed = True
    double_click_armed = True
    left_click.reset_state()
    right_click.reset_state()
    double_click.reset_state()
    scroll.reset_state()

def cancel_drag():
    """Abort an in-progress left-click/drag because another gesture took over.

    Releases the physical button if it was down, but deliberately does NOT emit
    a click: the user switched to a different gesture, they did not finish a
    click. Without this, is_dragging stayed True forever once scroll/right/
    double click won the dispatch below, the button stayed held down, and the
    `not is_dragging` guard blocked every further click.
    """
    global is_dragging, mouse_down_triggered, left_click_candidate_frames

    if is_dragging and mouse_down_triggered and not should_suppress_mouse_actions():
        pyautogui.mouseUp()
    is_dragging = False
    mouse_down_triggered = False
    left_click_candidate_frames = 0

def detect_gesture(frame, landmark_list, geom_list, processed):
    import cv2  
    global last_click_time, is_dragging, last_left_click_detected_time
    global left_click_start_time, mouse_down_triggered
    global calibration_gesture, calibration_idx_history, calibration_mid_history, calibration_thumb_history
    global current_state
    global wizard_active
    global left_click_candidate_frames, right_click_candidate_frames, double_click_candidate_frames
    global right_click_armed, double_click_armed

    current_time = time.time()
    action = "move"

    if len(landmark_list) >= 21 and len(geom_list) >= 21:
        hand_landmarks = processed.multi_hand_landmarks[0]
        palm_center = cursor_movement.get_palm_center(hand_landmarks)
        
        if calibration_gesture:
            idx_angle = util.get_angle(geom_list[5], geom_list[6], geom_list[8])
            mid_angle = util.get_angle(geom_list[9], geom_list[10], geom_list[12])
            thumb_dist = util.get_distance(geom_list, 4, 5)
            
            calibration_idx_history.append(idx_angle)
            calibration_mid_history.append(mid_angle)
            calibration_thumb_history.append(thumb_dist)
            
            # Record for ~1.5 seconds (45 frames) to get a stable average
            if len(calibration_idx_history) >= 45:
                completed_gesture = calibration_gesture
                calibrated = calibration_manager.auto_calibrate(completed_gesture, calibration_idx_history, calibration_mid_history, calibration_thumb_history)
                calibration_gesture = None
                current_state["action"] = "calibrated" if calibrated else "calibration_failed"
                current_state["calibration"] = {
                    "status": "saved" if calibrated else "failed",
                    "gesture": completed_gesture
                }
            else:
                current_state["action"] = "calibrating"
                current_state["calibration"] = {"status": "recording", "gesture": calibration_gesture}
                cv2.putText(frame, f"Calibrating...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return
     
        x, y = cursor_movement.get_normalized_position(palm_center, geom_list)
        
        if x is not None and y is not None:
            current_state["x"], current_state["y"] = x, y
            if not should_suppress_mouse_actions():
                screen_w, screen_h = get_screen_size()
                pyautogui.moveTo(int(x * screen_w), int(y * screen_h))

        thumb_index_dist = util.get_distance(geom_list, 4, 5)
        
        is_scrolling, scroll_amount = scroll.detect_scroll(landmark_list, geom_list)
        raw_is_db = double_click.is_double_click(geom_list, thumb_index_dist)
        raw_is_rc = right_click.is_right_click(geom_list, thumb_index_dist)
        raw_is_lc = left_click.is_left_click(geom_list, thumb_index_dist, is_dragging)

        if is_scrolling:
            raw_is_db = False
            raw_is_rc = False
            raw_is_lc = False

        if is_dragging:
            is_lc = raw_is_lc
        else:
            if raw_is_lc:
                left_click_candidate_frames += 1
            else:
                left_click_candidate_frames = 0
            is_lc = left_click_candidate_frames >= click_confirmation_frames

        if raw_is_rc:
            right_click_candidate_frames += 1
        else:
            right_click_candidate_frames = 0
            right_click_armed = True

        if raw_is_db:
            double_click_candidate_frames += 1
        else:
            double_click_candidate_frames = 0
            double_click_armed = True

        is_rc = right_click_candidate_frames >= click_confirmation_frames
        is_db = double_click_candidate_frames >= click_confirmation_frames

        if is_scrolling:
            cancel_drag()
            action = "scroll"
            if scroll_amount != 0 and not should_suppress_mouse_actions():
                pyautogui.scroll(scroll_amount)
        elif is_db:
            cancel_drag()
            action = "double_click"
        elif is_rc:
            cancel_drag()
            action = "right_click"
        elif is_lc:
            last_left_click_detected_time = current_time
            if not is_dragging:
                is_dragging = True
                left_click_candidate_frames = 0
                left_click_start_time = current_time
                mouse_down_triggered = False
                action = "left_click_intent"
            else:
                if current_time - left_click_start_time > drag_hold_threshold and not mouse_down_triggered:
                    if not should_suppress_mouse_actions():
                        pyautogui.mouseDown()
                    mouse_down_triggered = True
                action = "drag" if mouse_down_triggered else "holding_click"
        else:
            if is_dragging:
                if current_time - last_left_click_detected_time > 0.2:
                    is_dragging = False
                    if mouse_down_triggered:
                        action = "release_drag"
                        if not should_suppress_mouse_actions():
                            pyautogui.mouseUp()
                        mouse_down_triggered = False
                    else:
                        action = "left_click"
                        if not should_suppress_mouse_actions():
                            pyautogui.click()
                    last_click_time = current_time
                    left_click_candidate_frames = 0
                else:
                    action = "drag" if mouse_down_triggered else "holding_click"

        if not is_dragging and current_time - last_click_time > click_cooldown:
            if action == "right_click" and right_click_armed:
                if not should_suppress_mouse_actions():
                    pyautogui.click(button='right')
                right_click_armed = False
                last_click_time = current_time
            elif action == "double_click" and double_click_armed:
                if not should_suppress_mouse_actions():
                    pyautogui.doubleClick()
                double_click_armed = False
                last_click_time = current_time

        current_state["action"] = action
        
        cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if cursor_movement.is_frozen(geom_list):
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            
    else:
        left_click_candidate_frames = 0
        right_click_candidate_frames = 0
        double_click_candidate_frames = 0
        right_click_armed = True
        double_click_armed = True
        left_click.reset_state()
        right_click.reset_state()
        double_click.reset_state()
        scroll.reset_state()
        if is_dragging:
            is_dragging = False
            if mouse_down_triggered:
                if not should_suppress_mouse_actions():
                    pyautogui.mouseUp()
                mouse_down_triggered = False

class HandController:
    """Hand mode: MediaPipe Hands. The open palm steers the cursor
    (cursor_movement.py); fingertip-to-thumb pinches click, drag and scroll
    (pinch_control.py)."""
    name = "hands"

    def __init__(self):
        self.model = None
        self.trace = None
        self.pinch = None
        self.cursor = None
        self.hand_seen = False

    def load(self):
        import pinch_control
        from control_common import CursorPump
        # The hand cursor is smoothed (One Euro) and moved at 120 Hz on its own
        # thread, instead of a fixed average stepped at the camera's ~30 fps.
        self.cursor = CursorPump(get_screen_size)
        self.pinch = pinch_control.PinchGestures(get_screen_size, cursor=self.cursor)
        path = os.environ.get("WGI_TRACE")
        if path:
            # Numbers only: thumb-to-fingertip distances (in hand sizes), the
            # learned open distances, and the resulting state.
            self.trace = open(path.replace(".csv", "_hands.csv"), "a", buffering=1)
            self.trace.write("t,d_index,d_middle,d_ring,open_index,open_middle,open_ring,"
                             "contact,action,hand_y,cur_x,cur_y,geom\n")
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.draw = mp.solutions.drawing_utils
        self.model = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            max_num_hands=1
        )

    def process(self, frame, rgb_frame, aspect):
        processed = self.model.process(rgb_frame)
        now = time.time()

        if not processed.multi_hand_landmarks:
            self.hand_seen = False
            dropped = self.pinch.dragging
            self.pinch.reset()
            return "release_drag" if dropped else "move"
        self.hand_seen = True

        hand = processed.multi_hand_landmarks[0]
        landmark_list = [(lm.x, lm.y) for lm in hand.landmark]   # image space: position
        world = getattr(processed, "multi_hand_world_landmarks", None)
        world_list = ([(lm.x, lm.y, lm.z) for lm in world[0].landmark] if world else None)
        geom_list = util.to_geometry_space(landmark_list, world_list, aspect)  # 3D: shape
        self.draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

        palm = cursor_movement.get_palm_center(hand)
        frozen = cursor_movement.is_frozen(geom_list)      # thumb on the palm: hold still
        self.cursor.freeze(frozen)
        if not frozen:
            self.cursor.set_target(*cursor_movement.get_target_position(palm))
        x, y = self.cursor.position()
        current_state["x"], current_state["y"] = x, y
        action = self.pinch.update(geom_list, x, y, landmark_list[0][1], now)

        if self.trace:
            d = self.pinch.last.get("d", {})
            o = {n: self.pinch.open_distance(n) for n in ("index", "middle", "ring")}
            fmt = lambda v: f"{v:.3f}" if v is not None else ""
            self.trace.write(f"{now:.3f},{fmt(d.get('index'))},{fmt(d.get('middle'))},{fmt(d.get('ring'))},"
                             f"{fmt(o['index'])},{fmt(o['middle'])},{fmt(o['ring'])},"
                             f"{self.pinch.last.get('contact', '')},{action},"
                             f"{landmark_list[0][1]:.4f},{x:.4f},{y:.4f},"
                             f"{' '.join(f'{v:.4f}' for p in geom_list for v in p)}\n")
        return action

    def recenter(self):
        pass

    def info(self):
        if self.pinch is None:
            return None
        if not self.hand_seen:
            return {"status": "lost", "progress": 1.0}
        p = self.pinch.learning_progress()
        return {"status": "centering" if p < 1.0 else "tracking", "progress": round(p, 2)}

    def close(self):
        if self.pinch is not None:
            self.pinch.reset()
        if getattr(self, "cursor", None) is not None:
            self.cursor.stop()
        # Hands owns a native TFLite graph that is NOT freed by the garbage
        # collector. Re-running start/stop without closing it leaked ~18 MB
        # per cycle.
        if self.model is not None:
            self.model.close()
            self.model = None
        if self.trace:
            self.trace.close()
            self.trace = None


def _make_controller(mode):
    # Imported here, not at module top: nothing MediaPipe-related loads until
    # the user actually picks a mode.
    if mode == "eyes":
        import eye_control
        return eye_control.EyeController(get_screen_size)
    if mode == "torso":
        import torso_control
        return torso_control.TorsoController(get_screen_size)
    return HandController()


class TrackingSession:
    """One start..stop run of one mode. Status lives on the session so a
    previous session that is still shutting down can never overwrite the
    state of the one that replaced it."""

    def __init__(self, mode, headless):
        self.mode = mode
        self.headless = headless
        self.stop_event = threading.Event()
        self.model_status = "loading"     # loading | ready | error
        self.camera_status = "opening"    # opening | ready | error
        self.error = None
        self.controller = None
        self.thread = None

    @property
    def active(self):
        return not self.stop_event.is_set()


_session = None
_session_lock = threading.Lock()


def tracking_active():
    s = _session
    return s is not None and s.active


def active_mode():
    s = _session
    return s.mode if s is not None else None


def _open_camera(cv2):
    if platform.system() == 'Linux':
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Aviso: Falha ao usar V4L2. Tentando backend padrão do OpenCV...")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    # cap.set() silently no-ops on unsupported modes, so always read back the
    # value the driver actually settled on rather than assuming it took.
    if CAPTURE_WIDTH and CAPTURE_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    return cap


# Actions that mark a single event. The loop runs faster than the extension
# polls /state, so these must not be overwritten by the next frame's "move"
# before someone has read them (/state resets them).
ONE_SHOT_ACTIONS = {"left_click", "right_click", "double_click", "release_drag"}


def _tracking_loop(session):
    """Loads the selected model on a worker thread while this thread opens the
    camera -- on Windows the camera alone can take ~35 s, so doing them in
    parallel means the model costs nothing extra."""
    global camera_size, latest_frame, latest_frame_id
    import cv2

    controller = _make_controller(session.mode)
    session.controller = controller
    load_errors = []

    def _load():
        t0 = time.time()
        try:
            controller.load()
            print(f"Model '{session.mode}' loaded in {time.time() - t0:.1f}s")
        except Exception as exc:
            traceback.print_exc()
            load_errors.append(exc)

    loader = threading.Thread(target=_load, daemon=True, name=f"wgi-load-{session.mode}")
    loader.start()

    cap = None
    try:
        cap = _open_camera(cv2)
        if cap is None:
            print("Erro: Nao foi possivel abrir a camera.")
            session.camera_status = "error"
            session.error = "Could not open the camera."
            return
        session.camera_status = "ready"
        camera_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera: {camera_size[0]}x{camera_size[1]}")

        while loader.is_alive():
            if not session.active:
                return
            loader.join(0.1)
        if load_errors:
            session.model_status = "error"
            session.error = f"Could not load the {session.mode} model: {load_errors[0]}"
            return
        session.model_status = "ready"
        if _session is session:
            camera_ready_event.set()

        while cap.isOpened() and session.active:
            ret, frame = cap.read()
            if not ret:
                session.error = "Camera stopped sending frames."
                break
            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            aspect = frame_w / frame_h if frame_h else 1.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False       # documented fast path

            action = controller.process(frame, rgb_frame, aspect)
            if action is not None and (action in ONE_SHOT_ACTIONS
                                       or current_state["action"] not in ONE_SHOT_ACTIONS):
                current_state["action"] = action

            latest_frame = frame
            latest_frame_id += 1

            if not session.headless:
                cv2.putText(frame, f"Mode: {session.mode}", (50, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow('WGI Engine', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as exc:
        traceback.print_exc()
        session.error = f"Tracking crashed: {exc}"
    finally:
        session.stop_event.set()
        if cap is not None:
            cap.release()
        if not session.headless:
            cv2.destroyAllWindows()
        # A model still loading cannot be interrupted; close it once it is built.
        loader.join()
        controller.close()
        if _session is session:
            if session.mode == "hands":
                reset_interaction_state()
            camera_ready_event.set()

def encode_frame(frame, width=0, quality=80):
    """JPEG-encodes a frame, optionally rescaled to `width`, keeping the aspect
    ratio. width=0 keeps the camera's native size."""
    import cv2
    h, w = frame.shape[:2]
    if width and width != w and w > 0:
        scale = width / w
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (width, max(1, int(round(h * scale)))), interpolation=interp)
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None

def _read_json(handler):
    try:
        length = int(handler.headers.get('Content-Length', 0))
    except (TypeError, ValueError):
        return {}
    if length <= 0:
        return {}
    try:
        data = json.loads(handler.rfile.read(length).decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class WGIServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        message = "Ação não reconhecida."
        extra = {}
        if self.path == '/start_tracking':
            # Returns immediately; the model and camera come up in the
            # background. Poll GET /state for model_status / camera_ready.
            mode = _read_json(self).get("mode", "hands")
            if mode not in MODES:
                message = f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}."
                extra["error"] = True
            else:
                start_tracking(headless=True, mode=mode)
                message = f"Loading {mode} model and starting the camera..."
                extra["mode"] = mode
        elif self.path == '/stop_tracking':
            stop_tracking()
            message = "Successfully stopped camera."
        elif self.path == '/recenter':
            s = _session
            if s is not None and s.active and s.controller is not None:
                s.controller.recenter()
                message = "Hold a neutral pose: re-centering..."
            else:
                message = "Start tracking before re-centering."
                extra["error"] = True
        elif self.path == '/config':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                new_config = json.loads(post_data.decode('utf-8'))
                calibration_manager.config.update(new_config)
                calibration_manager.config = calibration_manager.sanitize_config(calibration_manager.config)
                calibration_manager.save_config(calibration_manager.config)
                message = "Configuração atualizada com sucesso!"
        elif self.path == '/start_calibration':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                global calibration_gesture, calibration_idx_history, calibration_mid_history, calibration_thumb_history
                gesture = data.get("gesture")
                if not tracking_active() or active_mode() != "hands":
                    cancel_calibration()
                    current_state["action"] = "calibration_failed"
                    current_state["calibration"] = {"status": "failed", "gesture": gesture}
                    message = "Start the camera in Hands mode before calibration."
                elif gesture not in calibration_manager.HAND_GESTURES:
                    cancel_calibration()
                    current_state["action"] = "calibration_failed"
                    current_state["calibration"] = {"status": "failed", "gesture": gesture}
                    message = "Calibration gesture not recognized."
                else:
                    reset_interaction_state()
                    calibration_gesture = gesture
                    calibration_idx_history = []
                    calibration_mid_history = []
                    calibration_thumb_history = []
                    current_state["calibration"] = {"status": "recording", "gesture": calibration_gesture}
                    message = f"Calibrating {calibration_gesture}..."
        elif self.path == '/set_wizard_mode':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                global wizard_active
                requested_active = data.get("active", False)
                if not requested_active:
                    cancel_calibration()
                reset_interaction_state()
                wizard_active = requested_active
                message = f"Wizard mode set to {wizard_active}"
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers()
        self.wfile.write(json.dumps({"message": message, **extra}).encode())

    def do_GET(self):
        if self.path == '/state':
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers()
            state_data = current_state.copy()
            s = _session
            active = s is not None and s.active
            state_data["is_tracking"] = active
            state_data["modes"] = list(MODES)
            state_data["mode"] = s.mode if s else None
            # idle | loading | ready | error -- "error" survives the stop so the
            # popup can show why tracking ended.
            state_data["model_status"] = (s.model_status if active or (s and s.error) else "idle") if s else "idle"
            state_data["camera_status"] = (s.camera_status if active or (s and s.error) else "idle") if s else "idle"
            state_data["error"] = s.error if s else None
            ctrl = s.controller if active else None
            state_data["tracker"] = ctrl.info() if ctrl is not None else None
            screen_w, screen_h = get_screen_size()
            state_data["screen"] = {"width": screen_w, "height": screen_h}
            state_data["camera"] = {"width": camera_size[0], "height": camera_size[1]}
            state_data["camera_ready"] = (active and s.model_status == "ready"
                                          and s.camera_status == "ready")
            self.wfile.write(json.dumps(state_data).encode())
            if current_state["action"] not in ["move", "drag", "holding_click"]: current_state["action"] = "move"
        elif self.path == '/config':
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers()
            self.wfile.write(json.dumps(calibration_manager.config).encode())
        elif self.path.startswith('/video_feed'):
            query = parse_qs(urlparse(self.path).query)

            def _int(name, default, lo, hi):
                try:
                    value = int(query.get(name, [default])[0])
                except (TypeError, ValueError):
                    return default
                return max(lo, min(hi, value))

            # width=0 means "native": the client sizes the stream to what it is
            # actually displaying on, instead of a size hardcoded server-side.
            width = _int('width', 0, 0, 3840)
            quality = _int('quality', 80, 30, 95)
            fps = _int('fps', 15, 1, 60)

            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                sent_id = -1
                interval = 1.0 / fps
                while tracking_active():
                    frame, frame_id = latest_frame, latest_frame_id
                    if frame is not None and frame_id != sent_id:
                        sent_id = frame_id
                        payload = encode_frame(frame, width, quality)
                        if payload:
                            self.wfile.write(
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                b'Content-Length: ' + str(len(payload)).encode() +
                                b'\r\n\r\n' + payload + b'\r\n')
                    time.sleep(interval)
            except Exception:
                pass
        elif self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers()
            self.wfile.write(b"WGI Server is running! Endpoints: GET /state, GET /config, GET /video_feed, POST /start_tracking {\"mode\": \"hands|eyes|torso\"}, POST /stop_tracking, POST /recenter, POST /config")
        else:
            self.send_error(404, "Endpoint not found")

MODES = ("hands", "eyes", "torso")
# How long a mode switch waits for the previous session to release the camera.
SWITCH_JOIN_TIMEOUT = 15.0

def start_tracking(headless, mode="hands", wait=0):
    """Starts tracking in `mode` and returns the session immediately.

    The model loads on a worker thread while the camera opens (both can take
    seconds; the camera tens of seconds on some Windows drivers), so the HTTP
    request never blocks on them. Poll GET /state for model_status /
    camera_ready. Asking for a different mode while tracking switches modes:
    the old model is closed and the camera released before the new one starts.
    `wait` (seconds) optionally blocks until ready -- the CLI uses it.
    """
    global _session
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    with _session_lock:
        current = _session
        if current is not None and current.active and current.mode == mode:
            session = current
        else:
            if current is not None:
                current.stop_event.set()
                if current.thread is not None:
                    current.thread.join(timeout=SWITCH_JOIN_TIMEOUT)
            cancel_calibration()
            reset_interaction_state()
            camera_ready_event.clear()
            current_state["action"] = "move"
            session = TrackingSession(mode, headless)
            _session = session
            session.thread = threading.Thread(target=_tracking_loop, args=(session,),
                                              daemon=True, name=f"wgi-track-{mode}")
            session.thread.start()
    if wait:
        camera_ready_event.wait(timeout=wait)
    return session

def stop_tracking():
    with _session_lock:
        if _session is not None:
            _session.stop_event.set()
        cancel_calibration()
        reset_interaction_state()
        current_state["action"] = "move"

def check_environment():
    """Surface dependency mismatches instead of letting them fail obscurely."""
    import importlib.metadata as md
    problems = []

    def ver(pkg):
        try: return md.version(pkg)
        except md.PackageNotFoundError: return None

    numpy_v, mp_v = ver("numpy"), ver("mediapipe")
    if numpy_v and mp_v and numpy_v.startswith("2.") and mp_v.startswith("0.10."):
        problems.append(
            f"numpy {numpy_v} with mediapipe {mp_v}: these wheels are built against "
            f"the numpy 1.x ABI. Install numpy<2.0 (see requirements.txt).")

    if ver("opencv-python") and ver("opencv-contrib-python"):
        problems.append(
            f"both opencv-python ({ver('opencv-python')}) and opencv-contrib-python "
            f"({ver('opencv-contrib-python')}) are installed; two competing cv2 "
            f"modules are on the path. Uninstall opencv-python.")

    for msg in problems:
        print(f"AVISO / WARNING: {msg}")
    return not problems

def find_free_port(start_port=8765, max_port=8800):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Nenhuma porta livre encontrada entre {start_port} e {max_port}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="WGI gesture server.")
    parser.add_argument("--start", action="store_true",
                        help="Start tracking immediately instead of waiting for the "
                             "extension to POST /start_tracking. Without this the "
                             "server runs but the camera is off and the mouse never moves.")
    parser.add_argument("--show", action="store_true",
                        help="With --start, open the annotated camera window (press q to quit).")
    parser.add_argument("--mode", choices=MODES, default="hands",
                        help="Control mode used with --start (default: hands).")
    args = parser.parse_args()

    check_environment()
    port = find_free_port()
    server = ThreadingHTTPServer(('127.0.0.1', port), WGIServer)
    print(f"WGI Server Running on {port}...")
    if args.start:
        print(f"Loading {args.mode} model and opening camera (can take ~35s on some Windows drivers)...")
        session = start_tracking(headless=not args.show, mode=args.mode, wait=90)
        if session.model_status == "ready" and session.camera_status == "ready":
            print(f"Ready -- the mouse is now under {args.mode} control.")
        else:
            print(f"Tracking did not come up: {session.error or 'timed out'}")
    else:
        print("Camera is idle. The extension starts it, or re-run with --start [--mode eyes].")
    server.serve_forever()
