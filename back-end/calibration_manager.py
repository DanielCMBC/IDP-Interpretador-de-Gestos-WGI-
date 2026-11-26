import json
import os
import copy
import statistics
import sys

def get_runtime_dir():
    """Where calibration.json lives: next to the executable in the compiled
    (PyInstaller) app, the repository root when running from source."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_FILE = os.path.join(get_runtime_dir(), "calibration.json")

# Once a left click has started, the pose is allowed to relax by this many
# degrees before the drag releases. It is hysteresis, not a second pose: it must
# stay well BELOW a relaxed open hand, or the drag latches and no click is ever
# emitted. Measured rest-pose index angles run 100-175 deg, so 110 (the old
# value) sat inside the rest range and never released.
DRAG_HYSTERESIS = 25

DEFAULT_CONFIG = {
    "right_click": {
        "mid_angle_max": 50,
        "idx_angle_min": 90,
        "thumb_index_dist_min": 50,
        "stability_std_max": 25
    },
    "left_click": {
        "idx_angle_max": 50,
        "mid_angle_min": 90,
        "idx_drag_angle_max": 50 + DRAG_HYSTERESIS,
        "mid_drag_angle_min": 90 - DRAG_HYSTERESIS,
        "thumb_index_dist_min": 50,
        "stability_std_max": 25
    },
    "double_click": {
        "trigger_angle_max": 50,
        "release_angle_min": 90,
        "thumb_index_dist_min": 50,
        "stability_std_max": 25
    },
    "scroll": {
        "idx_angle_min": 90,
        "mid_angle_min": 90,
        "ring_angle_max": 50,
        "pinky_angle_max": 50,
        "stability_std_max": 25
    },
    # Eye/head mode (eye_control.py). Head signal = ~sin(head angle).
    "eyes": {
        "head_gain_x": 2.8,            # screen edge at ~10 deg of turn
        "head_gain_y": 4.0,            # screen edge at ~7 deg of tilt
        # The iris is the noisiest signal in the face mesh (and its vertical
        # position mostly tracks the eyelids), so it only nudges the cursor.
        "iris_gain_x": 0.4,
        "iris_gain_y": 0.0,
        # One Euro filter: lower min_cutoff = steadier when still; higher
        # beta = less lag on fast head moves.
        "smooth_min_cutoff": 0.2,
        "smooth_beta": 0.6,
        "deadband_px": 24,             # shake smaller than this never moves the cursor
        "blink_ratio": 0.5,            # eye "closed" below this fraction of its recent open level
        "wink_ratio": 0.7,             # wink = closed eye below this fraction of the other eye
        "wink_min": 0.05,              # seconds
        "drag_hold": 0.6,              # left eye shut this long -> press and drag
        "wink_max": 1.5,               # right eye shut longer than this -> ignored
        "double_window": 0.45,         # second left wink within this -> double click
        "mouth_open_ratio": 0.35,
        "mouth_hold": 0.35,            # mouth must stay open this long before scrolling
        "scroll_deadzone": 0.05,       # ~3 deg of nod before scrolling starts
        "scroll_speed": 500,
        # Laugh / cough / sneeze guard: freeze cursor and clicks for
        # disturbance_hold seconds after the head turns faster than jerk_speed
        # (~sin(angle) per second; 2.0 is ~115 deg/s -- above the user's
        # recorded deliberate head motion) or both eyes squint with the mouth open.
        "jerk_speed": 2.0,
        "squint_ratio": 0.8,
        "disturbance_hold": 0.4
    },
    # Torso mode (torso_control.py). Offsets are in shoulder widths.
    "torso": {
        "gain_x": 1.6,
        "gain_y": 2.5,
        "dwell_enabled": 1,
        "dwell_time": 1.2,
        "dwell_radius": 0.03,
        "shrug_ratio": 0.75,
        "lean_scroll": 0.12,
        "scroll_speed": 400
    }
}

# Gestures the hand calibration wizard can record (eyes/torso are settings, not poses).
HAND_GESTURES = ("right_click", "left_click", "double_click", "scroll")

VALID_RANGES = {
    "eyes": {
        "head_gain_x": (0.3, 15.0),
        "head_gain_y": (0.3, 20.0),
        "iris_gain_x": (0.0, 5.0),
        "iris_gain_y": (0.0, 5.0),
        "smooth_min_cutoff": (0.05, 5.0),
        "smooth_beta": (0.0, 20.0),
        "deadband_px": (0, 60),
        "blink_ratio": (0.3, 0.85),
        "wink_ratio": (0.3, 0.95),
        "wink_min": (0.0, 0.5),
        "drag_hold": (0.3, 2.0),
        "wink_max": (0.5, 4.0),
        "double_window": (0.2, 1.0),
        "mouth_open_ratio": (0.15, 0.8),
        "mouth_hold": (0.0, 1.5),
        "scroll_deadzone": (0.0, 0.3),
        "scroll_speed": (50, 6000),
        "jerk_speed": (0.5, 30.0),
        "squint_ratio": (0.5, 0.95),
        "disturbance_hold": (0.0, 2.0)
    },
    "torso": {
        "gain_x": (0.3, 6.0),
        "gain_y": (0.3, 8.0),
        "dwell_enabled": (0, 1),
        "dwell_time": (0.5, 4.0),
        "dwell_radius": (0.005, 0.1),
        "shrug_ratio": (0.4, 0.85),
        "lean_scroll": (0.05, 0.4),
        "scroll_speed": (50, 2000)
    },
    "right_click": {
        "mid_angle_max": (20, 95),
        "idx_angle_min": (65, 175),
        "thumb_index_dist_min": (25, 350),
        "stability_std_max": (8, 35)
    },
    "left_click": {
        "idx_angle_max": (20, 95),
        "mid_angle_min": (65, 175),
        "idx_drag_angle_max": (70, 130),
        "mid_drag_angle_min": (20, 80),
        "thumb_index_dist_min": (25, 350),
        "stability_std_max": (8, 35)
    },
    "double_click": {
        "trigger_angle_max": (20, 90),
        "release_angle_min": (65, 145),
        "thumb_index_dist_min": (25, 350),
        "stability_std_max": (8, 35)
    },
    "scroll": {
        "idx_angle_min": (65, 175),
        "mid_angle_min": (65, 175),
        "ring_angle_max": (20, 95),
        "pinky_angle_max": (20, 95),
        "stability_std_max": (8, 35)
    }
}

# Fraction of the thumb distance measured during calibration that still counts as
# "thumb deliberately extended". 0.6 leaves real headroom for tracking noise.
THUMB_GATE_MARGIN = 0.6

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def sanitize_config(raw_config):
    clean = copy.deepcopy(DEFAULT_CONFIG)

    if isinstance(raw_config, dict):
        for gesture_name, gesture_defaults in DEFAULT_CONFIG.items():
            raw_gesture = raw_config.get(gesture_name, {})
            if not isinstance(raw_gesture, dict):
                continue

            for key, default_value in gesture_defaults.items():
                value = raw_gesture.get(key, default_value)
                min_value, max_value = VALID_RANGES.get(gesture_name, {}).get(key, (None, None))

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = default_value

                if min_value is not None and max_value is not None:
                    value = clamp(value, min_value, max_value)

                clean[gesture_name][key] = value

    return clean

def load_config():
    if not os.path.exists(DB_FILE):
        save_config(DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)
        
    with open(DB_FILE, 'r') as f:
        try:
            loaded_config = json.load(f)
        except json.JSONDecodeError:
            return copy.deepcopy(DEFAULT_CONFIG)

    clean_config = sanitize_config(loaded_config)
    if clean_config != loaded_config:
        save_config(clean_config)
    return clean_config

def save_config(config_data):
    config_data = sanitize_config(config_data)
    with open(DB_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

def auto_calibrate(gesture_name, idx_raw_history, mid_raw_history, thumb_dist_history=None):
    global config

    if gesture_name not in HAND_GESTURES:
        return False
    
    if gesture_name not in config:
        config[gesture_name] = {}
        
    idx_mean = statistics.median(idx_raw_history)
    mid_mean = statistics.median(mid_raw_history)
    
    BUFFER = 12
    
    if gesture_name == "left_click":
        if idx_mean > 110 or mid_mean < 60:
            return False
        config[gesture_name]["idx_angle_max"] = idx_mean + BUFFER
        config[gesture_name]["mid_angle_min"] = mid_mean - BUFFER
        config[gesture_name]["idx_drag_angle_max"] = idx_mean + BUFFER + DRAG_HYSTERESIS
        config[gesture_name]["mid_drag_angle_min"] = mid_mean - BUFFER - DRAG_HYSTERESIS
    elif gesture_name == "right_click":
        if idx_mean < 60 or mid_mean > 110:
            return False
        config[gesture_name]["idx_angle_min"] = idx_mean - BUFFER
        config[gesture_name]["mid_angle_max"] = mid_mean + BUFFER
    elif gesture_name == "double_click":
        if idx_mean > 110 or mid_mean > 110:
            return False
        trigger = max(idx_mean, mid_mean) + BUFFER
        config[gesture_name]["trigger_angle_max"] = trigger
        # release must stay clear of the trigger or the gesture can never unlatch
        config[gesture_name]["release_angle_min"] = max(trigger + 2 * BUFFER,
                                                       DEFAULT_CONFIG["double_click"]["release_angle_min"])
    elif gesture_name == "scroll":
        if idx_mean < 60 or mid_mean < 60:
            return False
        config[gesture_name]["idx_angle_min"] = idx_mean - BUFFER
        config[gesture_name]["mid_angle_min"] = mid_mean - BUFFER
        
    if thumb_dist_history and len(thumb_dist_history) > 0:
        thumb_mean = statistics.median(thumb_dist_history)
        config[gesture_name]["thumb_index_dist_min"] = thumb_mean * THUMB_GATE_MARGIN
        
    config = sanitize_config(config)
    save_config(config)
    return True

config = load_config()
