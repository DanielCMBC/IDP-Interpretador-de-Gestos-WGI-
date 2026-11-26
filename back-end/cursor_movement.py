import numpy as np
import calculo_distancia as util

prev_x, prev_y = 0.5, 0.5
smoothening = 5

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_palm_center(hand_landmarks):
    points = [0, 5, 9, 13, 17]
    avg_x = sum(hand_landmarks.landmark[i].x for i in points) / len(points)
    avg_y = sum(hand_landmarks.landmark[i].y for i in points) / len(points)
    return Point(avg_x, avg_y)

def is_frozen(geom_list):
    """Freeze cursor if Thumb Tip (4) is close to Middle Finger MCP (9).

    Takes geometry-space landmarks (see calculo_distancia.to_geometry_space).
    With 2D landmarks a thumb held *in front of* the palm projects onto
    landmark 9 and reads as frozen when it is not; the 3D world landmarks
    do not have that failure mode.
    """
    if len(geom_list) < 21: return False
    scale = util.get_hand_scale(geom_list)
    dist = util.get_raw_dist(geom_list[4], geom_list[9]) / scale
    return dist < 0.35

def get_target_position(palm_center):
    """Palm centre (camera frame) -> unsmoothed screen position (fractions).
    Same mapping as get_normalized_position, without its smoothing: hand mode
    smooths in control_common.CursorPump instead."""
    boundary = 0.25
    x = np.interp(palm_center.x, [boundary, 1 - boundary], [0.0, 1.0])
    y = np.interp(palm_center.y, [boundary, 1 - boundary], [0.0, 1.0])
    return float(x), float(y)

def get_normalized_position(palm_center, geom_list):
    global prev_x, prev_y

    if is_frozen(geom_list):
        return prev_x, prev_y

    if palm_center is not None:
        boundary = 0.25
        x = np.interp(palm_center.x, [boundary, 1 - boundary], [0.0, 1.0])
        y = np.interp(palm_center.y, [boundary, 1 - boundary], [0.0, 1.0])

        curr_x = prev_x + (x - prev_x) / smoothening
        curr_y = prev_y + (y - prev_y) / smoothening
        
        prev_x, prev_y = curr_x, curr_y
        return curr_x, curr_y
    return prev_x, prev_y
