import numpy as np

def get_angle(a, b, c):
    """Unsigned angle ABC in degrees, in [0, 180]. 180 = straight, 0 = fully folded.

    Accepts 2D or 3D points. Prefer MediaPipe *world* landmarks: an angle taken
    from the raw normalised 2D landmarks changes with how the hand is tilted
    towards the camera and with the camera's aspect ratio, because MediaPipe
    normalises x by frame width and y by frame height.
    """
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def get_hand_scale(landmark_list):
    """Returns the reference size of the hand (wrist to middle finger MCP)."""
    if len(landmark_list) < 10:
        return 1.0
    scale = get_raw_dist(landmark_list[0], landmark_list[9])
    return scale if scale > 0 else 1.0

def get_distance(landmark_list, idx1=4, idx2=5):
    """Returns distance between two landmarks, normalized by hand size."""
    if len(landmark_list) <= max(idx1, idx2):
        return 0
    p1, p2 = landmark_list[idx1], landmark_list[idx2]
    scale = get_hand_scale(landmark_list)
    L = get_raw_dist(p1, p2) / scale
    return float(np.interp(L, [0, 2], [0, 1000]))

def get_raw_dist(p1, p2):
    """Returns Euclidean distance between two points (2D or 3D)."""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))

def to_geometry_space(landmark_list, world_landmark_list=None, aspect=1.0):
    """Picks the best available coordinate space for hand *geometry*.

    Prefers MediaPipe's metric 3D world landmarks (tilt- and aspect-invariant).
    Falls back to the 2D normalised landmarks with x rescaled by the frame
    aspect ratio, which at least removes the aspect distortion.

    NOTE: world landmarks are centred on the hand, so they carry no information
    about where the hand sits in the frame. Anything positional (cursor mapping,
    scroll displacement) must keep using the raw normalised landmarks.
    """
    if world_landmark_list and len(world_landmark_list) >= 21:
        return world_landmark_list
    return [(x * aspect, y) for (x, y) in landmark_list]
