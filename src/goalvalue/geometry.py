"""Pitch geometry in StatsBomb coordinates.

StatsBomb uses a 120 x 80 yard pitch with the origin at the top-left corner.
Event locations are always oriented so the acting team attacks towards
``x = 120``. The goal mouth is the segment from ``(120, 36)`` to ``(120, 44)``
(8 yards wide).

All functions are vectorised: they accept scalars or numpy arrays and
broadcast in the usual way.
"""
from __future__ import annotations

import numpy as np

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
LEFT_POST = np.array([120.0, 36.0])
RIGHT_POST = np.array([120.0, 44.0])
GOAL_CENTRE = np.array([120.0, 40.0])


def distance_to_goal(x, y):
    """Euclidean distance (yards) from ``(x, y)`` to the centre of the goal mouth."""
    return np.hypot(GOAL_X - np.asarray(x, float), GOAL_CENTRE[1] - np.asarray(y, float))


def goal_angle(x, y):
    """Angle (radians) subtended by the two goal posts as seen from ``(x, y)``.

    This is the "visible goal mouth": the angle between the rays shooter->left post
    and shooter->right post, in ``[0, pi]``. From a spot on the goal line
    outside the posts it is 0; standing between the posts on the line it is pi.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    a1 = np.arctan2(LEFT_POST[1] - y, LEFT_POST[0] - x)
    a2 = np.arctan2(RIGHT_POST[1] - y, RIGHT_POST[0] - x)
    ang = np.abs(a2 - a1)
    return np.where(ang > np.pi, 2 * np.pi - ang, ang)


def _cross(ox, oy, ax, ay, bx, by):
    """z-component of (a - o) x (b - o)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def point_in_triangle(px, py, a, b, c, *, inclusive: bool = True):
    """Is point ``(px, py)`` inside triangle ``abc``?

    Uses the sign-of-cross-product test: the point is inside when it lies on the
    same side of all three edges. ``a``, ``b``, ``c`` are ``(x, y)`` pairs (each
    coordinate may itself be an array to test many triangles at once). Points on
    an edge count as inside when ``inclusive`` is True. Degenerate (zero-area)
    triangles contain only points on their segment when inclusive.
    """
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    ax, ay = (np.asarray(v, float) for v in a)
    bx, by = (np.asarray(v, float) for v in b)
    cx, cy = (np.asarray(v, float) for v in c)
    d1 = _cross(ax, ay, bx, by, px, py)
    d2 = _cross(bx, by, cx, cy, px, py)
    d3 = _cross(cx, cy, ax, ay, px, py)
    if inclusive:
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)
    return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))


def point_segment_distance(px, py, ax, ay, bx, by):
    """Shortest distance from point ``p`` to the line segment ``a``-``b``.

    Projects ``p`` onto the segment, clamping the projection parameter to
    ``[0, 1]`` so points beyond an endpoint measure to that endpoint.
    """
    px, py, ax, ay, bx, by = (np.asarray(v, float) for v in (px, py, ax, ay, bx, by))
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(len2 > 0, ((px - ax) * dx + (py - ay) * dy) / len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def freeze_frame_features(shot_xy, frame) -> dict:
    """Defensive-geometry features from a StatsBomb shot freeze frame.

    Parameters
    ----------
    shot_xy : (x, y) location of the shooter.
    frame : list of dicts with keys ``location`` ([x, y]), ``teammate`` (bool) and
        ``position`` ({"name": ...}) as in StatsBomb's ``shot.freeze_frame``. The
        shooter is not part of the frame.

    Returns
    -------
    dict with
        ``n_def_triangle``     outfield opponents inside the shooter-post-post triangle
        ``n_att_triangle``     teammates inside that triangle
        ``gk_in_triangle``     1 if the opposing goalkeeper is inside it
        ``gk_dist_line``       goalkeeper's distance off the goal line (yards)
        ``gk_dist_shotline``   goalkeeper's distance to the segment shooter->goal centre
        ``nearest_def_dist``   distance from shooter to the closest outfield opponent
        ``n_def_within_3``     outfield opponents within 3 yards of the shooter
        ``gk_missing``         1 if no goalkeeper is visible in the frame
    Missing values (no frame / no GK / no defenders) are NaN.
    """
    sx, sy = float(shot_xy[0]), float(shot_xy[1])
    out = {
        "n_def_triangle": np.nan,
        "n_att_triangle": np.nan,
        "gk_in_triangle": np.nan,
        "gk_dist_line": np.nan,
        "gk_dist_shotline": np.nan,
        "nearest_def_dist": np.nan,
        "n_def_within_3": np.nan,
        "gk_missing": np.nan,
    }
    if not frame:
        return out
    loc = np.array([p["location"][:2] for p in frame], float)
    mate = np.array([bool(p.get("teammate")) for p in frame])
    is_gk = np.array([(p.get("position") or {}).get("name") == "Goalkeeper" for p in frame])
    opp_gk = (~mate) & is_gk
    defenders = (~mate) & (~is_gk)
    inside = point_in_triangle(loc[:, 0], loc[:, 1], (sx, sy), LEFT_POST, RIGHT_POST)
    out["n_def_triangle"] = float(np.sum(inside & defenders))
    out["n_att_triangle"] = float(np.sum(inside & mate))
    if defenders.any():
        d = np.hypot(loc[defenders, 0] - sx, loc[defenders, 1] - sy)
        out["nearest_def_dist"] = float(d.min())
        out["n_def_within_3"] = float(np.sum(d <= 3.0))
    else:
        out["n_def_within_3"] = 0.0
    if opp_gk.any():
        gx, gy = loc[np.argmax(opp_gk)]
        out["gk_missing"] = 0.0
        out["gk_in_triangle"] = float(inside[np.argmax(opp_gk)])
        out["gk_dist_line"] = float(GOAL_X - gx)
        out["gk_dist_shotline"] = float(
            point_segment_distance(gx, gy, sx, sy, GOAL_CENTRE[0], GOAL_CENTRE[1])
        )
    else:
        out["gk_missing"] = 1.0
    return out
