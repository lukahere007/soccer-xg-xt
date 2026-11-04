import numpy as np
import pytest

from goalvalue import geometry as geo


def test_distance_penalty_spot():
    assert geo.distance_to_goal(108, 40) == pytest.approx(12.0)


def test_distance_offset():
    # (114, 36) -> (120, 40): sqrt(6^2 + 4^2)
    assert geo.distance_to_goal(114, 36) == pytest.approx(np.sqrt(52))


def test_angle_penalty_spot_hand_computed():
    # symmetric: 2 * atan(half-width / distance) = 2 * atan(4 / 12) = 36.87 degrees
    assert geo.goal_angle(108, 40) == pytest.approx(2 * np.arctan(4 / 12))
    assert np.degrees(geo.goal_angle(108, 40)) == pytest.approx(36.8699, abs=1e-3)


def test_angle_in_line_with_post():
    # from (114, 36) the near post is straight ahead; far post at atan2(8, 6)
    assert geo.goal_angle(114, 36) == pytest.approx(np.arctan2(8, 6))


def test_angle_limits():
    assert geo.goal_angle(120, 30) == pytest.approx(0.0)   # goal line, outside the posts
    assert geo.goal_angle(120, 40) == pytest.approx(np.pi)  # on the line between the posts
    assert geo.goal_angle(60, 40) < geo.goal_angle(100, 40)  # further away -> smaller


def test_angle_symmetric_and_vectorised():
    a = geo.goal_angle(np.array([100, 100]), np.array([30, 50]))
    assert a[0] == pytest.approx(a[1])


def test_point_in_triangle_basic():
    a, b, c = (0, 0), (4, 0), (0, 4)
    assert geo.point_in_triangle(1, 1, a, b, c)
    assert not geo.point_in_triangle(3, 3, a, b, c)
    assert geo.point_in_triangle(2, 0, a, b, c)               # on an edge
    assert not geo.point_in_triangle(2, 0, a, b, c, inclusive=False)
    assert geo.point_in_triangle(0, 0, a, b, c)               # on a vertex


def test_point_in_triangle_orientation_independent():
    a, b, c = (0, 0), (4, 0), (0, 4)
    assert geo.point_in_triangle(1, 1, a, c, b)  # clockwise winding


def test_point_in_triangle_vectorised_shot_triangle():
    shooter = (108, 40)
    px = np.array([115, 115, 100, 119.9])
    py = np.array([40, 50, 40, 43.9])
    inside = geo.point_in_triangle(px, py, shooter, geo.LEFT_POST, geo.RIGHT_POST)
    assert inside.tolist() == [True, False, False, True]


def test_point_segment_distance():
    assert geo.point_segment_distance(1, 1, 0, 0, 2, 0) == pytest.approx(1.0)   # perpendicular
    assert geo.point_segment_distance(5, 0, 0, 0, 2, 0) == pytest.approx(3.0)   # beyond endpoint
    assert geo.point_segment_distance(-3, 4, 0, 0, 2, 0) == pytest.approx(5.0)  # before start
    assert geo.point_segment_distance(1, 1, 0, 0, 0, 0) == pytest.approx(np.sqrt(2))  # degenerate


def test_freeze_frame_features_hand_example():
    frame = [
        {"location": [115, 40], "teammate": False, "position": {"name": "Center Back"}},
        {"location": [115, 50], "teammate": False, "position": {"name": "Left Back"}},
        {"location": [110, 40.5], "teammate": True, "position": {"name": "Center Forward"}},
        {"location": [119, 41], "teammate": False, "position": {"name": "Goalkeeper"}},
    ]
    f = geo.freeze_frame_features((108, 40), frame)
    assert f["n_def_triangle"] == 1
    assert f["n_att_triangle"] == 1
    assert f["gk_in_triangle"] == 1
    assert f["gk_dist_line"] == pytest.approx(1.0)
    assert f["gk_dist_shotline"] == pytest.approx(1.0)
    assert f["nearest_def_dist"] == pytest.approx(7.0)
    assert f["n_def_within_3"] == 0
    assert f["gk_missing"] == 0


def test_freeze_frame_empty():
    f = geo.freeze_frame_features((108, 40), None)
    assert np.isnan(f["n_def_triangle"])
