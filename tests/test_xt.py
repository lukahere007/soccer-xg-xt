import numpy as np
import pandas as pd
import pytest

from goalvalue import xt

# Two-zone toy chain, solved by hand:
#   v0 = 0.2*0.1 + 0.8*(0.25 v0 + 0.5 v1)   ->  0.8 v0 - 0.4 v1 = 0.02
#   v1 = 0.5*0.4 + 0.5*(0.4  v0 + 0.4 v1)   -> -0.2 v0 + 0.8 v1 = 0.20
#   => v1 = 0.205 / 0.7 = 41/140,  v0 = 0.025 + v1/2 = 6/35
SHOOT = np.array([0.2, 0.5])
MOVE = np.array([0.8, 0.5])
GOAL = np.array([0.1, 0.4])
T = np.array([[0.25, 0.5], [0.4, 0.4]])  # rows sum < 1: the rest is lost possession
EXPECTED = np.array([6 / 35, 41 / 140])


def test_linear_solve_matches_closed_form():
    assert xt.solve_linear(SHOOT, MOVE, GOAL, T) == pytest.approx(EXPECTED)


def test_iterative_solve_matches_closed_form_and_linear():
    v, n_iter = xt.solve_iterative(SHOOT, MOVE, GOAL, T, tol=1e-13)
    assert v == pytest.approx(EXPECTED, abs=1e-11)
    assert n_iter > 1


def test_zone_index_corners_and_clipping():
    assert xt.zone_index(0, 0) == 0
    assert xt.zone_index(119.99, 79.99) == 16 * 12 - 1
    assert xt.zone_index(130, -5) == xt.zone_index(119.9, 0)


def test_estimate_components_counts():
    # 1x2 grid in x: zone 0 = x<60, zone 1 = x>=60 (ny=1)
    moves = pd.DataFrame({
        "x": [10, 10, 10, 70], "y": [40] * 4,
        "end_x": [70, 20, 70, 10], "end_y": [40] * 4,
        "success": [True, True, False, True],
    })
    shots = pd.DataFrame({"x": [100, 100, 30], "y": [40, 40, 40], "goal": [1, 0, 0]})
    s, m, g, T_hat = xt.estimate_components(moves, shots, nx=2, ny=1)
    assert s == pytest.approx([1 / 4, 2 / 3])
    assert m == pytest.approx([3 / 4, 1 / 3])
    assert g == pytest.approx([0.0, 0.5])
    assert T_hat == pytest.approx(np.array([[1 / 3, 1 / 3], [1.0, 0.0]]))


def test_fit_xt_value_of_action():
    # zone 0 loses the ball half the time, so being in zone 1 (where shots happen) is worth more
    moves = pd.DataFrame({"x": [10, 10, 70], "y": [40, 40, 40], "end_x": [70, 70, 10],
                          "end_y": [40, 40, 40], "success": [True, False, True]})
    shots = pd.DataFrame({"x": [100, 100], "y": [40, 40], "goal": [1, 0]})
    model = xt.fit_xt(moves, shots, nx=2, ny=1)
    v = model.value(10, 40, 70, 40)
    assert v == pytest.approx(model.xt[1] - model.xt[0])
    assert model.xt[1] > model.xt[0]
