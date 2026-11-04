"""Expected threat (xT), following Karun Singh (2018).

The pitch is cut into an ``nx`` x ``ny`` grid. With the ball in zone ``z`` a team
either shoots (probability ``s_z``) or moves the ball with a pass/carry
(``m_z = 1 - s_z``). A shot scores with probability ``g_z``. A move ends in zone
``z'`` with probability ``T[z, z']``; failed moves (lost ball) are the missing
mass in each row of ``T`` and are worth 0. The value of possessing the ball in
``z`` is then the fixed point of

    xT_z = s_z * g_z + m_z * sum_z' T[z, z'] * xT_z'

i.e. ``(I - diag(m) T) xT = s * g``. We solve it two ways (value iteration and a
direct linear solve) and check they agree. A successful action is worth
``xT(end) - xT(start)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import PITCH_LENGTH, PITCH_WIDTH


def zone_index(x, y, nx: int = 16, ny: int = 12) -> np.ndarray:
    """Flat zone id ``ix * ny + iy`` for pitch coordinates (clipped onto the pitch)."""
    ix = np.clip((np.asarray(x, float) / PITCH_LENGTH * nx).astype(int), 0, nx - 1)
    iy = np.clip((np.asarray(y, float) / PITCH_WIDTH * ny).astype(int), 0, ny - 1)
    return ix * ny + iy


@dataclass
class XTModel:
    """Fitted xT components and the solved surface (flat arrays of length nx*ny)."""
    nx: int
    ny: int
    shoot: np.ndarray
    move: np.ndarray
    goal: np.ndarray
    transition: np.ndarray
    xt: np.ndarray
    n_iter: int

    def surface(self) -> np.ndarray:
        """xT as an (ny, nx) array, row = y band, column = x band (for plotting)."""
        return self.xt.reshape(self.nx, self.ny).T

    def value(self, x, y, end_x, end_y) -> np.ndarray:
        """xT added by moving the ball from (x, y) to (end_x, end_y)."""
        a = zone_index(x, y, self.nx, self.ny)
        b = zone_index(end_x, end_y, self.nx, self.ny)
        return self.xt[b] - self.xt[a]


def solve_iterative(shoot, move, goal, T, tol: float = 1e-10, max_iter: int = 10_000):
    """Value iteration from xT = 0 until the largest change is below ``tol``."""
    shoot, move, goal, T = map(np.asarray, (shoot, move, goal, T))
    reward = shoot * goal
    xt = np.zeros_like(reward, dtype=float)
    for it in range(1, max_iter + 1):
        new = reward + move * (T @ xt)
        if np.max(np.abs(new - xt)) < tol:
            return new, it
        xt = new
    raise RuntimeError("xT value iteration did not converge")


def solve_linear(shoot, move, goal, T) -> np.ndarray:
    """Direct solve of ``(I - diag(move) T) xT = shoot * goal``."""
    shoot, move, goal, T = map(np.asarray, (shoot, move, goal, T))
    A = np.eye(len(shoot)) - move[:, None] * T
    return np.linalg.solve(A, shoot * goal)


def estimate_components(moves: pd.DataFrame, shots: pd.DataFrame, nx: int = 16, ny: int = 12,
                        shot_value: str | None = None):
    """Count-based estimates of shoot/move probabilities, goal probability and T.

    moves : columns x, y, end_x, end_y, success (passes and carries).
    shots : columns x, y and either ``goal`` or the column named by ``shot_value``
        (e.g. a model's out-of-fold xG, which gives a smoother goal-probability
        surface than raw goal counts in sparsely shot-at zones).
    """
    n = nx * ny
    zm = zone_index(moves["x"], moves["y"], nx, ny)
    zs = zone_index(shots["x"], shots["y"], nx, ny)
    n_move = np.bincount(zm, minlength=n).astype(float)
    n_shot = np.bincount(zs, minlength=n).astype(float)
    val = shots[shot_value or "goal"].to_numpy(float)
    goal_sum = np.bincount(zs, weights=val, minlength=n)
    total = n_move + n_shot
    with np.errstate(invalid="ignore", divide="ignore"):
        shoot = np.where(total > 0, n_shot / total, 0.0)
        move = np.where(total > 0, n_move / total, 0.0)
        goal = np.where(n_shot > 0, goal_sum / n_shot, 0.0)
    ok = moves["success"].to_numpy(bool)
    ze = zone_index(moves["end_x"], moves["end_y"], nx, ny)
    T = np.zeros((n, n))
    np.add.at(T, (zm[ok], ze[ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(n_move[:, None] > 0, T / n_move[:, None], 0.0)
    return shoot, move, goal, T


def fit_xt(moves: pd.DataFrame, shots: pd.DataFrame, nx: int = 16, ny: int = 12,
           shot_value: str | None = None, tol: float = 1e-10) -> XTModel:
    """Estimate the components and solve for the xT surface (checks both solvers agree)."""
    shoot, move, goal, T = estimate_components(moves, shots, nx, ny, shot_value)
    xt_it, n_iter = solve_iterative(shoot, move, goal, T, tol=tol)
    xt_lin = solve_linear(shoot, move, goal, T)
    if not np.allclose(xt_it, xt_lin, atol=1e-8):
        raise AssertionError("iterative and linear xT solutions disagree")
    return XTModel(nx, ny, shoot, move, goal, T, xt_lin, n_iter)


def player_xt_per90(actions: pd.DataFrame, minutes: pd.DataFrame, min_minutes: float = 270.0):
    """Sum xT added by successful actions per player and scale per 90 minutes played.

    ``actions`` needs player_id, success, xt_added and action; ``minutes`` needs
    player_id, player, minutes. Players under ``min_minutes`` are dropped.
    """
    ok = actions[actions["success"]]
    agg = ok.pivot_table(index="player_id", columns="action", values="xt_added",
                         aggfunc="sum", fill_value=0.0, observed=True)
    agg.columns = [f"xt_{c.lower()}" for c in agg.columns]
    agg["xt_total"] = agg.sum(axis=1)
    mins = minutes.groupby("player_id").agg(player=("player", "first"), minutes=("minutes", "sum"),
                                            matches=("match_id", "nunique"))
    out = mins.join(agg, how="inner")
    out = out[out["minutes"] >= min_minutes].copy()
    for c in [c for c in out.columns if c.startswith("xt_")]:
        out[c + "_p90"] = out[c] / out["minutes"] * 90
    return out.sort_values("xt_total_p90", ascending=False)
