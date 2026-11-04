"""Pitch drawing and the project figures (matplotlib, static PNGs for the README)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def draw_pitch(ax, half: bool = False, color: str = MUTED, lw: float = 1.0):
    """Draw a StatsBomb 120x80 pitch (y axis inverted so it matches the data's view)."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    kw = dict(fill=False, ec=color, lw=lw)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color=color, lw=lw)
    ax.add_patch(Arc((60, 40), 20, 20, theta1=0, theta2=360, ec=color, lw=lw))
    for x0, sign in ((0, 1), (120, -1)):
        ax.add_patch(Rectangle((x0 if sign > 0 else 102, 18), 18, 44, **kw))
        ax.add_patch(Rectangle((x0 if sign > 0 else 114, 30), 6, 20, **kw))
        ax.add_patch(Arc((x0 + sign * 12, 40), 20, 20,
                         theta1=-53 if sign > 0 else 127, theta2=53 if sign > 0 else 233, ec=color, lw=lw))
        ax.plot([x0, x0 - sign * 1.5], [36, 36], color=color, lw=lw)
        ax.plot([x0, x0 - sign * 1.5], [44, 44], color=color, lw=lw)
        ax.plot([x0 - sign * 1.5] * 2, [36, 44], color=color, lw=lw)
        ax.scatter([x0 + sign * 12], [40], s=4, color=color)
    ax.set_xlim(60 if half else -2, 122)
    ax.set_ylim(82, -2)
    ax.set_aspect("equal")
    return ax


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
