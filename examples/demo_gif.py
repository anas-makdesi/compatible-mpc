"""Short demo for the README and the website: the same run with and without certified inputs.

Same planner, same learned model, same data, same disturbances (seed 2, run 3 of 5 in
results/comparison.json). Needs the two runs first:

    python examples/four_zones_online.py --seed 2 --laps 2 --out /tmp/cm_comp
    python examples/four_zones_online.py --seed 2 --laps 2 --out /tmp/cm_base --no-safety
    python examples/demo_gif.py /tmp/cm_comp /tmp/cm_base   (writes docs/figures/demo.gif)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from compmpc.robots import FourZoneArena  # noqa: E402

INK, TEAL, ORANGE, MUTED, RED = "#0B1220", "#2a9d8f", "#FB923C", "#6B7280", "#C2410C"
PAPER = "#F7F7F5"
ZONE_COLORS = {"red": "#e03131", "yellow": "#f2c94c", "green": "#2f9e44", "blue": "#1c7ed6"}
DT = 0.2            # control period of the example [s]
SHOW_S = 16.0       # seconds shown
FRAME_MS = 100      # 2x speed
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def draw_arena(ax, arena):
    ax.add_patch(Rectangle((0, 0), arena.size, arena.size, fill=False, lw=2.5, ec=INK))
    for (x0, x1), (y0, y1) in arena.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=INK))
    for name, ((x0, x1), (y0, y1)) in arena.zones.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=ZONE_COLORS[name], alpha=0.3,
                               lw=0))
    ax.set(xlim=(-0.1, arena.size + 0.1), ylim=(-0.1, arena.size + 0.1), aspect="equal")
    ax.axis("off")


def robot(ax, x, y, th, color):
    pts = np.array([[0.22, 0], [-0.14, 0.12], [-0.14, -0.12]])
    c, s = np.cos(th), np.sin(th)
    pts = pts @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(pts, closed=True, color=color, zorder=5))


def main(comp_dir, base_dir):
    arena = FourZoneArena()
    runs = [(np.load(Path(comp_dir) / "four_zones.npz"), "With certified inputs", TEAL),
            (np.load(Path(base_dir) / "four_zones.npz"), "Learned model only", ORANGE)]
    n_frames = int(SHOW_S / DT) + 1

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=100)
    fig.patch.set_facecolor(PAPER)
    frames = []
    for k in range(n_frames):
        t = k * DT
        for ax, (d, title, color) in zip(axs, runs):
            ax.clear()
            draw_arena(ax, arena)
            S = d["S"]
            last = min(k, len(S) - 1)
            ax.plot(S[: last + 1, 0], S[: last + 1, 1], color=color, lw=1.6, alpha=0.8)
            crashed = k >= len(S) - 1 and len(S) - 1 < n_frames - 1
            robot(ax, *S[last], RED if crashed else color)
            if crashed:
                ax.plot(S[last, 0], S[last, 1], "X", color=RED, ms=16, mec="white", mew=1.5,
                        zorder=6)
                status, scol = f"crashed at {(len(S) - 1) * DT:.1f} s", RED
            else:
                status, scol = f"speed {d['U'][min(last, len(d['U']) - 1), 0]:.1f} m/s", INK
            ax.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=11)
            ax.text(arena.size, -0.35, status, ha="right", va="top", color=scol, fontsize=9.5)
        fig.suptitle(f"Same planner, learned model, data and disturbances   ·   t = {t:4.1f} s "
                     "(2x speed)", x=0.02, ha="left", fontsize=9.5, color=MUTED)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.08)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    durations = [FRAME_MS] * len(frames)
    durations[-1] = 3000
    images = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
              for f in frames]
    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"{len(frames)} frames; saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
