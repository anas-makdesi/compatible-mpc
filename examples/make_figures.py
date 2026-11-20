"""Figures for the README and the project page, from results/four_zones.npz.

    python examples/four_zones_online.py --out results
    python examples/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from compmpc.robots import FourZoneArena  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
INK, MUTED, ACCENT = "#0d121c", "#6b7280", "#e8590c"
ZONE_COLORS = {"red": "#e03131", "yellow": "#f2c94c", "green": "#2f9e44", "blue": "#1c7ed6"}

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlelocation": "left",
})


def draw_arena(ax, arena: FourZoneArena):
    ax.add_patch(Rectangle((0, 0), arena.size, arena.size, fill=False, lw=3, ec=INK))
    for (x0, x1), (y0, y1) in arena.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=INK))
    for name, ((x0, x1), (y0, y1)) in arena.zones.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=ZONE_COLORS[name], alpha=0.35,
                               lw=0))
    ax.set_xlim(-0.05, arena.size + 0.05)
    ax.set_ylim(-0.05, arena.size + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def speed_path(ax, S, U, vmin=0.1, vmax=2.0):
    pts = S[:, :2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", norm=plt.Normalize(vmin, vmax), lw=1.6)
    lc.set_array(U[:, 0])
    ax.add_collection(lc)
    return lc


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = np.load(RES / "four_zones.npz")
    summary = json.loads((RES / "four_zones_summary.json").read_text())
    S, U, msv = r["S"], r["U"], r["max_safe_v"]
    visits = r["visits"]
    dt = 0.2
    t = np.arange(len(U)) * dt
    arena = FourZoneArena()
    lap_ends = visits[3::4]

    # ------------------------------------------------------------- hero (wide)
    fig = plt.figure(figsize=(21, 6.9), dpi=100)
    ax1 = fig.add_axes([0.035, 0.11, 0.25, 0.78])
    draw_arena(ax1, arena)
    lc = speed_path(ax1, S, U)
    ax1.plot(*S[0, :2], "o", color=INK, ms=8)
    cax = fig.add_axes([0.29, 0.11, 0.008, 0.78])
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label("speed [m/s]")
    ax1.set_title("10 laps, 0 collisions")

    ax2 = fig.add_axes([0.38, 0.11, 0.6, 0.78])
    ax2.plot(t, U[:, 0], color=INK, lw=1.2, label="speed chosen by the MPC")
    ax2.step(t, msv, where="post", color=ACCENT, lw=2,
             label="highest certified-safe speed at the current state")
    ax2.axhline(1.0, color=MUTED, ls="--", lw=1)
    ax2.text(t[-1], 1.03, "fastest speed in the training data", ha="right", va="bottom",
             color=MUTED, fontsize=11)
    for k, e in enumerate(lap_ends):
        ax2.axvline(e * dt, color=MUTED, lw=0.6, alpha=0.6)
        start = 0 if k == 0 else lap_ends[k - 1]
        ax2.text((start + e) / 2 * dt, 2.08, f"{(e - start) * dt:.1f} s", ha="center",
                 fontsize=10, color=MUTED)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, 2.2)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("speed [m/s]")
    ax2.set_title("Trained below 1 m/s, the robot learns to use the full certified speed range"
                  " (lap times on top)")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none")
    fig.savefig(FIG / "hero.png", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------- arena only
    fig, ax = plt.subplots(figsize=(6.2, 6), dpi=130)
    draw_arena(ax, arena)
    lc = speed_path(ax, S, U)
    fig.colorbar(lc, ax=ax, label="speed [m/s]", fraction=0.046)
    ax.set_title("Ten laps red → yellow → green → blue")
    fig.tight_layout()
    fig.savefig(FIG / "laps.png", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------- turn-rate model width
    e1, e2 = r["u1_edges"], r["u2_edges"]
    w0, w1 = r["width_w_offline"], r["width_w_online"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130, sharey=True)
    vmax = np.percentile(w0, 95)
    for ax, w, title in ((axs[0], w0, "Before: low-speed data only"),
                         (axs[1], w1, "After ten laps of online learning")):
        im = ax.pcolormesh(e1, e2, w.T, cmap="magma_r", vmin=0, vmax=vmax, shading="flat")
        ax.axvline(1.0, color="white", ls="--", lw=1.2)
        ax.set_xlabel("speed v [m/s]")
        ax.set_title(title)
    axs[0].set_ylabel("steering δ [rad]")
    fig.colorbar(im, ax=axs, label="width of the guaranteed turn-rate bounds [rad/s]",
                 fraction=0.03)
    fig.savefig(FIG / "turn_model_uncertainty.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- comparison bars
    cmp_path = RES / "comparison.json"
    if cmp_path.exists():
        c = json.loads(cmp_path.read_text())
        keys = ["compatible", "baseline"]
        names = ["with certified inputs", "learned model only"]
        cols = [ACCENT, MUTED]
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), dpi=130)
        n = c["compatible"]["runs"]
        panels = [
            ([c[k]["runs_with_collision"] for k in keys], f"runs that crashed (of {n})",
             "Collisions", "{:.0f}"),
            ([c[k]["mean_time_s_runs_without_collision"] for k in keys],
             "time for 10 laps [s] (runs without a crash)", "Cost of the guarantee", "{:.1f}"),
            ([c[k]["min_clearance_m"] for k in keys],
             "closest approach to a wall [m] (< 0: inside a block)",
             "Margin kept (worst run)", "{:.2f}"),
        ]
        for i, (ax, (vals, xlabel, title, fmt)) in enumerate(zip(axs, panels)):
            ax.barh(names, vals, color=cols)
            ax.axvline(0, color=INK, lw=0.8)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            for p, v in zip(ax.patches, vals):
                ax.text(max(p.get_width(), 0), p.get_y() + p.get_height() / 2,
                        " " + fmt.format(v), va="center")
            if i == 0:
                ax.set_xlim(0, n)
                ax.set_xticks(range(n + 1))
            else:
                ax.set_yticklabels([])
        fig.tight_layout()
        fig.savefig(FIG / "comparison.png", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------- the crash
    cb, cc = RES / "crash_baseline" / "four_zones.npz", RES / "crash_compatible" / "four_zones.npz"
    if cb.exists() and cc.exists():
        fig, axs = plt.subplots(1, 2, figsize=(12, 6.2), dpi=130)
        for ax, path, title in ((axs[0], cc, "With certified inputs: safe"),
                                (axs[1], cb, "Learned model only: crash after 12.8 s")):
            d = np.load(path)
            draw_arena(ax, arena)
            lc = speed_path(ax, d["S"], d["U"])
            ax.plot(*d["S"][0, :2], "o", color=INK, ms=7)
            ax.set_title(title)
        end = np.load(cb)["S"][-1]
        axs[1].plot(end[0], end[1], "X", color="#e03131", ms=16, mec="white", mew=1.5)
        fig.colorbar(lc, ax=axs, label="speed [m/s]", fraction=0.025)
        fig.suptitle("Same planner, same learned model, same data, same disturbances (run 3 of 5)",
                     x=0.06, ha="left", fontweight="bold")
        fig.savefig(FIG / "crash.png", facecolor="white", bbox_inches="tight")
        plt.close(fig)
    print("figures written to", FIG, "| laps:", summary["laps"])


if __name__ == "__main__":
    main()
