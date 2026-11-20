"""Compare the three MPC optimizers (shooting, CEM, gradient per safe cell).

1. Open loop: the same 200 states (taken from a ten-lap run), the same models; for each
   optimizer, the cost of the plan it returns and the time it takes. A long CEM run
   (10x more samples, 3x more iterations) serves as the best-known reference.
2. Closed loop: ten laps of the online demo per optimizer and seed.

    python examples/four_zones_online.py --out results      # provides the states
    python examples/compare_planners.py --seeds 3

Writes results/planners.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from four_zones_online import build_models
from four_zones_online import main as run_demo
from scipy.ndimage import map_coordinates

from compmpc import CompatibleMPC, PlanarSafetyController
from compmpc.robots import FourZoneArena, Unicycle

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("shooting", "cem", "gradient")
SETTINGS = {"shooting": {"n_samples": 600}, "cem": {"n_samples": 400},
            "gradient": {"n_samples": 600}}


def open_loop(n_states=200, seed=0):
    rng = np.random.default_rng(seed)
    robot, arena = Unicycle(), FourZoneArena()
    grid = arena.planar_grid(robot=robot)
    Z, Y = robot.sample(10_000, rng, v_max=1.0)
    models, est = build_models(robot, grid, Z, Y)
    safety = PlanarSafetyController(grid, *models, safe=arena.safe_mask(grid)).solve()
    fields, h = arena.distance_fields()
    run = np.load(ROOT / "results" / "four_zones.npz")
    S_run, visits = run["S"], run["visits"]
    # target zone of each step of the recorded run
    targets = np.empty(len(S_run), dtype=object)
    k, order = 0, list(arena.order)
    for t in range(len(S_run)):
        targets[t] = order[k % 4]
        if k < len(visits) and t + 1 >= visits[k]:
            k += 1
    idx = [t for t in rng.permutation(len(S_run)) if safety.in_invariant(S_run[t])][:n_states]
    current = {"target": None}

    def stage_cost(S, U, i):
        coords = np.vstack([S[:, 0] / h - 0.5, S[:, 1] / h - 0.5])
        return (map_coordinates(fields[current["target"]], coords, order=1, mode="nearest")
                + 0.5 / np.maximum(U[:, 0], 1e-3))

    configs = {m: dict(method=m, **SETTINGS[m]) for m in METHODS}
    configs["reference"] = dict(method="cem", n_samples=4000, cem_iters=15)
    out = {name: {"cost": [], "feasible": [], "ms": [], "evals": []} for name in configs}
    for t in idx:
        current["target"] = targets[t]
        for name, cfg in configs.items():
            mpc = CompatibleMPC(safety, *est, stage_cost=stage_cost, horizon=5, seed=int(t),
                                **cfg)
            t0 = time.perf_counter()
            res = mpc(S_run[t])
            out[name]["ms"].append(1e3 * (time.perf_counter() - t0))
            out[name]["cost"].append(res.cost)
            out[name]["feasible"].append(res.cost < mpc.unsafe_penalty)
            out[name]["evals"].append(res.n_evaluations)
    ref = np.minimum.reduce([np.array(out[n]["cost"]) for n in configs])
    summary = {}
    for name in configs:
        c = np.array(out[name]["cost"])
        gap = 100 * (c - ref) / np.abs(ref)
        summary[name] = {
            "median_gap_to_best_pct": float(np.median(gap)),
            "mean_gap_to_best_pct": float(np.mean(gap)),
            "p90_gap_to_best_pct": float(np.percentile(gap, 90)),
            "best_or_tied_pct": float(np.mean(gap < 0.01) * 100),
            "feasible_pct": float(np.mean(out[name]["feasible"]) * 100),
            "median_ms": float(np.median(out[name]["ms"])),
            "p95_ms": float(np.percentile(out[name]["ms"], 95)),
            "plans_evaluated": float(np.mean(out[name]["evals"])),
        }
    return {"n_states": len(idx), "summary": summary}


def closed_loop(seeds, laps):
    rows = []
    for method in METHODS:
        for seed in range(seeds):
            s = run_demo(["--seed", str(seed), "--laps", str(laps), "--method", method,
                          "--out", f"/tmp/planners_{method}_{seed}"])
            rows.append({"method": method, "seed": seed, "time_s": s["time_s"],
                         "laps": s["laps"], "collisions": s["collisions"],
                         "left_invariant_set": s["left_invariant_set"],
                         "min_clearance_m": s["min_clearance_m"],
                         "mpc_step_median_ms": s["mpc_step_median_ms"],
                         "mpc_step_p95_ms": s["mpc_step_p95_ms"],
                         "lap_times_s": s["lap_times_s"]})
    summary = {}
    for method in METHODS:
        r = [x for x in rows if x["method"] == method]
        summary[method] = {
            "mean_time_10_laps_s": float(np.mean([x["time_s"] for x in r])),
            "mean_lap_time_laps_3_10_s": float(np.mean([np.mean(x["lap_times_s"][2:])
                                                        for x in r])),
            "collisions": int(sum(x["collisions"] for x in r)),
            "steps_outside_safe_set": int(sum(x["left_invariant_set"] for x in r)),
            "min_clearance_m": float(min(x["min_clearance_m"] for x in r)),
            "median_ms": float(np.median([x["mpc_step_median_ms"] for x in r])),
            "p95_ms": float(np.max([x["mpc_step_p95_ms"] for x in r])),
        }
    return {"runs": rows, "summary": summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--laps", type=int, default=10)
    p.add_argument("--states", type=int, default=200)
    args = p.parse_args()
    result = {"open_loop": open_loop(args.states), "closed_loop": closed_loop(args.seeds,
                                                                              args.laps)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "planners.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"open_loop": result["open_loop"]["summary"],
                      "closed_loop": result["closed_loop"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
