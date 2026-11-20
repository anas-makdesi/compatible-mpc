"""Online safe learning on the four-zone course (ADHS 2024, Section 5).

A robot whose dynamics are unknown is trained on low-speed data only (v <= 1 m/s). It then
drives laps red -> yellow -> green -> blue as fast as it can. Every measurement updates the
guaranteed model and the compatible estimate; the safe set is recomputed periodically.

    python examples/four_zones_online.py            # 10 laps, figures in docs/figures
    python examples/four_zones_online.py --laps 2   # quick run

Prints a summary and saves results/four_zones.npz.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from compmpc import CompatibleEstimate, CompatibleMPC, GrowthMap, PlanarSafetyController
from compmpc.robots import FourZoneArena, Unicycle


def build_models(robot, grid, Z, Y):
    (gv_lo, gv_hi), (gw_lo, gw_hi) = robot.derivative_bounds()
    d = robot.d_max
    zv, zw = Z[:, :2], Z[:, 1:]
    mx = GrowthMap.from_data(zv, Y[:, 0], grid.velocity_grid(), gv_lo, gv_hi, -d, d)
    my = GrowthMap.from_data(zv, Y[:, 1], grid.velocity_grid(), gv_lo, gv_hi, -d, d)
    mw = GrowthMap.from_data(zw, Y[:, 2], grid.turn_grid(), gw_lo, gw_hi, -d, d)
    ex = CompatibleEstimate.fit(zv, Y[:, 0], mx)
    ey = CompatibleEstimate.fit(zv, Y[:, 1], my)
    ew = CompatibleEstimate.fit(zw, Y[:, 2], mw)
    return [mx, my, mw], [ex, ey, ew]


def model_error(robot, grid, est, v_range, rng, n=20000):
    """RMS error of the learned velocities on random inputs with speed in v_range."""
    th = rng.uniform(-np.pi, np.pi, n)
    v = rng.uniform(*v_range, n)
    de = rng.uniform(*grid.u2_lim, n)
    true = robot.velocities(th, v, de)
    pred = np.c_[est[0](np.c_[th, v]), est[1](np.c_[th, v]), est[2](np.c_[v, de])]
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--laps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-offline", type=int, default=10_000)
    p.add_argument("--v-train", type=float, default=1.0, help="max speed in the offline data")
    p.add_argument("--resolve-every", type=int, default=25, help="steps between safe-set updates")
    p.add_argument("--speed-weight", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--d-max", type=float, default=0.05, help="disturbance bound")
    p.add_argument("--method", default="cem", choices=["cem", "shooting", "gradient"],
                   help="MPC optimizer")
    p.add_argument("--no-safety", action="store_true",
                   help="baseline: plan with the learned model only, no certified inputs")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    robot, arena = Unicycle(d_max=args.d_max), FourZoneArena()
    grid = arena.planar_grid(robot=robot)

    # ---------------------------------------------------------------- offline phase
    t0 = time.perf_counter()
    Z, Y = robot.sample(args.n_offline, rng, v_max=args.v_train)
    models, est = build_models(robot, grid, Z, Y)
    t_models = time.perf_counter() - t0
    t0 = time.perf_counter()
    safety = PlanarSafetyController(grid, *models, safe=arena.safe_mask(grid)).solve()
    t_safe_first = time.perf_counter() - t0          # includes numba compilation
    t0 = time.perf_counter()
    safety.solve(restart=True)
    t_safe = time.perf_counter() - t0
    w_offline = [m.widths().copy() for m in models]
    est_offline = [e.values.copy() for e in est]
    err_offline_fast = model_error(robot, grid, est, (1.0, 2.0), np.random.default_rng(1))

    fields, h = arena.distance_fields()
    targets = list(arena.order)

    current = {"target": targets[0]}

    def stage_cost(S, U, i):
        f = fields[current["target"]]
        coords = np.vstack([S[:, 0] / h - 0.5, S[:, 1] / h - 0.5])
        dist = map_coordinates(f, coords, order=1, mode="nearest")
        return dist + args.speed_weight / np.maximum(U[:, 0], 1e-3)

    mpc = CompatibleMPC(safety, *est, stage_cost=stage_cost, horizon=5, method=args.method,
                        seed=args.seed,
                        enforce_safety=not args.no_safety)

    # ---------------------------------------------------------------- online phase
    s = np.array([2.5, 0.6, np.pi])
    assert safety.in_invariant(s), "start state must be certified safe"
    k_target, laps, visits = 0, 0, []
    current["target"] = targets[k_target]
    log = {k: [] for k in ("s", "u", "max_safe_v", "n_safe", "in_inv", "collide",
                           "t_mpc", "t_update", "n_eval")}
    resolve_times, inv_frac, width_fast = [], [], []
    for step in range(args.max_steps):
        t0 = time.perf_counter()
        res = mpc(s)
        t_mpc = time.perf_counter() - t0
        s_next, meas = robot.step(s, res.u, grid.dt, rng)
        # learn from the new transition: guaranteed models first, then the estimates
        t0 = time.perf_counter()
        zv, zw = np.array([[s[2], res.u[0]]]), np.array([[res.u[0], res.u[1]]])
        for m, e, z, yv in zip(models, est, (zv, zv, zw), meas):
            m.update(z, [yv])
            e.update(z[0], yv, m)
        t_update = time.perf_counter() - t0
        for k, v in (("s", s), ("u", res.u), ("max_safe_v", res.max_safe_u1),
                     ("n_safe", res.n_safe_cells), ("in_inv", safety.in_invariant(s_next)),
                     ("collide", arena.collides(s_next)), ("t_mpc", t_mpc),
                     ("t_update", t_update), ("n_eval", res.n_evaluations)):
            log[k].append(v)
        s = s_next
        if args.no_safety and arena.collides(s):
            print(f"collision at t = {(step + 1) * grid.dt:.1f} s, position {s[:2].round(2)}")
            break
        if (step + 1) % args.resolve_every == 0:
            t0 = time.perf_counter()
            safety.set_models(*models)
            safety.solve(restart=True)
            resolve_times.append(time.perf_counter() - t0)
            inv_frac.append((step + 1, safety.invariant_fraction()))
            width_fast.append((step + 1, float(np.mean(models[2].widths()[20:]))))
        if arena.in_zone(s, targets[k_target]):
            visits.append((step + 1, targets[k_target]))
            k_target = (k_target + 1) % 4
            current["target"] = targets[k_target]
            if k_target == 0:
                laps += 1
                print(f"lap {laps} done at t = {(step + 1) * grid.dt:.1f} s")
                if laps >= args.laps:
                    break
    log["s"].append(s)

    S = np.array(log["s"])
    U = np.array(log["u"])
    err_online_fast = model_error(robot, grid, est, (1.0, 2.0), np.random.default_rng(1))
    summary = {
        "laps": laps,
        "steps": len(U),
        "time_s": len(U) * grid.dt,
        "collisions": int(np.sum(log["collide"])),
        "left_invariant_set": int(len(U) - np.sum(log["in_inv"])),
        "max_speed_used": float(U[:, 0].max()),
        "min_clearance_m": float(min(arena.clearance(x) for x in S)),
        "offline_models_s": t_models,
        "safe_set_first_s_incl_compile": t_safe_first,
        "safe_set_s": t_safe,
        "safe_set_resolve_median_s": float(np.median(resolve_times)) if resolve_times else None,
        "method": args.method,
        "mpc_step_median_ms": 1e3 * float(np.median(log["t_mpc"])),
        "mpc_step_p95_ms": 1e3 * float(np.percentile(log["t_mpc"], 95)),
        "plans_evaluated_per_step": float(np.mean(log["n_eval"])),
        "learning_update_median_ms": 1e3 * float(np.median(log["t_update"])),
        "invariant_fraction_start": float(inv_frac[0][1]) if inv_frac else None,
        "invariant_fraction_end": safety.invariant_fraction(),
        "rmse_fast_offline": err_offline_fast.tolist(),
        "rmse_fast_online": err_online_fast.tolist(),
        "turn_model_width_fast_offline": float(np.mean(w_offline[2][20:])),
        "turn_model_width_fast_end": float(np.mean(models[2].widths()[20:])),
    }
    # lap times
    lap_ends = [t for t, name in visits if name == "blue"]
    starts = [0] + lap_ends[:-1]
    summary["lap_times_s"] = [round((b - a) * grid.dt, 1) for a, b in zip(starts, lap_ends)]
    summary["mean_speed_per_lap"] = [round(float(U[a:b, 0].mean()), 2)
                                     for a, b in zip(starts, lap_ends)]
    summary["min_certified_top_speed_per_lap"] = [
        round(float(np.min(log["max_safe_v"][a:b])), 2) for a, b in zip(starts, lap_ends)]
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "four_zones.npz", S=S, U=U, max_safe_v=np.array(log["max_safe_v"]),
             n_safe=np.array(log["n_safe"]), inv_frac=np.array(inv_frac),
             width_fast=np.array(width_fast), visits=np.array([v[0] for v in visits]),
             est_w_offline=est_offline[2], est_w_online=est[2].values,
             width_w_offline=w_offline[2], width_w_online=models[2].widths(),
             u1_edges=grid.u1_edges, u2_edges=grid.u2_edges)
    (out / "four_zones_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
