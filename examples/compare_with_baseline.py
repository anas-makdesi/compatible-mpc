"""Same planner and learned model, with and without the certified input set, over seeds.

    python examples/compare_with_baseline.py --seeds 5

Writes results/comparison.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from four_zones_online import main as run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--laps", type=int, default=10)
    args = p.parse_args()
    rows = []
    for seed in range(args.seeds):
        for mode in ("compatible", "baseline"):
            extra = ["--no-safety"] if mode == "baseline" else []
            s = run(["--seed", str(seed), "--laps", str(args.laps), "--out", f"/tmp/cmp_{mode}"]
                    + extra)
            rows.append({"seed": seed, "mode": mode, "time_s": s["time_s"], "laps": s["laps"],
                         "collisions": s["collisions"], "min_clearance_m": s["min_clearance_m"],
                         "left_invariant_set": s["left_invariant_set"]})
    out = {"runs": rows}
    for mode in ("compatible", "baseline"):
        r = [x for x in rows if x["mode"] == mode]
        done = [x for x in r if x["collisions"] == 0]
        out[mode] = {
            "runs": len(r),
            "runs_with_collision": len(r) - len(done),
            "mean_time_s_runs_without_collision": float(np.mean([x["time_s"] for x in done]))
            if done else None,
            "min_clearance_m": float(min(x["min_clearance_m"] for x in r)),
            "mean_min_clearance_m": float(np.mean([x["min_clearance_m"] for x in r])),
        }
    res = Path(__file__).resolve().parent.parent / "results"
    res.mkdir(exist_ok=True)
    (res / "comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
