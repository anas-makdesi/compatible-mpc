"""Safety controller and MPC: brute-force fixed point, closed-loop guarantees."""

import itertools

import numpy as np
import pytest

from compmpc import (
    CompatibleEstimate,
    CompatibleMPC,
    GrowthMap,
    PlanarSafetyController,
)
from compmpc.robots import FourZoneArena, Unicycle


def small_setup(seed=0, n_data=5000, v_max=1.0):
    """The four-zone arena with a coarser input grid (invariance needs a fine state grid:
    the vehicle can never stop, so it must be able to circle inside the safe set)."""
    rng = np.random.default_rng(seed)
    robot, arena = Unicycle(), FourZoneArena()
    grid = arena.planar_grid(n_u=16, robot=robot)
    Z, Y = robot.sample(n_data, rng, v_max=v_max)
    (a, b), (c, d) = robot.derivative_bounds()
    dm = robot.d_max
    models = [GrowthMap.from_data(Z[:, :2], Y[:, 0], grid.velocity_grid(), a, b, -dm, dm),
              GrowthMap.from_data(Z[:, :2], Y[:, 1], grid.velocity_grid(), a, b, -dm, dm),
              GrowthMap.from_data(Z[:, 1:], Y[:, 2], grid.turn_grid(), c, d, -dm, dm)]
    return robot, grid, models, arena.safe_mask(grid), rng


def has_safe_input(ctrl, inv, i, j, k):
    """Independent check with explicit array slicing: some input maps cell (i,j,k) into inv."""
    g = ctrl.grid
    for a, b in itertools.product(range(g.n_u1), range(g.n_u2)):
        if not ctrl.valid[k, a, b]:
            continue
        x0, x1 = i + ctrl.xo[k, a, 0], i + ctrl.xo[k, a, 1]
        y0, y1 = j + ctrl.yo[k, a, 0], j + ctrl.yo[k, a, 1]
        if x0 < 0 or y0 < 0 or x1 >= g.n_x or y1 >= g.n_y:
            continue
        t0, tl = ctrl.to[k, a, b]
        ths = [(t0 + q) % g.n_theta for q in range(tl)]
        if inv[x0:x1 + 1, y0:y1 + 1][:, :, ths].all():
            return True
    return False


def test_fixed_point_is_invariant_and_maximal():
    _, grid, models, safe, rng = small_setup()
    ctrl = PlanarSafetyController(grid, *models, safe=safe).solve()
    inv = ctrl.inv
    assert 0 < inv.sum() < ctrl.safe.sum()
    assert not np.any(inv & ~ctrl.safe)
    kept = np.argwhere(inv)
    for i, j, k in kept[rng.choice(len(kept), 300, replace=False)]:
        assert has_safe_input(ctrl, inv, i, j, k)        # invariance
    removed = np.argwhere(ctrl.safe & ~inv)
    for i, j, k in removed[rng.choice(len(removed), min(150, len(removed)), replace=False)]:
        assert not has_safe_input(ctrl, inv, i, j, k)    # maximality


def test_safe_inputs_keep_true_system_in_invariant_set():
    robot, grid, models, safe, rng = small_setup(seed=1)
    ctrl = PlanarSafetyController(grid, *models, safe=safe).solve()
    cells = np.argwhere(ctrl.inv)
    hx, hy, ht = grid.hx, grid.hy, grid.htheta
    checked = 0
    for _ in range(3000):
        i, j, k = cells[rng.integers(len(cells))]
        s = np.array([(i + rng.uniform()) * hx, (j + rng.uniform()) * hy,
                      -np.pi + (k + rng.uniform()) * ht])
        mask = ctrl.safe_inputs(s)
        assert mask.any(), "every invariant state must have a safe input"
        a, b = np.argwhere(mask)[rng.integers(mask.sum())]
        lo, hi = grid.input_cell_bounds(a, b)
        u = rng.uniform(lo, hi)
        s2, _ = robot.step(s, u, grid.dt, rng)
        assert ctrl.in_invariant(s2)
        checked += 1
    assert checked == 3000


def test_refined_models_never_shrink_the_invariant_set():
    robot, grid, models, safe, rng = small_setup(seed=2, n_data=500)
    ctrl = PlanarSafetyController(grid, *models, safe=safe).solve()
    before = ctrl.inv.copy()
    Z, Y = robot.sample(5000, rng)
    for m, z, y in zip(models, (Z[:, :2], Z[:, :2], Z[:, 1:]), Y.T):
        m.update(z, y)
    ctrl.set_models(*models)
    ctrl.solve(restart=True)
    assert np.all(ctrl.inv[before])
    assert ctrl.inv.sum() >= before.sum()


@pytest.mark.parametrize("method", ["shooting", "cem", "gradient"])
def test_mpc_first_input_is_always_safe_and_closed_loop_safe(method):
    robot = Unicycle()
    arena = FourZoneArena()
    grid = arena.planar_grid(n_u=16, robot=robot)
    rng = np.random.default_rng(3)
    Z, Y = robot.sample(4000, rng, v_max=1.0)
    (a, b), (c, d) = robot.derivative_bounds()
    dm = robot.d_max
    zs = (Z[:, :2], Z[:, :2], Z[:, 1:])
    grids = (grid.velocity_grid(), grid.velocity_grid(), grid.turn_grid())
    bnds = ((a, b), (a, b), (c, d))
    models = [GrowthMap.from_data(z, y, gg, lo, hi, -dm, dm)
              for z, y, gg, (lo, hi) in zip(zs, Y.T, grids, bnds)]
    est = [CompatibleEstimate.fit(z, y, m) for z, y, m in zip(zs, Y.T, models)]
    ctrl = PlanarSafetyController(grid, *models, safe=arena.safe_mask(grid)).solve()
    goal = np.array([4.5, 3.6])

    def cost(S, U, i):
        return np.hypot(S[:, 0] - goal[0], S[:, 1] - goal[1]) + 0.3 / U[:, 0]

    mpc = CompatibleMPC(ctrl, *est, stage_cost=cost, horizon=4, n_samples=200, seed=0,
                        method=method, grad_cells=3, grad_iters=10)
    s = np.array([2.5, 0.6, np.pi / 2])
    assert ctrl.in_invariant(s)
    for _ in range(60):
        res = mpc(s)
        mask = ctrl.safe_inputs(s)
        e1, e2 = grid.u1_edges, grid.u2_edges
        i1 = min(np.searchsorted(e1, res.u[0], "right") - 1, grid.n_u1 - 1)
        i2 = min(np.searchsorted(e2, res.u[1], "right") - 1, grid.n_u2 - 1)
        assert mask[i1, i2]
        zv = np.array([[s[2], res.u[0]]])
        s, meas = robot.step(s, res.u, grid.dt, rng)
        assert ctrl.in_invariant(s) and not arena.collides(s)
        for m, e, z, yv in zip(models, est, (zv, zv, np.array([res.u])), meas):
            m.update(z, [yv])
            e.update(z[0], yv, m)
