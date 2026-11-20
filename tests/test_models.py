"""Guaranteed models and compatible estimates: soundness, minimal growth, compatibility."""

import numpy as np

from compmpc import (
    CompatibleEstimate,
    GrowthMap,
    RectangularPartition,
    interpolate,
    intersect,
    monotone_split_model,
    vertex_bounds,
)
from compmpc.grid import locate


def random_system(rng, n):
    """A smooth function with known derivative bounds on [-1, 1]^n."""
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(0.5, 2.0, n)
    c = rng.uniform(-1, 1, n)

    def f(Z):
        return (np.sin(b * Z) * a).sum(axis=1) + Z @ c

    g_lo = c - np.abs(a) * b
    g_hi = c + np.abs(a) * b
    return f, g_lo, g_hi


def test_growth_models_of_the_robot_are_sound():
    """The guaranteed models used by the demo (ddma.GrowthMap) contain the true velocities."""
    from compmpc.robots import FourZoneArena, Unicycle

    rng = np.random.default_rng(0)
    robot = Unicycle()
    grid = FourZoneArena().planar_grid(robot=robot)
    Z, Y = robot.sample(5000, rng, v_max=1.0)
    (a, b), (c, d) = robot.derivative_bounds()
    dm = robot.d_max
    T = rng.uniform([-np.pi, 0.1, -np.pi / 4], [np.pi, 2.0, np.pi / 4], (50000, 3))
    true = robot.velocities(T[:, 0], T[:, 1], T[:, 2])
    for k, (cols, part, lo_b, hi_b) in enumerate(
            [((0, 1), grid.velocity_grid(), a, b), ((0, 1), grid.velocity_grid(), a, b),
             ((1, 2), grid.turn_grid(), c, d)]):
        M = GrowthMap.from_data(Z[:, cols], Y[:, k], part, lo_b, hi_b, -dm, dm)
        lo, hi = M.bounds(T[:, cols])
        assert np.all(true[:, k] - dm >= lo - 1e-9) and np.all(true[:, k] + dm <= hi + 1e-9)


def test_monotone_split_model_is_sound_and_intersects():

    rng = np.random.default_rng(2)
    f, g_lo, g_hi = random_system(rng, 2)
    grid = RectangularPartition.uniform([-1, -1], [1, 1], [10, 10])
    Z = rng.uniform(-1, 1, (2000, 2))
    d = 0.02
    y = f(Z) + rng.uniform(-d, d, 2000)
    M = monotone_split_model(Z, y, grid, g_lo, g_hi, -d, d)
    T = rng.uniform(-1, 1, (20000, 2))
    lo, hi = M.bounds(T)
    assert np.all(f(T) - d >= lo - 1e-9) and np.all(f(T) + d <= hi + 1e-9)
    G = GrowthMap.from_data(Z, y, grid, g_lo, g_hi, -d, d)
    both = intersect(M, G)
    assert np.all(both.upper <= np.minimum(M.upper, G.upper))
    lo, hi = both.bounds(T)
    assert np.all(f(T) - d >= lo - 1e-9) and np.all(f(T) + d <= hi + 1e-9)


def test_multilinear_interpolation_is_exact_for_multiaffine_functions():
    rng = np.random.default_rng(3)
    grid = RectangularPartition([np.sort(rng.uniform(0, 1, 6)), np.sort(rng.uniform(0, 1, 5))])
    c = rng.normal(size=4)

    def g(Z):
        return c[0] + c[1] * Z[:, 0] + c[2] * Z[:, 1] + c[3] * Z[:, 0] * Z[:, 1]

    V = np.stack(np.meshgrid(*grid.edges, indexing="ij"), -1).reshape(-1, 2)
    vals = g(V).reshape(6, 5)
    lo = [e[0] for e in grid.edges]
    hi = [e[-1] for e in grid.edges]
    T = rng.uniform(lo, hi, (1000, 2))
    np.testing.assert_allclose(interpolate(grid, vals, T), g(T), atol=1e-12)


def _check_compatible(E, M, rng, n_pts=20000):
    lo = [e[0] for e in M.partition.edges]
    hi = [e[-1] for e in M.partition.edges]
    T = rng.uniform(lo, hi, (n_pts, len(lo)))
    f_lo, f_hi = M.f_bounds()
    idx = tuple(locate(M.partition, T).T)
    v = E(T)
    assert np.all(v >= f_lo[idx] - 1e-9) and np.all(v <= f_hi[idx] + 1e-9)


def test_offline_fit_is_compatible_and_accurate():
    rng = np.random.default_rng(4)
    f, g_lo, g_hi = random_system(rng, 2)
    grid = RectangularPartition.uniform([-1, -1], [1, 1], [20, 20])
    Z = rng.uniform(-1, 1, (5000, 2))
    d = 0.05
    y = f(Z) + rng.uniform(-d, d, 5000)
    M = GrowthMap.from_data(Z, y, grid, g_lo, g_hi, -d, d)
    E = CompatibleEstimate.fit(Z, y, M)
    assert E.compatibility_margin(M) >= -1e-12
    _check_compatible(E, M, rng)
    T = rng.uniform(-1, 1, (5000, 2))
    assert np.sqrt(np.mean((E(T) - f(T)) ** 2)) < 0.05


def test_online_updates_stay_compatible_and_learn():
    rng = np.random.default_rng(5)
    f, g_lo, g_hi = random_system(rng, 2)
    grid = RectangularPartition.uniform([-1, -1], [1, 1], [15, 15])
    d = 0.02
    M = GrowthMap(grid, g_lo, g_hi, -d, d)
    Z0 = rng.uniform(-1, 1, (50, 2))
    M.update(Z0, f(Z0) + rng.uniform(-d, d, 50))
    E = CompatibleEstimate.midpoint(M)
    T = rng.uniform(-1, 1, (5000, 2))
    err0 = np.sqrt(np.mean((E(T) - f(T)) ** 2))
    for _ in range(3000):
        z = rng.uniform(-1, 1, 2)
        y = f(z[None])[0] + rng.uniform(-d, d)
        M.update(z[None], [y])
        E.update(z, y, M, window=2)
        assert E.compatibility_margin(M) >= -1e-12
    _check_compatible(E, M, rng)
    assert np.sqrt(np.mean((E(T) - f(T)) ** 2)) < 0.5 * err0


def test_vertex_bounds_contain_true_function():
    rng = np.random.default_rng(6)
    f, g_lo, g_hi = random_system(rng, 2)
    grid = RectangularPartition.uniform([-1, -1], [1, 1], [8, 8])
    Z = rng.uniform(-1, 1, (300, 2))
    M = GrowthMap.from_data(Z, f(Z), grid, g_lo, g_hi, 0, 0)
    lo, hi = vertex_bounds(M)
    V = np.stack(np.meshgrid(*grid.edges, indexing="ij"), -1).reshape(-1, 2)
    fv = f(V).reshape(lo.shape)
    assert np.all(fv >= lo - 1e-12) and np.all(fv <= hi + 1e-12)
