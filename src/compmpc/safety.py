"""Safety controllers for planar vehicles with learned velocity models.

The vehicle state is (x, y, theta) with theta periodic, the input is u = (u1, u2), and the
Euler-discretized dynamics are

    x+ = x + dt * vx(theta, u1),   y+ = y + dt * vy(theta, u1),   theta+ = theta + dt * w(u1, u2)

where (vx, vy, w) are unknown and only known through guaranteed interval models
(:class:`ddma.GrowthMap`, or any model with ``partition``, ``lower``, ``upper``) of
vx + d, vy + d, w + d. This covers the unicycle
and kinematic bicycle of the EJC 2023 and ADHS 2024 papers (u1 = speed, u2 = steering).

The symbolic abstraction has one state per cell of a uniform (x, y, theta) grid and one
input per cell of the (u1, u2) grid. Because the dynamics are translation invariant in
(x, y), the successor box of a cell is the cell shifted by integer offsets that depend
only on (theta cell, input cell). The maximal controlled invariant set inside a safe
set is the usual greatest fixed point; it is computed with 3-D summed-area tables and a
witness input per state, compiled with numba.

Guarantee: if F contains the true velocities plus disturbance, then from every state in
the invariant set, every input returned by :meth:`PlanarSafetyController.safe_inputs` keeps
the true system in the invariant set at the next step (alternating simulation, TAC 2024
Thm. 3), so it stays in the safe set forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from ddma import RectangularPartition

__all__ = ["PlanarGrid", "PlanarSafetyController"]


@dataclass(frozen=True)
class PlanarGrid:
    """Uniform grid of the vehicle state (x, y, theta) and of the input (u1, u2)."""

    x_lim: tuple
    y_lim: tuple
    n_x: int
    n_y: int
    n_theta: int
    u1_lim: tuple
    u2_lim: tuple
    n_u1: int
    n_u2: int
    dt: float

    @property
    def hx(self):
        return (self.x_lim[1] - self.x_lim[0]) / self.n_x

    @property
    def hy(self):
        return (self.y_lim[1] - self.y_lim[0]) / self.n_y

    @property
    def htheta(self):
        return 2 * np.pi / self.n_theta

    @property
    def theta_edges(self):
        return np.linspace(-np.pi, np.pi, self.n_theta + 1)

    @property
    def u1_edges(self):
        return np.linspace(*self.u1_lim, self.n_u1 + 1)

    @property
    def u2_edges(self):
        return np.linspace(*self.u2_lim, self.n_u2 + 1)

    def velocity_grid(self) -> RectangularPartition:
        """Grid on which vx and vy must be learned: (theta, u1)."""
        return RectangularPartition([self.theta_edges, self.u1_edges])

    def turn_grid(self) -> RectangularPartition:
        """Grid on which the turn rate w must be learned: (u1, u2)."""
        return RectangularPartition([self.u1_edges, self.u2_edges])

    def state_cell(self, s):
        """Cell (ix, iy, itheta) of a state; ix/iy may be out of range (outside the arena)."""
        x, y, th = s
        ix = int(np.floor((x - self.x_lim[0]) / self.hx))
        iy = int(np.floor((y - self.y_lim[0]) / self.hy))
        it = int(np.floor((wrap(th) + np.pi) / self.htheta)) % self.n_theta
        return ix, iy, it

    def input_cell_bounds(self, i1, i2):
        e1, e2 = self.u1_edges, self.u2_edges
        return np.array([e1[i1], e2[i2]]), np.array([e1[i1 + 1], e2[i2 + 1]])

    def cell_mask(self, region) -> np.ndarray:
        """Boolean (n_x, n_y) mask of cells whose closure meets the rectangle region."""
        (x0, x1), (y0, y1) = region
        xe = np.linspace(*self.x_lim, self.n_x + 1)
        ye = np.linspace(*self.y_lim, self.n_y + 1)
        mx = (xe[1:] >= x0) & (xe[:-1] <= x1)
        my = (ye[1:] >= y0) & (ye[:-1] <= y1)
        return mx[:, None] & my[None, :]


def wrap(theta):
    """Angle in [-pi, pi)."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------- kernels
@nb.njit(cache=True)
def _sat3(bad):
    """Summed-area table of `bad` with the theta axis doubled (for cyclic ranges)."""
    nx, ny, nt = bad.shape
    S = np.zeros((nx + 1, ny + 1, 2 * nt + 1), dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            for k in range(2 * nt):
                S[i + 1, j + 1, k + 1] = (bad[i, j, k % nt] + S[i, j + 1, k + 1]
                                          + S[i + 1, j, k + 1] + S[i + 1, j + 1, k]
                                          - S[i, j, k + 1] - S[i, j + 1, k] - S[i + 1, j, k]
                                          + S[i, j, k])
    return S


@nb.njit(cache=True)
def _box_ok(S, x0, x1, y0, y1, t0, tl, nx, ny):
    """True iff the box [x0,x1]x[y0,y1]x(t0..t0+tl-1 cyclic) is inside the grid and all good."""
    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        return False
    a, b, c, d, e, f = x0, x1 + 1, y0, y1 + 1, t0, t0 + tl
    cnt = (S[b, d, f] - S[a, d, f] - S[b, c, f] - S[b, d, e]
           + S[a, c, f] + S[a, d, e] + S[b, c, e] - S[a, c, e])
    return cnt == 0


@nb.njit(cache=True)
def _max_invariant(inv, xo, yo, to, valid, witness):
    """Greatest fixed point inside `inv` (modified in place). Returns the number of sweeps."""
    nx, ny, nt = inv.shape
    n1, n2 = valid.shape[1], valid.shape[2]
    sweeps = 0
    changed = True
    while changed:
        changed = False
        sweeps += 1
        bad = np.empty(inv.shape, dtype=np.int32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nt):
                    bad[i, j, k] = 0 if inv[i, j, k] else 1
        S = _sat3(bad)
        for k in range(nt):
            for i in range(nx):
                for j in range(ny):
                    if not inv[i, j, k]:
                        continue
                    w = witness[i, j, k]
                    if w >= 0:
                        a, b = w // n2, w % n2
                        if valid[k, a, b] and _box_ok(
                                S, i + xo[k, a, 0], i + xo[k, a, 1], j + yo[k, a, 0],
                                j + yo[k, a, 1], to[k, a, b, 0], to[k, a, b, 1], nx, ny):
                            continue
                    found = -1
                    for a in range(n1):
                        for b in range(n2):
                            if valid[k, a, b] and _box_ok(
                                    S, i + xo[k, a, 0], i + xo[k, a, 1], j + yo[k, a, 0],
                                    j + yo[k, a, 1], to[k, a, b, 0], to[k, a, b, 1], nx, ny):
                                found = a * n2 + b
                                break
                        if found >= 0:
                            break
                    witness[i, j, k] = found
                    if found < 0:
                        inv[i, j, k] = False
                        changed = True
    return sweeps


@nb.njit(cache=True)
def _safe_inputs(S, i, j, k, xo, yo, to, valid, nx, ny):
    n1, n2 = valid.shape[1], valid.shape[2]
    out = np.zeros((n1, n2), dtype=np.bool_)
    for a in range(n1):
        for b in range(n2):
            if valid[k, a, b] and _box_ok(S, i + xo[k, a, 0], i + xo[k, a, 1], j + yo[k, a, 0],
                                          j + yo[k, a, 1], to[k, a, b, 0], to[k, a, b, 1],
                                          nx, ny):
                out[a, b] = True
    return out


# ---------------------------------------------------------------------------- controller
class PlanarSafetyController:
    """Maximal controlled invariant set and safe inputs for a planar vehicle.

    Parameters
    ----------
    grid : PlanarGrid.
    model_vx, model_vy : guaranteed models on ``grid.velocity_grid()`` (theta, u1).
    model_w : guaranteed model on ``grid.turn_grid()`` (u1, u2).
    safe : boolean (n_x, n_y) or (n_x, n_y, n_theta) mask of the safe set.
    """

    def __init__(self, grid: PlanarGrid, model_vx, model_vy, model_w, safe: np.ndarray):
        self.grid = grid
        for m, g in ((model_vx, grid.velocity_grid()), (model_vy, grid.velocity_grid()),
                     (model_w, grid.turn_grid())):
            if not m.partition.same_as(g):
                raise ValueError("models must be learned on grid.velocity_grid()/turn_grid()")
        safe = np.asarray(safe, bool)
        if safe.ndim == 2:
            safe = np.repeat(safe[:, :, None], grid.n_theta, axis=2)
        if safe.shape != (grid.n_x, grid.n_y, grid.n_theta):
            raise ValueError("safe mask has the wrong shape")
        self.safe = safe
        self.inv = safe.copy()
        self.witness = np.full(safe.shape, -1, dtype=np.int64)
        self.set_models(model_vx, model_vy, model_w)
        self.sweeps = 0

    def set_models(self, model_vx, model_vy, model_w):
        """Recompute the successor offsets from (possibly updated) models."""
        g = self.grid
        nt = g.n_theta
        dt = g.dt
        vx_lo, vx_hi = model_vx.lower, model_vx.upper          # (nt, n1)
        vy_lo, vy_hi = model_vy.lower, model_vy.upper
        w_lo, w_hi = model_w.lower, model_w.upper              # (n1, n2)
        fin_v = np.isfinite(vx_lo) & np.isfinite(vx_hi) & np.isfinite(vy_lo) & np.isfinite(vy_hi)
        fin_w = np.isfinite(w_lo) & np.isfinite(w_hi)
        with np.errstate(invalid="ignore"):
            xo = np.stack([np.floor(dt * vx_lo / g.hx), 1 + np.floor(dt * vx_hi / g.hx)], -1)
            yo = np.stack([np.floor(dt * vy_lo / g.hy), 1 + np.floor(dt * vy_hi / g.hy)], -1)
            t0 = np.floor(dt * w_lo / g.htheta)                 # (n1, n2) offset of first cell
            t1 = 1 + np.floor(dt * w_hi / g.htheta)
        xo = np.where(fin_v[..., None], xo, 0).astype(np.int64)
        yo = np.where(fin_v[..., None], yo, 0).astype(np.int64)
        length = np.where(fin_w, t1 - t0 + 1, 0).astype(np.int64)
        start = (np.arange(nt)[:, None, None] + np.where(fin_w, t0, 0).astype(np.int64)[None])
        start = np.mod(start, nt)
        length = np.minimum(np.broadcast_to(length[None], start.shape), nt)
        start = np.where(length >= nt, 0, start)
        self.xo, self.yo = xo, yo
        self.to = np.ascontiguousarray(np.stack([start, length], -1))
        self.valid = np.ascontiguousarray(fin_v[:, :, None] & fin_w[None, :, :])
        self.models = (model_vx, model_vy, model_w)
        self._S = None

    def solve(self, restart: bool = False) -> PlanarSafetyController:
        """Maximal controlled invariant set inside the safe set.

        After the models were refined with new data, the new maximal invariant set
        contains the old one. ``restart=True`` recomputes it from the whole safe set
        (and so finds every state that became safe); ``restart=False`` only removes
        states, which is enough when the models did not change.
        """
        if restart:
            self.inv = self.safe.copy()
        self.sweeps = _max_invariant(self.inv, self.xo, self.yo, self.to, self.valid,
                                     self.witness)
        self._S = None
        return self

    def _sat(self):
        if self._S is None:
            self._S = _sat3((~self.inv).astype(np.int32))
        return self._S

    def in_invariant(self, s) -> bool:
        g = self.grid
        i, j, k = g.state_cell(s)
        return 0 <= i < g.n_x and 0 <= j < g.n_y and bool(self.inv[i, j, k])

    def safe_inputs(self, s) -> np.ndarray:
        """Boolean (n_u1, n_u2) mask of input cells that keep the state in the invariant set."""
        g = self.grid
        i, j, k = g.state_cell(s)
        if not (0 <= i < g.n_x and 0 <= j < g.n_y and self.inv[i, j, k]):
            return np.zeros((g.n_u1, g.n_u2), dtype=bool)
        return _safe_inputs(self._sat(), i, j, k, self.xo, self.yo, self.to, self.valid,
                            g.n_x, g.n_y)

    def invariant_fraction(self) -> float:
        """Share of the safe set that is certified (controlled invariant)."""
        return float(self.inv.sum() / max(self.safe.sum(), 1))
