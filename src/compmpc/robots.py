"""The mobile-robot benchmark of the EJC 2023 and ADHS 2024 papers.

A unicycle (kinematic bicycle with wheelbase L) in a 5 x 5 m room:

    x' = v cos(theta) + d1,   y' = v sin(theta) + d2,   theta' = v tan(delta) / L + d3,

discretized with the Euler method, d in [-0.05, 0.05]^3. The inputs are the speed v and
the steering angle delta. The controller does not know these equations: it only sees
measured transitions and the derivative bounds below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .safety import PlanarGrid, wrap

__all__ = ["Unicycle", "FourZoneArena"]


@dataclass
class Unicycle:
    L: float = 0.1
    d_max: float = 0.05
    v_lim: tuple = (0.1, 2.0)
    delta_lim: tuple = (-np.pi / 4, np.pi / 4)

    # ------------------------------------------------------------------ true system
    def velocities(self, theta, v, delta):
        theta, v, delta = np.broadcast_arrays(theta, v, delta)
        return np.stack([v * np.cos(theta), v * np.sin(theta), v * np.tan(delta) / self.L], -1)

    def step(self, s, u, dt, rng):
        """One Euler step of the true system; returns (next state, measured velocities)."""
        meas = self.velocities(s[2], u[0], u[1]) + rng.uniform(-self.d_max, self.d_max, 3)
        nxt = np.array([s[0] + dt * meas[0], s[1] + dt * meas[1], wrap(s[2] + dt * meas[2])])
        return nxt, meas

    def sample(self, n, rng, v_max=None):
        """Independent measured transitions (theta, v, delta) -> velocities + disturbance."""
        v_hi = self.v_lim[1] if v_max is None else v_max
        th = rng.uniform(-np.pi, np.pi, n)
        v = rng.uniform(self.v_lim[0], v_hi, n)
        de = rng.uniform(*self.delta_lim, n)
        Y = self.velocities(th, v, de) + rng.uniform(-self.d_max, self.d_max, (n, 3))
        return np.c_[th, v, de], Y

    # ------------------------------------------------------------------ prior knowledge
    def derivative_bounds(self):
        """Bounds on the partial derivatives over the input box (all the controller knows).

        vx = v cos(theta):  d/dtheta in [-v_max, v_max],  d/dv in [-1, 1]  (same for vy)
        w  = v tan(delta)/L: d/dv in [tan(delta_min)/L, tan(delta_max)/L],
                             d/ddelta in [v_min/L, v_max / (L cos^2(delta_max))]
        """
        v0, v1 = self.v_lim
        d0, d1 = self.delta_lim
        vxy = (np.array([-v1, -1.0]), np.array([v1, 1.0]))
        w = (np.array([np.tan(d0) / self.L, v0 / self.L]),
             np.array([np.tan(d1) / self.L, v1 / (self.L * np.cos(d1) ** 2)]))
        return vxy, w


@dataclass
class FourZoneArena:
    """The ADHS 2024 course: two blocks, four coloured target zones, walls all around."""

    size: float = 5.0
    obstacles: list = field(default_factory=lambda: [((0.0, 1.5), (2.1, 3.1)),
                                                     ((3.5, 5.0), (2.1, 3.1))])
    zones: dict = field(default_factory=lambda: {
        "red": ((0.0, 1.0), (1.1, 2.1)), "yellow": ((4.0, 5.0), (3.1, 4.1)),
        "green": ((0.0, 1.0), (3.1, 4.1)), "blue": ((4.0, 5.0), (1.1, 2.1))})
    order: tuple = ("red", "yellow", "green", "blue")

    def planar_grid(self, n_xy=50, n_theta=40, n_u=40, dt=0.2, robot: Unicycle = None):
        r = robot or Unicycle()
        return PlanarGrid((0.0, self.size), (0.0, self.size), n_xy, n_xy, n_theta,
                          r.v_lim, r.delta_lim, n_u, n_u, dt)

    def safe_mask(self, grid: PlanarGrid) -> np.ndarray:
        """Cells of the room that do not overlap an obstacle (touching an edge is allowed)."""
        xe = np.linspace(*grid.x_lim, grid.n_x + 1)
        ye = np.linspace(*grid.y_lim, grid.n_y + 1)
        safe = np.ones((grid.n_x, grid.n_y), bool)
        for (x0, x1), (y0, y1) in self.obstacles:
            mx = (xe[1:] > x0) & (xe[:-1] < x1)
            my = (ye[1:] > y0) & (ye[:-1] < y1)
            safe &= ~(mx[:, None] & my[None, :])
        return safe

    def collides(self, s) -> bool:
        x, y = s[0], s[1]
        if not (0.0 <= x <= self.size and 0.0 <= y <= self.size):
            return True
        return any(x0 < x < x1 and y0 < y < y1 for (x0, x1), (y0, y1) in self.obstacles)

    def clearance(self, s) -> float:
        """Distance from the point to the nearest wall or obstacle (negative inside one)."""
        x, y = s[0], s[1]
        d = min(x, y, self.size - x, self.size - y)
        for (x0, x1), (y0, y1) in self.obstacles:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            out = np.hypot(dx, dy)
            inside = -min(x - x0, x1 - x, y - y0, y1 - y) if (dx == 0 and dy == 0) else out
            d = min(d, inside)
        return float(d)

    def in_zone(self, s, name) -> bool:
        (x0, x1), (y0, y1) = self.zones[name]
        return x0 <= s[0] <= x1 and y0 <= s[1] <= y1

    def distance_fields(self, h=0.05, margin=0.15):
        """Shortest-path distance (around the obstacles) to each zone, on a fine grid.

        Used as the MPC's cost-to-go so that a short horizon does not get stuck behind
        an obstacle. Returns {zone: array (n, n)} and the grid spacing.
        """
        n = int(round(self.size / h))
        c = (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(c, c, indexing="ij")
        free = np.ones((n, n), bool)
        for (x0, x1), (y0, y1) in self.obstacles:
            free &= ~((X > x0 - margin) & (X < x1 + margin) & (Y > y0 - margin) & (Y < y1 + margin))
        idx = np.arange(n * n).reshape(n, n)
        rows, cols, w = [], [], []
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            i2, j2 = ii + di, jj + dj
            ok = (i2 >= 0) & (i2 < n) & (j2 >= 0) & (j2 < n)
            a, b = idx[ii[ok], jj[ok]], idx[i2[ok], j2[ok]]
            keep = free.ravel()[a] & free.ravel()[b]
            rows += [a[keep]]
            cols += [b[keep]]
            w += [np.full(keep.sum(), h * np.hypot(di, dj))]
        G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n * n, n * n)).tocsr()
        fields = {}
        for name, ((x0, x1), (y0, y1)) in self.zones.items():
            src = idx[(X >= x0 + 0.1) & (X <= x1 - 0.1) & (Y >= y0 + 0.1) & (Y <= y1 - 0.1) & free]
            dist = dijkstra(G, directed=False, indices=src, min_only=True)
            d = dist.reshape(n, n)
            d[~np.isfinite(d)] = d[np.isfinite(d)].max() + 1.0
            fields[name] = d
        return fields, h
