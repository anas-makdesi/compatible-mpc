"""Safe learning-based MPC with compatible models (EJC 2023, Theorem 1).

At each step the controller

1. asks the safety controller for the input cells that keep the state in the controlled
   invariant set (computed from the *guaranteed* model): the set C_F(x);
2. solves the MPC problem with the *learned* (compatible) model, with the first input
   restricted to C_F(x);
3. applies the first input.

Because the first input is always in C_F(x), the closed loop stays in the invariant set for
every realization of the disturbance, whatever the quality of the learned model or of the
optimizer. The learned model and the optimizer only affect performance.

C_F(x) is a union of boxes (input cells), generally non-convex and disconnected. Three
optimizers are available; all of them only ever evaluate plans whose first input lies in
a safe cell:

``method="shooting"``  random shooting: sample many plans, keep the cheapest, then two rounds
                       of random perturbations around it (the first input stays in its cell).
``method="cem"``       cross-entropy method: sample plans from a Gaussian, refit the Gaussian
                       to the best 10 %, repeat; first inputs are projected onto the nearest
                       safe cell.
``method="gradient"``  the MPC problem solved per safe cell: for the most promising cells,
                       projected gradient descent with the first input bounded to the
                       cell's box (all cells in one batch).

The predicted-state constraint x(i|t) in X_s of Theorem 1 is implemented as a penalty larger
than any feasible cost, so a plan that keeps its predictions in the invariant set always
beats one that does not (a hard constraint with a fallback when no sampled plan is feasible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .estimate import CompatibleEstimate
from .safety import PlanarSafetyController, wrap

__all__ = ["CompatibleMPC", "MPCResult"]

METHODS = ("shooting", "cem", "gradient")


@dataclass
class MPCResult:
    u: np.ndarray            # applied input
    cost: float              # predicted cost of the chosen plan (penalty included)
    n_safe_cells: int        # number of safe input cells at this state
    max_safe_u1: float       # largest u1 (e.g. speed) among the safe input cells
    plan: np.ndarray         # predicted states (N+1, 3)
    n_evaluations: int       # number of plans whose cost was computed


class CompatibleMPC:
    """MPC for a planar vehicle: plans with the learned model, acts within the safe inputs.

    Parameters
    ----------
    safety : PlanarSafetyController (guaranteed model -> safe inputs).
    est_vx, est_vy, est_w : CompatibleEstimate of vx(theta, u1), vy(theta, u1), w(u1, u2).
    stage_cost : vectorized function (states (M, 3), inputs (M, 2), step i) -> (M,) costs.
    horizon : prediction horizon N.
    method : "cem" (default), "shooting" or "gradient" (see the module docstring).
    n_samples : plans per sampling round (default 400 for CEM, 600 otherwise).
    """

    def __init__(self, safety: PlanarSafetyController, est_vx: CompatibleEstimate,
                 est_vy: CompatibleEstimate, est_w: CompatibleEstimate,
                 stage_cost: Callable, horizon: int = 5, method: str = "cem",
                 n_samples: int = None, n_refine: int = 2, cem_iters: int = 5,
                 cem_elite: float = 0.1, grad_cells: int = 8, grad_iters: int = 30,
                 unsafe_penalty: float = 1e3, seed: int = 0, enforce_safety: bool = True):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.safety = safety
        self.est = (est_vx, est_vy, est_w)
        self.stage_cost = stage_cost
        self.N = horizon
        self.method = method
        # default population: 400 per CEM iteration (5 iterations = 2,000 plans);
        # 600 per round for shooting and gradient
        self.n_samples = n_samples or (400 if method == "cem" else 600)
        self.n_refine = n_refine
        self.cem_iters, self.cem_elite = cem_iters, cem_elite
        self.grad_cells, self.grad_iters = grad_cells, grad_iters
        self.unsafe_penalty = unsafe_penalty
        self.rng = np.random.default_rng(seed)
        # False gives the usual learning-based MPC baseline: same planner and learned model,
        # obstacles as state constraints, but no certified input set (no guarantee).
        self.enforce_safety = enforce_safety
        g = safety.grid
        self._lo = np.array([g.u1_lim[0], g.u2_lim[0]])
        self._hi = np.array([g.u1_lim[1], g.u2_lim[1]])
        self._e = (g.u1_edges, g.u2_edges)

    # ================================================================== model and cost
    def predict(self, s0, U):
        """Roll out input sequences U (M, N, 2) from s0 with the learned model."""
        g = self.safety.grid
        M = U.shape[0]
        S = np.empty((M, self.N + 1, 3))
        S[:, 0] = s0
        evx, evy, ew = self.est
        for i in range(self.N):
            th = S[:, i, 2]
            u = np.clip(U[:, i], self._lo, self._hi)
            zv = np.c_[th, u[:, 0]]
            S[:, i + 1, 0] = S[:, i, 0] + g.dt * evx(zv)
            S[:, i + 1, 1] = S[:, i, 1] + g.dt * evy(zv)
            S[:, i + 1, 2] = wrap(th + g.dt * ew(u))
        return S

    def _violations(self, S):
        """Number of predicted states x(1|t)..x(N|t) outside the invariant (or safe) set."""
        g = self.safety.grid
        inv = self.safety.inv if self.enforce_safety else self.safety.safe
        P = S[:, 1:].reshape(-1, 3)
        ix = np.floor((P[:, 0] - g.x_lim[0]) / g.hx).astype(int)
        iy = np.floor((P[:, 1] - g.y_lim[0]) / g.hy).astype(int)
        it = np.floor((P[:, 2] + np.pi) / g.htheta).astype(int) % g.n_theta
        inside = (ix >= 0) & (ix < g.n_x) & (iy >= 0) & (iy < g.n_y)
        ok = np.zeros(P.shape[0], bool)
        ok[inside] = inv[ix[inside], iy[inside], it[inside]]
        return (~ok).reshape(S.shape[0], self.N).sum(axis=1)

    def performance_cost(self, S, U):
        """Sum of the stage costs (the MPC objective, without the state constraint)."""
        total = np.zeros(S.shape[0])
        for i in range(1, self.N + 1):
            total += self.stage_cost(S[:, i], U[:, i - 1], i)
        return total

    def cost(self, S, U):
        """Objective plus a penalty per predicted state outside the invariant set."""
        return self.performance_cost(S, U) + self.unsafe_penalty * self._violations(S)

    # ================================================================== safe first inputs
    def _cell_of(self, u):
        """Input cell index (i1, i2) of each input u (M, 2)."""
        e1, e2 = self._e
        i1 = np.clip(np.searchsorted(e1, u[:, 0], "right") - 1, 0, e1.size - 2)
        i2 = np.clip(np.searchsorted(e2, u[:, 1], "right") - 1, 0, e2.size - 2)
        return np.c_[i1, i2]

    def _cell_box(self, cells):
        e1, e2 = self._e
        lo = np.c_[e1[cells[:, 0]], e2[cells[:, 1]]]
        hi = np.c_[e1[cells[:, 0] + 1], e2[cells[:, 1] + 1]]
        return lo, hi

    def _sample_in_cells(self, cells, M):
        """M random inputs, each uniform inside a random safe cell."""
        pick = cells[self.rng.integers(len(cells), size=M)]
        lo, hi = self._cell_box(pick)
        return self.rng.uniform(lo, hi)

    def _project_to_cells(self, u, cells, mask=None):
        """Nearest point of the union of the safe cells (distances scaled by the input range).

        With ``mask`` given, inputs already inside a safe cell are left unchanged.
        """
        if mask is not None:
            c = self._cell_of(u)
            out = u.copy()
            bad = ~mask[c[:, 0], c[:, 1]]
            if np.any(bad):
                out[bad] = self._project_to_cells(u[bad], cells)
            return out
        lo, hi = self._cell_box(cells)                               # (K, 2)
        scale = self._hi - self._lo
        cand = np.clip(u[:, None, :], lo[None], hi[None])            # (M, K, 2)
        d = (((cand - u[:, None, :]) / scale) ** 2).sum(axis=2)      # (M, K)
        return cand[np.arange(len(u)), np.argmin(d, axis=1)]

    def _random_plans(self, s, cells, M):
        """Plans whose first input is in a safe cell and whose later inputs random-walk."""
        U = np.repeat(self._sample_in_cells(cells, M)[:, None, :], self.N, axis=1)
        step = np.array([0.15, 0.25]) * (self._hi - self._lo)
        U[:, 1:] += np.cumsum(self.rng.normal(0, 1, (M, self.N - 1, 2)) * step, axis=1)
        return np.clip(U, self._lo, self._hi)

    # ================================================================== optimizers
    def _plan_shooting(self, s, cells, mask=None):
        U = self._random_plans(s, cells, self.n_samples)
        S = self.predict(s, U)
        J = self.cost(S, U)
        b = int(np.argmin(J))
        best_U, best_S, best_J = U[b], S[b], J[b]
        n_eval = len(U)
        for r in range(self.n_refine):
            noise = np.array([0.1, 0.15]) * (self._hi - self._lo) * 0.5 ** (r + 1)
            Ur = best_U[None] + self.rng.normal(0, 1, (self.n_samples, self.N, 2)) * noise
            lo, hi = self._cell_box(self._cell_of(best_U[None, 0]))
            Ur[:, 0] = np.clip(Ur[:, 0], lo, hi)          # first input stays in its safe cell
            Ur = np.clip(Ur, self._lo, self._hi)
            Sr = self.predict(s, Ur)
            Jr = self.cost(Sr, Ur)
            k = int(np.argmin(Jr))
            n_eval += len(Ur)
            if Jr[k] < best_J:
                best_U, best_S, best_J = Ur[k], Sr[k], Jr[k]
        return best_U, best_S, best_J, n_eval

    def _plan_cem(self, s, cells, mask):
        M = self.n_samples
        n_elite = max(2, int(self.cem_elite * M))
        U = self._random_plans(s, cells, M)                # iteration 0: same as shooting
        best_U, best_S, best_J, n_eval = None, None, np.inf, 0
        min_std = 0.01 * (self._hi - self._lo)
        mean = std = None
        for it in range(self.cem_iters):
            if it > 0:
                U = mean[None] + std[None] * self.rng.normal(0, 1, (M, self.N, 2))
                U = np.clip(U, self._lo, self._hi)
                U[:, 0] = self._project_to_cells(U[:, 0], cells, mask)  # first input made safe
            S = self.predict(s, U)
            J = self.cost(S, U)
            n_eval += M
            order = np.argsort(J)
            if J[order[0]] < best_J:
                best_U, best_S, best_J = U[order[0]], S[order[0]], J[order[0]]
            elite = U[order[:n_elite]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), min_std)
        return best_U, best_S, best_J, n_eval

    def _plan_gradient(self, s, cells, mask=None):
        # 1) one round of random plans to rank the safe cells
        U0 = self._random_plans(s, cells, self.n_samples)
        S0 = self.predict(s, U0)
        J0 = self.cost(S0, U0)
        n_eval = len(U0)
        first_cells = self._cell_of(U0[:, 0])
        order = np.argsort(J0)
        starts, seen = [], set()
        for k in order:                                   # best plan of each distinct cell
            key = tuple(first_cells[k])
            if key not in seen:
                seen.add(key)
                starts.append(k)
            if len(starts) == self.grad_cells:
                break
        b = int(order[0])
        best_U, best_S, best_J = U0[b], S0[b], J0[b]
        dim = 2 * self.N
        counter = [0]                                     # objective evaluations (plans)

        # 2) for these cells at once: projected gradient descent on the objective, with the
        #    first input bounded to its cell's box (one box-constrained problem per cell).
        #    Variables are scaled to [0, 1]; gradients by forward differences, all cells and
        #    all perturbations evaluated in one batch per iteration.
        K = len(starts)
        span = np.tile(self._hi - self._lo, self.N)                   # (dim,)
        base = np.tile(self._lo, self.N)
        lo_b = np.zeros((K, dim))
        hi_b = np.ones((K, dim))
        c_lo, c_hi = self._cell_box(first_cells[starts])
        lo_b[:, :2] = (c_lo - self._lo) / (self._hi - self._lo)
        hi_b[:, :2] = (c_hi - self._lo) / (self._hi - self._lo)
        X = (U0[starts].reshape(K, dim) - base) / span
        h = 1e-4

        def objective(Xs):
            Us = (Xs * span + base).reshape(-1, self.N, 2)
            return self.performance_cost(self.predict(s, Us), Us)

        f = objective(X)
        step = np.full(K, 0.05)
        for _ in range(self.grad_iters):
            P = np.repeat(X[:, None, :], dim, axis=1) + h * np.eye(dim)[None]   # (K, dim, dim)
            G = (objective(P.reshape(-1, dim)).reshape(K, dim) - f[:, None]) / h
            counter[0] += K * dim
            Xn = np.clip(X - step[:, None] * G / (np.linalg.norm(G, axis=1, keepdims=True)
                                                 + 1e-12), lo_b, hi_b)
            fn = objective(Xn)
            counter[0] += K
            better = fn < f
            X[better], f[better] = Xn[better], fn[better]
            step = np.where(better, step * 1.5, step * 0.5)            # simple step adaptation
            if np.all(step < 1e-4):
                break
        Uk = (X * span + base).reshape(K, self.N, 2)
        Sk = self.predict(s, Uk)
        Jk = self.cost(Sk, Uk)                            # constraint checked after solving
        k = int(np.argmin(Jk))
        if Jk[k] < best_J:
            best_U, best_S, best_J = Uk[k], Sk[k], Jk[k]
        return best_U, best_S, best_J, n_eval + counter[0]

    # ================================================================== control
    def safe_cells(self, s):
        """Input cells in C_F(x) for the current state (all cells for the baseline)."""
        g = self.safety.grid
        if self.enforce_safety:
            mask = self.safety.safe_inputs(s)
        else:
            mask = np.ones((g.n_u1, g.n_u2), dtype=bool)
        return mask, np.argwhere(mask)

    def __call__(self, s) -> MPCResult:
        mask, cells = self.safe_cells(s)
        if cells.size == 0:
            raise RuntimeError("state is outside the controlled invariant set")
        plan = {"shooting": self._plan_shooting, "cem": self._plan_cem,
                "gradient": self._plan_gradient}[self.method]
        U, S, J, n_eval = plan(s, cells, mask)
        u = U[0].copy()
        # the guarantee: the applied input lies in a safe cell (snap inside on a cell edge)
        c = self._cell_of(u[None])[0]
        if not mask[c[0], c[1]]:
            u = self._project_to_cells(u[None], cells)[0]
            c = self._cell_of(u[None])[0]
            if not mask[c[0], c[1]]:                      # exactly on an edge: move inside
                lo, hi = self._cell_box(cells[[np.argmin(((cells - c) ** 2).sum(1))]])
                u = np.clip(u, lo[0], np.nextafter(hi[0], -np.inf))
        e1 = self._e[0]
        return MPCResult(u=u, cost=float(J), n_safe_cells=int(len(cells)),
                         max_safe_u1=float(e1[cells[:, 0].max() + 1]), plan=S,
                         n_evaluations=int(n_eval))
