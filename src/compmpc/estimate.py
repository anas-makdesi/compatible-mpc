"""Compatible estimates: a smooth best-fit model kept inside the guaranteed model.

The estimate f_hat is piecewise multi-affine on the same grid as the guaranteed model F and
is stored by its values at the grid vertices. It is *compatible* with F when

    f_hat(z) + D  is contained in  F(cell of z)    for every z,

i.e. f_hat(z) in [lower - d_lo, upper - d_hi] on every cell. A multi-affine function on a
box takes its extreme values at the vertices (Belta and Habets 2006, Lemma 2), so this
holds as soon as every vertex value lies in the bounds of all cells that share the vertex.
Compatibility is therefore a set of *box constraints* on the vertex values:

* offline (EJC 2023, Prop. 5): a sparse least-squares fit with these box constraints;
* online (ADHS 2024, Section 4.2): a stochastic-gradient step on the vertices around each
  new data point, followed by a projection onto the box.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear

from .grid import corner_offsets, interpolate, locate, multilinear_weights, vertex_shape

__all__ = ["CompatibleEstimate", "vertex_bounds"]


def vertex_bounds(model):
    """Box [lo, hi] for the vertex values that makes an estimate compatible with ``model``."""
    f_lo, f_hi = model.f_bounds()
    n = f_lo.ndim
    lo = np.full(vertex_shape(model.partition), -np.inf)
    hi = np.full(vertex_shape(model.partition), np.inf)
    for c in corner_offsets(n):  # each cell writes to its 2^n vertices
        sl = tuple(slice(int(o), int(o) + s) for o, s in zip(c, f_lo.shape))
        np.maximum(lo[sl], f_lo, out=lo[sl])
        np.minimum(hi[sl], f_hi, out=hi[sl])
    return lo, hi


class CompatibleEstimate:
    """Piecewise multi-affine estimate of one scalar output, compatible with a model.

    ``model`` arguments are guaranteed models on the same partition: a
    :class:`ddma.GrowthMap` or an :class:`compmpc.bounds.IntervalModel`.

    Attributes
    ----------
    grid : the grid of the guaranteed model.
    values : vertex values, shape ``vertex_shape(grid)``.
    """

    def __init__(self, grid, values):
        self.grid = grid
        self.values = np.asarray(values, float)
        if self.values.shape != vertex_shape(grid):
            raise ValueError(f"values must have shape {vertex_shape(grid)}")
        self.n_updates = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def midpoint(cls, model) -> CompatibleEstimate:
        """Middle of the compatible box at every vertex (0 where it is unbounded)."""
        lo, hi = vertex_bounds(model)
        v = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                     np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0)))
        return cls(model.partition, v)

    @classmethod
    def fit(cls, Z, y, model, smooth: float = 1e-3) -> CompatibleEstimate:
        """Offline compatible fit (EJC 2023, Prop. 5).

        Minimizes  sum_k (y_k - f_hat(z_k))^2 + smooth * K * |differences of f_hat|^2
        subject to the compatibility box. The small smoothness term (an addition to the
        paper) fills in vertices that no data point touches; set ``smooth=0`` to get the
        paper's problem exactly.
        """
        grid = model.partition
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).reshape(-1)
        vshape = vertex_shape(grid)
        V = int(np.prod(vshape))
        w, verts = multilinear_weights(grid, Z)
        cols = np.ravel_multi_index(tuple(np.moveaxis(verts, -1, 0)), vshape)
        rows = np.repeat(np.arange(y.size), w.shape[1])
        A = sp.csr_matrix((w.ravel(), (rows, cols.ravel())), shape=(y.size, V))
        blocks, rhs = [A], [y]
        if smooth > 0:
            scale = np.sqrt(smooth * y.size / max(V, 1))
            idx = np.arange(V).reshape(vshape)
            for j in range(len(vshape)):
                a = np.take(idx, np.arange(vshape[j] - 1), axis=j).ravel()
                b = np.take(idx, np.arange(1, vshape[j]), axis=j).ravel()
                D = sp.csr_matrix(
                    (np.r_[np.ones(a.size), -np.ones(a.size)],
                     (np.r_[np.arange(a.size), np.arange(a.size)], np.r_[a, b])),
                    shape=(a.size, V))
                blocks.append(scale * D)
                rhs.append(np.zeros(a.size))
        lo, hi = vertex_bounds(model)
        res = lsq_linear(sp.vstack(blocks).tocsr(), np.concatenate(rhs),
                         bounds=(lo.ravel(), hi.ravel()), method="trf", lsmr_tol="auto",
                         tol=1e-10)
        vals = np.clip(res.x, lo.ravel(), hi.ravel()).reshape(vshape)
        return cls(grid, vals)

    # ------------------------------------------------------------------ online update
    def update(self, z, y: float, model, window: int = 2,
               lr: float = 0.5, sigma: float = None) -> CompatibleEstimate:
        """One online step for a new data point (ADHS 2024, Section 4.2); in place.

        Every cell r' in the (2*window+1)^n neighbourhood of the point's cell contributes
        one equation y = f_hat_{r'}(z), with f_hat_{r'} the multi-affine function of cell r'
        (extrapolated outside the cell). The equations are weighted by a Gaussian of the
        cell distance, a normalized gradient step is taken on the vertices involved, and
        the result is projected onto the compatibility box of ``model``.
        """
        z = np.asarray(z, float).reshape(1, -1)
        n = z.shape[1]
        cell = locate(self.grid, z)[0]
        shape = np.array(self.grid.shape)
        sigma = max(window, 1) / 2.0 if sigma is None else sigma
        offs = np.stack(np.meshgrid(*[np.arange(-window, window + 1)] * n, indexing="ij"),
                        axis=-1).reshape(-1, n)
        cells = cell[None, :] + offs
        ok = np.all((cells >= 0) & (cells < shape), axis=1)
        cells, offs = cells[ok], offs[ok]
        wts = np.exp(-0.5 * (offs ** 2).sum(axis=1) / sigma ** 2)
        a, verts = multilinear_weights(self.grid, np.repeat(z, len(cells), 0), cells)
        vidx = tuple(np.moveaxis(verts, -1, 0))                        # (R, 2^n) each
        resid = (a * self.values[vidx]).sum(axis=1) - y                # (R,)
        grad = (wts * resid)[:, None] * a                              # (R, 2^n)
        denom = float((wts[:, None] * a ** 2).sum()) + 1e-12
        np.add.at(self.values, vidx, -lr * grad / denom)
        self.project(model)
        self.n_updates += 1
        return self

    def project(self, model) -> CompatibleEstimate:
        """Clip the vertex values into the compatibility box of ``model`` (in place)."""
        lo, hi = vertex_bounds(model)
        np.clip(self.values, lo, hi, out=self.values)
        return self

    # ------------------------------------------------------------------ queries
    def __call__(self, Z) -> np.ndarray:
        return interpolate(self.grid, self.values, Z)

    def compatibility_margin(self, model) -> float:
        """min over vertices of the distance to the box (>= 0 iff compatible)."""
        lo, hi = vertex_bounds(model)
        return float(np.min(np.minimum(self.values - lo, hi - self.values)))
