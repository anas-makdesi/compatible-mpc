"""Guaranteed (set-valued) models of the unknown dynamics.

A guaranteed model of one scalar output f stores, for every cell of a rectangular
partition of its inputs, an interval [lower, upper] that contains f(z) + d for every z in
the cell and every disturbance d in [d_lo, d_hi]. Two constructions are used:

* :class:`ddma.GrowthMap` (ADHS 2024; thesis Ch. 3, growth cones): every data point bounds
  f by a cone whose slopes are the derivative bounds; the model is the intersection of all
  cones seen so far. ``update`` adds new data exactly (one point at a time online),
  ``update_local`` only around each new point (the windowed update of the paper).
* :func:`monotone_split_model` (EJC 2023, Section 3): the data are transformed into two
  monotone problems, solved with the minimal simulating maps of ddma, and intersected.

Both are sound; they can be intersected with :func:`intersect`. Everything downstream
(compatible estimates, safety controller) only needs ``partition``, ``lower``, ``upper``,
``d_lo``, ``d_hi`` and ``f_bounds()``, which both provide.
"""

from __future__ import annotations

import numpy as np
from ddma import GrowthMap, PartitionedSimulatingMap, RectangularPartition

__all__ = ["GrowthMap", "IntervalModel", "monotone_split_model", "intersect"]


class IntervalModel:
    """A guaranteed model given by its cell intervals (no data kept).

    Attributes: ``partition``, ``lower``, ``upper`` (shape ``partition.shape``), ``d_lo``,
    ``d_hi``.
    """

    def __init__(self, partition: RectangularPartition, lower, upper, d_lo: float,
                 d_hi: float):
        self.partition = partition
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        if self.lower.shape != partition.shape or self.upper.shape != partition.shape:
            raise ValueError("lower and upper must have shape partition.shape")
        self.d_lo, self.d_hi = float(d_lo), float(d_hi)

    def bounds(self, Z):
        """Bounds on f(z) + d at the cells of the points Z: (lower (K,), upper (K,))."""
        from .grid import locate

        idx = tuple(locate(self.partition, Z).T)
        return self.lower[idx], self.upper[idx]

    def f_bounds(self):
        """Bounds on f itself per cell: (lower - d_lo, upper - d_hi)."""
        return self.lower - self.d_lo, self.upper - self.d_hi

    def widths(self) -> np.ndarray:
        return self.upper - self.lower


def intersect(a, b) -> IntervalModel:
    """Intersection of two sound models on the same partition (it stays sound)."""
    if not a.partition.same_as(b.partition):
        raise ValueError("models must be on the same partition")
    if (a.d_lo, a.d_hi) != (b.d_lo, b.d_hi):
        raise ValueError("models must assume the same disturbance bounds")
    return IntervalModel(a.partition, np.maximum(a.lower, b.lower),
                         np.minimum(a.upper, b.upper), a.d_lo, a.d_hi)


def monotone_split_model(Z, y, partition, g_lo, g_hi, d_lo, d_hi) -> IntervalModel:
    """Guaranteed model of the EJC 2023 paper (Section 3), computed with ddma.

    With A- = min(g_lo, 0) and A+ = max(g_hi, 0), the functions f- = f - A- z and
    f+ = A+ z - f are monotone. Their minimal simulating maps on ``partition`` (TAC paper)
    give two interval models of f + d, which are intersected (EJC 2023, Prop. 2).
    """
    Z = np.atleast_2d(np.asarray(Z, float))
    y = np.asarray(y, float).reshape(-1)
    n = Z.shape[1]
    g_lo = np.broadcast_to(np.asarray(g_lo, float), (n,))
    g_hi = np.broadcast_to(np.asarray(g_hi, float), (n,))
    Am, Ap = np.minimum(g_lo, 0.0), np.maximum(g_hi, 0.0)
    # y- = f- + d ;  y+ = f+ - d  with  -d in [-d_hi, -d_lo]
    s_minus = PartitionedSimulatingMap.from_data(Z, (y - Z @ Am)[:, None], partition,
                                                 d_lo, d_hi)
    s_plus = PartitionedSimulatingMap.from_data(Z, (Z @ Ap - y)[:, None], partition,
                                                -d_hi, -d_lo)

    def lin_range(A):  # range of A z over each cell (separable)
        lo = np.zeros(partition.shape)
        hi = np.zeros(partition.shape)
        for j, e in enumerate(partition.edges):
            s = [1] * n
            s[j] = e.size - 1
            v1, v2 = (A[j] * e[:-1]).reshape(s), (A[j] * e[1:]).reshape(s)
            lo = lo + np.minimum(v1, v2)
            hi = hi + np.maximum(v1, v2)
        return lo, hi

    am_lo, am_hi = lin_range(Am)
    ap_lo, ap_hi = lin_range(Ap)
    lower = np.maximum(am_lo + s_minus.lower[..., 0], ap_lo - s_plus.upper[..., 0])
    upper = np.minimum(am_hi + s_minus.upper[..., 0], ap_hi - s_plus.lower[..., 0])
    return IntervalModel(partition, lower, upper, d_lo, d_hi)
