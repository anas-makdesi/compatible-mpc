"""Rectangular grids and piecewise multi-affine (multilinear) interpolation.

A grid is a :class:`ddma.RectangularPartition` (product of 1-D edge arrays). A piecewise
multi-affine function on it is stored by its values at the grid vertices; inside a cell it
is the unique multi-affine function through the cell's 2^n vertex values
(Belta and Habets, 2006, Prop. 1). It is continuous across cells (EJC 2023, Prop. 4).
"""

from __future__ import annotations

import itertools

import numpy as np
from ddma import RectangularPartition

__all__ = ["RectangularPartition", "locate", "corner_offsets", "multilinear_weights",
           "vertex_shape", "interpolate"]


def vertex_shape(grid: RectangularPartition) -> tuple[int, ...]:
    """Shape of the array of vertex values: (n_cells_j + 1) along each axis."""
    return tuple(s + 1 for s in grid.shape)


def locate(grid: RectangularPartition, Z) -> np.ndarray:
    """Cell index (K, n) of each point; points outside the box are clipped to the edge cell."""
    Z = np.atleast_2d(np.asarray(Z, float))
    idx = np.empty(Z.shape, dtype=np.intp)
    for j, e in enumerate(grid.edges):
        idx[:, j] = np.clip(np.searchsorted(e, Z[:, j], side="right") - 1, 0, e.size - 2)
    return idx


def corner_offsets(n: int) -> np.ndarray:
    """The 2^n corners of a cell as 0/1 offsets, shape (2^n, n)."""
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.intp)


def multilinear_weights(grid: RectangularPartition, Z, idx=None):
    """Weights of the 2^n vertices of cell ``idx`` for the points ``Z``.

    Returns (weights (K, 2^n), vertex indices (K, 2^n, n)). The weights are those of
    Prop. 1 of Belta and Habets (2006); they are nonnegative and sum to 1 for points
    inside the cell. For points outside it they extrapolate the cell's multi-affine
    function (used by the windowed online update, ADHS 2024 Prop. 4).
    """
    Z = np.atleast_2d(np.asarray(Z, float))
    if idx is None:
        idx = locate(grid, Z)
    idx = np.atleast_2d(np.asarray(idx, dtype=np.intp))
    n = Z.shape[1]
    t = np.empty(Z.shape)
    for j, e in enumerate(grid.edges):
        lo, hi = e[idx[:, j]], e[idx[:, j] + 1]
        t[:, j] = (Z[:, j] - lo) / (hi - lo)
    C = corner_offsets(n)                                  # (2^n, n)
    w = np.where(C[None, :, :] == 1, t[:, None, :], 1.0 - t[:, None, :]).prod(axis=2)
    verts = idx[:, None, :] + C[None, :, :]
    return w, verts


def interpolate(grid: RectangularPartition, values: np.ndarray, Z) -> np.ndarray:
    """Evaluate the piecewise multi-affine function with vertex ``values`` at points ``Z``."""
    w, verts = multilinear_weights(grid, Z)
    return (w * values[tuple(np.moveaxis(verts, -1, 0))]).sum(axis=1)
