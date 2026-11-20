"""compmpc: safe learning-based MPC with the compatible models approach.

Two models are learned from the same data: a guaranteed (set-valued) model that certifies
safety, and a compatible estimate inside it that drives an MPC. See the README.
"""

from .bounds import GrowthMap, IntervalModel, intersect, monotone_split_model
from .estimate import CompatibleEstimate, vertex_bounds
from .grid import RectangularPartition, interpolate, multilinear_weights
from .mpc import CompatibleMPC, MPCResult
from .safety import PlanarGrid, PlanarSafetyController, wrap

__version__ = "0.1.0"

__all__ = [
    "GrowthMap", "IntervalModel", "intersect", "monotone_split_model", "CompatibleEstimate",
    "vertex_bounds", "RectangularPartition", "interpolate", "multilinear_weights", "CompatibleMPC",
    "MPCResult", "PlanarGrid", "PlanarSafetyController", "wrap",
]
