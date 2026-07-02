"""Mini Flows: a tiny equinox-only library of neural bijections.

Each bijector exposes an ``fwd_logdet``/``inv_logdet`` interface and is meant to
be combined with a base density to define a flow model.
"""

from .causal_mlp import CausalLinear, CausalMLP
from .coupling_aff import AffineCoupling
from .coupling_nsf import SplineCoupling
from .maf import ARAffine
from .mansf import ARSpline
from .plu import PLU
from .spline import (
    RationalQuadratic,
    RationalQuadraticSpline,
    spline_fwd,
    spline_inv,
)

__version__ = "0.1.1"

__all__ = [
    # supporting characters
    "CausalLinear",
    "CausalMLP",
    # bijectors
    "PLU",
    "AffineCoupling",
    "SplineCoupling",
    "ARAffine",
    "ARSpline",
    # spline primitives
    "RationalQuadratic",
    "RationalQuadraticSpline",
    "spline_fwd",
    "spline_inv",
]
