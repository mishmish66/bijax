"""Bijax: a tiny equinox-only library of neural bijections.

Each bijector exposes an ``fwd_logdet``/``inv_logdet`` interface and can be
combined with a base density to define a flow model.
"""

__version__ = "0.1.3"
__docformat__ = "numpy"

from .causal_mlp import CausalLinear, CausalMLP
from .coupling_aff import AffineCoupling
from .coupling_nsf import SplineCoupling
from .maf import ARAffine
from .mansf import ARSpline
from .plu import PLU
from .spline import spline_fwd, spline_inv

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
    "spline_fwd",
    "spline_inv",
]
