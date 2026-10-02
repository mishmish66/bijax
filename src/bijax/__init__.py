"""Bijax: a tiny equinox-only library of composable flow primitives.

Every bijector exposes ``fwd_logdet(x, c=None, *, rng=None)`` and
``inv_logdet(y, c=None, *, rng=None)``, each returning the mapped value and a
scalar log-determinant. Structures (`Coupling`, `MAF`, `IAF`) combine an
`Elementwise` transform with a user-supplied conditioner.
"""

__version__ = "0.1.4"
__docformat__ = "numpy"

from .autoregressive import IAF, MAF
from .causal_linear import CausalLinear
from .coupling import Coupling
from .elementwise import RQS, Affine, Elementwise
from .plu import PLU
from .rational_quadratic_spline import rqs_fwd, rqs_inv

__all__ = [
    # bijectors
    "PLU",
    "Coupling",
    "MAF",
    "IAF",
    # elementwise transforms
    "Elementwise",
    "Affine",
    "RQS",
    # spline primitives
    "rqs_fwd",
    "rqs_inv",
    # conditioner building blocks
    "CausalLinear",
]
