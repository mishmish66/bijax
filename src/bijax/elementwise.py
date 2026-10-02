"""Scalar bijections parameterized by a vector of reals."""

import abc

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float

from bijax.rational_quadratic_spline import rqs_fwd, rqs_inv

Scalar = Float[Array, ""]
Params = Float[Array, " p"]


class Elementwise(eqx.Module):
    """A scalar bijection whose shape is set by ``n_params`` unconstrained reals."""

    @property
    @abc.abstractmethod
    def n_params(self) -> int:
        """Length of the parameter vector."""

    @abc.abstractmethod
    def fwd(self, x: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Map ``x`` to ``y`` and return ``log|dy/dx|``."""

    @abc.abstractmethod
    def inv(self, y: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Map ``y`` to ``x`` and return ``log|dx/dy|``."""


class Affine(Elementwise):
    """``y = x * exp(s) + t`` for ``p = (s_raw, t)``.

    ``s`` is ``s_raw`` soft-clamped into ``(log min_scale, -log min_scale)``.
    """

    min_scale: float = eqx.field(static=True, default=1e-3)

    @property
    def n_params(self) -> int:
        """Log-scale and shift."""
        return 2

    def _log_scale(self, s_raw: Scalar) -> Scalar:
        return s_raw / (1 + jnp.abs(s_raw / jnp.log(self.min_scale)))

    def fwd(self, x: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Scale then shift ``x``."""
        s = self._log_scale(p[0])
        return x * jnp.exp(s) + p[1], s

    def inv(self, y: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Unshift then unscale ``y``."""
        s = self._log_scale(p[0])
        return (y - p[1]) * jnp.exp(-s), -s


class RQS(Elementwise):
    """Rational-quadratic spline on ``[lower, upper]``, the identity outside it.

    See `bijax.rqs_fwd` for the parameter layout.
    """

    n_bins: int = eqx.field(static=True)
    lower: float = eqx.field(static=True, default=-5.0)
    upper: float = eqx.field(static=True, default=5.0)
    min_bin_size: float = eqx.field(static=True, default=1e-4)
    min_knot_slope: float = eqx.field(static=True, default=1e-4)

    @property
    def n_params(self) -> int:
        """``B`` widths, ``B`` heights and ``B + 1`` knot derivatives."""
        return 3 * self.n_bins + 1

    def fwd(self, x: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Evaluate the spline."""
        return rqs_fwd(
            x, p, self.lower, self.upper, self.min_bin_size, self.min_knot_slope
        )

    def inv(self, y: Scalar, p: Params) -> tuple[Scalar, Scalar]:
        """Invert the spline."""
        return rqs_inv(
            y, p, self.lower, self.upper, self.min_bin_size, self.min_knot_slope
        )
