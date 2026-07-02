"""Monotone rational-quadratic spline primitives, following Neural Spline Flows.

@inproceedings{durkan_neural_2019,
    title = {Neural {Spline} {Flows}},
    url = {https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html},
    booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 32 ({NeurIPS} 2019)},
    author = {Durkan, Conor and Bekasov, Artur and Murray, Iain and Papamakarios, George},
    editor = {Wallach, Hanna M. and Larochelle, Hugo and Beygelzimer, Alina and d'Alché-Buc, Florence and Fox, Emily B. and Garnett, Roman},
    year = {2019},
    pages = {7509--7520},
}
"""

from jaxtyping import Float, Array

import jax
from jax import numpy as jnp
import equinox as eqx


class RationalQuadratic(eqx.Module):
    """Monotone rational-quadratic segment on [0,1] -> [0,1]"""

    e0: Float[Array, ""]  # derivative at 0
    e1: Float[Array, ""]  # derivative at 1

    def fwd(self, z: Float[Array, ""]) -> Float[Array, ""]:
        zc = z * (1 - z)
        return (z**2 + self.e0 * zc) / (1 + (self.e0 + self.e1 - 2) * zc)

    def dydx(self, z: Float[Array, ""]) -> Float[Array, ""]:
        zc = z * (1 - z)
        num = self.e1 * z**2 + 2 * zc + self.e0 * (1 - z) ** 2
        den = (1 + (self.e0 + self.e1 - 2) * zc) ** 2
        return num / den

    def inverse(self, w: Float[Array, ""]) -> Float[Array, ""]:
        D = self.e0 + self.e1 - 2
        a = 1 - self.e0 + w * D
        b = self.e0 - w * D
        c = -w
        return 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))


class RationalQuadraticSpline(eqx.Module):
    """Rational Quadratic Spline"""

    k_x0s: Float[Array, " k"]
    k_x1s: Float[Array, " k"]
    k_y0s: Float[Array, " k"]
    k_y1s: Float[Array, " k"]
    k_d0s: Float[Array, " k"]
    k_d1s: Float[Array, " k"]

    @staticmethod
    def decode(
        p: Float[Array, " p"], min_slope: float = 1e-3
    ) -> "RationalQuadraticSpline":
        if len(p) % 3 != 2:
            msg = (
                f"param length must be 3*B - 1 for B bins, got {len(p)}; "
                "layout is B widths, B heights, B-1 interior derivatives"
            )
            raise ValueError(msg)
        if not 0.0 < min_slope < 1.0:
            msg = f"min_slope must be in (0, 1), got {min_slope}"
            raise ValueError(msg)

        raw_ws, raw_hs, raw_ds = jnp.array_split(p, 3)
        log_ds = raw_ds / (1 + jnp.abs(raw_ds / jnp.log(min_slope)))

        safe_ws = raw_ws / (1 + jnp.abs(2 * raw_ws / jnp.log(min_slope)))
        safe_hs = raw_hs / (1 + jnp.abs(2 * raw_hs / jnp.log(min_slope)))

        ws = jax.nn.softmax(safe_ws)
        hs = jax.nn.softmax(safe_hs)
        k_xs = jnp.concat([jnp.zeros((1,)), jnp.cumsum(ws)])
        k_ys = jnp.concat([jnp.zeros((1,)), jnp.cumsum(hs)])
        k_ds = jnp.concat([jnp.ones((1,)), jnp.exp(log_ds), jnp.ones((1,))])

        return RationalQuadraticSpline(
            k_x0s=k_xs[:-1],
            k_x1s=k_xs[1:],
            k_y0s=k_ys[:-1],
            k_y1s=k_ys[1:],
            k_d0s=k_ds[:-1],
            k_d1s=k_ds[1:],
        )

    def fwd_logdet(self, x: Float[Array, ""]):
        x_inside = (x < 1) & (x > 0)
        xu = x  # Store unclipped value
        x = x.clip(0.0, 1.0)

        n = self.k_x0s.shape[0]
        k = jnp.clip(jnp.searchsorted(self.k_x0s, x, side="right") - 1, 0, n - 1)
        x0, x1 = self.k_x0s[k], self.k_x1s[k]
        y0, y1 = self.k_y0s[k], self.k_y1s[k]
        s = (y1 - y0) / (x1 - x0)
        seg = RationalQuadratic(self.k_d0s[k] / s, self.k_d1s[k] / s)
        z = (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * seg.fwd(z)
        ld = jnp.log(s * seg.dydx(z))
        # apply tails
        y = jax.lax.select(x_inside, y, xu)
        ld = jax.lax.select(x_inside, ld, 0.0)
        return y, ld

    def inv_logdet(self, y: Float[Array, ""]):
        y_inside = (y < 1) & (y > 0)
        yu = y  # store unclipped value
        y = y.clip(0.0, 1.0)

        n = self.k_y0s.shape[0]
        k = jnp.clip(jnp.searchsorted(self.k_y0s, y, side="right") - 1, 0, n - 1)
        x0, x1 = self.k_x0s[k], self.k_x1s[k]
        y0, y1 = self.k_y0s[k], self.k_y1s[k]
        s = (y1 - y0) / (x1 - x0)
        seg = RationalQuadratic(self.k_d0s[k] / s, self.k_d1s[k] / s)
        w = (y - y0) / (y1 - y0)
        z = seg.inverse(w)
        x = x0 + z * (x1 - x0)
        slope = s * seg.dydx(z)
        # rounding in y is amplified by 1/slope; refuse to return an inverse
        # that has lost more than ~3 decimal digits of the input range
        msg = (
            "spline inverse is ill-conditioned (near-flat segment); "
            "use float64 or a larger min_slope"
        )
        x = eqx.error_if(x, y_inside & (slope < 1e3 * jnp.finfo(x.dtype).eps), msg)
        ld = -jnp.log(slope)
        # apply tails
        x = jax.lax.select(y_inside, x, yu)
        ld = jax.lax.select(y_inside, ld, 0.0)
        return x, ld


def spline_fwd(
    x: Float[Array, ""],
    params: Float[Array, " p"],
    low: Float[Array, " #l"],
    high: Float[Array, " #h"],
    min_slope: float = 1e-3,
):
    spline = RationalQuadraticSpline.decode(params, min_slope=min_slope)
    irange = high - low
    yu, ld = spline.fwd_logdet((x - low) / irange)
    return yu * irange + low, ld


def spline_inv(
    y: Float[Array, ""],
    params: Float[Array, " p"],
    low: Float[Array, "  #l"],
    high: Float[Array, " #h"],
    min_slope: float = 1e-3,
):
    spline = RationalQuadraticSpline.decode(params, min_slope=min_slope)
    irange = high - low
    xu, ld = spline.inv_logdet((y - low) / irange)
    return xu * irange + low, ld
