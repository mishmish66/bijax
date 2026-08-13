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

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


class RationalQuadratic(eqx.Module):
    """Monotone rational-quadratic segment on [0,1] -> [0,1]"""

    le0: Float[Array, ""]  # log derivative at 0
    le1: Float[Array, ""]  # log derivative at 1

    def fwd(self, z: Float[Array, ""]) -> Float[Array, ""]:
        zc = z * (1 - z)
        e0, e1 = jnp.exp(self.le0), jnp.exp(self.le1)
        return (z**2 + e0 * zc) / (1 + (e0 + e1 - 2) * zc)

    def log_dydx(self, z: Float[Array, ""]) -> Float[Array, ""]:
        zc = z * (1 - z)

        # The weird `logsumexp` is for numerical stability
        num_lse_arg = [self.le1, 0, self.le0]
        num_lse_scl = [z**2, 2 * zc, (1 - z) ** 2]
        log_num = jax.nn.logsumexp(a=jnp.array(num_lse_arg), b=jnp.array(num_lse_scl))

        e0, e1 = jnp.exp(self.le0), jnp.exp(self.le1)
        log_den = 2 * jnp.log1p((e0 + e1 - 2) * zc)
        return log_num - log_den

    def inverse(self, w: Float[Array, ""]) -> Float[Array, ""]:
        e0, e1 = jnp.exp(self.le0), jnp.exp(self.le1)
        D = e0 + e1 - 2
        a = 1 - e0 + w * D
        b = e0 - w * D
        c = -w
        return 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))


class RationalQuadraticSpline(eqx.Module):
    """Rational Quadratic Spline"""

    log_k_ws: Float[Array, " k"]
    log_k_hs: Float[Array, " k"]
    log_k_d0s: Float[Array, " k"]
    log_k_d1s: Float[Array, " k"]

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

        log_ws = jax.nn.log_softmax(safe_ws)
        log_hs = jax.nn.log_softmax(safe_hs)
        log_ds = jnp.concat([jnp.zeros((1,)), log_ds, jnp.zeros((1,))])

        return RationalQuadraticSpline(
            log_k_ws=log_ws,
            log_k_hs=log_hs,
            log_k_d0s=log_ds[:-1],
            log_k_d1s=log_ds[1:],
        )

    def fwd_logdet(self, x: Float[Array, ""]):
        x_inside = (x < 1) & (x > 0)
        xu = x  # Store unclipped value
        x = x.clip(0.0, 1.0)

        n = self.log_k_ws.shape[0]

        bin_x_bounds = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_ws))])
        bin_y_bounds = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_hs))])
        k = jnp.clip(jnp.searchsorted(bin_x_bounds, x, side="right") - 1, 0, n - 1)
        bin_xlb = bin_x_bounds[k]
        bin_ylb = bin_y_bounds[k]
        bin_log_w = self.log_k_ws[k]
        bin_log_h = self.log_k_hs[k]
        log_s = bin_log_h - bin_log_w
        seg = RationalQuadratic(self.log_k_d0s[k] - log_s, self.log_k_d1s[k] - log_s)
        z = (x - bin_xlb) / jnp.exp(bin_log_w)
        y = bin_ylb + jnp.exp(bin_log_h) * seg.fwd(z)
        ld = log_s + seg.log_dydx(z)
        # apply tails
        y = jax.lax.select(x_inside, y, xu)
        ld = jax.lax.select(x_inside, ld, 0.0)
        return y, ld

    def inv_logdet(self, y: Float[Array, ""]):
        y_inside = (y < 1) & (y > 0)
        yu = y  # store unclipped value
        y = y.clip(0.0, 1.0)

        n = self.log_k_hs.shape[0]

        bin_x_bounds = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_ws))])
        bin_y_bounds = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_hs))])
        k = jnp.clip(jnp.searchsorted(bin_y_bounds, y, side="right") - 1, 0, n - 1)
        bin_log_w = self.log_k_ws[k]
        bin_log_h = self.log_k_hs[k]
        bin_xlb = bin_x_bounds[k]
        bin_ylb = bin_y_bounds[k]
        log_s = bin_log_h - bin_log_w
        seg = RationalQuadratic(self.log_k_d0s[k] - log_s, self.log_k_d1s[k] - log_s)
        w = (y - bin_ylb) / jnp.exp(bin_log_h)
        z = seg.inverse(w)
        x = bin_xlb + z * jnp.exp(bin_log_w)
        log_slope = log_s + seg.log_dydx(z)
        # rounding in y is amplified by 1/slope; refuse to return an inverse
        # that has lost more than ~3 decimal digits of the input range
        msg = (
            "spline inverse is ill-conditioned (near-flat segment); "
            "use float64 or a larger min_slope"
        )
        x = eqx.error_if(
            x,
            y_inside & (log_slope < jnp.log(1e3) + jnp.log(jnp.finfo(x.dtype).eps)),
            msg,
        )
        ld = -log_slope
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
