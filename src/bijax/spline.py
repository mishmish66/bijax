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
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
class _RationalQuadraticSpline(eqx.Module):
    log_k_ws: Float[Array, " k"]
    log_k_hs: Float[Array, " k"]
    log_k_dls: Float[Array, " k"]
    log_k_drs: Float[Array, " k"]
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    @staticmethod
    def decode(
        p: Float[Array, " p"],
        min_slope: float,
        lower: float,
        upper: float,
    ) -> "_RationalQuadraticSpline":
        if len(p) % 3 != 2:
            msg = (
                f"param length must be 3*B - 1 for B bins, got {len(p)}; "
                "layout is B widths, B heights, B-1 interior derivatives"
            )
            raise ValueError(msg)
        if not 0.0 < min_slope < 1.0:
            msg = f"min_slope must be in (0, 1), got {min_slope}"
            raise ValueError(msg)

        raw_ws, raw_hs, raw_ds = p[::3], p[1::3], p[2::3]
        log_ds = raw_ds / (1.0 + jnp.abs(raw_ds / jnp.log(min_slope)))

        safe_ws = raw_ws / (1.0 + jnp.abs(2 * raw_ws / jnp.log(min_slope)))
        safe_hs = raw_hs / (1.0 + jnp.abs(2 * raw_hs / jnp.log(min_slope)))

        log_wf = jax.nn.log_softmax(safe_ws)
        log_hf = jax.nn.log_softmax(safe_hs)
        log_ds = jnp.concat([jnp.zeros((1,)), log_ds, jnp.zeros((1,))])

        return _RationalQuadraticSpline(
            log_k_ws=log_wf,
            log_k_hs=log_hf,
            log_k_dls=log_ds[:-1],
            log_k_drs=log_ds[1:],
            lower=lower,
            upper=upper,
        )

    def fwd_logdydx(
        self, x: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        n = self.log_k_ws.shape[0]
        bin_x_bounds, bin_y_bounds = self.bounds()
        ku = (
            jnp.searchsorted(bin_x_bounds, x, side="right") - 1
        )  # offset the prepended 0

        x_inside = (ku >= 0) & (ku < n)
        xu = x  # Store unclipped value
        x = x.clip(bin_x_bounds[0], bin_x_bounds[-1])
        k = ku.clip(0, n - 1)

        xlb, xrb = bin_x_bounds[k], bin_x_bounds[k + 1]
        ylb, yrb = bin_y_bounds[k], bin_y_bounds[k + 1]
        # log_s and s may disagree. s went through cumsum and log_s didn't.
        # preserving log_s makes the slope math stay in log space
        log_s, s = self.log_k_hs[k] - self.log_k_ws[k], (yrb - ylb) / (xrb - xlb)
        log_dl, log_dr = self.log_k_dls[k], self.log_k_drs[k]
        dl, dr = jnp.exp(log_dl), jnp.exp(log_dr)
        ζomζ = (ζ := (x - xlb) / (xrb - xlb)) * (omζ := (xrb - x) / (xrb - xlb))

        den = s + (dl + dr - 2 * s) * ζomζ
        y = ylb + (yrb - ylb) * ((s * ζ**2 + dl * ζomζ) / den)  # eq 19
        ld = (  # eq 22
            2 * log_s
            + jnp.log(dr * ζ**2 + 2 * s * ζomζ + dl * omζ**2)
            - 2 * jnp.log(den)
        )

        # apply tails
        y = jax.lax.select(x_inside, y, xu)
        ld = jax.lax.select(x_inside, ld, 0.0)
        return y, ld

    def bounds(self) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
        span = self.upper - self.lower
        bin_x_01 = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_ws))])
        bin_y_01 = jnp.concat([jnp.zeros(1), jnp.cumsum(jnp.exp(self.log_k_hs))])
        return bin_x_01 * span + self.lower, bin_y_01 * span + self.lower

    def inv_logdxdy(self, y: Float[Array, ""]):
        n = self.log_k_hs.shape[0]
        bin_x_bounds, bin_y_bounds = self.bounds()
        ku = jnp.searchsorted(bin_y_bounds, y, side="right") - 1

        y_inside = (ku >= 0) & (ku < n)
        yu = y  # store unclipped value
        y = y.clip(bin_y_bounds[0], bin_y_bounds[-1])
        k = ku.clip(0, n - 1)

        log_dl, log_dr = self.log_k_dls[k], self.log_k_drs[k]
        xlb, xrb = bin_x_bounds[k], bin_x_bounds[k + 1]
        ylb, yrb = bin_y_bounds[k], bin_y_bounds[k + 1]
        # log_s and s may disagree. s went through cumsum and log_s didn't.
        # preserving log_s makes the slope math stay in log space
        log_s, s = self.log_k_hs[k] - self.log_k_ws[k], (yrb - ylb) / (xrb - xlb)
        dl, dr = jnp.exp(log_dl), jnp.exp(log_dr)

        # eqs 29, 30, 31, 32
        a = (yrb - ylb) * (s - dl) + (y - ylb) * (dr + dl - 2 * s)
        b = (yrb - ylb) * dl - (y - ylb) * (dr + dl - 2 * s)
        c = -s * (y - ylb)
        disc = jnp.sqrt(jnp.maximum(b**2 - 4 * a * c, 0.0))
        q = -0.5 * (b + jnp.sign(b) * disc)
        a_safe = jax.lax.select(b > 0, 1.0, a)
        ζomζ = (ζ := jax.lax.select(b > 0, c / q, q / a_safe)) * (omζ := 1 - ζ)

        x = xlb + ζ * (xrb - xlb)
        ld = (
            -2 * log_s
            - jnp.log(dr * ζ**2 + 2 * s * ζomζ + dl * omζ**2)
            + 2 * jnp.log(s + (dl + dr - 2 * s) * ζomζ)
        )
        # apply tails
        x = jax.lax.select(y_inside, x, yu)
        ld = jax.lax.select(y_inside, ld, 0.0)
        return x, ld


def spline_fwd(
    x: Float[Array, ""],
    params: Float[Array, " p"],
    lower: float = -5.0,
    upper: float = 5.0,
    min_slope: float = 1e-3,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Evaluate rational quadratic spline.

    Rational quadratic splines from Durkan et al. are a parametric
    constant time invertible transform from R to R with desirable
    stability properties. Params is a block of parameters designed for
    a neural network to output, they are mostly in log space and every
    real set of parameters will generate a valid spline.

    Parameters
    ----------
    y : Float[Array, ""]
        Output for inversion.
    params : Float[Array, " p"]
        Array of real parameters.
    lower : float
        Lower limit of spline beyond which the transform is linear
    upper : float
        Upper bound of spline beyond which the transform is linear
    min_slope : float
        a slope constraint for the slope parameters to keep them
        valid.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        the y this spline maps from the input x.

    Examples
    --------
    FIXME: Add docs.

    """
    spline = _RationalQuadraticSpline.decode(
        params, min_slope=min_slope, lower=lower, upper=upper
    )
    return spline.fwd_logdydx(x)


def spline_inv(
    y: Float[Array, ""],
    params: Float[Array, " p"],
    lower: float = -5.0,
    upper: float = 5.0,
    min_slope: float = 1e-3,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Invert rational quadratic spline.

    Rational quadratic splines from Durkan et al. are a parametric
    constant time invertible transform from R to R with desirable
    stability properties. Params is a block of parameters designed for
    a neural network to output, they are mostly in log space and every
    real set of parameters will generate a valid spline.

    Parameters
    ----------
    y : Float[Array, ""]
        Output for inversion.
    params : Float[Array, " p"]
        Array of real parameters.
    lower : float
        Lower limit of spline beyond which the transform is linear
    upper : float
        Upper bound of spline beyond which the transform is linear
    min_slope : float
        a slope constraint for the slope parameters to keep them
        valid.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        the input to this spline producing the passed output y.

    Examples
    --------
    FIXME: Add docs.

    """
    spline = _RationalQuadraticSpline.decode(
        params, min_slope=min_slope, lower=lower, upper=upper
    )
    return spline.inv_logdxdy(y)
