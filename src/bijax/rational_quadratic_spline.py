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


def _knot_distance(
    θ: Float[Array, ""],
    θc: Float[Array, ""],
    s: Float[Array, ""],
    d: Float[Array, ""],
    dc: Float[Array, ""],
) -> Float[Array, ""]:
    """Fraction of a bin's width between a knot and the x that lies ``θ`` from it in y.

    ``θc`` is the y distance to the bin's other knot, ``s`` the bin's mean slope,
    and ``d`` and ``dc`` the slopes at this knot and the other. Solves eqs 29-32 of
    Neural Spline Flows with the coefficients written in ``θ`` and ``θc``.
    """
    a = θc * (s - d) + θ * (dc - s)
    b = θc * d + θ * (2 * s - dc)
    c = -s * θ
    bsqm4ac = b**2 - 4 * a * c
    safe_sqin = jnp.where(bsqm4ac > 0, bsqm4ac, 1.0)
    disc = jnp.where(bsqm4ac > 0, jnp.sqrt(safe_sqin), 0.0)
    bsign = jnp.where(b >= 0, 1.0, -1.0)
    q = -0.5 * (b + bsign * disc)
    a_safe = jax.lax.select(b >= 0, 1.0, a)
    return jnp.clip(jax.lax.select(b >= 0, c / q, q / a_safe), 0.0, 1.0)


@jaxtyped(typechecker=beartype)
class _RQSpline(eqx.Module):
    """Rational Quadratic Spline."""

    k_ws: Float[Array, " k"]
    k_hs: Float[Array, " k"]
    log_k_dls: Float[Array, " k"]
    log_k_drs: Float[Array, " k"]
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    @staticmethod
    def decode(
        p: Float[Array, " p"],
        min_bin_size: float,
        min_knot_slope: float,
        lower: float,
        upper: float,
    ) -> "_RQSpline":
        if len(p) % 3 != 1 or len(p) < 7:
            msg = (
                f"param length must be 3*B+1 for B >= 2 bins, got {len(p)}; "
                "layout is B widths, B heights, B+1 knot derivatives"
            )
            raise ValueError(msg)
        if not 0.0 < min_knot_slope < 1.0:
            msg = f"min_knot_slope must be in (0, 1), got {min_knot_slope}"
            raise ValueError(msg)
        if not 0.0 < min_bin_size < 1.0:
            msg = f"min_bin_size must be in (0, 1), got {min_bin_size}"
            raise ValueError(msg)

        nbin = len(p) // 3
        if min_bin_size * nbin >= upper - lower:
            msg = (
                f"min_bin_size {min_bin_size} leaves no span for {nbin} bins "
                f"in range ({lower}, {upper}); it must be below "
                f"{(upper - lower) / nbin}"
            )
            raise ValueError(msg)

        raw_ds, raw_ws, raw_hs = p[::3], p[1::3], p[2::3]
        # From distrax. Offset exactly makes slope=1 when raw_slope=0
        offset = jnp.log(jnp.expm1(1.0 - min_knot_slope))
        log_ds = jnp.log(jax.nn.softplus(raw_ds + offset) + min_knot_slope)
        # Softmax over the span left once every bin has taken min_bin_size
        safe_ws = raw_ws / (1.0 + jnp.abs(2 * raw_ws / jnp.log(min_knot_slope)))
        safe_hs = raw_hs / (1.0 + jnp.abs(2 * raw_hs / jnp.log(min_knot_slope)))
        wf = (
            jax.nn.softmax(safe_ws) * ((upper - lower) - min_bin_size * nbin)
            + min_bin_size
        )
        hf = (
            jax.nn.softmax(safe_hs) * ((upper - lower) - min_bin_size * nbin)
            + min_bin_size
        )

        return _RQSpline(
            k_ws=wf,
            k_hs=hf,
            log_k_dls=log_ds[:-1],
            log_k_drs=log_ds[1:],
            lower=lower,
            upper=upper,
        )

    def fwd_logdydx(
        self, x: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        n = self.k_ws.shape[0]
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
        s = (yrb - ylb) / (xrb - xlb)
        log_dl, log_dr = self.log_k_dls[k], self.log_k_drs[k]
        dl, dr = jnp.exp(log_dl), jnp.exp(log_dr)
        ζomζ = (ζ := (x - xlb) / (xrb - xlb)) * (omζ := (xrb - x) / (xrb - xlb))

        den = s + (dl + dr - 2 * s) * ζomζ
        y = ylb + (yrb - ylb) * ((s * ζ**2 + dl * ζomζ) / den)  # eq 19
        ld = (  # eq 22
            2 * jnp.log(s)
            + jnp.log(dr * ζ**2 + 2 * s * ζomζ + dl * omζ**2)
            - 2 * jnp.log(den)
        )

        # apply tails
        y = jax.lax.select(x_inside, y, xu)
        ld = jax.lax.select(x_inside, ld, 0.0)
        return y, ld

    def bounds(self) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
        """Knot positions; both tables end at one shared knot no higher than upper."""
        bin_xs = self.lower + jnp.concat([jnp.zeros(1), jnp.cumsum(self.k_ws)])
        bin_ys = self.lower + jnp.concat([jnp.zeros(1), jnp.cumsum(self.k_hs)])
        top = jnp.minimum(bin_xs[-1], self.upper)
        return bin_xs.at[-1].set(top), bin_ys.at[-1].set(top)

    def inv_logdxdy(self, y: Float[Array, ""]):
        n = self.k_hs.shape[0]
        bin_x_bounds, bin_y_bounds = self.bounds()
        ku = jnp.searchsorted(bin_y_bounds, y, side="right") - 1

        y_inside = (ku >= 0) & (ku < n)
        yu = y  # store unclipped value
        y = y.clip(bin_y_bounds[0], bin_y_bounds[-1])
        k = ku.clip(0, n - 1)

        log_dl, log_dr = self.log_k_dls[k], self.log_k_drs[k]
        xlb, xrb = bin_x_bounds[k], bin_x_bounds[k + 1]
        ylb, yrb = bin_y_bounds[k], bin_y_bounds[k + 1]
        s = (yrb - ylb) / (xrb - xlb)
        dl, dr = jnp.exp(log_dl), jnp.exp(log_dr)

        # ζ and 1 - ζ, each solved from the knot it is the distance from
        ζl = _knot_distance(y - ylb, yrb - y, s, dl, dr)
        ζr = _knot_distance(yrb - y, y - ylb, s, dr, dl)
        from_left = ζl <= ζr
        ζ = jnp.where(from_left, ζl, 1 - ζr)
        omζ = jnp.where(from_left, 1 - ζl, ζr)
        ζomζ = ζ * omζ

        x = jnp.where(from_left, xlb + ζ * (xrb - xlb), xrb - omζ * (xrb - xlb))
        ld = (
            -2 * jnp.log(s)
            - jnp.log(dr * ζ**2 + 2 * s * ζomζ + dl * omζ**2)
            + 2 * jnp.log(s + (dl + dr - 2 * s) * ζomζ)
        )
        # apply tails
        x = jax.lax.select(y_inside, x, yu)
        ld = jax.lax.select(y_inside, ld, 0.0)
        return x, ld


def rqs_fwd(
    x: Float[Array, ""],
    params: Float[Array, " p"],
    lower: float = -5.0,
    upper: float = 5.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Evaluate rational quadratic spline.

    Rational quadratic splines from Durkan et al. are a parametric constant time
    invertible transform from R to R with desirable stability properties. Params
    is a block of parameters designed for a neural network to output, they are
    mostly in log space and every real set of parameters will generate a valid
    spline.

    Parameters
    ----------
    x : Float[Array, ""]
        Input to transform.
    params : Float[Array, " p"]
        ``3*B + 1`` reals for ``B >= 2`` bins, interleaved as
        ``d_0, w_0, h_0, d_1, ..., w_{B-1}, h_{B-1}, d_B``: knot derivatives
        ``d`` (including both range ends), bin widths ``w`` and heights ``h``.
    lower : float
        Lower limit of spline beyond which the transform is linear
    upper : float
        Upper bound of spline beyond which the transform is linear
    min_bin_size : float
        Bin size constraint to prevent unstable tiny bins.
    min_knot_slope : float
        Slope constraint for the slope parameters to keep them valid.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        the y this spline maps from the input x.

    Examples
    --------
    FIXME: Add docs.

    """
    spline = _RQSpline.decode(
        params,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
        lower=lower,
        upper=upper,
    )
    return spline.fwd_logdydx(x)


def rqs_inv(
    y: Float[Array, ""],
    params: Float[Array, " p"],
    lower: float = -5.0,
    upper: float = 5.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Invert rational quadratic spline.

    Rational quadratic splines from Durkan et al. are a parametric constant time
    invertible transform from R to R with desirable stability properties. Params
    is a block of parameters designed for a neural network to output, they are
    mostly in log space and every real set of parameters will generate a valid
    spline.

    Parameters
    ----------
    y : Float[Array, ""]
        Output for inversion.
    params : Float[Array, " p"]
        ``3*B + 1`` reals for ``B >= 2`` bins, interleaved as
        ``d_0, w_0, h_0, d_1, ..., w_{B-1}, h_{B-1}, d_B``: knot derivatives
        ``d`` (including both range ends), bin widths ``w`` and heights ``h``.
    lower : float
        Lower limit of spline beyond which the transform is linear
    upper : float
        Upper bound of spline beyond which the transform is linear
    min_bin_size : float
        Bin size constraint to prevent unstable tiny bins.
    min_knot_slope : float
        Slope constraint for the slope parameters to keep them valid.

    Returns
    -------
    tuple[Float[Array, ""], Float[Array, ""]]
        the input to this spline producing the passed output y.

    Examples
    --------
    FIXME: Add docs.

    """
    spline = _RQSpline.decode(
        params,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
        lower=lower,
        upper=upper,
    )
    return spline.inv_logdxdy(y)
