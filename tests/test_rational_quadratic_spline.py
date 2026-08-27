"""Tests for the rational-quadratic spline primitives."""

import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from bijax.rational_quadratic_spline import _RQSpline as RQS
from bijax.rational_quadratic_spline import spline_fwd, spline_inf

# A diagonal through (bin count x range x parameter scale), not the full cross product:
# the axes are independent, so one representative combination each is enough.
CASES = [
    (1, (-5.0, 5.0), 1.0),
    (2, (0.0, 1.0), 3.0),
    (4, (-1.0, 3.0), 0.1),
    (8, (-5.0, 5.0), 2.0),
    (16, (0.0, 1.0), 1.0),
    (32, (-1.0, 3.0), 1.0),
]
CASE = pytest.mark.parametrize(
    ("n_bins", "bounds", "scale"), CASES, ids=[f"bins{n}" for n, _, _ in CASES]
)
SHAPE = pytest.mark.parametrize(  # for tests where the parameter scale is irrelevant
    ("n_bins", "bounds"),
    [(n, b) for n, b, _ in CASES],
    ids=[f"bins{n}" for n, _, _ in CASES],
)
BOTH = pytest.mark.parametrize(
    "transform", [spline_fwd, spline_inv], ids=["fwd", "inv"]
)
MIN_SLOPE = 1e-3  # what tests that decode directly, but do not probe the slope, use


def _params(seed, n_bins, scale):
    return scale * jr.normal(jr.key(seed), (3 * n_bins - 1,))


def _span(bounds):
    return bounds[1] - bounds[0]


def _grid(n, bounds, pad_frac=0.0):
    pad = pad_frac * _span(bounds)
    return jnp.linspace(bounds[0] + pad, bounds[1] - pad, n)


def _sweep(transform, x, p, bounds):
    return jax.vmap(transform, in_axes=(0, None, None, None))(x, p, *bounds)


def _finite(tree):
    return all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(tree))


def _grads(transform, x, p, bounds):
    """d(out)/dx, d(out)/dparams, d(logdet)/dx, d(logdet)/dparams."""
    return [
        jax.grad(lambda a, b, i=i: transform(a, b, *bounds)[i], argnums=arg)(x, p)
        for i in (0, 1)
        for arg in (0, 1)
    ]


def _short_table_spline(n_bins, bounds, shortfall=1e-5):
    """Spline whose bin sizes sum to just under one, leaving a sliver below ``upper``."""
    sizes = jnp.full((n_bins,), (1.0 - shortfall) / n_bins)
    log_d = jnp.linspace(-1.0, 1.0, n_bins + 1)
    return RQS(
        k_ws=sizes,
        k_hs=sizes,
        log_k_dls=log_d[:-1],
        log_k_drs=log_d[1:],
        lower=bounds[0],
        upper=bounds[1],
    )


@CASE
@BOTH
def test_maps_the_range_onto_itself_monotonically(n_bins, bounds, scale, transform):
    for seed in range(4):
        out, ld = _sweep(
            transform, _grid(257, bounds), _params(seed, n_bins, scale), bounds
        )
        assert _finite((out, ld))
        assert jnp.allclose(out[0], bounds[0], atol=1e-5 * _span(bounds))
        assert jnp.allclose(out[-1], bounds[1], atol=1e-5 * _span(bounds))
        assert jnp.all(jnp.diff(out) > 0)


@CASE
def test_inverse_undoes_forward_and_negates_the_logdet(n_bins, bounds, scale):
    for seed in range(4):
        p = _params(seed, n_bins, scale)
        x = _grid(257, bounds)
        y, ld = _sweep(spline_fwd, x, p, bounds)
        xr, ldi = _sweep(spline_inv, y, p, bounds)
        assert jnp.allclose(xr, x, atol=1e-3 * _span(bounds))
        assert jnp.allclose(ld + ldi, 0.0, atol=2e-3)


@CASE
@BOTH
def test_logdet_matches_the_autodiff_derivative(n_bins, bounds, scale, transform):
    p = _params(6, n_bins, scale)
    x = _grid(41, bounds, pad_frac=0.02)
    slope = jax.vmap(jax.grad(lambda v: transform(v, p, *bounds)[0]))(x)
    _, ld = _sweep(transform, x, p, bounds)
    assert jnp.allclose(jnp.log(slope), ld, atol=1e-3)


@SHAPE
@BOTH
def test_zero_params_are_the_identity(n_bins, bounds, transform):
    x = _grid(101, bounds)
    out, ld = _sweep(transform, x, jnp.zeros(3 * n_bins - 1), bounds)
    assert jnp.allclose(out, x, atol=1e-5 * _span(bounds))
    assert jnp.allclose(ld, 0.0, atol=1e-5)


@CASE
@BOTH
def test_outside_the_range_is_the_identity_with_frozen_gradients(
    n_bins, bounds, scale, transform
):
    p = _params(3, n_bins, 10.0 * scale)
    span = _span(bounds)
    for x in (
        bounds[0] - 1e-3 * span,
        bounds[0] - span,
        bounds[1] + 1e-3 * span,
        bounds[1] + span,
    ):
        x = jnp.array(x)
        out, ld = transform(x, p, *bounds)
        assert out == x
        assert ld == 0.0
        d_out_dx, d_out_dp, d_ld_dx, d_ld_dp = _grads(transform, x, p, bounds)
        assert d_out_dx == 1.0 and jnp.all(d_out_dp == 0.0)
        assert d_ld_dx == 0.0 and jnp.all(d_ld_dp == 0.0)


@pytest.mark.parametrize(
    ("n_params", "min_slope"),
    [(n, 1e-3) for n in (0, 1, 3, 4, 6, 9, 10)]
    + [(11, s) for s in (0.0, 1.0, -0.5, 2.0)],
)
def test_rejects_invalid_params(n_params, min_slope):
    with pytest.raises(ValueError):
        spline_fwd(jnp.array(0.0), jnp.zeros(n_params), -5.0, 5.0, min_slope)


@CASE
@pytest.mark.parametrize("min_slope", [1e-4, 1e-3, 1e-2])
def test_decoded_bins_partition_the_range_without_degeneracy(
    n_bins, bounds, scale, min_slope
):
    spline = RQS.decode(
        _params(9, n_bins, 100.0 * scale),
        min_slope=min_slope,
        lower=bounds[0],
        upper=bounds[1],
    )
    for sizes in (spline.k_ws, spline.k_hs):
        assert jnp.allclose(sizes.sum(), 1.0, atol=1e-5)
        assert jnp.all(sizes >= min_slope / n_bins)
    for knots in spline.bounds():
        assert knots[0] == bounds[0]
        assert jnp.allclose(knots[-1], bounds[1], atol=1e-5 * _span(bounds))
        assert jnp.all(jnp.diff(knots) > 0)
    d = jnp.exp(jnp.concatenate([spline.log_k_dls, spline.log_k_drs]))
    assert jnp.all(d >= min_slope) and jnp.all(d <= 1.0 / min_slope)


@CASE
def test_param_slots_map_to_widths_heights_and_derivatives(n_bins, bounds, scale):
    p = _params(7, n_bins, scale)
    knots = RQS.decode(p, MIN_SLOPE, bounds[0], bounds[1]).bounds()

    def shift(slot):
        moved = RQS.decode(p.at[slot].add(5.0), MIN_SLOPE, bounds[0], bounds[1])
        return [
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(moved.bounds(), knots, strict=True)
        ]

    dx_from_width, dy_from_width = shift(0)
    dx_from_height, dy_from_height = shift(1)
    assert dy_from_width == 0.0 and dx_from_height == 0.0
    if n_bins > 1:  # a single bin spans the whole range whatever its params
        assert shift(2) == [0.0, 0.0]  # derivative slots never move the bin table
        assert min(dx_from_width, dy_from_height) > 0.01 * _span(bounds)


def test_omitting_the_bounds_gives_a_working_spline():
    p, x = _params(0, 8, 1.0), jnp.array(0.25)
    y, ld = spline_fwd(x, p)
    xr, ldi = spline_inv(y, p)
    assert _finite((y, ld, xr, ldi))
    assert jnp.allclose(xr, x, atol=1e-3)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-3)


@BOTH
def test_jit_and_vmap_agree_with_the_unbatched_call(transform):
    n_bins, bounds = 8, (-5.0, 5.0)
    p = jnp.stack([_params(s, n_bins, 1.0) for s in range(4)])
    x = _grid(4, (-6.0, 6.0))
    batched = jax.jit(
        jax.vmap(transform, in_axes=(0, 0, None, None)), static_argnums=(2, 3)
    )
    got = batched(x, p, *bounds)
    want = [transform(x[i], p[i], *bounds) for i in range(4)]
    for g, w in zip(got, zip(*want, strict=True), strict=True):
        assert jnp.allclose(g, jnp.stack(w), atol=1e-5)


@CASE
@BOTH
def test_continuous_across_the_range_boundary(n_bins, bounds, scale, transform):
    p = _params(4, n_bins, 10.0 * scale)
    eps = 1e-5 * _span(bounds)
    for edge in bounds:
        inner, _ = transform(jnp.array(edge - eps), p, *bounds)
        outer, _ = transform(jnp.array(edge + eps), p, *bounds)
        assert jnp.abs(outer - inner) < 1e-3 * _span(bounds)


@CASE
@BOTH
def test_gradients_are_finite_exactly_on_the_knots(n_bins, bounds, scale, transform):
    p = _params(12, n_bins, 3.0 * scale)
    x_knots, y_knots = RQS.decode(p, MIN_SLOPE, bounds[0], bounds[1]).bounds()
    knots = x_knots if transform is spline_fwd else y_knots
    grads = jax.vmap(_grads, in_axes=(None, 0, None, None))(transform, knots, p, bounds)
    assert _finite(grads)


@CASE
@BOTH
def test_values_and_gradients_are_finite_for_extreme_params(
    n_bins, bounds, scale, transform
):
    p = _params(11, n_bins, 50.0 * scale)
    span = _span(bounds)
    x = _grid(401, (bounds[0] - span, bounds[1] + span))
    assert _finite(_sweep(transform, x, p, bounds))
    assert _finite(
        jax.vmap(_grads, in_axes=(None, 0, None, None))(transform, x, p, bounds)
    )


@pytest.mark.parametrize("n_bins", [n for n, _, _ in CASES])
def test_inverse_gradients_are_finite_when_a_bin_is_linear(n_bins):
    assert _finite(
        _grads(spline_inv, jnp.array(0.25), jnp.zeros(3 * n_bins - 1), (-5.0, 5.0))
    )


@SHAPE
@pytest.mark.parametrize("at", ["ulp_below_upper", "midway"])
def test_finite_and_near_identity_in_the_sliver_above_the_last_knot(n_bins, bounds, at):
    spline = _short_table_spline(n_bins, bounds)
    top = float(spline.bounds()[0][-1])
    assert top < bounds[1], "bin table must stop short of the upper bound"
    upper = jnp.float32(bounds[1])
    x = (
        jnp.nextafter(upper, jnp.float32(0.0))
        if at == "ulp_below_upper"
        else (top + upper) / 2
    )
    for name in ("fwd_logdydx", "inv_logdxdy"):
        evaluate = getattr(type(spline), name)
        out, ld = evaluate(spline, x)
        assert jnp.allclose(out, x, atol=1e-5 * _span(bounds))
        assert jnp.allclose(ld, 0.0, atol=1e-5)
        assert _finite(jax.grad(lambda v, f=evaluate: f(spline, v)[1] ** 2)(x))
        assert _finite(jax.grad(lambda s, f=evaluate: f(s, x)[1] ** 2)(spline))


@CASE
def test_float64_roundtrip_is_exact_for_extreme_params(n_bins, bounds, scale):
    with jax.enable_x64():
        for seed in range(4):
            p = _params(seed, n_bins, 10.0 * scale)
            x = _grid(1001, bounds)
            y, ld = _sweep(spline_fwd, x, p, bounds)
            xr, ldi = _sweep(spline_inv, y, p, bounds)
            assert jnp.all(jnp.diff(y) > 0)
            assert jnp.max(jnp.abs(xr - x)) < 1e-7 * _span(bounds)
            assert jnp.max(jnp.abs(ld + ldi)) < 1e-6
