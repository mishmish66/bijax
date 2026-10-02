"""Tests for the rational-quadratic spline primitives."""

import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from bijax.rational_quadratic_spline import _RQSpline, rqs_fwd, rqs_inv

# A diagonal through (bin count x range x parameter scale), not the full cross product:
# the axes are independent, so one representative combination each is enough.
CASES = [
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
BOTH = pytest.mark.parametrize("transform", [rqs_fwd, rqs_inv], ids=["fwd", "inv"])
# what tests that decode directly, but do not probe the constraints, use
MIN_BIN_SIZE = 1e-4
MIN_KNOT_SLOPE = 1e-4


def _params(seed, n_bins, scale):
    return scale * jr.normal(jr.key(seed), (3 * n_bins + 1,))


def _span(bounds):
    return bounds[1] - bounds[0]


def _decode(p, bounds, min_bin_size=MIN_BIN_SIZE, min_knot_slope=MIN_KNOT_SLOPE):
    return _RQSpline.decode(p, min_bin_size, min_knot_slope, bounds[0], bounds[1])


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
    """Spline whose bins stop short of ``upper``, leaving a sliver above the table."""
    sizes = jnp.full((n_bins,), (_span(bounds) - shortfall) / n_bins)
    log_d = jnp.linspace(-1.0, 1.0, n_bins + 1)
    return _RQSpline(
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
        y, ld = _sweep(rqs_fwd, x, p, bounds)
        xr, ldi = _sweep(rqs_inv, y, p, bounds)
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
    out, ld = _sweep(transform, x, jnp.zeros(3 * n_bins + 1), bounds)
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
    ("n_params", "min_bin_size", "min_knot_slope"),
    # 4 would be a single bin
    [(n, MIN_BIN_SIZE, MIN_KNOT_SLOPE) for n in (0, 1, 2, 3, 4, 5, 6, 8, 9, 11)]
    + [(13, s, MIN_KNOT_SLOPE) for s in (0.0, 1.0, -0.5, 2.0)]
    + [(13, MIN_BIN_SIZE, s) for s in (0.0, 1.0, -0.5, 2.0)],
)
def test_rejects_invalid_params(n_params, min_bin_size, min_knot_slope):
    with pytest.raises(ValueError):
        rqs_fwd(
            jnp.array(0.0),
            jnp.zeros(n_params),
            -5.0,
            5.0,
            min_bin_size,
            min_knot_slope,
        )


@pytest.mark.parametrize("n_bins", [2, 4, 8, 16, 32])
def test_rejects_a_bin_size_floor_that_cannot_fit_the_range(n_bins):
    bounds = (0.0, 1.0)  # a floor inside (0, 1) can still overshoot a unit range
    with pytest.raises(ValueError):
        rqs_fwd(
            jnp.array(0.5),
            jnp.zeros(3 * n_bins + 1),
            *bounds,
            1.5 / n_bins,
            MIN_KNOT_SLOPE,
        )


@CASE
@pytest.mark.parametrize("min_bin_size", [1e-4, 1e-3, 1e-2])
def test_decoded_bins_partition_the_range_without_degeneracy(
    n_bins, bounds, scale, min_bin_size
):
    span = _span(bounds)
    spline = _decode(
        _params(9, n_bins, 100.0 * scale), bounds, min_bin_size=min_bin_size
    )
    for sizes in (spline.k_ws, spline.k_hs):
        assert jnp.allclose(sizes.sum(), span, atol=1e-5 * span)
        assert jnp.all(sizes >= min_bin_size)
    for knots in spline.bounds():
        assert knots[0] == bounds[0]
        assert jnp.allclose(knots[-1], bounds[1], atol=1e-5 * span)
        assert jnp.all(jnp.diff(knots) > 0)


@CASE
@pytest.mark.parametrize("min_knot_slope", [1e-4, 1e-3, 1e-2])
def test_decoded_knot_slopes_respect_the_floor(n_bins, bounds, scale, min_knot_slope):
    spline = _decode(
        _params(9, n_bins, 100.0 * scale), bounds, min_knot_slope=min_knot_slope
    )
    log_d = jnp.concatenate([spline.log_k_dls, spline.log_k_drs])
    assert _finite(log_d)
    assert jnp.all(log_d >= jnp.log(min_knot_slope))


@SHAPE
def test_zero_params_decode_to_unit_knot_slopes(n_bins, bounds):
    spline = _decode(jnp.zeros(3 * n_bins + 1), bounds)
    d = jnp.exp(jnp.concatenate([spline.log_k_dls, spline.log_k_drs]))
    assert jnp.allclose(d, 1.0, atol=1e-5)


@CASE
def test_param_slots_map_to_derivatives_widths_and_heights(n_bins, bounds, scale):
    p = _params(7, n_bins, scale)
    knots = _decode(p, bounds).bounds()

    def shift(slot):  # largest move per table, excluding the shared last knot
        moved = _decode(p.at[slot].add(5.0), bounds)
        return [
            float(jnp.max(jnp.abs(a[:-1] - b[:-1])))
            for a, b in zip(moved.bounds(), knots, strict=True)
        ]

    dx_from_width, dy_from_width = shift(1)
    dx_from_height, dy_from_height = shift(2)
    assert dy_from_width == 0.0 and dx_from_height == 0.0
    for slot in (0, 3, 3 * n_bins):  # derivative slots never move the bin table
        assert shift(slot) == [0.0, 0.0]
    assert min(dx_from_width, dy_from_height) > 0.01 * _span(bounds)


@CASE
def test_tables_share_a_last_knot_no_higher_than_upper(n_bins, bounds, scale):
    for seed in range(8):
        x_knots, y_knots = _decode(_params(seed, n_bins, 10.0 * scale), bounds).bounds()
        assert x_knots[-1] == y_knots[-1]
        assert x_knots[-1] <= bounds[1]


@CASE
def test_the_upper_bound_roundtrips_with_cancelling_logdets(n_bins, bounds, scale):
    x = jnp.array(bounds[1])
    for seed in range(8):
        p = _params(seed, n_bins, 10.0 * scale)
        y, ld = rqs_fwd(x, p, *bounds)
        xr, ldi = rqs_inv(y, p, *bounds)
        assert jnp.allclose(xr, x, atol=1e-6 * _span(bounds))
        assert jnp.allclose(ld + ldi, 0.0, atol=1e-6)


@CASE
def test_end_slots_set_the_slopes_at_the_range_ends(n_bins, bounds, scale):
    p = _params(8, n_bins, scale).at[0].set(2.0).at[-1].set(-2.0)
    spline = _decode(p, bounds)
    log_d_lo, log_d_hi = spline.log_k_dls[0], spline.log_k_drs[-1]
    assert not jnp.allclose(log_d_lo, 0.0) and not jnp.allclose(log_d_hi, 0.0)
    x_knots = spline.bounds()[0]
    _, ld_lo = rqs_fwd(x_knots[0], p, *bounds)
    _, ld_hi = rqs_fwd(jnp.nextafter(x_knots[-1], x_knots[0]), p, *bounds)
    assert jnp.allclose(ld_lo, log_d_lo, atol=1e-4)
    assert jnp.allclose(ld_hi, log_d_hi, atol=1e-4)


def test_omitting_the_bounds_gives_a_working_spline():
    p, x = _params(0, 8, 1.0), jnp.array(0.25)
    y, ld = rqs_fwd(x, p)
    xr, ldi = rqs_inv(y, p)
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
    with jax.enable_x64():
        p = _params(4, n_bins, 10.0 * scale)
        eps = 1e-12 * _span(bounds)
        for edge in bounds:
            inner, _ = transform(jnp.array(edge - eps), p, *bounds)
            outer, _ = transform(jnp.array(edge + eps), p, *bounds)
            assert jnp.abs(outer - inner) < 1e-3 * _span(bounds)


@CASE
@BOTH
def test_gradients_are_finite_exactly_on_the_knots(n_bins, bounds, scale, transform):
    p = _params(12, n_bins, 3.0 * scale)
    x_knots, y_knots = _decode(p, bounds).bounds()
    knots = x_knots if transform is rqs_fwd else y_knots
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
        _grads(rqs_inv, jnp.array(0.25), jnp.zeros(3 * n_bins + 1), (-5.0, 5.0))
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
            y, ld = _sweep(rqs_fwd, x, p, bounds)
            xr, ldi = _sweep(rqs_inv, y, p, bounds)
            assert jnp.all(jnp.diff(y) > 0)
            assert jnp.max(jnp.abs(xr - x)) < 1e-7 * _span(bounds)
            assert jnp.max(jnp.abs(ld + ldi)) < 1e-6
