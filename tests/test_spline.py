"""Tests for the rational-quadratic spline primitives."""

import jax
from jax import numpy as jnp, random as jr
import pytest

from bijax.spline import RationalQuadratic, RationalQuadraticSpline, spline_fwd

N_BINS = 8


def _random_segment(key):
    le0, le1 = 0.5 * jr.normal(key, (2,))
    return RationalQuadratic(le0=le0, le1=le1)


def _random_params(key, n_bins=N_BINS):
    return jr.normal(key, (3 * n_bins - 1,))


def test_segment_maps_unit_endpoints():
    seg = RationalQuadratic(le0=jnp.log(jnp.array(1.3)), le1=jnp.log(jnp.array(0.7)))
    assert jnp.allclose(seg.fwd(jnp.array(0.0)), 0.0, atol=1e-6)
    assert jnp.allclose(seg.fwd(jnp.array(1.0)), 1.0, atol=1e-6)


def test_segment_roundtrip():
    for s in range(5):
        seg = _random_segment(jr.key(s))
        z = jnp.linspace(0.01, 0.99, 25)
        w = jax.vmap(seg.fwd)(z)
        zr = jax.vmap(seg.inverse)(w)
        assert jnp.allclose(zr, z, atol=1e-4)


def test_segment_log_dydx_matches_autodiff():
    seg = _random_segment(jr.key(0))
    for z in jnp.linspace(0.05, 0.95, 10):
        ad = jax.grad(lambda z: seg.fwd(z))(z)
        assert jnp.allclose(jnp.log(ad), seg.log_dydx(z), atol=1e-4)


def test_segment_monotone():
    seg = _random_segment(jr.key(1))
    y = jax.vmap(seg.fwd)(jnp.linspace(0.0, 1.0, 50))
    assert jnp.all(jnp.diff(y) > 0)


@pytest.mark.parametrize("bad_len", [9, 10])
def test_decode_rejects_bad_param_length(bad_len):
    with pytest.raises(ValueError):
        RationalQuadraticSpline.decode(jnp.zeros(bad_len))


def test_decode_accepts_valid_length():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(0)))
    ws, hs = jnp.exp(spline.log_k_ws), jnp.exp(spline.log_k_hs)
    # bins are positive and partition [x0, x0+1] x [y0, y0+1]
    assert jnp.all(ws > 0) and jnp.all(hs > 0)
    assert jnp.allclose(ws.sum(), 1.0, atol=1e-5)
    assert jnp.allclose(hs.sum(), 1.0, atol=1e-5)


def test_spline_maps_unit_interval():
    # the cumulative knots must span [0,1] on both axes, so the last bin ends
    # exactly at the identity tails
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(0)))
    y_lo, _ = spline.fwd_logdet(jnp.array(1e-6))
    y_hi, _ = spline.fwd_logdet(jnp.array(1.0 - 1e-6))
    assert jnp.allclose(y_lo, 0.0, atol=1e-5)
    assert jnp.allclose(y_hi, 1.0, atol=1e-5)


def test_spline_roundtrip_inside_unit():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(3)))
    x = jnp.linspace(0.01, 0.99, 50)
    y, ld = jax.vmap(spline.fwd_logdet)(x)
    xr, ldi = jax.vmap(spline.inv_logdet)(y)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_spline_logdet_matches_autodiff():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(4)))
    for x in jnp.linspace(0.05, 0.95, 12):
        dy = jax.grad(lambda x: spline.fwd_logdet(x)[0])(x)
        _, ld = spline.fwd_logdet(x)
        assert jnp.allclose(jnp.log(dy), ld, atol=1e-4)


def test_spline_tails_are_identity():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(5)))
    for x in [jnp.array(-2.0), jnp.array(3.5)]:
        y, ld = spline.fwd_logdet(x)
        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0, atol=1e-6)


def test_spline_is_monotone_on_unit():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(6)))
    y, _ = jax.vmap(spline.fwd_logdet)(jnp.linspace(0.0, 1.0, 100))
    assert jnp.all(jnp.diff(y) > 0)


@pytest.mark.parametrize("bad_slope", [0.0, 1.0])
def test_decode_rejects_bad_min_slope(bad_slope):
    # min_slope >= 1 makes log(min_slope) >= 0 and the soft-clamps NaN out
    with pytest.raises(ValueError):
        RationalQuadraticSpline.decode(_random_params(jr.key(0)), min_slope=bad_slope)


def test_tail_gradients_are_zero_and_finite():
    # outside [low, high] the transform is the identity: gradients w.r.t.
    # params must be exactly zero (no NaN leaking through the select) and the
    # gradient w.r.t. x must be 1.
    p = 10.0 * _random_params(jr.key(7))
    low, high = jnp.array(-5.0), jnp.array(5.0)
    for xt in [jnp.array(-7.0), jnp.array(7.0)]:
        g = jax.grad(lambda p: spline_fwd(xt, p, low, high)[0])(p)
        assert jnp.all(g == 0.0)
        gx = jax.grad(lambda x: spline_fwd(x, p, low, high)[0])(xt)
        assert jnp.allclose(gx, 1.0)


def test_spline_is_continuous_at_range_boundaries():
    # boundary derivatives are pinned to 1, so the spline must meet the
    # identity tails without a jump
    p = 10.0 * _random_params(jr.key(8))
    low, high = jnp.array(-5.0), jnp.array(5.0)
    eps = 1e-3
    for b in [-5.0, 5.0]:
        ya, _ = spline_fwd(jnp.array(b - eps), p, low, high)
        yb, _ = spline_fwd(jnp.array(b + eps), p, low, high)
        assert jnp.abs(yb - ya) < 0.05


def test_spline_roundtrip_extreme_params_float64():
    # large raw params make wide, near-flat bins; in float64 the algorithm
    # must pick the right bin and invert tightly (bin-selection bugs would
    # otherwise hide inside float32 conditioning noise)
    with jax.enable_x64():
        for s in range(5):
            spline = RationalQuadraticSpline.decode(10.0 * _random_params(jr.key(s)))
            x = jnp.linspace(1e-4, 1 - 1e-4, 1001)
            y, ld = jax.vmap(spline.fwd_logdet)(x)
            xr, ldi = jax.vmap(spline.inv_logdet)(y)
            assert jnp.max(jnp.abs(xr - x)) < 1e-8
            assert jnp.max(jnp.abs(ld + ldi)) < 1e-8


def test_spline_inverse_errors_when_ill_conditioned():
    # float32 inverses lose most of their precision in wide near-flat
    # segments (y - y0 cancels catastrophically); the eqx.error_if guard must
    # turn the silently wrong result into a loud failure
    spline = RationalQuadraticSpline.decode(10.0 * _random_params(jr.key(12)))
    x = jnp.linspace(1e-4, 1 - 1e-4, 1001)
    y, _ = jax.vmap(spline.fwd_logdet)(x)
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        jax.block_until_ready(jax.vmap(spline.inv_logdet)(y))
