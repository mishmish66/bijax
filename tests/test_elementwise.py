"""Tests for the elementwise transforms and the Elementwise contract."""

import equinox as eqx
import jax
import pytest
from conditioners import MLP, MaskedAffine
from jax import numpy as jnp
from jax import random as jr

from bijax import (
    IAF,
    MAF,
    RQS,
    Affine,
    Coupling,
    Elementwise,
    rqs_fwd,
    rqs_inv,
)

TRANSFORMS = pytest.mark.parametrize(
    "transform",
    [
        pytest.param(Affine(), id="affine"),
        pytest.param(Affine(min_scale=1e-2), id="affine-tight"),
        pytest.param(RQS(8), id="rqs"),
        pytest.param(RQS(4, lower=0.0, upper=1.0), id="rqs-unit"),
    ],
)


def _x(transform, n=33):
    lo, hi = (
        (transform.lower, transform.upper) if isinstance(transform, RQS) else (-3, 3)
    )
    return jnp.linspace(lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo), n)


def _params(transform, seed=0):
    return jr.normal(jr.key(seed), (transform.n_params,))


@pytest.mark.parametrize("n_bins", [2, 8, 32])
def test_rqs_takes_widths_heights_and_every_knot_derivative(n_bins):
    assert RQS(n_bins).n_params == 3 * n_bins + 1


def test_affine_takes_a_log_scale_and_a_shift():
    assert Affine().n_params == 2


@TRANSFORMS
def test_inverse_undoes_forward_and_negates_the_logdet(transform):
    p, x = _params(transform), _x(transform)
    y, ld = jax.vmap(transform.fwd, in_axes=(0, None))(x, p)
    xr, ldi = jax.vmap(transform.inv, in_axes=(0, None))(y, p)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@TRANSFORMS
def test_logdet_matches_the_autodiff_derivative(transform):
    p, x = _params(transform, 1), _x(transform)
    slope = jax.vmap(jax.grad(lambda v: transform.fwd(v, p)[0]))(x)
    _, ld = jax.vmap(transform.fwd, in_axes=(0, None))(x, p)
    assert jnp.allclose(jnp.log(slope), ld, atol=1e-3)


@TRANSFORMS
def test_zero_params_are_the_identity(transform):
    x = _x(transform)
    y, ld = jax.vmap(transform.fwd, in_axes=(0, None))(x, jnp.zeros(transform.n_params))
    assert jnp.allclose(y, x, atol=1e-5)
    assert jnp.allclose(ld, 0.0, atol=1e-5)


@pytest.mark.parametrize("min_scale", [1e-3, 1e-1])
@pytest.mark.parametrize("s_raw", [-1e4, -3.0, 3.0, 1e4])
def test_affine_log_scale_stays_inside_the_min_scale_bound(min_scale, s_raw):
    _, ld = Affine(min_scale).fwd(jnp.array(0.5), jnp.array([s_raw, 0.0]))
    assert jnp.abs(ld) < -jnp.log(min_scale)


def test_rqs_is_the_spline_functions_with_its_config():
    t = RQS(6, lower=-2.0, upper=3.0, min_bin_size=1e-3, min_knot_slope=1e-2)
    p, x = _params(t, 2), jnp.array(0.7)
    config = (t.lower, t.upper, t.min_bin_size, t.min_knot_slope)
    for got, want in (
        (t.fwd(x, p), rqs_fwd(x, p, *config)),
        (t.inv(x, p), rqs_inv(x, p, *config)),
    ):
        assert jnp.array_equal(got[0], want[0]) and jnp.array_equal(got[1], want[1])


def test_elementwise_is_abstract():
    with pytest.raises(TypeError):
        Elementwise()


class _Shift(Elementwise):
    @property
    def n_params(self):
        return 1

    def fwd(self, x, p):
        return x + p[0], jnp.zeros(())

    def inv(self, y, p):
        return y - p[0], jnp.zeros(())


@pytest.mark.parametrize("structure", ["coupling", MAF, IAF])
def test_a_user_defined_transform_works_in_every_structure(structure):
    t = _Shift()
    if structure == "coupling":
        m = Coupling(MLP(2, (2, 1), rng=jr.key(0)), t, (0, 2), (1, 3))
    else:
        m = structure(MaskedAffine(4, 1, rng=jr.key(0)), t)
    x = jr.normal(jr.key(1), (4,))
    y, ld = eqx.filter_jit(lambda m, x: m.fwd_logdet(x))(m, x)
    xr, _ = m.inv_logdet(y)
    assert not jnp.allclose(y, x)
    assert jnp.allclose(xr, x, atol=1e-5)
    assert ld == 0.0
