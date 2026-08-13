"""Tests for ARSpline's `direction` flag.

"maf" -> fwd_logdet = _slow, inv_logdet = _fast
"iaf" -> fwd_logdet = _fast, inv_logdet = _slow
"""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr
from jax.flatten_util import ravel_pytree

from bijax.causal_mlp import CausalMLP
from bijax.mansf import ARSpline

N_BINS = 8
DIRECTIONS = ["maf", "iaf"]
FAST_METHOD = {"maf": "inv_logdet", "iaf": "fwd_logdet"}
SLOW_METHOD = {"maf": "fwd_logdet", "iaf": "inv_logdet"}


def make_arspline(
    key,
    dim: int = 3,
    width: int = 32,
    depth: int = 2,
    cond_dim: int | None = None,
    direction: str = "maf",
    **kwargs,
) -> ARSpline:
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=3 * N_BINS - 1,
        width=width,
        depth=depth,
        cond_dim=cond_dim,
        rng=key,
    )
    return ARSpline(net=net, direction=direction, **kwargs)


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def test_default_direction_is_maf():
    assert make_arspline(jr.key(0)).direction == "maf"


def test_direction_is_a_static_field():
    m = make_arspline(jr.key(0))
    _, static = eqx.partition(m, eqx.is_inexact_array)
    assert static.direction == "maf"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_fast_path_is_one_net_call_slow_path_is_num_ranks_plus_one(
    monkeypatch, direction
):
    dim = 4
    m = make_arspline(jr.key(0), dim=dim, direction=direction)
    x = jr.normal(jr.key(1), (dim,))

    calls = []
    original = CausalMLP.__call__
    monkeypatch.setattr(
        CausalMLP,
        "__call__",
        lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1],
    )

    calls.clear()
    getattr(m, FAST_METHOD[direction])(x)
    assert len(calls) == 1

    calls.clear()
    getattr(m, SLOW_METHOD[direction])(x)
    assert len(calls) == dim + 1


def test_each_direction_is_the_other_wired_backwards():
    maf = make_arspline(jr.key(0), direction="maf")
    iaf = make_arspline(jr.key(0), direction="iaf")
    x = jr.normal(jr.key(1), (3,))

    for a, b in (
        (maf.fwd_logdet(x), iaf.inv_logdet(x)),
        (maf.inv_logdet(x), iaf.fwd_logdet(x)),
    ):
        assert jnp.array_equal(a[0], b[0])
        assert jnp.array_equal(a[1], b[1])


def test_direction_changes_the_composite_function():
    maf = make_arspline(jr.key(0), direction="maf")
    iaf = make_arspline(jr.key(0), direction="iaf")
    x = jr.normal(jr.key(1), (3,))
    assert not jnp.allclose(maf.fwd_logdet(x)[0], iaf.fwd_logdet(x)[0])


@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_unknown_direction_raises(method):
    """Validation may live in the constructor or in the call."""
    x = jr.normal(jr.key(1), (3,))
    with pytest.raises(ValueError):
        m = make_arspline(jr.key(0), direction="MAF")
        getattr(m, method)(x)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_roundtrip_both_ways(direction, dim):
    m = make_arspline(jr.key(0), dim=dim, direction=direction)
    x = jr.normal(jr.key(1), (dim,))

    z, ldf = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ldf + ldi, 0.0, atol=1e-4)

    w, ldi2 = m.inv_logdet(x)
    xr2, ldf2 = m.fwd_logdet(w)
    assert jnp.allclose(xr2, x, atol=1e-4)
    assert jnp.allclose(ldi2 + ldf2, 0.0, atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_logdet_matches_autodiff_jacobian(direction, method):
    dim = 3
    m = make_arspline(jr.key(0), dim=dim, direction=direction)
    x = jr.normal(jr.key(2), (dim,))
    f = getattr(m, method)
    _, ld = f(x)
    J = jax.jacobian(lambda v: f(v)[0])(x)
    assert jnp.allclose(ld, jnp.log(jnp.abs(jnp.linalg.det(J))), atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_jacobian_is_lower_triangular(direction, method):
    dim = 4
    m = make_arspline(jr.key(0), dim=dim, direction=direction)
    x = jr.normal(jr.key(3), (dim,))
    J = jax.jacobian(lambda v: getattr(m, method)(v)[0])(x)
    assert jnp.all(jnp.triu(jnp.abs(J), k=1) < 1e-6)
    assert jnp.all(jnp.abs(jnp.diag(J)) > 0)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_monotone_in_each_coordinate(direction):
    m = make_arspline(jr.key(0), dim=3, direction=direction)
    x = jr.normal(jr.key(4), (3,))
    J = jax.jacobian(lambda v: m.fwd_logdet(v)[0])(x)
    assert jnp.all(jnp.diag(J) > 0)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_conditional_roundtrip(direction):
    dim, cdim = 3, 4
    m = make_arspline(jr.key(0), dim=dim, cond_dim=cdim, direction=direction)
    x = jr.normal(jr.key(1), (dim,))
    c = jr.normal(jr.key(2), (cdim,))
    z, ld = m.fwd_logdet(x, c)
    xr, ldi = m.inv_logdet(z, c)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_every_coordinate_depends_on_the_condition(direction):
    dim, cdim = 3, 4
    m = make_arspline(jr.key(0), dim=dim, cond_dim=cdim, direction=direction)
    x = jr.normal(jr.key(1), (dim,))
    z1 = m.fwd_logdet(x, jr.normal(jr.key(11), (cdim,)))[0]
    z2 = m.fwd_logdet(x, jr.normal(jr.key(12), (cdim,)))[0]
    for i in range(dim):
        assert not jnp.allclose(z1[i], z2[i]), f"coordinate {i} ignores c"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_custom_domain_roundtrip(direction):
    m = make_arspline(jr.key(0), direction=direction, low=0.0, high=1.0)
    x = jr.uniform(jr.key(1), (3,), minval=0.05, maxval=0.95)
    z, ld = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.all((z > 0.0) & (z < 1.0))
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_outside_the_domain_is_the_identity_with_zero_logdet(direction):
    m = make_arspline(jr.key(0), direction=direction, low=0.0, high=1.0)
    x = jnp.array([-3.0, 2.5, 7.0])
    z, ld = m.fwd_logdet(x)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(ld, 0.0, atol=1e-6)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_works_under_jit(direction):
    m = make_arspline(jr.key(0), direction=direction)
    x = jr.normal(jr.key(1), (3,))
    z, ld = eqx.filter_jit(lambda m, v: m.fwd_logdet(v))(m, x)
    xr, ldi = eqx.filter_jit(lambda m, v: m.inv_logdet(v))(m, z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_works_under_vmap(direction):
    m = make_arspline(jr.key(0), direction=direction)
    xs = jr.normal(jr.key(1), (16, 3))
    zs, lds = jax.vmap(m.fwd_logdet)(xs)
    xrs, ldis = jax.vmap(m.inv_logdet)(zs)
    assert zs.shape == xs.shape and lds.shape == (16,)
    assert jnp.allclose(xrs, xs, atol=1e-4)
    assert jnp.allclose(lds + ldis, 0.0, atol=1e-4)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_slow_path_gradient_is_finite(direction):
    m = make_arspline(jr.key(0), direction=direction)
    x = jr.normal(jr.key(1), (3,))
    g = jax.grad(lambda v: getattr(m, SLOW_METHOD[direction])(v)[1])(x)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_slow_path_logdet_gradient_matches_finite_differences(x64, direction):
    """Autodiff through the unrolled solve must be exact."""
    dim = 3
    m = make_arspline(jr.key(0), dim=dim, direction=direction)
    x = jr.uniform(jr.key(1), (dim,), minval=-2.0, maxval=2.0)
    slow_name = SLOW_METHOD[direction]

    arrays, static = eqx.partition(m, eqx.is_inexact_array)
    flat, unflat = ravel_pytree(arrays)

    def logdet(theta):
        return getattr(eqx.combine(unflat(theta), static), slow_name)(x)[1]

    grad = jax.grad(logdet)(flat)

    h = 1e-6
    for j in range(0, flat.size, max(flat.size // 12, 1)):
        e = jnp.zeros_like(flat).at[j].set(h)
        fd = (logdet(flat + e) - logdet(flat - e)) / (2 * h)
        assert jnp.allclose(grad[j], fd, rtol=1e-5, atol=1e-7), (
            f"param {j}: autodiff {grad[j]} vs finite difference {fd}"
        )


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_roundtrip_logdets_cancel_for_every_parameter_value(x64, direction):
    m = make_arspline(jr.key(0), dim=3, direction=direction)
    x = jr.uniform(jr.key(1), (3,), minval=-2.0, maxval=2.0)
    arrays, static = eqx.partition(m, eqx.is_inexact_array)
    flat, unflat = ravel_pytree(arrays)

    def total(theta):
        model = eqx.combine(unflat(theta), static)
        z, ldf = model.fwd_logdet(x)
        _, ldi = model.inv_logdet(z)
        return ldf + ldi

    assert jnp.allclose(total(flat), 0.0, atol=1e-8)
    assert jnp.allclose(jnp.linalg.norm(jax.grad(total)(flat)), 0.0, atol=1e-6)
