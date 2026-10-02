"""Tests for the MAF and IAF layers over each elementwise transform."""

import equinox as eqx
import jax
import pytest
from conditioners import MaskedAffine, Noisy
from jax import numpy as jnp
from jax import random as jr
from jax.flatten_util import ravel_pytree

from bijax import IAF, MAF, RQS, Affine

LAYERS = pytest.mark.parametrize("layer", [MAF, IAF], ids=["maf", "iaf"])
TRANSFORMS = pytest.mark.parametrize(
    "transform",
    [pytest.param(Affine(), id="affine"), pytest.param(RQS(8), id="rqs")],
)
METHODS = pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
FAST_METHOD = {MAF: "inv_logdet", IAF: "fwd_logdet"}
SLOW_METHOD = {MAF: "fwd_logdet", IAF: "inv_logdet"}


def _make(transform, dim=3, cond_dim=None, layer=MAF, seed=0):
    return layer(
        MaskedAffine(dim, transform.n_params, cond_dim, rng=jr.key(seed)), transform
    )


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


@TRANSFORMS
@LAYERS
def test_fast_path_is_one_conditioner_call_slow_path_is_dim_plus_one(
    monkeypatch, transform, layer
):
    dim = 4
    m = _make(transform, dim=dim, layer=layer)
    x = jr.normal(jr.key(1), (dim,))

    calls = []
    original = MaskedAffine.__call__
    monkeypatch.setattr(
        MaskedAffine,
        "__call__",
        lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1],
    )

    calls.clear()
    getattr(m, FAST_METHOD[layer])(x)
    assert len(calls) == 1

    calls.clear()
    getattr(m, SLOW_METHOD[layer])(x)
    assert len(calls) == dim + 1


@TRANSFORMS
def test_maf_is_iaf_wired_backwards(transform):
    maf = _make(transform, layer=MAF)
    iaf = _make(transform, layer=IAF)
    x = jr.normal(jr.key(1), (3,))
    for a, b in (
        (maf.fwd_logdet(x), iaf.inv_logdet(x)),
        (maf.inv_logdet(x), iaf.fwd_logdet(x)),
    ):
        assert jnp.array_equal(a[0], b[0])
        assert jnp.array_equal(a[1], b[1])


@TRANSFORMS
def test_maf_and_iaf_are_different_maps(transform):
    maf = _make(transform, layer=MAF)
    iaf = _make(transform, layer=IAF)
    x = jr.normal(jr.key(1), (3,))
    assert not jnp.allclose(maf.fwd_logdet(x)[0], iaf.fwd_logdet(x)[0])


@TRANSFORMS
@LAYERS
@METHODS
def test_rejects_a_conditioner_output_of_the_wrong_shape(transform, layer, method):
    net = MaskedAffine(3, transform.n_params + 1, rng=jr.key(0))
    with pytest.raises(ValueError):
        getattr(layer(net, transform), method)(jr.normal(jr.key(1), (3,)))


class _NeedsRng(eqx.Module):
    inner: MaskedAffine

    def __call__(self, x, c, *, rng):
        return self.inner(x, c)


@LAYERS
@METHODS
def test_conditioner_is_always_given_rng(layer, method):
    m = layer(_NeedsRng(MaskedAffine(3, 2, rng=jr.key(0))), Affine())
    getattr(m, method)(jr.normal(jr.key(1), (3,)))


@TRANSFORMS
@LAYERS
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_roundtrip_both_ways(transform, layer, dim):
    m = _make(transform, dim=dim, layer=layer)
    x = jr.normal(jr.key(1), (dim,))

    z, ldf = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ldf + ldi, 0.0, atol=1e-4)

    w, ldi2 = m.inv_logdet(x)
    xr2, ldf2 = m.fwd_logdet(w)
    assert jnp.allclose(xr2, x, atol=1e-4)
    assert jnp.allclose(ldi2 + ldf2, 0.0, atol=1e-4)


@TRANSFORMS
@LAYERS
@METHODS
def test_logdet_matches_autodiff_jacobian(transform, layer, method):
    m = _make(transform, layer=layer)
    x = jr.normal(jr.key(2), (3,))
    f = getattr(m, method)
    _, ld = f(x)
    J = jax.jacobian(lambda v: f(v)[0])(x)
    assert jnp.allclose(ld, jnp.log(jnp.abs(jnp.linalg.det(J))), atol=1e-4)


@TRANSFORMS
@LAYERS
@METHODS
def test_jacobian_is_lower_triangular(transform, layer, method):
    m = _make(transform, dim=4, layer=layer)
    x = jr.normal(jr.key(3), (4,))
    J = jax.jacobian(lambda v: getattr(m, method)(v)[0])(x)
    assert jnp.all(jnp.triu(jnp.abs(J), k=1) < 1e-6)
    assert jnp.all(jnp.abs(jnp.diag(J)) > 0)


@TRANSFORMS
@LAYERS
def test_monotone_in_each_coordinate(transform, layer):
    m = _make(transform, layer=layer)
    x = jr.normal(jr.key(4), (3,))
    J = jax.jacobian(lambda v: m.fwd_logdet(v)[0])(x)
    assert jnp.all(jnp.diag(J) > 0)


@TRANSFORMS
@LAYERS
def test_conditional_roundtrip(transform, layer):
    m = _make(transform, cond_dim=4, layer=layer)
    x = jr.normal(jr.key(1), (3,))
    c = jr.normal(jr.key(2), (4,))
    z, ld = m.fwd_logdet(x, c)
    xr, ldi = m.inv_logdet(z, c)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@TRANSFORMS
@LAYERS
def test_every_coordinate_depends_on_the_condition(transform, layer):
    m = _make(transform, cond_dim=4, layer=layer)
    x = jr.normal(jr.key(1), (3,))
    z1 = m.fwd_logdet(x, jr.normal(jr.key(11), (4,)))[0]
    z2 = m.fwd_logdet(x, jr.normal(jr.key(12), (4,)))[0]
    for i in range(3):
        assert not jnp.allclose(z1[i], z2[i]), f"coordinate {i} ignores c"


@TRANSFORMS
def test_stacked_layers_condition_every_coordinate(transform):
    layers = [_make(transform, cond_dim=4, seed=i) for i in range(3)]

    def flow(x, c):
        for lay in layers:
            x, _ = lay.fwd_logdet(x, c)
        return x

    x = jr.normal(jr.key(10), (3,))
    z1 = flow(x, jr.normal(jr.key(11), (4,)))
    z2 = flow(x, jr.normal(jr.key(12), (4,)))
    for i in range(3):
        assert not jnp.allclose(z1[i], z2[i]), f"coordinate {i} ignores c"


@TRANSFORMS
@LAYERS
def test_rng_reaches_every_conditioner_call_unchanged(transform, layer):
    m = _make(transform, layer=layer, seed=5)
    m = layer(Noisy(m.conditioner), m.transform)
    x, rng = jr.normal(jr.key(1), (3,)), jr.key(2)
    for there, back in (("fwd_logdet", "inv_logdet"), ("inv_logdet", "fwd_logdet")):
        y, ld = getattr(m, there)(x, rng=rng)
        xr, ldr = getattr(m, back)(y, rng=rng)
        assert jnp.allclose(xr, x, atol=1e-4)
        assert jnp.allclose(ld + ldr, 0.0, atol=1e-4)
    assert not jnp.allclose(m.fwd_logdet(x, rng=rng)[0], m.fwd_logdet(x)[0])


@LAYERS
def test_custom_domain_roundtrip(layer):
    m = _make(RQS(8, lower=0.0, upper=1.0), layer=layer)
    x = jr.uniform(jr.key(1), (3,), minval=0.05, maxval=0.95)
    z, ld = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.all((z > 0.0) & (z < 1.0))
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@LAYERS
def test_outside_the_domain_is_the_identity_with_zero_logdet(layer):
    m = _make(RQS(8, lower=0.0, upper=1.0), layer=layer)
    x = jnp.array([-3.0, 2.5, 7.0])
    z, ld = m.fwd_logdet(x)
    assert jnp.allclose(z, x, atol=1e-6)
    assert jnp.allclose(ld, 0.0, atol=1e-6)


@TRANSFORMS
@LAYERS
def test_works_under_jit(transform, layer):
    m = _make(transform, layer=layer)
    x = jr.normal(jr.key(1), (3,))
    z, ld = eqx.filter_jit(lambda m, v: m.fwd_logdet(v))(m, x)
    xr, ldi = eqx.filter_jit(lambda m, v: m.inv_logdet(v))(m, z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


@TRANSFORMS
@LAYERS
def test_works_under_vmap(transform, layer):
    m = _make(transform, layer=layer)
    xs = jr.normal(jr.key(1), (16, 3))
    zs, lds = jax.vmap(m.fwd_logdet)(xs)
    xrs, ldis = jax.vmap(m.inv_logdet)(zs)
    assert zs.shape == xs.shape and lds.shape == (16,)
    assert jnp.allclose(xrs, xs, atol=1e-4)
    assert jnp.allclose(lds + ldis, 0.0, atol=1e-4)


@TRANSFORMS
@LAYERS
def test_slow_path_gradient_is_finite(transform, layer):
    m = _make(transform, layer=layer)
    x = jr.normal(jr.key(1), (3,))
    g = jax.grad(lambda v: getattr(m, SLOW_METHOD[layer])(v)[1])(x)
    assert jnp.all(jnp.isfinite(g))


@TRANSFORMS
@LAYERS
def test_slow_path_logdet_gradient_matches_finite_differences(x64, transform, layer):
    m = _make(transform, layer=layer)
    x = jr.uniform(jr.key(1), (3,), minval=-2.0, maxval=2.0)
    slow_name = SLOW_METHOD[layer]

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


@TRANSFORMS
@LAYERS
def test_roundtrip_logdets_cancel_for_every_parameter_value(x64, transform, layer):
    m = _make(transform, layer=layer)
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
