"""Every bijector composes into one flow through the shared interface."""

import equinox as eqx
import jax
import pytest
from conditioners import MLP, MaskedAffine
from jax import numpy as jnp
from jax import random as jr

from bijax import IAF, MAF, PLU, RQS, Affine, Coupling

DIM, COND = 4, 2
ATOL = 1e-10


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def _chain():
    k = iter(jr.split(jr.key(0), 8))
    return [
        PLU(DIM, rng=next(k)),
        Coupling(MLP(2, (2, 2), COND, rng=next(k)), Affine(), (0, 2), (1, 3)),
        MAF(MaskedAffine(DIM, 25, COND, rng=next(k)), RQS(8)),
        PLU(DIM, rng=next(k)),
        Coupling(MLP(2, (2, 25), COND, rng=next(k)), RQS(8), (1, 3), (0, 2)),
        IAF(MaskedAffine(DIM, 2, COND, rng=next(k)), Affine()),
    ]


def _fwd(layers, x, c):
    total = 0.0
    for lay in layers:
        x, ld = lay.fwd_logdet(x, c)
        total = total + ld
    return x, total


def _inv(layers, y, c):
    total = 0.0
    for lay in reversed(layers):
        y, ld = lay.inv_logdet(y, c)
        total = total + ld
    return y, total


def test_chain_roundtrips_and_its_logdets_cancel():
    layers = _chain()
    x, c = jr.normal(jr.key(1), (DIM,)), jr.normal(jr.key(2), (COND,))
    y, ld = _fwd(layers, x, c)
    xr, ldi = _inv(layers, y, c)
    assert jnp.allclose(xr, x, atol=ATOL)
    assert jnp.allclose(ld + ldi, 0.0, atol=ATOL)


def test_chain_logdet_matches_autodiff():
    layers = _chain()
    x, c = 0.5 * jr.normal(jr.key(3), (DIM,)), jr.normal(jr.key(4), (COND,))
    _, ld = _fwd(layers, x, c)
    J = jax.jacobian(lambda v: _fwd(layers, v, c)[0])(x)
    assert jnp.allclose(ld, jnp.linalg.slogdet(J)[1], atol=ATOL)


def test_chain_works_under_jit_and_vmap():
    layers = _chain()
    xs, c = jr.normal(jr.key(5), (8, DIM)), jr.normal(jr.key(6), (COND,))
    fwd = eqx.filter_jit(jax.vmap(_fwd, in_axes=(None, 0, None)))
    inv = eqx.filter_jit(jax.vmap(_inv, in_axes=(None, 0, None)))
    ys, lds = fwd(layers, xs, c)
    xrs, ldis = inv(layers, ys, c)
    assert lds.shape == (8,)
    assert jnp.allclose(xrs, xs, atol=ATOL)
    assert jnp.allclose(lds + ldis, 0.0, atol=ATOL)
