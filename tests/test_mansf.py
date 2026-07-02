"""Tests for the masked-autoregressive spline flow (ARSpline)"""

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr

from bijax.causal_mlp import CausalMLP
from bijax.mansf import ARSpline

N_BINS = 8


def make_arspline(
    key, dim: int = 2, width: int = 64, depth: int = 2, cond_dim: int | None = None
) -> ARSpline:
    nparams = 3 * N_BINS - 1
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=nparams,
        width=width,
        depth=depth,
        cond_dim=cond_dim,
        rng=key,
    )
    return ARSpline(net=net)


def test_arspline_roundtrip_and_logdets_cancel():
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(1), (2,))
    z, ld = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_arspline_logdet_matches_autodiff():
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(2), (2,))
    _, ld = m.fwd_logdet(x)
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    assert jnp.allclose(ld, jnp.log(jnp.abs(jnp.linalg.det(J))), atol=1e-4)


def test_arspline_forward_is_autoregressive():
    dim = 4
    m = make_arspline(jr.key(0), dim=dim)
    x = jr.normal(jr.key(3), (dim,))
    # z_i depends on x_{<=i} only -> forward jacobian is lower triangular.
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    upper = jnp.triu(jnp.abs(J), k=1)
    assert jnp.all(upper < 1e-6)
    assert jnp.all(jnp.abs(jnp.diag(J)) > 0)


def test_stacked_arspline_conditions_every_coordinate():
    # a pure-AR stack (no mixing layers) must not leave p(x_0 | c)
    # independent of c
    dim = 3
    layers = [make_arspline(jr.key(i), dim=dim, width=32, cond_dim=4) for i in range(3)]

    def flow(x, c):
        for lay in layers:
            x, _ = lay.fwd_logdet(x, c)
        return x

    x = jr.normal(jr.key(10), (dim,))
    c1 = jr.normal(jr.key(11), (4,))
    c2 = jr.normal(jr.key(12), (4,))
    z1, z2 = flow(x, c1), flow(x, c2)
    for i in range(dim):
        assert not jnp.allclose(z1[i], z2[i]), f"coordinate {i} ignores c"


def test_arspline_conditional_roundtrip_and_logdets():
    dim = 3
    m = make_arspline(jr.key(0), dim=dim, width=32, cond_dim=4)
    x = jr.normal(jr.key(1), (dim,))
    c = jr.normal(jr.key(2), (4,))
    z, ld = m.fwd_logdet(x, c)
    xr, ldi = m.inv_logdet(z, c)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_arspline_works_under_jit():
    # the model must be jit-able as a traced argument, the way a training
    # step receives it
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(1), (2,))
    z, ld = eqx.filter_jit(lambda m, x: m.fwd_logdet(x))(m, x)
    xr, ldi = eqx.filter_jit(lambda m, z: m.inv_logdet(z))(m, z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)
