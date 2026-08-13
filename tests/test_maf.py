"""Tests for the masked-autoregressive affine flow (ARAffine)"""

import jax
from jax import numpy as jnp
from jax import random as jr

from bijax.causal_mlp import CausalMLP
from bijax.maf import ARAffine


def make_araffine(key, dim: int = 2, width: int = 64, depth: int = 2) -> ARAffine:
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=2,  # log-scale and shift per coordinate
        width=width,
        depth=depth,
        rng=key,
    )
    return ARAffine(net=net)


def test_araffine_roundtrip_and_logdets_cancel():
    m = make_araffine(jr.key(0))
    x = jr.normal(jr.key(1), (2,))
    z, ld = m.fwd_logdet(x)
    xr, ldi = m.inv_logdet(z)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_araffine_logdet_matches_autodiff():
    m = make_araffine(jr.key(0))
    x = jr.normal(jr.key(2), (2,))
    _, ld = m.fwd_logdet(x)
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    assert jnp.allclose(ld, jnp.log(jnp.abs(jnp.linalg.det(J))), atol=1e-4)


def test_araffine_forward_is_autoregressive():
    dim = 4
    m = make_araffine(jr.key(0), dim=dim)
    x = jr.normal(jr.key(3), (dim,))
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    upper = jnp.triu(jnp.abs(J), k=1)
    assert jnp.all(upper < 1e-6)
    assert jnp.all(jnp.abs(jnp.diag(J)) > 0)


def test_araffine_conditioning_changes_output():
    dim = 3
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=2,
        width=16,
        depth=2,
        cond_dim=4,
        rng=jr.key(0),
    )
    m = ARAffine(net=net)
    x = jr.normal(jr.key(1), (dim,))
    c1 = jr.normal(jr.key(2), (4,))
    c2 = jr.normal(jr.key(3), (4,))
    z1, _ = m.fwd_logdet(x, c1)
    z2, _ = m.fwd_logdet(x, c2)
    for i in range(dim):
        assert not jnp.allclose(z1[i], z2[i]), f"coordinate {i} ignores c"
