"""Tests for the masked linear layer (CausalLinear)."""

import jax
from jax import numpy as jnp
from jax import random as jr

from bijax import CausalLinear


def test_causal_linear_respects_ranks():
    # reads are inclusive: rank r reads every input of rank <= r
    lay = CausalLinear([0, 0, 1], [0, 1], rng=jr.key(0))
    x = jnp.array([1.0, 1.0, 1.0])
    J = jax.jacobian(lay)(x)
    assert jnp.all(jnp.abs(J[0, :2]) > 0)
    assert jnp.allclose(J[0, 2], 0.0)
    assert jnp.all(jnp.abs(J[1]) > 0)


def test_causal_linear_shapes():
    lay = CausalLinear([0, 1, 2], [0, 1, 2, 3], rng=jr.key(0))
    out = lay(jnp.ones(3))
    assert out.shape == (4,)
