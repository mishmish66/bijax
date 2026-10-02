"""Minimal conditioners for exercising flow structures."""

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr


class MaskedAffine(eqx.Module):
    """Row ``i`` of the output is affine in ``x[:i]`` and ``c``."""

    w: jnp.ndarray
    b: jnp.ndarray
    v: jnp.ndarray | None

    def __init__(self, dim, n_params, cond_dim=None, *, scale=0.5, rng):
        kw, kb, kv = jr.split(rng, 3)
        self.w = scale * jr.normal(kw, (dim, dim, n_params))
        self.b = scale * jr.normal(kb, (dim, n_params))
        self.v = (
            None
            if cond_dim is None
            else scale * jr.normal(kv, (cond_dim, dim, n_params))
        )

    def __call__(self, x, c=None, *, rng=None):
        mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0])), k=-1)
        h = jnp.einsum("ij,ijp,j->ip", mask, self.w, x) + self.b
        if c is not None:
            h = h + jnp.einsum("k,kip->ip", c, self.v)
        return h


class MLP(eqx.Module):
    """``eqx.nn.MLP`` on ``concat(x, c)``, reshaped to ``shape``."""

    mlp: eqx.nn.MLP
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, in_dim, shape, cond_dim=0, *, width=32, depth=2, rng):
        self.shape = shape
        self.mlp = eqx.nn.MLP(
            in_dim + cond_dim, shape[0] * shape[1], width, depth, key=rng
        )

    def __call__(self, x, c=None, *, rng=None):
        h = x if c is None else jnp.concat([x, c])
        return self.mlp(h).reshape(self.shape)


class Noisy(eqx.Module):
    """Adds ``rng``-seeded noise to the wrapped conditioner's output."""

    inner: eqx.Module

    def __call__(self, x, c=None, *, rng=None):
        out = self.inner(x, c, rng=None)
        return out if rng is None else out + jr.normal(rng, out.shape)
