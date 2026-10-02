"""Tests for the example MADE conditioner (CausalMLP)."""

import jax
import pytest
from jax import numpy as jnp
from jax import random as jr
from made_spline_flow import CausalMLP

from bijax import CausalLinear


def test_causal_mlp_output_shape_vector_out():
    m = CausalMLP(
        num_ranks=4,
        in_rank_dim="scalar",
        out_rank_dim=5,
        width=16,
        depth=2,
        rng=jr.key(0),
    )
    assert m(jnp.arange(4.0), None).shape == (4, 5)


def test_causal_mlp_output_shape_scalar_out():
    m = CausalMLP(
        num_ranks=4,
        in_rank_dim="scalar",
        out_rank_dim="scalar",
        width=16,
        depth=2,
        rng=jr.key(0),
    )
    assert m(jnp.arange(4.0), None).shape == (4,)


def test_causal_mlp_is_strictly_autoregressive():
    dim = 5
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=3,
        width=24,
        depth=3,
        rng=jr.key(1),
    )
    x = jr.normal(jr.key(2), (dim,))
    J = jax.jacobian(lambda x: m(x, None))(x)  # J[i, j, k] = d(param i,j) / d(x k)
    dep = jnp.abs(J).sum(1)  # (dim, dim): coord i vs input k
    expected = jnp.tril(jnp.ones((dim, dim)), k=-1)
    assert jnp.allclose((dep > 1e-7).astype(float), expected)


def test_causal_mlp_conditioning_changes_output():
    dim = 3
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=2,
        width=16,
        depth=2,
        cond_dim=4,
        rng=jr.key(3),
    )
    x = jr.normal(jr.key(4), (dim,))
    c1 = jr.normal(jr.key(5), (4,))
    c2 = jr.normal(jr.key(6), (4,))
    out1, out2 = m(x, c1), m(x, c2)
    assert out1.shape == (dim, 2)
    for i in range(dim):
        assert not jnp.allclose(out1[i], out2[i]), f"coordinate {i} ignores c"


def test_causal_mlp_conditional_is_strictly_autoregressive():
    dim = 4
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=3,
        width=24,
        depth=2,
        cond_dim=4,
        rng=jr.key(7),
    )
    x = jr.normal(jr.key(8), (dim,))
    c = jr.normal(jr.key(9), (4,))
    J = jax.jacobian(lambda x: m(x, c))(x)
    dep = jnp.abs(J).sum(1)  # (dim, dim): coord i vs input k
    expected = jnp.tril(jnp.ones((dim, dim)), k=-1)
    assert jnp.allclose((dep > 1e-7).astype(float), expected)


def _connectivity_mask(lay: CausalLinear):
    """(in_dim, out_dim) boolean mask of unmasked weights."""
    return (
        jnp.zeros((lay.in_dim, lay.out_dim), dtype=bool).at[lay.unmasked_idxs].set(True)
    )


@pytest.mark.parametrize("cond_dim", [None, 4])
def test_causal_mlp_has_no_dead_hidden_units(cond_dim):
    dim = 5
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=2,
        width=15,
        depth=2,
        cond_dim=cond_dim,
        rng=jr.key(10),
    )
    live = jnp.ones(m.layers[-1].out_dim, dtype=bool)
    for lay in reversed(m.layers):
        # after this step `live` describes the layer's inputs
        live = (_connectivity_mask(lay) & live[None, :]).any(axis=1)
        if lay is not m.layers[0]:
            assert bool(live.all()), "dead hidden units found"
    # only the last coordinate may be unread
    assert bool(live[: dim - 1].all())
    if cond_dim is not None:
        assert bool(live[dim:].all()), "conditioner inputs are unread"


def test_causal_mlp_requires_two_ranks():
    with pytest.raises(ValueError):
        CausalMLP(
            num_ranks=1,
            in_rank_dim="scalar",
            out_rank_dim=2,
            width=8,
            depth=1,
            rng=jr.key(0),
        )


def _dropout_pair(rate):
    kwargs = {
        "num_ranks": 3,
        "in_rank_dim": "scalar",
        "out_rank_dim": 2,
        "width": 16,
        "depth": 2,
    }
    plain = CausalMLP(**kwargs, rng=jr.key(0))
    drop = CausalMLP(**kwargs, dropout=rate, rng=jr.key(0))
    return plain, drop


def test_dropout_is_off_without_rng():
    plain, drop = _dropout_pair(0.5)
    x = jr.normal(jr.key(1), (3,))
    assert jnp.array_equal(drop(x, None), plain(x, None))


def test_dropout_with_rng_changes_the_output():
    plain, drop = _dropout_pair(0.5)
    x = jr.normal(jr.key(1), (3,))
    assert not jnp.allclose(drop(x, None, rng=jr.key(2)), plain(x, None))
