"""Tests for the affine-coupling layer (AffineCoupling).

The layer's primary interface is `fwd_logdet` / `inv_logdet`, mirroring the
RQ-spline `SplineCoupling`.  `s` is a log-scale, so the forward applies
`x -> x * exp(s) + t` on the transformed dims with log-det `s.sum()`.
"""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from bijax.coupling_aff import AffineCoupling

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(flow_dim: int, cond_dim: int = 0, seed: int = 0) -> AffineCoupling:
    """Build an AffineCoupling using mlp_dims (handles cond_dim == 0 too)."""
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, cond_dim)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=jr.key(seed))
    # even dims are identity (pass-through / MLP input), odd dims transformed
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    return AffineCoupling(mlp=mlp, id_idxs=id_idxs, tr_idxs=tr_idxs)


# ---------------------------------------------------------------------------
# mlp_dims static helper
# ---------------------------------------------------------------------------


def test_mlp_dims_unconditional():
    flow_dim = 4
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, 0)
    assert in_dim == flow_dim // 2
    assert out_dim == (flow_dim - flow_dim // 2) * 2


def test_mlp_dims_conditional_includes_cond():
    in_dim, out_dim = AffineCoupling.mlp_dims(4, 3)
    assert in_dim == 4 // 2 + 3
    assert out_dim == (4 - 4 // 2) * 2


# ---------------------------------------------------------------------------
# Unconditional — roundtrip / log-dets cancel / autodiff / id-dims passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_roundtrip(flow_dim):
    m = _make(flow_dim, seed=10)
    x = jr.normal(jr.key(3), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    xr, _ = m.inv_logdet(y, None)
    assert jnp.allclose(xr, x, atol=1e-5), f"max err={jnp.abs(xr - x).max()}"


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdets_cancel(flow_dim):
    m = _make(flow_dim, seed=20)
    x = jr.normal(jr.key(4), (flow_dim,))
    y, ld_fwd = m.fwd_logdet(x, None)
    _, ld_inv = m.inv_logdet(y, None)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdet_matches_autodiff(flow_dim):
    m = _make(flow_dim, seed=30)
    x = 0.5 * jr.normal(jr.key(5), (flow_dim,))
    _, ld = m.fwd_logdet(x, None)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, None)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=1e-5), f"ld={ld}, expected={expected}"


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_id_dims_unchanged(flow_dim):
    m = _make(flow_dim, seed=40)
    x = jr.normal(jr.key(6), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    id_ix = jnp.array(m.id_idxs)
    assert jnp.allclose(y[id_ix], x[id_ix], atol=1e-6)


# ---------------------------------------------------------------------------
# Conditional — roundtrip / log-dets / autodiff / varies with c
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_roundtrip(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=50)
    x = jr.normal(jr.key(11), (flow_dim,))
    c = jr.normal(jr.key(12), (cond_dim,))
    y, _ = m.fwd_logdet(x, c)
    xr, _ = m.inv_logdet(y, c)
    assert jnp.allclose(xr, x, atol=1e-5), f"max err={jnp.abs(xr - x).max()}"


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_logdets_cancel(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=60)
    x = jr.normal(jr.key(13), (flow_dim,))
    c = jr.normal(jr.key(14), (cond_dim,))
    y, ld_fwd = m.fwd_logdet(x, c)
    _, ld_inv = m.inv_logdet(y, c)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_logdet_matches_autodiff(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=70)
    x = 0.5 * jr.normal(jr.key(15), (flow_dim,))
    c = jr.normal(jr.key(16), (cond_dim,))
    _, ld = m.fwd_logdet(x, c)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, c)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=1e-5), f"ld={ld}, expected={expected}"


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_output_varies_with_c(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=80)
    x = jr.normal(jr.key(17), (flow_dim,))
    c1 = jr.normal(jr.key(18), (cond_dim,))
    c2 = jr.normal(jr.key(19), (cond_dim,))
    y1, _ = m.fwd_logdet(x, c1)
    y2, _ = m.fwd_logdet(x, c2)
    # transformed dims must differ; identity dims stay the same
    id_ix, tr_ix = jnp.array(m.id_idxs), jnp.array(m.tr_idxs)
    assert not jnp.allclose(y1[tr_ix], y2[tr_ix])
    assert jnp.allclose(y1[id_ix], y2[id_ix])


# ---------------------------------------------------------------------------
# MLP-output-shape validation
# ---------------------------------------------------------------------------


def _make_bad(flow_dim: int = 4, seed: int = 0) -> AffineCoupling:
    """Layer whose MLP emits the wrong number of params."""
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, 0)
    mlp = eqx.nn.MLP(in_dim, out_dim + 1, width_size=8, depth=1, key=jr.key(seed))
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    return AffineCoupling(mlp=mlp, id_idxs=id_idxs, tr_idxs=tr_idxs)


def test_fwd_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.fwd_logdet(jr.normal(jr.key(1), (4,)), None)


def test_inv_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.inv_logdet(jr.normal(jr.key(1), (4,)), None)


# ---------------------------------------------------------------------------
# jit compatibility
# ---------------------------------------------------------------------------


def test_affine_coupling_works_under_jit():
    m = _make(4, seed=90)
    x = jr.normal(jr.key(20), (4,))
    y, _ = eqx.filter_jit(lambda m, x: m.fwd_logdet(x, None))(m, x)
    xr, _ = eqx.filter_jit(lambda m, y: m.inv_logdet(y, None))(m, y)
    assert jnp.allclose(xr, x, atol=1e-4)
