"""Tests for the RQ-spline coupling layer (SplineCoupling)."""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from bijax.coupling_nsf import SplineCoupling

N_BINS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unconditional(
    flow_dim: int, n_bins: int = N_BINS, seed: int = 0
) -> SplineCoupling:
    """Build an unconditional SplineCoupling using mlp_dims."""
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    key = jr.key(seed)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=key)
    # even dims are identity (pass-through / MLP input), odd dims transformed
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    return SplineCoupling(mlp=mlp, id_idxs=id_idxs, tr_idxs=tr_idxs, n_bins=n_bins)


def _make_conditional(
    flow_dim: int, cond_dim: int, n_bins: int = N_BINS, seed: int = 0
) -> SplineCoupling:
    """Build a conditional SplineCoupling using mlp_dims."""
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, cond_dim, n_bins)
    key = jr.key(seed)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=key)
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    return SplineCoupling(mlp=mlp, id_idxs=id_idxs, tr_idxs=tr_idxs, n_bins=n_bins)


# ---------------------------------------------------------------------------
# mlp_dims static helper
# ---------------------------------------------------------------------------


def test_mlp_dims_unconditional():
    flow_dim, n_bins = 4, N_BINS
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    assert in_dim == flow_dim // 2
    assert out_dim == (flow_dim - flow_dim // 2) * (n_bins * 3 - 1)


def test_mlp_dims_returns_ints():
    in_dim, out_dim = SplineCoupling.mlp_dims(6, 0, N_BINS)
    assert isinstance(in_dim, int)
    assert isinstance(out_dim, int)


# ---------------------------------------------------------------------------
# Unconditional — roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_roundtrip(flow_dim):
    m = _make_unconditional(flow_dim, seed=10)
    x = jr.normal(jr.key(3), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    xr, _ = m.inv_logdet(y, None)
    assert jnp.allclose(xr, x, atol=1e-4), f"max err={jnp.abs(xr - x).max()}"


# ---------------------------------------------------------------------------
# Unconditional — forward/inverse log-dets cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_logdets_cancel(flow_dim):
    m = _make_unconditional(flow_dim, seed=20)
    x = jr.normal(jr.key(4), (flow_dim,))
    y, ld_fwd = m.fwd_logdet(x, None)
    _, ld_inv = m.inv_logdet(y, None)
    # ld_fwd and ld_inv are per-transformed-dim arrays; sum over dims
    assert jnp.allclose(ld_fwd.sum() + ld_inv.sum(), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Unconditional — log-det matches autodiff Jacobian determinant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_logdet_matches_autodiff(flow_dim):
    m = _make_unconditional(flow_dim, seed=30)
    # Use a small input so spline is inside [low, high] = [-5, 5]
    x = 0.5 * jr.normal(jr.key(5), (flow_dim,))
    _, ld = m.fwd_logdet(x, None)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, None)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld.sum(), expected, atol=1e-4), (
        f"ld.sum()={ld.sum()}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Unconditional — identity mask dims are truly passed through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_id_dims_unchanged(flow_dim):
    m = _make_unconditional(flow_dim, seed=40)
    x = jr.normal(jr.key(6), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    id_ix = jnp.array(m.id_idxs)
    assert jnp.allclose(y[id_ix], x[id_ix], atol=1e-6)


# ---------------------------------------------------------------------------
# Conditional — roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_roundtrip(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=50)
    x = jr.normal(jr.key(11), (flow_dim,))
    c = jr.normal(jr.key(12), (cond_dim,))
    y, _ = m.fwd_logdet(x, c)
    xr, _ = m.inv_logdet(y, c)
    assert jnp.allclose(xr, x, atol=1e-4), f"max err={jnp.abs(xr - x).max()}"


# ---------------------------------------------------------------------------
# Conditional — forward/inverse log-dets cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_logdets_cancel(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=60)
    x = jr.normal(jr.key(13), (flow_dim,))
    c = jr.normal(jr.key(14), (cond_dim,))
    y, ld_fwd = m.fwd_logdet(x, c)
    _, ld_inv = m.inv_logdet(y, c)
    assert jnp.allclose(ld_fwd.sum() + ld_inv.sum(), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Conditional — log-det matches autodiff Jacobian determinant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_logdet_matches_autodiff(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=70)
    x = 0.5 * jr.normal(jr.key(15), (flow_dim,))
    c = jr.normal(jr.key(16), (cond_dim,))
    _, ld = m.fwd_logdet(x, c)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, c)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld.sum(), expected, atol=1e-4), (
        f"ld.sum()={ld.sum()}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Conditional — output changes with conditioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_output_varies_with_c(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=80)
    x = jr.normal(jr.key(17), (flow_dim,))
    c1 = jr.normal(jr.key(18), (cond_dim,))
    c2 = jr.normal(jr.key(19), (cond_dim,))
    y1, _ = m.fwd_logdet(x, c1)
    y2, _ = m.fwd_logdet(x, c2)
    # The transformed dims must differ; identity dims stay the same
    id_ix, tr_ix = jnp.array(m.id_idxs), jnp.array(m.tr_idxs)
    assert not jnp.allclose(y1[tr_ix], y2[tr_ix])
    assert jnp.allclose(y1[id_ix], y2[id_ix])


# ---------------------------------------------------------------------------
# MLP-output-shape validation
# ---------------------------------------------------------------------------


def _make_bad(flow_dim: int = 4, n_bins: int = N_BINS, seed: int = 0) -> SplineCoupling:
    """Layer whose MLP emits the wrong number of params."""
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    mlp = eqx.nn.MLP(in_dim, out_dim + 1, width_size=8, depth=1, key=jr.key(seed))
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    return SplineCoupling(mlp=mlp, id_idxs=id_idxs, tr_idxs=tr_idxs, n_bins=n_bins)


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


def test_spline_coupling_works_under_jit():
    m = _make_unconditional(4, seed=90)
    x = jr.normal(jr.key(20), (4,))
    y, _ = eqx.filter_jit(lambda m, x: m.fwd_logdet(x, None))(m, x)
    xr, _ = eqx.filter_jit(lambda m, y: m.inv_logdet(y, None))(m, y)
    assert jnp.allclose(xr, x, atol=1e-4)
