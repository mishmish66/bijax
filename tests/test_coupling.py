"""Tests for the coupling layer over each elementwise transform."""

import equinox as eqx
import jax
import pytest
from conditioners import MLP, Noisy
from jax import numpy as jnp
from jax import random as jr

from bijax import RQS, Affine, Coupling

TRANSFORMS = pytest.mark.parametrize(
    ("transform", "atol"),
    [pytest.param(Affine(), 1e-5, id="affine"), pytest.param(RQS(8), 1e-4, id="rqs")],
)


def _make(transform, flow_dim, cond_dim=0, seed=0, extra_params=0):
    # even dims pass through, odd dims are transformed
    id_idxs = tuple(range(0, flow_dim, 2))
    tr_idxs = tuple(range(1, flow_dim, 2))
    shape = (len(tr_idxs), transform.n_params + extra_params)
    net = MLP(len(id_idxs), shape, cond_dim, rng=jr.key(seed))
    return Coupling(net, transform, id_idxs, tr_idxs)


@TRANSFORMS
@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_roundtrip(transform, atol, flow_dim):
    m = _make(transform, flow_dim, seed=10)
    x = jr.normal(jr.key(3), (flow_dim,))
    y, _ = m.fwd_logdet(x)
    xr, _ = m.inv_logdet(y)
    assert jnp.allclose(xr, x, atol=atol), f"max err={jnp.abs(xr - x).max()}"


@TRANSFORMS
@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdets_cancel(transform, atol, flow_dim):
    m = _make(transform, flow_dim, seed=20)
    x = jr.normal(jr.key(4), (flow_dim,))
    y, ld_fwd = m.fwd_logdet(x)
    _, ld_inv = m.inv_logdet(y)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=atol)


@TRANSFORMS
@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdet_matches_autodiff(transform, atol, flow_dim):
    m = _make(transform, flow_dim, seed=30)
    x = 0.5 * jr.normal(jr.key(5), (flow_dim,))
    _, ld = m.fwd_logdet(x)
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=atol), f"ld={ld}, expected={expected}"


@TRANSFORMS
@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_logdet_is_a_scalar(transform, atol, method):
    m = _make(transform, 6, cond_dim=2)
    _, ld = getattr(m, method)(jr.normal(jr.key(1), (6,)), jnp.ones(2))
    assert ld.shape == ()


@TRANSFORMS
@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_id_dims_unchanged(transform, atol, flow_dim):
    m = _make(transform, flow_dim, seed=40)
    x = jr.normal(jr.key(6), (flow_dim,))
    y, _ = m.fwd_logdet(x)
    id_ix = jnp.array(m.id_idxs)
    assert jnp.allclose(y[id_ix], x[id_ix], atol=1e-6)


@TRANSFORMS
def test_each_transformed_coordinate_reads_only_itself_among_transformed(
    transform, atol
):
    m = _make(transform, 6, seed=45)
    x = 0.5 * jr.normal(jr.key(7), (6,))
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    tr_ix = jnp.array(m.tr_idxs)
    block = J[tr_ix][:, tr_ix]
    assert jnp.allclose(block - jnp.diag(jnp.diag(block)), 0.0)
    assert jnp.all(jnp.diag(block) > 0)


@TRANSFORMS
@pytest.mark.parametrize(("flow_dim", "cond_dim"), [(4, 2), (6, 3)])
def test_cond_roundtrip(transform, atol, flow_dim, cond_dim):
    m = _make(transform, flow_dim, cond_dim, seed=50)
    x = jr.normal(jr.key(11), (flow_dim,))
    c = jr.normal(jr.key(12), (cond_dim,))
    y, _ = m.fwd_logdet(x, c)
    xr, _ = m.inv_logdet(y, c)
    assert jnp.allclose(xr, x, atol=atol), f"max err={jnp.abs(xr - x).max()}"


@TRANSFORMS
@pytest.mark.parametrize(("flow_dim", "cond_dim"), [(4, 2), (6, 3)])
def test_cond_logdets_cancel(transform, atol, flow_dim, cond_dim):
    m = _make(transform, flow_dim, cond_dim, seed=60)
    x = jr.normal(jr.key(13), (flow_dim,))
    c = jr.normal(jr.key(14), (cond_dim,))
    y, ld_fwd = m.fwd_logdet(x, c)
    _, ld_inv = m.inv_logdet(y, c)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=atol)


@TRANSFORMS
@pytest.mark.parametrize(("flow_dim", "cond_dim"), [(4, 2), (6, 3)])
def test_cond_logdet_matches_autodiff(transform, atol, flow_dim, cond_dim):
    m = _make(transform, flow_dim, cond_dim, seed=70)
    x = 0.5 * jr.normal(jr.key(15), (flow_dim,))
    c = jr.normal(jr.key(16), (cond_dim,))
    _, ld = m.fwd_logdet(x, c)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, c)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=atol), f"ld={ld}, expected={expected}"


@TRANSFORMS
@pytest.mark.parametrize(("flow_dim", "cond_dim"), [(4, 2), (6, 3)])
def test_cond_output_varies_with_c(transform, atol, flow_dim, cond_dim):
    m = _make(transform, flow_dim, cond_dim, seed=80)
    x = jr.normal(jr.key(17), (flow_dim,))
    c1 = jr.normal(jr.key(18), (cond_dim,))
    c2 = jr.normal(jr.key(19), (cond_dim,))
    y1, _ = m.fwd_logdet(x, c1)
    y2, _ = m.fwd_logdet(x, c2)
    id_ix, tr_ix = jnp.array(m.id_idxs), jnp.array(m.tr_idxs)
    assert not jnp.allclose(y1[tr_ix], y2[tr_ix])
    assert jnp.allclose(y1[id_ix], y2[id_ix])


@TRANSFORMS
@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_rejects_a_conditioner_output_of_the_wrong_shape(transform, atol, method):
    m = _make(transform, 4, extra_params=1)
    with pytest.raises(ValueError):
        getattr(m, method)(jr.normal(jr.key(1), (4,)))


class _NeedsRng(eqx.Module):
    inner: MLP

    def __call__(self, x, c, *, rng):
        return self.inner(x, c)


@pytest.mark.parametrize("method", ["fwd_logdet", "inv_logdet"])
def test_conditioner_is_always_given_rng(method):
    m = _make(Affine(), 4)
    m = Coupling(_NeedsRng(m.conditioner), m.transform, m.id_idxs, m.tr_idxs)
    getattr(m, method)(jr.normal(jr.key(1), (4,)))


@TRANSFORMS
def test_rng_reaches_the_conditioner(transform, atol):
    m = _make(transform, 4, seed=85)
    m = Coupling(Noisy(m.conditioner), m.transform, m.id_idxs, m.tr_idxs)
    x, rng = jr.normal(jr.key(21), (4,)), jr.key(22)
    y, ld = m.fwd_logdet(x, rng=rng)
    xr, ldi = m.inv_logdet(y, rng=rng)
    assert jnp.allclose(xr, x, atol=atol)
    assert jnp.allclose(ld + ldi, 0.0, atol=atol)
    assert not jnp.allclose(y, m.fwd_logdet(x)[0])


@TRANSFORMS
def test_works_under_jit(transform, atol):
    m = _make(transform, 4, seed=90)
    x = jr.normal(jr.key(20), (4,))
    y, _ = eqx.filter_jit(lambda m, x: m.fwd_logdet(x))(m, x)
    xr, _ = eqx.filter_jit(lambda m, y: m.inv_logdet(y))(m, y)
    assert jnp.allclose(xr, x, atol=atol)
