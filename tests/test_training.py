"""End-to-end: the ARSpline flow should learn the two-moons density.

Maximum-likelihood training should drive the mean negative log-likelihood well
below the standard-Gaussian baseline (~2.84 nats on the standardised data).
"""

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.stats import norm

from bijax.causal_mlp import CausalMLP
from bijax.mansf import ARSpline

N_BINS = 8
GAUSSIAN_NLL = 2.84  # mean NLL of a unit Gaussian on the standardised data
TARGET_NLL = 2.3  # comfortably better than the Gaussian baseline


def two_moons(key, n: int, noise: float = 0.1):
    """Standardised sklearn-style two-moons cloud, shape (n, 2)."""
    n_out = n // 2
    n_in = n - n_out
    t_out = jnp.linspace(0, jnp.pi, n_out)
    outer = jnp.stack([jnp.cos(t_out), jnp.sin(t_out)], axis=-1)
    t_in = jnp.linspace(0, jnp.pi, n_in)
    inner = jnp.stack([1.0 - jnp.cos(t_in), 0.5 - jnp.sin(t_in)], axis=-1)
    x = jnp.concatenate([outer, inner], axis=0)
    x = x + noise * jr.normal(key, x.shape)
    return (x - x.mean(0)) / x.std(0)


def make_arspline(key, dim: int = 2, width: int = 64, depth: int = 2) -> ARSpline:
    nparams = 3 * N_BINS - 1
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=nparams,
        width=width,
        depth=depth,
        rng=key,
    )
    return ARSpline(net=net)


def arspline_log_prob(model: ARSpline, x):
    """Standard-normal-base log density of an ARSpline at a single point."""
    z, ld = model.fwd_logdet(x)
    return norm.logpdf(z).sum() + ld


def mean_nll(model, data):
    return -jax.vmap(lambda x: arspline_log_prob(model, x))(data).mean()


def train(model, data, *, steps=600, lr=5e-3, batch=256, seed=0):
    """Maximum-likelihood fit with Adam. Returns (model, init_nll, final_nll)."""
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, opt_state, xb):
        loss, grads = eqx.filter_value_and_grad(mean_nll)(m, xb)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(m, eqx.is_inexact_array)
        )
        return eqx.apply_updates(m, updates), opt_state, loss

    init_nll = float(mean_nll(model, data))
    key = jr.key(seed)
    for _ in range(steps):
        key, k = jr.split(key)
        idx = jr.randint(k, (batch,), 0, data.shape[0])
        model, opt_state, _ = step(model, opt_state, data[idx])
    final_nll = float(mean_nll(model, data))
    return model, init_nll, final_nll


def test_arspline_learns_two_moons():
    data = two_moons(jr.key(42), 2000)
    model = make_arspline(jr.key(0))
    _, init_nll, final_nll = train(model, data, steps=600, lr=5e-3, batch=256)
    assert final_nll < init_nll - 0.5, "arspline did not learn (no improvement)"
    assert final_nll < TARGET_NLL, f"arspline final NLL {final_nll:.3f} too high"
    assert final_nll < GAUSSIAN_NLL, "arspline no better than a Gaussian"
