"""Masked autoregressive spline flow on two moons, with a MADE conditioner.

`CausalMLP` builds a MADE network from `bijax.CausalLinear`; `make_flow` plugs
it into `bijax.MAF` layers with `bijax.PLU` mixing between them.
Run ``uv run python examples/made_spline_flow.py`` to train and report the NLL.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import Literal, overload

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.stats import norm
from jaxtyping import Array, Float, Key

from bijax import MAF, PLU, RQS, CausalLinear


class CausalMLP(eqx.Module):
    """Autoregressive MLP over a rank axis: x of shape (l, di) -> (l, do)."""

    layers: list[CausalLinear]
    activation: Callable[[Array], Array] = eqx.field(static=True)
    num_ranks: int = eqx.field(static=True)
    cond_dim: int | None = eqx.field(static=True)
    in_rank_dim: int | Literal["scalar"] = eqx.field(static=True)
    out_rank_dim: int | Literal["scalar"] = eqx.field(static=True)
    dropout: eqx.nn.Dropout | None

    @overload
    def __init__(
        self,
        num_ranks: int,
        in_rank_dim: int | Literal["scalar"],
        out_rank_dim: int | Literal["scalar"],
        width: int,
        depth: int,
        *,
        dropout: float = 0.0,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        """Randomly intialize a CausalMLP.

        A CausalMLP is a special kind of MLP where each output of a rank depends
        only on the inputs of at most its rank. The conditioning dimension
        is separate because sometimes more processing of the condition might
        be desirable, but the number of parameters involved in processing the
        condition will be lower for lower ranks because they don't make use of
        the higher rank's parameters.

        Args:
          num_ranks: Number of ranks to consider (not counting the condition)
          in_rank_dim: Dimension of each rank's input
          out_rank_dim: Dimension of each rank's output
          width: The width of the hidden dim for (non condition)
          depth: The number of layers (0 makes it just a CausalLinear)
          dropout: dropout rate for hidden activations
          activation: The nonlinearity to apply between layers
          rng: A random key to generate the parameters
        """
        ...

    @overload
    def __init__(
        self,
        num_ranks: int,
        in_rank_dim: int | Literal["scalar"],
        out_rank_dim: int | Literal["scalar"],
        width: int,
        depth: int,
        *,
        cond_dim: int,
        cond_width: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        """Randomly intialize a CausalMLP.

        A CausalMLP is a special kind of MLP where each output of a rank depends
        only on the inputs of at most its rank. The conditioning dimension
        is separate because sometimes more processing of the condition might
        be desirable, but the number of parameters involved in processing the
        condition will be lower for lower ranks because they don't make use of
        the higher rank's parameters.

        Args:
          num_ranks: Number of ranks to consider (not counting the condition)
          in_rank_dim: Dimension of each rank's input
          out_rank_dim: Dimension of each rank's output
          width: Width of the hidden dim for (non condition)
          depth: Number of layers (0 makes it just a CausalLinear)
          cond_width: Width of hidden dims for condition inputs
          cond_dim: Dimension of conditioning input
          dropout: dropout rate for hidden activations
          activation: Nonlinearity between layers
          rng: Random key generating parameters
        """
        ...

    def __init__(
        self,
        num_ranks: int,
        in_rank_dim: int | Literal["scalar"],
        out_rank_dim: int | Literal["scalar"],
        width: int,
        depth: int,
        *,
        cond_width: int | None = None,
        cond_dim: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        if num_ranks < 2:
            msg = f"need num_ranks >= 2 for autoregressive structure, got {num_ranks}"
            raise ValueError(msg)

        self.in_rank_dim = in_rank_dim
        self.out_rank_dim = out_rank_dim
        self.num_ranks = num_ranks
        self.cond_dim = cond_dim
        self.dropout = eqx.nn.Dropout(dropout) if dropout > 0.0 else None

        if in_rank_dim == "scalar":
            in_rank_dim = 1
        if out_rank_dim == "scalar":
            out_rank_dim = 1

        # create layer ranks
        in_ranks = [r + 1 for r in range(num_ranks) for _ in range(in_rank_dim)]
        hidden_ranks = [r + 1 for r in range(num_ranks - 1) for _ in range(width)]
        if cond_dim is not None:
            if cond_width is None:
                cond_width = width
            in_ranks.extend([0] * cond_dim)
            hidden_ranks.extend([0] * cond_width)

        # omit +1 so out r0 reads only hidden r0 (cond)
        out_ranks = [r for r in range(num_ranks) for _ in range(out_rank_dim)]
        ranks = [in_ranks] + [hidden_ranks] * depth + [out_ranks]

        # layers
        layers = []
        for ranks_in, ranks_out in pairwise(ranks):
            rng, key = jr.split(rng)
            layers.append(CausalLinear(ranks_in, ranks_out, rng=key))

        self.layers = layers
        self.activation = activation

    def __call__(
        self,
        x: Float[Array, "l di"] | Float[Array, " l"],
        c: Float[Array, " c"] | None,
        rng: Key[Array, ""] | None = None,
    ) -> Float[Array, "l do"] | Float[Array, " l"]:
        # scalar input is already the flat (l,) vector CausalLinear wants;
        # otherwise flatten the (l, di) grid row-major.
        x = x if self.in_rank_dim == "scalar" else x.reshape(-1)
        if c is None:
            h = x
        else:
            h = jnp.concat([x, c])

        for i, lay in enumerate(self.layers):
            h = lay(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout is not None:
                    if rng is not None:
                        rng, key = jr.split(rng)
                        h = self.dropout(h, key=key)
                    else:
                        self.dropout(h)
        # scalar output stays (l,); otherwise unflatten to (l, do).
        if self.out_rank_dim == "scalar":
            return h
        return h.reshape(self.num_ranks, self.out_rank_dim)  # (l*do,) -> (l, do)


def two_moons(key: Key[Array, ""], n: int, noise: float = 0.1) -> Float[Array, "n 2"]:
    """Sample a standardized two-moons cloud."""
    n_out = n // 2
    n_in = n - n_out
    t_out = jnp.linspace(0, jnp.pi, n_out)
    outer = jnp.stack([jnp.cos(t_out), jnp.sin(t_out)], axis=-1)
    t_in = jnp.linspace(0, jnp.pi, n_in)
    inner = jnp.stack([1.0 - jnp.cos(t_in), 0.5 - jnp.sin(t_in)], axis=-1)
    x = jnp.concatenate([outer, inner], axis=0)
    x = x + noise * jr.normal(key, x.shape)
    return (x - x.mean(0)) / x.std(0)


def make_flow(
    key: Key[Array, ""],
    dim: int = 2,
    n_bins: int = 8,
    width: int = 64,
    depth: int = 2,
    n_layers: int = 2,
) -> list[eqx.Module]:
    """Build spline layers ordered from base to data, with PLU mixing between."""
    transform = RQS(n_bins)
    layers = []
    for i, k in enumerate(jr.split(key, n_layers)):
        k_net, k_mix = jr.split(k)
        if i > 0:
            layers.append(PLU(dim, rng=k_mix))
        net = CausalMLP(
            num_ranks=dim,
            in_rank_dim="scalar",
            out_rank_dim=transform.n_params,
            width=width,
            depth=depth,
            rng=k_net,
        )
        layers.append(MAF(net, transform))
    return layers


def log_prob(layers: list[eqx.Module], x: Float[Array, " d"]) -> Float[Array, ""]:
    """Log density at ``x`` under a standard normal base."""
    total = 0.0
    for layer in reversed(layers):
        x, ld = layer.inv_logdet(x)
        total = total + ld
    return norm.logpdf(x).sum() + total


def sample(
    layers: list[eqx.Module], key: Key[Array, ""], n: int, dim: int = 2
) -> Float[Array, "n d"]:
    """Draw ``n`` samples by pushing base noise through the layers."""

    def push(z):
        for layer in layers:
            z, _ = layer.fwd_logdet(z)
        return z

    return jax.vmap(push)(jr.normal(key, (n, dim)))


def mean_nll(layers: list[eqx.Module], data: Float[Array, "n d"]) -> Float[Array, ""]:
    """Mean negative log-likelihood of ``data``."""
    return -jax.vmap(lambda x: log_prob(layers, x))(data).mean()


def train(
    layers: list[eqx.Module],
    data: Float[Array, "n d"],
    *,
    steps: int = 600,
    lr: float = 5e-3,
    batch: int = 256,
    seed: int = 0,
    log_every: int | None = None,
) -> tuple[list[eqx.Module], float, float]:
    """Fit by maximum likelihood with Adam; return the layers and initial/final NLL."""
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(layers, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, opt_state, xb):
        loss, grads = eqx.filter_value_and_grad(mean_nll)(m, xb)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(m, eqx.is_inexact_array)
        )
        return eqx.apply_updates(m, updates), opt_state, loss

    init_nll = float(mean_nll(layers, data))
    key = jr.key(seed)
    for i in range(steps):
        key, k = jr.split(key)
        idx = jr.randint(k, (batch,), 0, data.shape[0])
        layers, opt_state, loss = step(layers, opt_state, data[idx])
        if log_every and (i + 1) % log_every == 0:
            print(f"step {i + 1}/{steps} | nll={float(loss):.3f}", flush=True)
    return layers, init_nll, float(mean_nll(layers, data))


if __name__ == "__main__":
    data = two_moons(jr.key(42), 2000)
    steps = 2000
    _, init_nll, final_nll = train(
        make_flow(jr.key(0)), data, steps=steps, log_every=steps // 20
    )
    print(f"nll {init_nll:.3f} -> {final_nll:.3f}")
