"""Autoregressive masked linear / MLP, following MADE.

@inproceedings{DBLP:conf/nips/BengioB99,
  author       = {Yoshua Bengio and
                  Samy Bengio},
  editor       = {Sara A. Solla and
                  Todd K. Leen and
                  Klaus{-}Robert M{\"{u}}ller},
  title        = {Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks},
  booktitle    = {Advances in Neural Information Processing Systems 12, {[NIPS} Conference,
                  Denver, Colorado, USA, November 29 - December 4, 1999]},
  pages        = {400--406},
  publisher    = {The {MIT} Press},
  year         = {1999},
  url          = {http://papers.nips.cc/paper/1679-modeling-high-dimensional-discrete-data-with-multi-layer-neural-networks},
  timestamp    = {Mon, 16 May 2022 15:41:51 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/BengioB99.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{germain_made_2015,
    address = {Lille, France},
    series = {Proceedings of {Machine} {Learning} {Research}},
    title = {{MADE}: {Masked} {Autoencoder} for {Distribution} {Estimation}},
    volume = {37},
    url = {https://proceedings.mlr.press/v37/germain15.html},
    booktitle = {Proceedings of the 32nd {International} {Conference} on {Machine} {Learning}},
    publisher = {PMLR},
    author = {Germain, Mathieu and Gregor, Karol and Murray, Iain and Larochelle, Hugo},
    editor = {Bach, Francis and Blei, David},
    month = jul,
    year = {2015},
    pages = {881--889},
}
"""

from collections.abc import Callable
from typing import Literal, overload
from itertools import pairwise

from jax import numpy as jnp
import jax
from jax import random as jr

import equinox as eqx
from jaxtyping import Float, Array, Key, Int


class CausalLinear(eqx.Module):
    """Masked linear with per-unit autoregressive ranks."""

    w_flat: Float[Array, " n"]
    bias: Float[Array, " out"]
    unmasked_idxs: tuple[Int[Array, " n"], ...]

    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_ranks: list[int],
        out_ranks: list[int],
        *,
        rng: Key[Array, ""],
    ):
        """A flexible masked linear layer.

        This layer supports flexible dependencies between inputs and outputs in
        the form of ranks. Each output can depend only on inputs with the same
        or lower rank. This also means that for lower ranking outputs the
        effective number of parameters involved in the computation is quite low.

        Args:
          in_ranks: List with length of input dim specifying the rank of each
          out_ranks: List with length of output dim specifying the rank of each
          rng: key for random parameter generation
        """
        self.in_dim = len(in_ranks)
        self.out_dim = len(out_ranks)

        # mask[i, j] is True when output j may read input i, laid out as
        # (in_dim, out_dim) so it indexes the weight matrix directly.
        mask = jnp.array(in_ranks)[:, None] <= jnp.array(out_ranks)[None, :]
        fan_in = mask.sum(axis=0)
        lim = 1.0 / jnp.sqrt(fan_in.clip(min=1))

        rng, wkey, bkey = jr.split(rng, 3)

        w_unif = jr.uniform(
            wkey, (len(in_ranks), len(out_ranks)), minval=-1.0, maxval=1.0
        )
        w_full = w_unif * lim[None, :]

        self.unmasked_idxs = jnp.nonzero(mask)
        self.w_flat = w_full[self.unmasked_idxs]
        self.bias = jr.uniform(bkey, (len(out_ranks),))

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out"]:
        w_full = (
            jnp.zeros((self.in_dim, self.out_dim))
            .at[self.unmasked_idxs]
            .set(self.w_flat)
        )
        return jnp.einsum("i,io->o", x, w_full) + self.bias


class CausalMLP(eqx.Module):
    """Autoregressive MLP over a rank axis: x of shape (l, di) -> (l, do)."""

    layers: list[CausalLinear]
    activation: Callable[[Array], Array] = eqx.field(static=True)
    num_ranks: int = eqx.field(static=True)
    cond_dim: int | None = eqx.field(static=True)
    in_rank_dim: int | Literal["scalar"] = eqx.field(static=True)
    out_rank_dim: int | Literal["scalar"] = eqx.field(static=True)

    @overload
    def __init__(
        self,
        num_ranks: int,
        in_rank_dim: int | Literal["scalar"],
        out_rank_dim: int | Literal["scalar"],
        width: int,
        depth: int,
        *,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        """Randomly intialize a CausalMLP

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
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        """Randomly intialize a CausalMLP

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
        self, x: Float[Array, "l di"] | Float[Array, " l"], c: Float[Array, " c"] | None
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
        # scalar output stays (l,); otherwise unflatten to (l, do).
        if self.out_rank_dim == "scalar":
            return h
        return h.reshape(self.num_ranks, self.out_rank_dim)  # (l*do,) -> (l, do)
