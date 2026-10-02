r"""Masked linear layer with rank-ordered dependencies, following MADE.

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

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key


class CausalLinear(eqx.Module):
    """A linear layer with a configurable dependency structure.

    By zeroing weights relating higher rank inputs to lower rank outputs this
    linear layer makes outputs depend only on inputs lower ranked than them.

    Examples
    --------
    FIXME: Add docs.

    """

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
        """Randomly initialize a `CausalLinear`.

        supports flexible dependencies between inputs and outputs in
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
        w_full = jnp.zeros((self.in_dim, self.out_dim))
        w_full = w_full.at[self.unmasked_idxs].set(self.w_flat)
        return jnp.einsum("i,io->o", x, w_full) + self.bias
