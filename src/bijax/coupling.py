"""Coupling layer, following RealNVP and Neural Spline Flows.

@inproceedings{dinh_density_2017,
    title = {Density estimation using {Real} {NVP}},
    url = {https://openreview.net/forum?id=HkpbnH9lx},
    booktitle = {5th {International} {Conference} on {Learning} {Representations} ({ICLR} 2017)},
    publisher = {OpenReview.net},
    author = {Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
    year = {2017},
}

@inproceedings{durkan_neural_2019,
    title = {Neural {Spline} {Flows}},
    url = {https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html},
    booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 32 ({NeurIPS} 2019)},
    author = {Durkan, Conor and Bekasov, Artur and Murray, Iain and Papamakarios, George},
    editor = {Wallach, Hanna M. and Larochelle, Hugo and Beygelzimer, Alina and d'Alché-Buc, Florence and Fox, Emily B. and Garnett, Roman},
    year = {2019},
    pages = {7509--7520},
}
"""

from collections.abc import Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from bijax._conditioner import condition
from bijax.elementwise import Elementwise


class Coupling(eqx.Module):
    """Transform ``x[tr_idxs]`` elementwise, conditioned on ``x[id_idxs]``.

    ``conditioner(x[id_idxs], c, rng=rng)`` returns
    ``(len(tr_idxs), transform.n_params)`` params, row ``i`` for coordinate
    ``tr_idxs[i]``.
    """

    conditioner: Callable[..., Float[Array, "n p"]]
    transform: Elementwise
    id_idxs: tuple[int, ...] = eqx.field(static=True)
    tr_idxs: tuple[int, ...] = eqx.field(static=True)

    def fwd_logdet(
        self,
        x: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``x`` forward and return ``log|det J|``."""
        return self._apply(self.transform.fwd, x, c, rng)

    def inv_logdet(
        self,
        y: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``y`` back and return ``log|det J|`` of the inverse."""
        return self._apply(self.transform.inv, y, c, rng)

    def _apply(self, f, x, c, rng):
        id_ix = jnp.array(self.id_idxs, dtype=int)
        tr_ix = jnp.array(self.tr_idxs, dtype=int)
        shape = (len(self.tr_idxs), self.transform.n_params)
        params = condition(self.conditioner, x[id_ix], c, rng, shape)
        out, ld = jax.vmap(f)(x[tr_ix], params)
        return x.at[tr_ix].set(out), ld.sum()
