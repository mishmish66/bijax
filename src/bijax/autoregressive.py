"""Masked autoregressive layers, following MAF, IAF and Neural Spline Flows.

@inproceedings{papamakarios_masked_2017,
    title = {Masked {Autoregressive} {Flow} for {Density} {Estimation}},
    volume = {30},
    url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf},
    booktitle = {Advances in {Neural} {Information} {Processing} {Systems}},
    publisher = {Curran Associates, Inc.},
    author = {Papamakarios, George and Pavlakou, Theo and Murray, Iain},
    editor = {Guyon, I. and Luxburg, U. Von and Bengio, S. and Wallach, H. and Fergus, R. and Vishwanathan, S. and Garnett, R.},
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

@inproceedings{kingma_improved_2016,
    title = {Improved {Variational} {Inference} with {Inverse} {Autoregressive} {Flow}},
    url = {https://arxiv.org/abs/1606.04934},
    booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 29 ({NeurIPS} 2016)},
    author = {Kingma, Diederik P. and Salimans, Tim and Jozefowicz, Rafal and Chen, Xi and Sutskever, Ilya and Welling, Max},
    year = {2016},
}
"""

from collections.abc import Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from bijax._conditioner import condition
from bijax.elementwise import Elementwise

Conditioner = Callable[..., Float[Array, "n p"]]


class MAF(eqx.Module):
    """``inv_logdet`` is one conditioner call; ``fwd_logdet`` solves one coordinate at a time.

    ``conditioner(x, c, rng=rng)`` returns ``(len(x), transform.n_params)``
    params whose row ``i`` depends only on ``x[:i]`` and ``c``.
    """

    conditioner: Conditioner
    transform: Elementwise

    def fwd_logdet(
        self,
        x: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``x`` forward and return ``log|det J|``."""
        return _solve(self.conditioner, self.transform, x, c, rng)

    def inv_logdet(
        self,
        y: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``y`` back and return ``log|det J|`` of the inverse."""
        return _one_pass(self.conditioner, self.transform, y, c, rng)


class IAF(eqx.Module):
    """``fwd_logdet`` is one conditioner call; ``inv_logdet`` solves one coordinate at a time.

    ``conditioner(x, c, rng=rng)`` returns ``(len(x), transform.n_params)``
    params whose row ``i`` depends only on ``x[:i]`` and ``c``.
    """

    conditioner: Conditioner
    transform: Elementwise

    def fwd_logdet(
        self,
        x: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``x`` forward and return ``log|det J|``."""
        return _one_pass(self.conditioner, self.transform, x, c, rng)

    def inv_logdet(
        self,
        y: Float[Array, " d"],
        c: Float[Array, " c"] | None = None,
        *,
        rng: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        """Map ``y`` back and return ``log|det J|`` of the inverse."""
        return _solve(self.conditioner, self.transform, y, c, rng)


def _params(conditioner, transform, x, c, rng):
    return condition(conditioner, x, c, rng, (x.shape[0], transform.n_params))


def _one_pass(conditioner, transform, inp, c, rng):
    """Apply ``transform.fwd`` to every coordinate after one conditioner call."""
    params = _params(conditioner, transform, inp, c, rng)
    out, ld = jax.vmap(transform.fwd)(inp, params)
    return out, ld.sum()


def _solve(conditioner, transform, inp, c, rng):
    """Invert `_one_pass` one coordinate at a time."""
    out = jnp.zeros_like(inp)
    for i in range(inp.shape[0]):
        params = _params(conditioner, transform, out, c, rng)
        out_i, _ = transform.inv(inp[i], params[i])
        out = out.at[i].set(out_i)
    # log-det of the inverse is minus that of the forward at the solved point
    _, ld = _one_pass(conditioner, transform, out, c, rng)
    return out, -ld
