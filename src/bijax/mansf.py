"""Masked-autoregressive RQ spline flow: spline transform with MAF-style masking.

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
"""

from jaxtyping import Float, Array

import jax
from jax import numpy as jnp
import equinox as eqx

from bijax.spline import spline_fwd, spline_inv
from bijax.causal_mlp import CausalMLP


class ARSpline(eqx.Module):
    """Masked-autoregressive RQ spline layer"""

    net: CausalMLP
    low: float = eqx.field(static=True, default=-5.0)
    high: float = eqx.field(static=True, default=5.0)
    min_slope: float = eqx.field(static=True, default=1e-3)

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        params = self.net(x, c)  # (dim,) scalar-per-row -> (dim, n_params)
        z, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None, None))(
            x,
            params,
            jnp.array(self.low),
            jnp.array(self.high),
            self.min_slope,
        )
        return z, ld.sum()

    def inv_logdet(self, z: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        x = jnp.zeros(self.net.num_ranks)
        for i in range(self.net.num_ranks):
            params = self.net(x, c)  # (dim, n_params)
            xi, _ = spline_inv(
                z[i],
                params[i],
                jnp.array(self.low),
                jnp.array(self.high),
                self.min_slope,
            )
            x = x.at[i].set(xi)
        # log-det of the inverse is minus that of the forward at the solved x
        params = self.net(x, c)
        _, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None, None))(
            x,
            params,
            jnp.array(self.low),
            jnp.array(self.high),
            self.min_slope,
        )
        return x, -ld.sum()
