"""Masked-autoregressive affine flow (MAF).

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
"""

from jaxtyping import Float, Array

from jax import numpy as jnp
import equinox as eqx

from bijax.causal_mlp import CausalMLP


class ARAffine(eqx.Module):
    """Masked-autoregressive affine layer.

    `net` maps each coordinate's predecessors to a (log-scale, shift) pair, so
    the forward pass is a single vectorized call and the inverse is solved one
    coordinate at a time.
    """

    net: CausalMLP
    min_scale: float = eqx.field(static=True, default=1e-3)

    def _log_scale(self, s_raw: Float[Array, " d"]) -> Float[Array, " d"]:
        # soft-clamp the log-scale to (log min_scale, -log min_scale) for stability
        return s_raw / (1 + jnp.abs(s_raw / jnp.log(self.min_scale)))

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        params = self.net(x, c)  # (dim, 2): log-scale and shift per coordinate
        s = self._log_scale(params[:, 0])
        t = params[:, 1]
        z = x * jnp.exp(s) + t
        return z, s.sum()

    def inv_logdet(self, z: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        x = jnp.zeros(self.net.num_ranks)
        for i in range(self.net.num_ranks):
            params = self.net(x, c)  # coordinate i reads only x_{<i}, so far set
            s = self._log_scale(params[i, 0])
            x = x.at[i].set((z[i] - params[i, 1]) * jnp.exp(-s))
        # log-det of the inverse is minus the forward log-det at the solved x
        s = self._log_scale(self.net(x, c)[:, 0])
        return x, -s.sum()
