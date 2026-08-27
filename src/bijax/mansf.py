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

from typing import Literal

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from bijax.causal_mlp import CausalMLP
from bijax.rational_quadratic_spline import rqs_fwd, rqs_inv


class ARSpline(eqx.Module):
    """Masked-autoregressive RQ spline layer."""

    net: CausalMLP
    low: float = eqx.field(static=True, default=-5.0)
    high: float = eqx.field(static=True, default=5.0)
    min_slope: float | None = eqx.field(static=True, default=1e-3)
    direction: Literal["maf"] | Literal["iaf"] = eqx.field(static=True, default="maf")

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        r"""Apply the mapping represented by this `ARSpline` and get the log determinant.

        Computes the inverse of the bijection represented by this
        `ARSpline` along with the log jacobian determinant
        \\(\log\left|\frac{\partial f(x)}{\partial x}\right|\\) for downstream flows to use
        it in the change of variables formula.

        Parameters
        ----------
        x : Float[Array, " d"]
            The value to map.
        c : Float[Array, " c"], optional
            Conditioning data for the neural network.

        Raises
        ------
        ValueError
            Raises when direction is is invalid

        Examples
        --------
        FIXME: Add docs.

        """
        if self.direction == "maf":
            return self._slow(x, c)
        if self.direction == "iaf":
            return self._fast(x, c)
        msg = f"direction must be maf or iaf not {self.direction}"
        raise ValueError(msg)

    def inv_logdet(self, y: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        r"""Invert the mapping represented by this `ARSpline` and get the log determinant.

        Computes the inverse of the bijection represented by this
        `ARSpline` along with the log jacobian determinant
        \\(\log\left|\frac{\partial f^{-1}(y)}{\partial y}\right|\\) for downstream flows to use
        it in the change of variables formula.

        Parameters
        ----------
        y : Float[Array, " d"]
            The value to map.
        c : Float[Array, " c"], optional
            Conditioning data for the neural network.

        Raises
        ------
        ValueError
            Raises when direction is is invalid

        Examples
        --------
        FIXME: Add docs.

        """
        if self.direction == "maf":
            return self._fast(y, c)
        if self.direction == "iaf":
            return self._slow(y, c)
        msg = f"direction must be maf or iaf not {self.direction}"
        raise ValueError(msg)

    def _fast(self, inp: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        params = self.net(inp, c)  # (dim,) scalar-per-row -> (dim, n_params)
        outp, ld = jax.vmap(rqs_fwd, in_axes=(0, 0, None, None, None))(
            inp,
            params,
            self.low,
            self.high,
            self.min_slope,
        )
        return outp, ld.sum()

    def _slow(self, inp: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        outp = jnp.zeros(self.net.num_ranks)
        for i in range(self.net.num_ranks):
            params = self.net(outp, c)  # (dim, n_params)
            outp_i, _ = rqs_inv(
                inp[i],
                params[i],
                self.low,
                self.high,
                self.min_slope,
            )
            outp = outp.at[i].set(outp_i)
        # log-det of the inverse is minus that of the forward at the solved x
        params = self.net(outp, c)
        _, ld = jax.vmap(rqs_fwd, in_axes=(0, 0, None, None, None))(
            outp,
            params,
            self.low,
            self.high,
            self.min_slope,
        )
        return outp, -ld.sum()
