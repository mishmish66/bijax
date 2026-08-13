"""GLOW-style neural spline coupling flow.

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

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from bijax.spline import spline_fwd, spline_inv


class SplineCoupling(eqx.Module):
    """One RQ-spline coupling layer"""

    mlp: eqx.nn.MLP
    id_idxs: tuple[int, ...] = eqx.field(static=True)
    tr_idxs: tuple[int, ...] = eqx.field(static=True)

    n_bins: int = eqx.field(static=True)

    low: float = eqx.field(static=True, default=-5.0)
    high: float = eqx.field(static=True, default=5.0)

    @staticmethod
    def mlp_dims(flow_dim: int, cond_dim: int, n_bins: int) -> tuple[int, int]:
        in_dim = flow_dim // 2 + cond_dim
        out_dim = (flow_dim - flow_dim // 2) * (n_bins * 3 - 1)
        return in_dim, out_dim

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None):
        id_ix, tr_ix = jnp.array(self.id_idxs), jnp.array(self.tr_idxs)
        tform_dim = len(self.tr_idxs)
        if c is None:
            h = x[id_ix]
        else:
            h = jnp.concat([x[id_ix], c])

        mlp_out = self.mlp(h)

        param_dim = tform_dim * (self.n_bins * 3 - 1)
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        params = mlp_out.reshape(tform_dim, -1)
        for_tform = x[tr_ix]
        tformed, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None))(
            for_tform, params, jnp.array(self.low), jnp.array(self.high)
        )
        return x.at[tr_ix].set(tformed), ld

    def inv_logdet(self, z: Float[Array, " d"], c: Float[Array, " c"] | None):
        id_ix, tr_ix = jnp.array(self.id_idxs), jnp.array(self.tr_idxs)
        tform_dim = len(self.tr_idxs)
        if c is None:
            h = z[id_ix]
        else:
            h = jnp.concat([z[id_ix], c])
        mlp_out = self.mlp(h)

        param_dim = tform_dim * (self.n_bins * 3 - 1)
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        params = mlp_out.reshape(tform_dim, -1)
        for_tform = z[tr_ix]
        tformed, ld = jax.vmap(spline_inv, in_axes=(0, 0, None, None))(
            for_tform, params, jnp.array(self.low), jnp.array(self.high)
        )

        return z.at[tr_ix].set(tformed), ld
