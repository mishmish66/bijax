"""GLOW-style PLU decomposed bijector.

@inproceedings{kingma_glow_2018,
    title = {Glow: {Generative} {Flow} with {Invertible} 1x1 {Convolutions}},
    url = {https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html},
    booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 31 ({NeurIPS} 2018)},
    author = {Kingma, Diederik P. and Dhariwal, Prafulla},
    editor = {Bengio, Samy and Wallach, Hanna M. and Larochelle, Hugo and Grauman, Kristen and Cesa-Bianchi, Nicolò and Garnett, Roman},
    year = {2018},
    pages = {10236--10245},
}
"""

from jax import numpy as jnp
from jax import random as jr
from jax.lax import stop_gradient as sg
from jax.scipy import linalg as jla

import equinox as eqx
from jaxtyping import Float, Array, Key


class PLU(eqx.Module):
    """PLU-decomposition linear bijector (GLOW-style invertible mixing)"""

    dim: int = eqx.field(static=True)
    lflat: Float[Array, " ldim"]
    uflat: Float[Array, " udim"]
    logs: Float[Array, " dim"]
    signs: Float[Array, " dim"]
    p: Float[Array, "dim dim"]

    def __init__(self, dim: int, *, rng: Key[Array, ""]):
        w = jr.orthogonal(rng, dim)
        self.p, lmat, umat = jla.lu(w)
        self.lflat = lmat[jnp.tril_indices(dim, -1)]
        self.uflat = umat[jnp.triu_indices(dim, 1)]
        self.logs = jnp.log(jnp.abs(jnp.diag(umat)))
        self.signs = jnp.sign(jnp.diag(umat))
        self.dim = dim

    @property
    def l(self) -> Float[Array, " {self.dim} {self.dim}"]:
        return jnp.eye(self.dim).at[jnp.tril_indices(self.dim, -1)].set(self.lflat)

    @property
    def u(self) -> Float[Array, " {self.dim} {self.dim}"]:
        ud = jnp.diag(sg(self.signs) * jnp.exp(self.logs))
        return ud.at[jnp.triu_indices(self.dim, 1)].set(self.uflat)

    def fwd_logdet(self, x: Float[Array, " d"]):
        return sg(self.p) @ self.l @ self.u @ x, self.logs.sum()

    def inv_logdet(self, y: Float[Array, " d"]):
        z = sg(self.p.T) @ y
        ux = jla.solve_triangular(self.l, z, lower=True, unit_diagonal=True)
        x = jla.solve_triangular(self.u, ux, lower=False)
        return x, -self.logs.sum()
