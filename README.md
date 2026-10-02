# bijax

A small equinox library of composable normalizing-flow primitives.

Every bijector exposes `fwd_logdet(x, c=None, *, rng=None)` and `inv_logdet(y, c=None, *, rng=None)`, each returning the mapped value and a scalar log-determinant, so layers stack by chaining calls and summing log-determinants.
A flow model combines them with a base density.

```python
from bijax import RQS, Coupling

layer = Coupling(conditioner, RQS(n_bins=8), id_idxs=(0, 2), tr_idxs=(1, 3))
y, ld = layer.fwd_logdet(x, c)
```

## Primitives

### Bijectors
- PLU: PLU-decomposed linear bijector ([Glow][glow])
- Coupling: transforms some coordinates elementwise, conditioned on the rest ([RealNVP][realnvp], [Neural Spline Flows][nsf])
- MAF: transforms each coordinate elementwise, conditioned on the ones before it; `inv_logdet` is one conditioner call and `fwd_logdet` solves one coordinate at a time ([MAF][maf], [Neural Spline Flows][nsf])
- IAF: the same layer with the fast and slow directions swapped ([IAF][iaf])

### Elementwise transforms
Subclass `Elementwise` (`n_params`, `fwd`, `inv`) to add one; it then works in every structure.
- Affine: `x * exp(s) + t` with a bounded log-scale ([RealNVP][realnvp])
- RQS: rational-quadratic spline on `[lower, upper]` with identity tails ([Neural Spline Flows][nsf]); `rqs_fwd`/`rqs_inv` are its underlying functions

### Conditioners
`Coupling`, `MAF` and `IAF` take any callable `conditioner(x, c, rng=rng)` returning an `(n, transform.n_params)` array; `rng` is `None` unless you pass a key.
For `MAF` and `IAF`, row `i` must depend only on `x[:i]` and `c`.
- CausalLinear: linear layer with rank-ordered dependencies, the building block of MADE conditioners ([MADE][made], building on [Bengio & Bengio 1999][bengio99])

## Example
`examples/made_spline_flow.py` builds a MADE conditioner from `CausalLinear`, stacks `MAF(..., RQS(...))` layers with `PLU` mixing between them, and fits two moons:

```sh
uv run python examples/made_spline_flow.py
```

## References
- <a id="made"></a>[MADE: Masked Autoencoder for Distribution Estimation][made] — Germain, Gregor, Murray & Larochelle, ICML 2015
- <a id="bengio99"></a>[Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks][bengio99] — Bengio & Bengio, NeurIPS 1999
- <a id="glow"></a>[Glow: Generative Flow with Invertible 1x1 Convolutions][glow] — Kingma & Dhariwal, NeurIPS 2018
- <a id="realnvp"></a>[Density estimation using Real NVP][realnvp] — Dinh, Sohl-Dickstein & Bengio, ICLR 2017
- <a id="nsf"></a>[Neural Spline Flows][nsf] — Durkan, Bekasov, Murray & Papamakarios, NeurIPS 2019
- <a id="maf"></a>[Masked Autoregressive Flow for Density Estimation][maf] — Papamakarios, Pavlakou & Murray, NeurIPS 2017
- <a id="iaf"></a>[Improved Variational Inference with Inverse Autoregressive Flow][iaf] — Kingma, Salimans, Jozefowicz, Chen, Sutskever & Welling, NeurIPS 2016

[made]: https://proceedings.mlr.press/v37/germain15.html
[bengio99]: http://papers.nips.cc/paper/1679-modeling-high-dimensional-discrete-data-with-multi-layer-neural-networks
[glow]: https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html
[realnvp]: https://openreview.net/forum?id=HkpbnH9lx
[nsf]: https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html
[maf]: https://proceedings.neurips.cc/paper_files/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf
[iaf]: https://arxiv.org/abs/1606.04934
