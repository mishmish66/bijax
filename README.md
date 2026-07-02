# bijax

This is a really small equinox flows library that only depends on equinox.
The scope of this is very small, it's really just meant to give some basic neural bijections.

The interface for these bijections is `fwd_logdet`/`inv_logdet`.
To actually define a flow model usually our bijections would have to be combined with a base density.

## Modules

### Supporting characters
- CausalLinear: autoregressive Linear which maintains causal dependency between inputs and outputs ([MADE][made], building on [Bengio & Bengio 1999][bengio99])
- CausalMLP: autoregressive MLP as per [MADE][made]

### Bijectors
- PLU: PLU decomposed linear bijector ([Glow][glow])
- AffineCoupling: Invertible affine coupling transform with an MLP ([RealNVP][realnvp])
- SplineCoupling: Invertible spline coupling transform with an MLP ([Neural Spline Flows][nsf])
- ARSpline: Invertible autoregressive spline with a causal MLP ([Neural Spline Flows][nsf] + [MAF][maf])
- ARAffine: Invertible autoregressive affine with a causal MLP ([MAF][maf])

## References
- <a id="made"></a>[MADE: Masked Autoencoder for Distribution Estimation][made] — Germain, Gregor, Murray & Larochelle, ICML 2015
- <a id="bengio99"></a>[Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks][bengio99] — Bengio & Bengio, NeurIPS 1999
- <a id="glow"></a>[Glow: Generative Flow with Invertible 1x1 Convolutions][glow] — Kingma & Dhariwal, NeurIPS 2018
- <a id="realnvp"></a>[Density estimation using Real NVP][realnvp] — Dinh, Sohl-Dickstein & Bengio, ICLR 2017
- <a id="nsf"></a>[Neural Spline Flows][nsf] — Durkan, Bekasov, Murray & Papamakarios, NeurIPS 2019
- <a id="maf"></a>[Masked Autoregressive Flow for Density Estimation][maf] — Papamakarios, Pavlakou & Murray, NeurIPS 2017

[made]: https://proceedings.mlr.press/v37/germain15.html
[bengio99]: http://papers.nips.cc/paper/1679-modeling-high-dimensional-discrete-data-with-multi-layer-neural-networks
[glow]: https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html
[realnvp]: https://openreview.net/forum?id=HkpbnH9lx
[nsf]: https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html
[maf]: https://proceedings.neurips.cc/paper_files/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf
