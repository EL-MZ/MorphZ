# Examples

The notebooks below demonstrate MorphZ on standard benchmark problems.
Each notebook is self-contained: it defines the model, draws posterior
samples, runs `evidence(...)`, and inspects the result.

| Notebook | Description |
|---|---|
| **NumPyro MorphZ ln(Z)** | Additional NumPyro example focusing on log-evidence comparison across morph types. |
| **Eggbox** | Multimodal likelihood on a 2-D periodic "eggbox" surface. Tests MorphZ's ability to handle complex, disconnected posterior modes. |
| **Gaussian Shell** | Two concentric Gaussian shells — a classic nested-sampling benchmark with a thin, curved posterior. |
| **Peak Sampling** | Sharply peaked posterior; demonstrates the effect of bandwidth selection and morph type on evidence accuracy. |
| **NumPyro Gaussian Shell** | End-to-end integration with [NumPyro](https://num.pyro.ai): run NUTS inside NumPyro, pass the samples to MorphZ, estimate the evidence. |
| **JAXNS Gaussian Shell** | End-to-end integration with [JAXNS](https://jaxns.readthedocs.io/en/latest/): run nested sampling, resample posterior draws, pass them to MorphZ, and compare evidence estimates. |


```{tableofcontents}
```
