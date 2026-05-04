<p align="center">
  <img src="_static/mz_final.png" alt="MorphZ logo" width="360">
</p>

# MorphZ

## What MorphZ Computes

MorphZ estimates the Bayesian marginal likelihood (evidence) $z$ from posterior
samples.  It is a post-processing tool: it does not run its own sampler, and
places no requirements on which sampler produced the posterior draws — nested
sampling (e.g. dynesty), parallel-tempered MCMC, HMC, NUTS (e.g. NumPyro), or
anything else.

Given posterior samples, the prior, and the likelihood, MorphZ returns
$\log \hat z$ together with a relative-error diagnostic.

MorphZ targets the integral

$$
z(X \mid \mathcal{M}) = \int_{\Theta}
  L(X \mid \boldsymbol{\theta}, \mathcal{M})\,
  \pi(\boldsymbol{\theta} \mid \mathcal{M})\,
  d\boldsymbol{\theta},
$$

which underpins Bayesian model comparison through the Bayes factor

$$
\mathrm{BF}(\mathcal{M}_0 / \mathcal{M}_1) =
\frac{z(X \mid \mathcal{M}_0)}{z(X \mid \mathcal{M}_1)}.
$$

---

## How It Works

MorphZ has two components.

### The Morph Approximation

MorphZ first builds a tractable approximation to the posterior, called the
**Morph approximation**, by factorising the $d$-dimensional posterior into a
product of low-dimensional joint factors over **disjoint blocks** of
parameters, plus marginals over leftover singleton parameters:

$$
\mathcal{M}_{B_L}(\boldsymbol{\theta}) =
  \prod_{b \in B_L} P_b(\boldsymbol{\theta}_b)
  \;\prod_{s \in S(B_L)} P_s(\theta_s).
$$

Here $L$ is the block length, $B_L$ is a partition of the parameter indices
$\Gamma = \{1,\ldots,d\}$ into disjoint blocks of length $L$, and $S(B_L)$
collects any leftover (singleton) indices.

Block membership is chosen by maximising the sum of **block total correlations**

$$
C(\boldsymbol{\theta}_b) =
  \sum_{k \in b} H(\theta_k) - H(\boldsymbol{\theta}_b),
$$

an information-theoretic measure of joint dependence (linear and non-linear)
inside the block.  This is principled: maximising
$\sum_b C(\boldsymbol{\theta}_b)$ is equivalent to minimising
$D_{\mathrm{KL}}(P\,\|\,\mathcal{M}_{B_L})$, so the Morph approximation is
the closest product-of-blocks approximation to the true posterior at order $L$.

The optimisation is solved with the **Seeded Greedy Maximisation (SGM)**
algorithm.  Block and singleton densities are fit with multivariate
Gaussian-kernel KDEs.

### Optimal Bridge Sampling

MorphZ then uses $\mathcal{M}_{B_L}$ as the proposal $g(\boldsymbol{\theta})$
inside an **optimal bridge sampling** estimator,

$$
z =
\frac{\mathbb{E}_{g(\boldsymbol{\theta})}\!\left[
  L(X\mid\boldsymbol{\theta})\,\pi(\boldsymbol{\theta})\,h(\boldsymbol{\theta})
\right]}
{\mathbb{E}_{\mathrm{post}}\!\left[
  h(\boldsymbol{\theta})\,g(\boldsymbol{\theta})
\right]},
$$

with the minimum-variance bridge function

$$
h(\boldsymbol{\theta}) = C \cdot
\frac{1}{f_1\,L(X\mid\boldsymbol{\theta})\,\pi(\boldsymbol{\theta})
\;+\; f_2\,z\,g(\boldsymbol{\theta})},
$$

where $f_1 = N_1/(N_1+N_2)$ and $f_2 = N_2/(N_1+N_2)$ are the posterior and
proposal sample fractions.  Because $h$ depends on $z$, MorphZ solves for
$\hat z$ iteratively from an initial guess.

To avoid reuse bias, MorphZ splits the supplied posterior samples into one
batch for fitting $\mathcal{M}_{B_L}$ and a separate batch for bridge
sampling.

---

## What MorphZ Returns

For each call, MorphZ produces:

- $\log\hat z$, the log-evidence estimate.
- A bridge-sampling **relative mean-squared error** diagnostic on $\hat z$,
  computed from the same posterior and proposal samples, so no additional
  likelihood evaluations are needed.

---

## Citation

If you use MorphZ, please cite:

```bibtex
@article{1554-y6ns,
  title   = {Enhancing evidence estimation through informed probability density approximation},
  author  = {Zahraoui, El Mehdi and Maturana-Russel, Patricio and Vajpeyi, Avi
             and van Straten, Willem and Meyer, Renate and Gulyaev, Sergei},
  journal = {Phys. Rev. D},
  volume  = {113},
  issue   = {8},
  pages   = {083014},
  year    = {2026},
  month   = {Apr},
  publisher = {American Physical Society},
  doi     = {10.1103/1554-y6ns},
  url     = {https://link.aps.org/doi/10.1103/1554-y6ns}
}
```

---

```{include} _auto/README.md
:relative-docs: .
:relative-images:
```

```{note}
The README content above is copied automatically from the project root each
time the documentation is built.  To refresh it locally, run
`./docs/build_docs.sh`.
```
