# MorphZ

[![Python versions](https://img.shields.io/pypi/pyversions/morphZ.svg)](https://pypi.org/project/morphZ/)
[![PyPI version](https://img.shields.io/pypi/v/morphZ.svg)](https://pypi.org/project/morphZ/)
[![CI](https://github.com/EL-MZ/MorphZ/actions/workflows/ci.yml/badge.svg)](https://github.com/EL-MZ/MorphZ/actions/workflows/ci.yml)
[![Docs](https://github.com/EL-MZ/MorphZ/actions/workflows/docs.yml/badge.svg)](https://github.com/EL-MZ/MorphZ/actions/workflows/docs.yml)
[![GitHub Repo](https://img.shields.io/badge/GitHub-EL--MZ%2FMorphZ-181717?logo=github)](https://github.com/EL-MZ/MorphZ)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10283-B31B1B.svg)](https://arxiv.org/abs/2512.10283)

MorphZ for high accuracy marginal likelihood estimation and morphological density approximation toolkit for scientific workflows, with utilities for dependency analysis.

- Flexible Morph backends: independent, pairwise, grouped, and "tree-structured".
- Bandwidth selection: Scott, Silverman, Botev ISJ, and cross-validation variants.
- Evidence estimation via bridge sampling with robust diagnostics.
- Mutual information and Total correlation estimation.
- Mutual information and Chow–Liu dependency tree visualisation.

## Installation

Python 3.10+ is recommended.

```bash
pip install morphz
```

From source (editable):

```bash
pip install -e .
```

## Run The Examples

Interactive notebooks live in `examples/`:

- `examples/eggbox.ipynb` — eggbox likelihood (dynesty nested sampling)
- `examples/gaussian shell.ipynb` — Gaussian shell (dynesty nested sampling)
- `examples/peak_sampling.ipynb` — sharply peaked posterior
- `examples/numpyro_gaussian_shell.ipynb` — Gaussian shell with NUTS via NumPyro
- `examples/numpyro_morphz_lnz.ipynb` — log-evidence comparison across morph types (NumPyro/NUTS)

## Documentation

Jupyter Book powers the project docs. During each build the helper script copies
`README.md` and the contents of `examples/` into `docs/_auto/` so that the book
always reflects the latest files without committing the generated copies.

```bash
python -m pip install 'jupyter-book<2'
./docs/build_docs.sh
```

HTML output is written to `docs/_build/html`, and GitHub Actions publishes it to
GitHub Pages automatically on pushes to `main`.

## API Highlights

- Morphs: `Morph_Indep`, `Morph_Pairwise`, `Morph_Group`, `Morph_Tree`.
- Bandwidths: `select_bandwidth`, `compute_and_save_bandwidths`.
- Evidence: `evidence`, `bridge_sampling_ln` (lower‑level), `compute_bridge_rmse`.
- Dependency analysis: `dependency_tree.compute_and_plot_mi_tree`.
- Total correlation: `Nth_TC.compute_and_save_tc`.

Notes:

- If you pass a numeric `kde_bw` (e.g., `0.9`) the library skips bandwidth JSONs.
- pair/group proposals will compute and cache `MI.json`/`params_*_TC.json` on first use.

## Dependencies

- Core: `numpy`, `scipy`, `matplotlib`, `corner`, `networkx`, `emcee`, `statsmodels`, `scikit-learn`
- Optional: `pandas` (CSV labels), `pygraphviz` (nicer tree layout), `scikit-sparse` (optional exception type)

## Development

- Build wheels/sdist: `python -m build`
- Check metadata: `twine check dist/*`
- Tests live in `tests/`

## Versioning & Release

Versioning is derived from git tags via `setuptools_scm`.

- Tag a release: `git tag vX.Y.Z && git push --tags`
- CI: publishes to TestPyPI on pushes to `main`/`master`; to PyPI on `v*` tags.
- Uses PyPI/TestPyPI Trusted Publishing (OIDC). You can also use API tokens if preferred.

## License

BSD-3-Clause. See `LICENSE` for details.
