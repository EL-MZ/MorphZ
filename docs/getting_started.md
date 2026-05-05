# Getting Started

## Installation

### From PyPI (recommended)

```bash
python -m pip install morphZ
```

### From a local checkout

Use an editable install if you plan to modify the source:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Estimate Evidence

Use `morphZ.evidence` with posterior samples and a log-posterior function:

```python
import morphZ

log_z = morphZ.evidence(
    samples,
    log_posterior_function=lp_fn,
    log_posterior_values=log_prob,
    n_resamples=5000,
    n_estimations=1,
    morph_type="2_group",
    output_path="./example/",
)
```

## Run The Examples

You can run the interactive notebooks in `examples/` to try MorphZ on the
included example problems:

- `examples/eggbox.ipynb`
- `examples/gaussian shell.ipynb`
- `examples/peak_sampling.ipynb`
- `examples/numpyro_gaussian_shell.ipynb`
- `examples/jaxns_gaussian_shell.ipynb`
- `examples/numpyro_morphz_lnz.ipynb`

