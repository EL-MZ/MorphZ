import os

import bilby
from pp_setup import load_simulation
import pandas as pd
import numpy as np
from typing import Optional, Union
from morphZ import evidence as morphz_evidence
from tqdm import trange


def get_morphz_evidence(
        result: bilby.result.Result,
        priors: bilby.prior.PriorDict,
        likelihood: bilby.likelihood.Likelihood,
        label: str = "",
        output_dir: Optional[Union[str, os.PathLike]] = None,
) -> dict:
    posterior = result.posterior
    param_names = list(priors.keys())
    fixed_params = priors.fixed_keys
    search_params = [p for p in param_names if p not in fixed_params]
    morph_priors = {p: priors[p] for p in search_params}
    morph_priors = bilby.prior.PriorDict(morph_priors)
    # remove 'mass_1' and 'mass_2' if 'chirp_mass' and 'mass_ratio' are present
    if 'chirp_mass' in search_params and 'mass_ratio' in search_params:
        search_params = [p for p in search_params if p not in ['mass_1', 'mass_2']]
    fixed_param_vals = {p: priors[p].peak for p in fixed_params}
    # remove the priors with fixed params

    print(f"Computing morphZ evidence for parameters: {search_params}")
    print("morph-prior:")
    for p in morph_priors:
        print(f"  {p}: {morph_priors[p]}")

    samples = posterior[search_params].to_numpy()
    log_likelihoods = posterior["log_likelihood"].to_numpy()
    log_priors = posterior["log_prior"].to_numpy()
    log_posterior_values = log_likelihoods + log_priors

    print(f"Number of posterior samples: {samples.shape[0]}")
    Npost = 5000
    # thin posteiror samples to Npost
    if samples.shape[0] > Npost:
        indices = np.random.choice(samples.shape[0], size=Npost, replace=False)
        samples = samples[indices]
        log_posterior_values = log_posterior_values[indices]
        log_likelihoods = log_likelihoods[indices]
        log_priors = log_priors[indices]
        print(f"Thinned posterior samples to {Npost}.")

    print("Number of params:", samples.shape[1])

    def log_posterior(theta: np.ndarray) -> float:
        params = dict(zip(search_params, theta))
        log_prior = morph_priors.ln_prob(params)
        params.update(fixed_param_vals)
        if not np.isfinite(log_prior):
            return log_prior
        likelihood.parameters.update(params)
        log_likelihood = likelihood.log_likelihood(params)
        return log_likelihood + log_prior

    # recompute Likelihoods for all samples to ensure consistency
    print("Recomputing log posterior values for all samples to ensure consistency...")
    size = samples.shape[0]
    for i in trange(size):
        log_posterior_values[i] = log_posterior(samples[i, :])

    if output_dir is not None:
        target_outdir = os.fspath(output_dir)
    else:
        target_outdir = os.fspath(result.outdir)
    os.makedirs(target_outdir, exist_ok=True)
    label_suffix = f"_{label}" if label else ""
    morphz_output_path = os.path.join(target_outdir, f"morphZ{label_suffix}")

    morphz_lnzs = morphz_evidence(
        post_samples=samples,
        log_posterior_values=log_posterior_values,
        log_posterior_function=log_posterior,
        n_resamples=5000,
        morph_type='2_group',
        kde_bw='silverman',
        param_names=search_params,
        output_path=morphz_output_path,
        n_estimations=100,
        verbose=True,
    )

    # get mean and std of logz from multiple runs
    morphz_lnzs = np.array(morphz_lnzs)  # shape (n_estimations, 2)
    morphz_lnz = np.mean(morphz_lnzs, axis=0)  # shape (2,)
    return dict(
        lnz_mean=float(morphz_lnz[0]),
        lnz_err=float(morphz_lnz[1])
    )


def collect_lnz(idx, outdir_override: Optional[Union[str, os.PathLike]] = None):
    # Load simulation data
    (
        likelihood, priors, outdir, label, _
    ) = load_simulation(idx)

    default_outdir = os.fspath(outdir)
    if outdir_override is not None:
        final_outdir = os.fspath(outdir_override)
    else:
        final_outdir = default_outdir
    os.makedirs(final_outdir, exist_ok=True)

    # load the two sets of results
    result_dynesty = bilby.result.read_in_result(
        os.path.join(default_outdir, "dynesty_result.json")
    )
    result_mcmc = bilby.result.read_in_result(
        os.path.join(default_outdir, "mcmc_result.json")
    )

    # compute morphz evidence for both
    morphz_dynesty = get_morphz_evidence(
        result_dynesty,
        priors,
        likelihood,
        label=f"{label}_dynesty",
        output_dir=final_outdir,
    )
    morphz_mcmc = get_morphz_evidence(
        result_mcmc,
        priors,
        likelihood,
        label=f"{label}_mcmc",
        output_dir=final_outdir,
    )

    # write LnZs to a csv
    lnz_data = {
        'method': ['dynesty', 'mcmc', 'dynesty_morphz', 'mcmc_morphz'],
        'lnz': [
            result_dynesty.log_evidence,
            result_mcmc.log_evidence,
            morphz_dynesty['lnz_mean'],
            morphz_mcmc['lnz_mean']
        ],
        'lnz_err': [
            result_dynesty.log_evidence_err,
            result_mcmc.log_evidence_err,
            morphz_dynesty['lnz_err'],
            morphz_mcmc['lnz_err']
        ]
    }
    lnz_df = pd.DataFrame(lnz_data)
    # print to console
    print("______________")
    print(lnz_df)
    print("______________")
    lnz_df.to_csv(
        os.path.join(final_outdir, f"{label}_lnz_comparison.csv"),
        index=False
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("Usage: python compute_morphz_evidence.py <injection_index> [<outdir>]")
        sys.exit(1)

    injection_index = int(sys.argv[1])
    custom_outdir = sys.argv[2] if len(sys.argv) == 3 else None
    collect_lnz(injection_index, custom_outdir)
