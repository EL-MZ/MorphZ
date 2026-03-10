"""
Compare posteriors from the LVK public release (GWTC2.1 mixed), dynesty and
emcee runs for GW150914:
  - compute Jensen–Shannon divergences for key parameters
  - produce an overlaid corner plot
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import bilby
import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from GW150914_setup import label, outdir


# Paths to the three posterior products we want to compare
RESULTS_FILES = dict(
    gwtc="IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5",
    dynesty="dynesty_result.json",
    mcmc="mcmc_result.json",
)

# Parameters to include in JSD calculations (only those present in all three are kept)
JSD_PARAMETERS = [
    "chirp_mass",
    "total_mass",
    "mass_1",
    "mass_2",
    "mass_ratio",
    "symmetric_mass_ratio",
    "chirp_mass_source",
    "total_mass_source",
    "luminosity_distance",
    "ra",
    "dec",
    "psi",
    "theta_jn",
    "chi_eff",
    "chi_p",
]

# Parameters to show on the corner plot (subset of the common parameters)
CORNER_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "chi_eff",
    "chi_p",
    "theta_jn",
]

JSD_BINS = 50
CORNER_SAMPLE_LIMIT = 4000  # thin large posteriors for plotting speed


def load_dynesty_or_mcmc(path: Path) -> pd.DataFrame:
    """Load a bilby JSON result into a posterior dataframe."""
    result = bilby.result.read_in_result(path)
    return result.posterior.copy()


def load_gwtc_mixed(path: Path, group: str = "C01:Mixed") -> pd.DataFrame:
    """Load LVK posterior samples (structured array) into a dataframe."""
    with h5py.File(path, "r") as f:
        samples = f[group]["posterior_samples"]
        data = {name: samples[name][...] for name in samples.dtype.names}
    return pd.DataFrame(data)


def intersect_params(dfs: Iterable[pd.DataFrame], candidates: List[str]) -> List[str]:
    """Return parameters present in every dataframe, preserving candidate order."""
    common = []
    for p in candidates:
        if all(p in df.columns for df in dfs):
            common.append(p)
    return common


def compute_jsd(values_a: np.ndarray, values_b: np.ndarray, low: float, high: float) -> float:
    """Jensen–Shannon divergence for two 1D arrays using a common histogram."""
    h_a, _ = np.histogram(values_a, bins=JSD_BINS, range=(low, high), density=True)
    h_b, _ = np.histogram(values_b, bins=JSD_BINS, range=(low, high), density=True)

    # convert to proper probability distributions and avoid zeros
    h_a = h_a / h_a.sum()
    h_b = h_b / h_b.sum()
    eps = np.finfo(float).eps
    h_a = np.clip(h_a, eps, None)
    h_b = np.clip(h_b, eps, None)
    h_a = h_a / h_a.sum()
    h_b = h_b / h_b.sum()

    return float(jensenshannon(h_a, h_b))


def compute_jsd_table(posteriors: Dict[str, pd.DataFrame], params: List[str]) -> pd.DataFrame:
    """Compute JSDs between each pair of posteriors for the selected parameters."""
    rows = []
    dynesty_df = posteriors["dynesty"]
    mcmc_df = posteriors["mcmc"]
    gwtc_df = posteriors["gwtc"]

    for p in params:
        a = dynesty_df[p].to_numpy()
        b = mcmc_df[p].to_numpy()
        c = gwtc_df[p].to_numpy()

        low = min(a.min(), b.min(), c.min())
        high = max(a.max(), b.max(), c.max())

        rows.append(
            dict(
                parameter=p,
                jsd_dynesty_gwtc=compute_jsd(a, c, low, high),
                jsd_mcmc_gwtc=compute_jsd(b, c, low, high),
                jsd_dynesty_mcmc=compute_jsd(a, b, low, high),
            )
        )

    return pd.DataFrame(rows)


def _thin_for_plotting(df: pd.DataFrame, params: List[str], limit: int) -> np.ndarray:
    subset = df[params]
    if limit and len(subset) > limit:
        subset = subset.sample(limit, random_state=1)
    return subset.to_numpy()


def make_corner_plot(posteriors: Dict[str, pd.DataFrame], params: List[str], output: Path) -> None:
    """Overlay corner plots for the selected parameters."""
    colors = dict(gwtc="C0", dynesty="C1", mcmc="C2")
    fig = None
    handles: List[Tuple[str, str]] = []

    for name in ["gwtc", "dynesty", "mcmc"]:
        df = posteriors[name]
        samples = _thin_for_plotting(df, params, CORNER_SAMPLE_LIMIT)
        fig = corner.corner(
            samples,
            labels=params if fig is None else None,
            color=colors[name],
            fig=fig,
            plot_density=False,
            hist_kwargs={"density": True, "linewidth": 1.0},
            contour_kwargs={"linewidths": 1.0},
        )
        handles.append((name, colors[name]))

    if fig is None:
        return

    # add legend using an empty handle on the first axis
    ax = fig.axes[0]
    for name, color in handles:
        ax.plot([], [], color=color, label=name)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    results_dir = Path(outdir)

    posteriors = {
        "gwtc": load_gwtc_mixed(results_dir / RESULTS_FILES["gwtc"]),
        "dynesty": load_dynesty_or_mcmc(results_dir / RESULTS_FILES["dynesty"]),
        "mcmc": load_dynesty_or_mcmc(results_dir / RESULTS_FILES["mcmc"]),
    }

    jsd_params = intersect_params(posteriors.values(), JSD_PARAMETERS)
    corner_params = intersect_params(posteriors.values(), CORNER_PARAMETERS)

    if not jsd_params:
        raise RuntimeError("No overlapping parameters found for JSD computation.")

    jsd_table = compute_jsd_table(posteriors, jsd_params)
    jsd_path = results_dir / f"{label}_jsd_comparison.csv"
    jsd_table.to_csv(jsd_path, index=False)
    print(f"Saved JSD table to {jsd_path} for parameters: {jsd_params}")

    if corner_params:
        corner_path = results_dir / f"{label}_posterior_corner.png"
        make_corner_plot(posteriors, corner_params, corner_path)
        print(f"Saved corner plot to {corner_path} for parameters: {corner_params}")
    else:
        print("No common parameters available for corner plot.")


if __name__ == "__main__":
    main()
