#!/usr/bin/env python3
"""
Compute n-order Total Correlation (TC) from continuous posterior samples via Gaussian KDE.
For n=2, this is equivalent to pairwise Mutual Information (MI).
Plots an MI heatmap for n=2, and saves the TC values for n>=2.

Dependencies: numpy, scipy, matplotlib, pandas (optional).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    import pandas as pd  # optional, only for CSV labels
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


###############################################################################
#                                  UTILITIES                                  #
###############################################################################

def _safe_gaussian_kde(
    data: np.ndarray,
    bw_method: str | float | None,
    rng: np.random.Generator,
    max_retries: int = 2,
) -> gaussian_kde:
    """Fit ``gaussian_kde`` robustly, adding jitter if covariance is singular."""
    attempt = 0
    jitter_scale = 1e-9
    while True:
        try:
            return gaussian_kde(data, bw_method=bw_method)
        except np.linalg.LinAlgError:
            attempt += 1
            if attempt > max_retries:
                raise
            warnings.warn(
                f"Singular covariance detected (attempt {attempt}). Adding jitter and retrying.",
                RuntimeWarning,
            )
            data = data + rng.normal(scale=jitter_scale, size=data.shape)
            jitter_scale *= 10  # back‑off


def _eval_log_pdf_at_samples(
    kde: gaussian_kde,
    samples_T: np.ndarray,
    loo: bool,
    eps: float = 1e-300,
) -> np.ndarray:
    """Evaluate log‑pdf at training samples with optional leave‑one‑out correction."""
    pdf_vals = kde.evaluate(samples_T)
    if loo:
        n = samples_T.shape[1]
        try:
            norm_factor = kde._norm_factor  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover – older SciPy fallback
            d = kde.d
            cov_det = np.linalg.det(kde.covariance)
            norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        self_term = norm_factor * (1.0 / n)
        pdf_vals = (n * pdf_vals - self_term) / (n - 1)
    return np.log(np.clip(pdf_vals, eps, None))


def stable_seed_for_indices(base_seed: int, indices: tuple[int, ...]) -> int:
    """Derive a deterministic 32‑bit seed from a base seed and an index tuple."""
    data = f"{base_seed}|" + ",".join(map(str, indices))
    digest = hashlib.md5(data.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def compute_marginal_log_p(
    samples: np.ndarray,
    bw_method: str | float | None,
    loo: bool,
    seed: int,
    eps: float,
) -> tuple[np.ndarray, np.random.Generator]:
    """
    Compute 1‑D marginal log‑densities at the provided sample points.

    Returns the log densities (shape ``(n_dims, n_samples)``) and the RNG used.
    """
    samples = np.ascontiguousarray(samples, dtype=float)
    samples_T = samples.T
    n_dims, n_samples = samples_T.shape
    rng = np.random.default_rng(seed)

    log_p = np.empty((n_dims, n_samples))
    for i in range(n_dims):
        xi = samples_T[i:i + 1, :]
        kde_i = _safe_gaussian_kde(xi, bw_method, rng)
        log_p[i] = _eval_log_pdf_at_samples(kde_i, xi, loo, eps)
    return log_p, rng


def compute_tc_for_indices(
    indices: tuple[int, ...],
    samples_T: np.ndarray,
    log_p: np.ndarray,
    bw_method: str | float,
    loo: bool,
    eps: float,
    rng: np.random.Generator,
) -> float:
    """Compute TC/MI for a single index group."""
    subset_samples = samples_T[list(indices), :]
    kde_n = _safe_gaussian_kde(subset_samples, bw_method, rng)
    log_p_joint = _eval_log_pdf_at_samples(kde_n, subset_samples, loo, eps)

    idx = list(indices)
    log_p_marginals_sum = log_p[idx[0]].copy()
    for k in idx[1:]:
        log_p_marginals_sum += log_p[k]

    diff = log_p_joint - log_p_marginals_sum
    tc_value = float(np.mean(diff))
    return max(0.0, tc_value)


def compute_total_correlation(
    samples: np.ndarray,
    n_order: int = 2,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    eps: float = 1e-300,
    n_workers: int = 0,
    show_progress: bool = True,
    auto_thin: bool = True,
) -> np.ndarray | list[tuple[tuple[int, ...], float]]:
    """
    Compute a symmetric matrix of pairwise mutual informations (n_order=2)
    or a list of n-order total correlations using KDE.

    Args:
        samples (np.ndarray): 2D array of shape (n_samples, n_dims).
        n_order (int, optional): Order of total correlation. Defaults to 2 (pairwise MI).
        bw_method (str or float, optional): Bandwidth method for KDE. Defaults to "silverman".
        loo (bool, optional): If True, use leave-one-out correction. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        eps (float, optional): Small value to clip PDF values to avoid log(0). Defaults to 1e-300.
        n_workers (int, optional): >1 enables threaded parallelism; 0/1 uses serial path.
        show_progress (bool, optional): If True and tqdm is available, show progress over groups. Defaults to True.
        auto_thin (bool, optional): If True, downsample large, high-order problems for speed. Defaults to True.

    Returns:
        np.ndarray | list: For n_order=2, a symmetric matrix of pairwise MIs.
                           For n_order>2, a list of (indices, tc_value) tuples.
    """
    samples = np.ascontiguousarray(np.asarray(samples), dtype=float)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2‑D array (n_samples, n_dims).")
    n_samples, n_dims = samples.shape

    if auto_thin and n_order >= 3 and n_dims >= 50:
        target_n = 300
        if n_dims >= 75:
            target_n = 200
        if n_dims >= 100:
            target_n = 100
        if n_samples > target_n:
            rng_thin = np.random.default_rng(seed)
            idx = rng_thin.choice(n_samples, size=target_n, replace=False)
            samples = np.ascontiguousarray(samples[idx], dtype=float)
            n_samples = target_n
            logger.info("Auto-thinning samples to %s for faster TC computation (n_dims=%s, n_order=%s).", target_n, n_dims, n_order)

    if not 2 <= n_order <= n_dims:
        raise ValueError(f"`n_order` must be between 2 and {n_dims}.")

    samples_T = samples.T
    log_p, rng = compute_marginal_log_p(samples, bw_method, loo, seed, eps)

    if n_order == 2:
        groups = list(combinations(range(n_dims), 2))
    else:
        groups = list(combinations(range(n_dims), n_order))

    progress = None
    if show_progress and tqdm is not None:
        desc = "TC groups" if n_order > 2 else "MI pairs"
        progress = tqdm(total=len(groups), desc=desc)

    try:
        # N‑D densities & TC (serial path)
        if n_workers <= 1:
            if n_order == 2:
                mi = np.zeros((n_dims, n_dims))
                for i, j in groups:
                    tc_val = compute_tc_for_indices((i, j), samples_T, log_p, bw_method, loo, eps, rng)
                    mi[i, j] = mi[j, i] = tc_val
                    if progress is not None:
                        progress.update(1)
                return mi

            results = []
            for indices in groups:
                tc_val = compute_tc_for_indices(indices, samples_T, log_p, bw_method, loo, eps, rng)
                results.append((indices, tc_val))
                if progress is not None:
                    progress.update(1)
            logger.info("The number of estimated Total correlations: %s", len(results))
            return results

        # Parallel execution
        def worker(indices: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
            local_rng = np.random.default_rng(stable_seed_for_indices(seed, indices))
            value = compute_tc_for_indices(
                indices=indices,
                samples_T=samples_T,
                log_p=log_p,
                bw_method=bw_method,
                loo=loo,
                eps=eps,
                rng=local_rng,
            )
            return indices, value

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results_iter = executor.map(worker, groups)

            if n_order == 2:
                mi = np.zeros((n_dims, n_dims))
                for indices, value in results_iter:
                    i, j = indices
                    mi[i, j] = mi[j, i] = value
                    if progress is not None:
                        progress.update(1)
                return mi

            results = []
            for indices, value in results_iter:
                results.append((indices, value))
                if progress is not None:
                    progress.update(1)
            logger.info("The number of estimated Total correlations: %s", len(results))
            return results
    finally:
        if progress is not None:
            progress.close()


###############################################################################
#                                VISUALISATION                                #
###############################################################################

def plot_tc_heatmap(
    mi: np.ndarray,
    labels: Sequence[str],
    outfile: str = "mi_heatmap.png",
    cmap: str = "magma",
) -> None:
    """
    Save a heat-map of the MI matrix (only for n_order=2).

    Args:
        mi (np.ndarray): The mutual information matrix.
        labels (Sequence[str]): The labels for the axes.
        outfile (str, optional): The output file path. Defaults to "mi_heatmap.png".
        cmap (str, optional): The colormap to use. Defaults to "magma".
    """
    n_labels = len(labels)
    # Dynamically adjust figure size and font size for readability
    # These factors can be tuned for better aesthetics
    fig_width = max(8, n_labels * 0.15)
    fig_height = max(6, n_labels * 0.15)
    font_size = max(4, 12 - n_labels * 0.08)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(mi, cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mutual Information", rotation=-90, va="bottom")

    ax.set_title("Pairwise Mutual Information")
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)



###############################################################################
#                            MAIN ORCHESTRATION                                #
###############################################################################

def compute_and_save_tc(
    samples: np.ndarray,
    names: Optional[Iterable[str]] = None,
    n_order: int = 2,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    out_path: Optional[str] = None,
    n_workers: int = 0,
    show_progress: bool = True,
    auto_thin: bool = True,
) -> np.ndarray | list:
    """
    Compute TC, visualise (for n=2), and save all TC values.

    Args:
        samples (np.ndarray): 2D array of shape (n_samples, n_dims).
        names (Optional[Iterable[str]], optional): Names of the parameters. Defaults to None.
        n_order (int, optional): Order of total correlation. Defaults to 2.
        bw_method (str | float, optional): Bandwidth method for KDE. Defaults to "silverman".
        loo (bool, optional): If True, use leave-one-out correction. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        out_path (Optional[str], optional): The output directory path. Defaults to None.
        n_workers (int, optional): >1 enables threaded parallelism; 0/1 uses serial path.
        show_progress (bool, optional): If True and tqdm is available, show progress over groups. Defaults to True.
        auto_thin (bool, optional): If True, downsample large, high-order problems for speed. Defaults to True.

    Returns:
        np.ndarray | list: The total correlation results.
    """
    n_dims = samples.shape[1]
    if names is not None:
        labels = [str(n) for n in names]  # ensure JSON‑serialisable strings
    else:
        labels = [f"x{i}" for i in range(n_dims)]

    if len(labels) != n_dims:
        raise ValueError("Length of `names` must match number of dimensions.")

    results = compute_total_correlation(
        samples,
        n_order=n_order,
        bw_method=bw_method,
        loo=loo,
        seed=seed,
        n_workers=n_workers,
        show_progress=show_progress,
        auto_thin=auto_thin,
    )
    
    if out_path:
        p = Path(out_path)
        p.mkdir(parents=True, exist_ok=True)
    else:
        p = Path(".")

    if n_order == 2:
        mi_matrix = results
        heatmap_path = p / "mi_heatmap.png"
        plot_tc_heatmap(mi_matrix, labels, outfile=heatmap_path)

        tri = np.triu_indices_from(mi_matrix, k=1)
        all_pairs = []
        for i, j in zip(tri[0], tri[1]):
            # Traverse only the upper triangle to skip diagonals and duplicates
            value = float(mi_matrix[i, j])
            all_pairs.append((i, j, value))
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info("Top 10 MI pairs:")
        for i, j, val in all_pairs[:10]:
            logger.info("  %s — %s : %.6f", labels[i], labels[j], val)

        mi_path = p / f"params_{n_order}-order_TC.json"
        mi_data = [
            [[labels[i], labels[j]], float(val)] for i, j, val in all_pairs
        ]
        with open(mi_path, "w", encoding="utf8") as f:
            json.dump(mi_data, f, indent=2)
        logger.info("All MI pairs saved to %s", mi_path)
    else:
        # Sort results for n > 2
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 %s-order Total Correlation groups:", n_order)
        for indices, val in sorted_results[:10]:
            group_labels = [labels[i] for i in indices]
            logger.info("  %s : %.6f", ", ".join(group_labels), val)

        tc_path = p / f"params_{n_order}-order_TC.json"
        tc_data = [
            [[labels[i] for i in indices], val] for indices, val in sorted_results
        ]
        with open(tc_path, "w", encoding="utf8") as f:
            json.dump(tc_data, f, indent=2)
        logger.info("All %s-order TC values saved to %s", n_order, tc_path)

    return results
