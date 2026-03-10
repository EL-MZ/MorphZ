from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["dynesty", "mcmc", "dynesty_morphz", "mcmc_morphz"]
LABELS = {
    "dynesty": "Dynesty",
    "mcmc": "SS(MCMC)",
    "dynesty_morphz": r"$MorphZ_{(NS)}$",
    "mcmc_morphz": r"$MorphZ_{(PT)}$",
}
DEFAULT_COLORS = {
    "dynesty": "#1b9e77",
    "mcmc": "#d95f02",
    "dynesty_morphz": "#c31e23",
    "mcmc_morphz": "#000000",
}
DEFAULT_ROOT_DIR = Path("/fred/oz303/ezahraoui/github/fast_pp/outdir")
DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "outdir/plots/"
CACHE_FILENAME = "evidences.csv"
COLUMN_MAP = {
    "dynesty": ("lnz_dynesty", "lnz_err_dynesty"),
    "mcmc": ("lnz_mcmc", "lnz_err_mcmc"),
    "dynesty_morphz": ("lnz_morph_dynesty", "lnz_err_morph_dynesty"),
    "mcmc_morphz": ("lnz_morph_mcmc", "lnz_err_morph_mcmc"),
}


def load_seed_comparisons(root_dir: Path) -> dict[str, pd.DataFrame]:
    """Return DataFrames keyed by seed folder names (e.g., seed_1)."""
    data: dict[str, pd.DataFrame] = {}
    for seed_dir in sorted(root_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / f"{seed_dir.name}_lnz_comparison.csv"
        if not csv_path.is_file():
            continue
        data[seed_dir.name] = pd.read_csv(csv_path)
    return data


def extract_seed_index(seed_label: str) -> int | None:
    match = re.search(r"(\d+)", seed_label)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def comparisons_to_wide_frame(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten the per-seed comparisons into a single wide table."""
    rows: list[dict[str, float | int]] = []
    fallback_index = 0
    for seed_name in sorted(comparisons):
        df = comparisons[seed_name]
        seed_value = extract_seed_index(seed_name)
        if seed_value is None:
            seed_value = fallback_index
            fallback_index += 1
        row: dict[str, float | int] = {"seed": seed_value}
        for method, (value_col, err_col) in COLUMN_MAP.items():
            row[value_col] = np.nan
            row[err_col] = np.nan
        for _, method_row in df.iterrows():
            method_name = str(method_row["method"]).strip().lower()
            if method_name not in COLUMN_MAP:
                continue
            value_col, err_col = COLUMN_MAP[method_name]
            row[value_col] = method_row["lnz"]
            row[err_col] = method_row.get("lnz_err", np.nan)
        rows.append(row)

    columns = ["seed"]
    for method, (value_col, err_col) in COLUMN_MAP.items():
        columns.extend([value_col, err_col])

    frame = pd.DataFrame(rows, columns=columns)
    return sort_wide_frame(frame)


def sort_wide_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in frame.columns:
        return frame
    return frame.sort_values(by="seed").reset_index(drop=True)


def load_cached_results(
    root_dir: Path, cache_path: Path, force_refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Load cached evidences or refresh from the source directory."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        frame = sort_wide_frame(pd.read_csv(cache_path))
        report_missing_seeds(frame)
        return wide_frame_to_comparisons(frame)

    comparisons = load_seed_comparisons(root_dir)
    combined = comparisons_to_wide_frame(comparisons)
    if combined.empty:
        raise FileNotFoundError(
            f"No lnz comparison CSV files found under {root_dir}"
        )
    combined.to_csv(cache_path, index=False)
    report_missing_seeds(combined)
    return wide_frame_to_comparisons(combined)


def wide_frame_to_comparisons(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the cached table back into a seed -> DataFrame mapping."""
    if frame.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}
    for _, row in frame.iterrows():
        seed_value = row.get("seed")
        if pd.isna(seed_value):
            continue
        seed_label = f"seed_{int(seed_value)}"
        entries = []
        for method, (value_col, err_col) in COLUMN_MAP.items():
            lnz = row.get(value_col)
            if pd.isna(lnz):
                continue
            entries.append(
                {
                    "method": method,
                    "lnz": float(lnz),
                    "lnz_err": float(row.get(err_col, np.nan)),
                }
            )
        if entries:
            results[seed_label] = pd.DataFrame(entries)
    return results


def report_missing_seeds(frame: pd.DataFrame) -> None:
    if "seed" not in frame.columns or frame.empty:
        print("No seed data found in cache.")
        return

    seeds = frame["seed"].dropna().astype(int)
    if seeds.empty:
        print("No valid seed numbers found.")
        return

    min_seed = seeds.min()
    max_seed = seeds.max()
    expected = set(range(min_seed, max_seed + 1))
    present = set(seeds.tolist())
    missing = sorted(expected - present)

    if missing:
        missing_str = ", ".join(str(m) for m in missing)
        print(f"Missing seed numbers: {missing_str}")
    else:
        print(f"All seeds present from {min_seed} to {max_seed}.")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.labelsize": 24,
            "axes.titlesize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )


def plot_delta_lnz_spaghetti(
    results_by_seed,
    baseline="dynesty",
    methods=("dynesty", "mcmc", "dynesty_morphz", "mcmc_morphz"),
    sample=None,
    propagate_baseline_error=False,
    colors=None,
    random_state=None,
):
    """
    results_by_seed: dict[str -> pd.DataFrame] with columns ['method','lnz','lnz_err']
    baseline: method name to anchor at x=0
    methods: order to plot
    sample: optional int to subsample seeds
    propagate_baseline_error: if True, xerr for non-baseline methods is
      sqrt(err_method^2 + err_baseline^2). If False (default), uses each method's own err.
    colors: optional dict to override colors by method
    """

    seeds = list(results_by_seed.keys())
    if sample is not None and sample < len(seeds):
        rng = np.random.default_rng(random_state)
        seeds = list(rng.choice(seeds, size=sample, replace=False))

    lnz_list, err_list, kept = [], [], []
    needed = set(methods)
    for s in seeds:
        df = results_by_seed[s]
        if not needed.issubset(set(df["method"])):  # ensure all methods present
            continue
        row = df.set_index("method").loc[list(methods)]
        lnz_list.append(row["lnz"].to_numpy())
        err_list.append(row["lnz_err"].to_numpy())
        kept.append(s)

    if not kept:
        raise ValueError("No seeds contain all requested methods.")

    lnz = np.asarray(lnz_list)
    err = np.asarray(err_list)
    seeds = kept
    base_idx = methods.index(baseline)
    delta = lnz - lnz[:, [base_idx]]

    if propagate_baseline_error:
        rel_err = np.sqrt(err**2 + err[:, [base_idx]] ** 2)
        rel_err[:, base_idx] = err[:, base_idx]
    else:
        rel_err = err

    order = np.argsort(lnz[:, base_idx])
    seeds = [seeds[i] for i in order]
    delta = delta[order]
    rel_err = rel_err[order]

    default_colors = DEFAULT_COLORS.copy()
    if colors:
        default_colors.update(colors)

    fig, (ax_hist, ax) = plt.subplots(
        2,
        1,
        figsize=(11, 26),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0},
    )
    y = np.arange(len(seeds))
    for j, method in enumerate(methods):
        if j == base_idx:
            continue
        if method == "mcmc_morphz":
            ax.errorbar(
                delta[:, j],
                y,
                xerr=rel_err[:, j],
                fmt="o",
                ms=9,
                capsize=5,
                elinewidth=1.5,
                color=default_colors.get(method),
                alpha=0.9,
            )
        ax.errorbar(
            delta[:, j],
            y,
            xerr=rel_err[:, j],
            fmt="o",
            ms=9,
            capsize=1,
            solid_capstyle="round",
            elinewidth=2,
            color=default_colors.get(method),
            alpha=0.9,
            label=LABELS.get(method, method),
        )

    baseline_method = methods[base_idx]
    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")   # optional
    leg.get_frame().set_alpha(1.0)
    ax.vlines(x=1, ymin=0, ymax=99, linestyles="dashed", alpha=0.15, colors="grey")
    ax.vlines(x=-1, ymin=0, ymax=99, linestyles="dashed", alpha=0.15, colors="grey")
    ax.vlines(x=0.5, ymin=0, ymax=99, linestyles="dashed", alpha=0.15, colors="grey")
    ax.vlines(x=-0.5, ymin=0, ymax=99, linestyles="dashed", alpha=0.15, colors="grey")
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
    tick_positions = np.arange(0, len(seeds), 5)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels((tick_positions + 1).astype(int))
    ax.set_xlabel(r"Δ $\log(\hat{z})$ relative to NS")
    ax.set_ylabel("Seed Number")

    ax.spines["top"].set_linestyle((0, (1, 10)))
    ax.spines["top"].set_linewidth(0.5)

    ax.set_ylim([-1, len(seeds)])

    ax.grid(axis="both", alpha=0.1)
    ax.tick_params(axis="y", which="both", length=0)
    ax_hist.grid(False)
    hist_indices = [j for j in range(len(methods)) if j != base_idx]
    if hist_indices:
        hist_samples = np.concatenate([delta[:, j] for j in hist_indices])
        hist_samples = hist_samples[np.isfinite(hist_samples)]
    else:
        hist_samples = np.array([])
    if hist_samples.size == 0:
        hist_bins = 15
    else:
        low, high = hist_samples.min(), hist_samples.max()
        if np.isclose(low, high):
            low -= 0.5
            high += 0.5
        hist_bins = np.linspace(low, high, 20)
    for j in hist_indices:
        method = methods[j]
        values = delta[:, j]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax_hist.hist(
            values,
            bins=hist_bins,
            histtype="step",
            density=True,
            color=default_colors.get(method),
            linewidth=1.8,
            label=LABELS.get(method, method),
        )
    ax_hist.set_ylabel("")
    ax_hist.vlines(x=1, ymin=0, ymax=2.5, linestyles="dashed", alpha=0.15, colors="grey")
    ax_hist.vlines(x=-1, ymin=0, ymax=2.5, linestyles="dashed", alpha=0.15, colors="grey")
    ax_hist.vlines(x=0.5, ymin=0, ymax=2.5, linestyles="dashed", alpha=0.15, colors="grey")
    ax_hist.vlines(x=-0.5, ymin=0, ymax=2.5, linestyles="dashed", alpha=0.15, colors="grey")
    ax_hist.tick_params(labelbottom=False, labelleft=False, length=0)
    ax_hist.spines["bottom"].set_linestyle("")
    ax_hist.spines["bottom"].set_linewidth(0.0)

    fig.tight_layout()
    return fig, ax


def plot_one_minus_cdf_delta_lnz(
    results_by_seed,
    baseline="dynesty",
    methods=("dynesty", "mcmc", "dynesty_morphz", "mcmc_morphz"),
    sample=None,
    random_state=None,
    colors=None,
    use_absolute_differences=True,
):
    """
    Plot 1 - CDF of Δlog(z) relative to a baseline method.

    By default this uses |Δlog(z)| (absolute differences) so that the
    curves represent the fraction of seeds with a difference of at
    least x in magnitude.

    results_by_seed: dict[str -> pd.DataFrame] with columns ['method', 'lnz', 'lnz_err']
    baseline: method name to anchor Δlog(z)
    methods: tuple of methods to include (baseline must be present)
    sample: optional int to subsample seeds
    use_absolute_differences: if True, use |Δlog(z)|; if False, use signed Δlog(z)
    """

    if baseline not in methods:
        raise ValueError(
            f"Baseline method '{baseline}' not found in methods {methods}."
        )

    seeds = list(results_by_seed.keys())
    if sample is not None and sample < len(seeds):
        rng = np.random.default_rng(random_state)
        seeds = list(rng.choice(seeds, size=sample, replace=False))

    lnz_list, kept = [], []
    needed = set(methods)
    for s in seeds:
        df = results_by_seed[s]
        if not needed.issubset(set(df["method"])):
            continue
        row = df.set_index("method").loc[list(methods)]
        lnz_list.append(row["lnz"].to_numpy())
        kept.append(s)

    if not kept:
        raise ValueError("No seeds contain all requested methods for CDF plot.")

    lnz = np.asarray(lnz_list)
    base_idx = methods.index(baseline)
    delta = lnz - lnz[:, [base_idx]]

    default_colors = DEFAULT_COLORS.copy()
    if colors:
        default_colors.update(colors)

    fig, ax = plt.subplots(figsize=(10, 8))
    min_survival = np.inf
    max_value = 0.0

    for j, method in enumerate(methods):
        if j == base_idx:
            continue

        values = delta[:, j]
        if use_absolute_differences:
            values = np.abs(values)

        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        values = np.sort(values)
        n = values.size
        ranks = np.arange(n)
        survival = (n - ranks) / n
        min_survival = min(min_survival, survival.min())
        max_value = max(max_value, values.max() if values.size else 0.0)

        ax.step(
            values,
            survival,
            where="post",
            label=LABELS.get(method, method),
            linewidth=2.0,
            color=default_colors.get(method),
        )

    for x in (0.5, 1.0):
        ax.axvline(x, color="grey", lw=1.0, ls="--", alpha=0.7)

    baseline_label = LABELS.get(baseline, baseline)
    if use_absolute_differences:
        ax.set_xlabel(rf"|Δ log(z)| relative to {baseline_label}")
    else:
        ax.set_xlabel(rf"Δ log(z) relative to {baseline_label}")
    ax.set_ylabel(r"$1 - \mathrm{CDF}$")

    if not np.isfinite(min_survival) or min_survival <= 0:
        min_survival = 1e-4
    ax.set_yscale("log")
    ax.set_ylim(bottom=min_survival-0.0002, top=1.05)
    max_value = max(1.0, max_value)
    ax.set_xlim(left=0.0, right=max_value)
    ax.set_xticks(np.arange(0.0, max_value + 0.001, 0.2))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.margins(x=0)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(1.0)

    fig.tight_layout()
    return fig, ax


def ensure_baseline(methods: tuple[str, ...], baseline: str) -> tuple[str, ...]:
    if baseline in methods:
        return methods
    return (baseline,) + tuple(m for m in methods if m != baseline)


def save_plot_variations(
    results_by_seed,
    baseline: str,
    sample: int | None,
    propagate_baseline_error: bool,
    outdir: Path,
) -> None:
    variations = [
        ("lnz_all_methods.pdf", tuple(METHODS)),
        ("lnz_morph_mcmc_only.pdf", ("mcmc_morphz",)),
        (
            "lnz_morph_vs_dynesty.pdf",
            ("dynesty", "mcmc_morphz", "dynesty_morphz"),
        ),
        (
            "lnz_morph_vs_morph.pdf",
            ("mcmc_morphz", "dynesty_morphz"),
        ),
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    for filename, subset in variations:
        method_order = ensure_baseline(tuple(subset), baseline)
        try:
            # Spaghetti plot
            fig, _ = plot_delta_lnz_spaghetti(
                results_by_seed,
                baseline=baseline,
                methods=method_order,
                sample=sample,
                propagate_baseline_error=propagate_baseline_error,
            )
            fig.savefig(outdir / filename)
            plt.close(fig)

            # 1 - CDF plot (same methods, same baseline)
            cdf_filename = filename.replace(".pdf", "_1minusCDF.pdf")
            fig_cdf, _ = plot_one_minus_cdf_delta_lnz(
                results_by_seed,
                baseline=baseline,
                methods=method_order,
                sample=sample,
            )
            fig_cdf.savefig(outdir / cdf_filename)
            plt.close(fig_cdf)

        except ValueError as exc:
            print(f"Skipping {filename} and its CDF variant: {exc}")
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot log-evidence comparisons across seeds."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Directory containing seed_* subfolders with lnz comparison CSVs.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent / CACHE_FILENAME,
        help="Path to the cached flattened evidences CSV.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional single plot output path (all methods).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory to store plot variations when --plot is omitted.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild the cache from the root directory even if it exists.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optionally subsample this many seeds before plotting.",
    )
    parser.add_argument(
        "--baseline",
        choices=METHODS,
        default="dynesty",
        help="Method to anchor Δlog(z).",
    )
    parser.add_argument(
        "--propagate-baseline-error",
        action="store_true",
        help="Propagate baseline uncertainty into Δlog(z) errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_by_seed = load_cached_results(
        root_dir=args.root,
        cache_path=args.cache,
        force_refresh=args.force_refresh,
    )

    if args.plot:
        fig, _ = plot_delta_lnz_spaghetti(
            results_by_seed,
            baseline=args.baseline,
            sample=args.sample,
            propagate_baseline_error=args.propagate_baseline_error,
        )
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        plt.close(fig)
    else:
        save_plot_variations(
            results_by_seed,
            baseline=args.baseline,
            sample=args.sample,
            propagate_baseline_error=args.propagate_baseline_error,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
