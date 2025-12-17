#!/usr/bin/env python3
"""Compare runtime of TC computation with and without thread pooling."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import logging 

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from morphZ.Nth_TC import compute_total_correlation as compute_tc_new  # type: ignore  # noqa: E402


def make_correlated_samples(n_samples: int, d: int, rho: float, seed: int) -> np.ndarray:
    """Generate correlated Gaussian samples with shared off-diagonal rho."""
    rng = np.random.default_rng(seed)
    cov = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    mean = np.zeros(d, dtype=float)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def tc_stats_array(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def compare_results(old_res, new_res, n_order: int) -> tuple[float, float, dict[str, float]]:
    """Return (max_abs_diff, fro_norm_diff, stats_of_diff)."""
    if n_order == 2:
        diff = new_res - old_res
    else:
        old_dict = {tuple(idx): val for idx, val in old_res}
        new_dict = {tuple(idx): val for idx, val in new_res}
        all_keys = set(old_dict) | set(new_dict)
        diff_vals = []
        for key in all_keys:
            diff_vals.append(new_dict.get(key, 0.0) - old_dict.get(key, 0.0))
        diff = np.asarray(diff_vals, dtype=float)
    max_abs = float(np.max(np.abs(diff)))
    fro = float(np.linalg.norm(diff.ravel()))
    stats = tc_stats_array(diff)
    return max_abs, fro, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TC runtime with and without thread pooling.")
    parser.add_argument("-d", "--dims", type=int, required=True, help="Number of dimensions.")
    parser.add_argument("-n", "--order", type=int, required=True, help="TC order to compute.")
    parser.add_argument("-s", "--samples", type=int, default=2000, help="Number of samples.")
    parser.add_argument("--rho", type=float, default=0.5, help="Off-diagonal correlation coefficient.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for the pooled run (<=1 forces serial).",
    )
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="loo",
        default=True,
        help="Disable leave-one-out correction.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.order < 2 or args.order > args.dims:
        raise SystemExit("`order` must satisfy 2 <= order <= dims.")

    samples = make_correlated_samples(args.samples, args.dims, args.rho, args.seed)
    loo = args.loo

    def timed_run(n_workers: int):
        start = time.perf_counter()
        res = compute_tc_new(
            samples,
            n_order=args.order,
            seed=args.seed,
            loo=loo,
            n_workers=n_workers,
        )
        elapsed = time.perf_counter() - start
        return res, elapsed

    serial_res, serial_time = timed_run(n_workers=args.workers)

    pooled_res, pooled_time = serial_res, serial_time
    # pooled_res, pooled_time = timed_run(n_workers=args.workers)


    max_abs_diff, fro_diff, stats = compare_results(serial_res, pooled_res, args.order)

    print("=== TC pool vs serial comparison ===")
    print(f"dims={args.dims}, order={args.order}, samples={args.samples}, rho={args.rho}, loo={loo}")
    print(f"Serial runtime (n_workers=1): {serial_time:0.4f} s")
    print(f"Pooled runtime (n_workers={args.workers}): {pooled_time:0.4f} s")
    speedup = serial_time / pooled_time if pooled_time > 0 else float("inf")
    print(f"Speedup (serial/pooled): {speedup:0.2f}x")
    print(f"Max abs diff: {max_abs_diff:0.6g}")
    print(f"Frobenius norm diff: {fro_diff:0.6g}")
    print(f"Diff stats: min={stats['min']:0.6g}, max={stats['max']:0.6g}, mean={stats['mean']:0.6g}, std={stats['std']:0.6g}")


if __name__ == "__main__":
    main()
