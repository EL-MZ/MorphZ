#!/usr/bin/env python

"""
pp_analysis.py

Load simulation produced by pp_setup.load_simulation
and run either Dynesty or Bilby-MCMC.

Usage:
    python pp_analysis.py --index 37 --sampler dynesty
"""

import argparse
import os
import multiprocessing as mp
import bilby
from pp_setup import load_simulation

NPOOL = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))



def run_dynesty(index):
    likelihood, priors, outdir, label, checkpoint_delta_t = load_simulation(index)
    print(f"Running Dynesty {index}...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label="dynesty",
        nlive=2000,
        nact=20,
        sample="rwalk",
        npool=NPOOL,
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()


def run_mcmc(index):
    likelihood, priors, outdir, label, checkpoint_delta_t = load_simulation(index)
    print(f"Running Bilby-MCMC {index}...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby_mcmc",
        outdir=outdir,
        label="mcmc",
        nsamples=2000,
        thin_by_nact=0.2,
        ntemps=8,   
        npool=NPOOL,
        Tmax_from_SNR=20,
        adapt=True,
        proposal_cycle="gwA",
        L1steps=100,
        L2steps=5,
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()



print(f"[pp_analysis] Using NPOOL = {NPOOL}")

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True)
parser.add_argument("--sampler", choices=["dynesty", "mcmc"], required=True)
args = parser.parse_args()

if args.sampler == "dynesty":
    run_dynesty(args.index)
else:
    run_mcmc(args.index)

print("Analysis complete.")
