#!/usr/bin/env python

"""
pp_setup.py

Defines load_simulation(index) which:
  - loads an injection row
  - generates strain data
  - constructs analysis priors with extrinsics fixed
  - builds the GW likelihood
  - returns objects needed for analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import bilby


def load_simulation(idx):
    bilby.core.utils.random.seed(idx)
    # --------------------------------------------------------------
    # Load injection
    # --------------------------------------------------------------
    inj_csv = Path("injections.csv")
    injections = pd.read_csv(inj_csv, index_col=0)
    if idx < 0 or idx >= len(injections):
        raise ValueError(f"Index {idx} out of range")

    inj = injections.iloc[idx].to_dict()
    label = f"seed_{idx}"
    outdir = Path("outdir") / label
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Analysis constants
    # --------------------------------------------------------------
    detectors = ["H1", "L1"]
    duration = 4.0
    sampling_frequency = 4096
    minimum_frequency = 20.0
    maximum_frequency = 1024.0

    # --------------------------------------------------------------
    # Waveform generator
    # --------------------------------------------------------------
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=20,
        ),
    )

    # --------------------------------------------------------------
    # Interferometers + injection
    # --------------------------------------------------------------
    ifos = bilby.gw.detector.InterferometerList(detectors)
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=inj["geocent_time"] - 2,
    )
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=inj)
    ifos.plot_data(outdir=outdir, label=label)

    # --------------------------------------------------------------
    # Priors with extrinsics fixed
    # --------------------------------------------------------------
    priors = bilby.gw.prior.BBHPriorDict('pp.prior')
    extrinsic = [
        "psi", "ra", "dec", "geocent_time",
        "phase", "luminosity_distance", "cos_theta_jn",
    ]
    for key in extrinsic:
        priors[key] = inj[key]

    priors.validate_prior(duration, minimum_frequency)

    # --------------------------------------------------------------
    # Likelihood
    # --------------------------------------------------------------
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos,
        waveform_generator,
        priors=priors,
        time_marginalization=False,
        phase_marginalization=False,
        distance_marginalization=False,
    )

    checkpoint_delta_t = 1800

    return likelihood, priors, outdir, label, checkpoint_delta_t
