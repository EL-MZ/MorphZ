#!/usr/bin/env python
"""
GW150914 analysis using bilby
Includes:
  - Cached GWOSC data loading
  - PSD loaded from text files (config-style)
  - Full likelihood marginalisations
"""

import sys
import numpy as np
from pathlib import Path

import bilby
from gwpy.timeseries import TimeSeries
from multiprocessing import cpu_count
import multiprocessing as mp
import os


# NPOOL IS OBTAINED FROM ENVIRONMENT VARIABLE SET BY SLURM 
NPOOL = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))



###############################################################################
# CONFIGURATION (matching your .ini)
###############################################################################
trigger_time = 1126259462.391
detectors = ["H1", "L1"]
duration = 4
post_trigger = 2  # time from trigger to end of segment

maximum_frequency = {"H1": 896, "L1": 896}
minimum_frequency = {"H1": 20, "L1": 20}

psd_files = {
    "H1": "psd_data/h1_psd.txt",
    "L1": "psd_data/l1_psd.txt",
}

prior_file = "GW150914.prior"
label = "GW150914"
outdir = "outdir"
checkpoint_delta_t = 1800  # seconds

###############################################################################
# SETUP
###############################################################################
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
cache_dir = Path(outdir) / "cached_data"
cache_dir.mkdir(parents=True, exist_ok=True)

logger = bilby.core.utils.logger

###############################################################################
# HELPER: load cached GWOSC data
###############################################################################
def fetch_cached_timeseries(detector, start, end, kind):
    """
    Load cached GWOSC data if present; otherwise download and cache it.
    kind = "analysis" or "psd"
    """
    start_tag = str(start).replace(".", "p")
    end_tag = str(end).replace(".", "p")
    cache_file = cache_dir / f"{detector}_{kind}_{start_tag}_{end_tag}.hdf5"

    if cache_file.exists():
        logger.info(f"[CACHE] Loading {kind} data for {detector} from {cache_file}")
        return TimeSeries.read(cache_file)

    logger.info(f"[FETCH] Downloading {kind} data for {detector}")
    data = TimeSeries.fetch_open_data(detector, start, end)
    data.write(cache_file, overwrite=True)

    return data

###############################################################################
# TIME WINDOWS
###############################################################################
end_time = trigger_time + post_trigger
start_time = end_time - duration

###############################################################################
# CONSTRUCT INTERFEROMETERS
###############################################################################
ifo_list = bilby.gw.detector.InterferometerList([])

for det in detectors:
    logger.info(f"Setting up {det}")

    # Create empty interferometer
    ifo = bilby.gw.detector.get_empty_interferometer(det)

    # Load cached analysis data
    data = fetch_cached_timeseries(det, start_time, end_time, "analysis")
    ifo.strain_data.set_from_gwpy_timeseries(data)

    # Load PSD from your psd_data/*.txt (two-column file)
    freqs, psd_vals = np.loadtxt(psd_files[det]).T
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freqs,
        psd_array=psd_vals,
    )

    # Set min/max frequencies
    ifo.minimum_frequency = minimum_frequency[det]
    ifo.maximum_frequency = maximum_frequency[det]

    ifo_list.append(ifo)

logger.info("Saving data diagnostic plots...")
ifo_list.plot_data(outdir=outdir, label=label)

###############################################################################
# PRIORS
###############################################################################
priors = bilby.gw.prior.BBHPriorDict(filename=prior_file)

priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

###############################################################################
# WAVEFORM GENERATOR
###############################################################################
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50,
    ),
)

###############################################################################
# LIKELIHOOD (matches config)
###############################################################################
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    distance_marginalization=True,
    phase_marginalization=False,
    jitter_time=True,
    reference_frame="sky",
    time_reference="geocent",
    # enforce_signal_duration=True,
)
