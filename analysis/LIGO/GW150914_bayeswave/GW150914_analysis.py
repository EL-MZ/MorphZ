import argparse
from GW150914_setup import likelihood, priors, outdir, checkpoint_delta_t, NPOOL, logger, bilby


def run_dynesty():
    label = 'dynesty'
    result_path = f"{outdir}/{label}_result.json"
    logger.info("Running Dynesty...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=label,
        nlive=2000,
        nact=20,
        sample="rwalk",
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        npool=NPOOL,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()
    logger.info(f"Dynesty LnZ = {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")

def run_mcmc():
    from GW150914_setup import likelihood, priors, outdir, checkpoint_delta_t, NPOOL, logger, bilby
    label = 'mcmc'
    result_path = f"{outdir}/{label}_result.json"
    logger.info("Running MCMC...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby_mcmc",
        outdir=outdir,
        label=label,
        nsamples=2000,
        thin_by_nact=0.2,
        ntemps=NPOOL,
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
    logger.info(f"MCMC LnZ = {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run GW150914 analysis with specified sampler.")
    parser.add_argument('--sampler', type=str, choices=['dynesty', 'mcmc'], required=True,
                        help='Sampler to use for the analysis: dynesty or mcmc')    
    args = parser.parse_args()
    if args.sampler == 'dynesty':
        run_dynesty()
    elif args.sampler == 'mcmc':
        run_mcmc()
    else:
        logger.error(f"Unknown sampler: {args.sampler}")



if __name__ == "__main__":
    main()
