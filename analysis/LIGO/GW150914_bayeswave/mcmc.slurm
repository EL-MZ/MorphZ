#!/bin/bash
#SBATCH --job-name=gw150914_MCMC_wave
#SBATCH --output=logs/mcmc_%j.out
#SBATCH --error=logs/mcmc_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --dependency=singleton

module purge
module load python-scientific/3.11.5-foss-2023b
source morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting GW150914 mcmc job ${SLURM_JOB_ID}"
date

srun python GW150914_analysis.py --sampler mcmc
rc=$?
date
echo "Finished GW150914 MCMC prod job with exit code ${rc}"
exit ${rc}
