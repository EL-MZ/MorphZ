#!/bin/bash
#SBATCH --job-name=pp_mcmc
#SBATCH --output=logs/mcmc/index_%a.out
#SBATCH --error=logs/mcmc/index_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1-100
#SBATCH --dependency=singleton

mkdir -p logs/mcmc

module purge
module load python-scientific/3.11.5-foss-2023b
source /morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "------------------------------------------------------------"
echo " MCMC PP ANALYSIS"
echo " SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo " SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo " SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo " Node:                $(hostname)"
echo "------------------------------------------------------------"
date

srun python pp_analysis.py --index ${SLURM_ARRAY_TASK_ID} --sampler mcmc
rc=$?

date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
