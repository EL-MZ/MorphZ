#!/bin/bash
#SBATCH --job-name=morphz_pp
#SBATCH --output=logs/morph/index_%a.out
#SBATCH --error=logs/morph/index_%a.err
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-100

mkdir -p logs/morph

module purge
module load python-scientific/3.11.5-foss-2023b
source /morphZ_casestudy_CBC_pe/.venv/bin/activate


srun python morphz_computation.py ${SLURM_ARRAY_TASK_ID}
rc=$?

date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
