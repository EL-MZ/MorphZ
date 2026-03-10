#!/bin/bash
#SBATCH --job-name=morphz_GW
#SBATCH --output=logs/morph/index_GW15.out
#SBATCH --error=logs/morph/index_GW15.err
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

mkdir -p logs/morph

module purge
module load python-scientific/3.11.5-foss-2023b
source morphZ_casestudy_CBC_pe/.venv/bin/activate


srun python GW150914_bayeswave/morphz_computation.py


