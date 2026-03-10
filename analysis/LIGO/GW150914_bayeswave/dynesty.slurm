#!/bin/bash
#SBATCH --job-name=gw150914_NS_wave
#SBATCH --output=logs/dynesty_%j.out
#SBATCH --error=logs/dynesty_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --dependency=singleton

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting GW150914 Dynesty prod job ${SLURM_JOB_ID}"
date

srun python GW150914_analysis.py --sampler dynesty

rc=$?
date
echo "Finished GW150914 Dynesty prod job with exit code ${rc}"
exit ${rc}
