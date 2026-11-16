#!/bin/bash
# Runs run_dl.py on the DCE as a batch job (the interactive 30 min srun is too short).
#
#   sinfo                                  # lists the partitions: pick a batch (non-interactive) one
#   sbatch src/dl/dce_job.sh fixed         # from the project root; also: expanding, rolling
#   sbatch src/dl/dce_job.sh expanding forex     # optional 2nd argument: dataset (default stocks)
#   squeue -u $USER                        # is it running?
#   tail -f logs/dl-<job id>.out           # progress, one line per fold
#
# Before the first run, once:   python src/dl/run_dl.py --quick     (1 to 2 min, checks the setup)
#
#SBATCH --job-name=dl-vol
#SBATCH --partition=cpu_prod         # 12 h limit; `sinfo` lists the others
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-%j.out

SPLIT=${1:-fixed}
DATASET=${2:-stocks}

mkdir -p logs
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python src/dl/run_dl.py --dataset "$DATASET" --split "$SPLIT"
