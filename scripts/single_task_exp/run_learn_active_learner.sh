#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="single_task_new"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=rmc-gpu03,rmc-gpu14
#SBATCH --output=/home/patr_jy/Github/ALPBNN/slogs/slurm-%j.out
#SBATCH --error=/home/patr_jy/Github/ALPBNN/slogs/slurm_error-%j.out


source ~/mambaforge/etc/profile.d/conda.sh
conda activate albpnn

export WANDB_CACHE_DIR=$HOME/wandbdir/
export MPLCONFIGDIR=$HOME/matplotlibdir

export yaml_file="dqn_exp"


python ~/Github/ALPBNN/scripts/single_task_exp_new/learn_active_learner.py \
    --yaml_file "$yaml_file" &

wait