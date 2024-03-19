#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="feextrct"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=rmc-gpu03,rmc-gpu14,rmc-gpu20
#SBATCH --output=/home/patr_jy/Github/ALPBNN/slogs/slurm-%j.out
#SBATCH --error=/home/patr_jy/Github/ALPBNN/slogs/slurm_error-%j.out


source ~/mambaforge/etc/profile.d/conda.sh
conda activate albpnn

export WANDB_CACHE_DIR=$HOME/wandbdir/
export MPLCONFIGDIR=$HOME/matplotlibdir


datasets=('mnist' 'kmnist' 'cifar')
seeds=(1124 343423 987 6789)
models=(dino_single)
#models=(dino_single)
wandb_logging=True

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for seed in "${seeds[@]}"
    do
      python ~/Github/ALPBNN/scripts/feature_exp/dino_vs_resnet.py \
      --dataset "$dataset" \
      --wandb_logging "$wandb_logging" \
      --model "$model" \
      --seed "$seed" &
    done
    wait
  done
done


wait
