#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="tsneexp"
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
conda activate albpnn01

export WANDB_CACHE_DIR=$HOME/wandbdir/
export MPLCONFIGDIR=$HOME/matplotlibdir


datasets=('mnist' 'kmnist' 'cifar')
#seeds=(1124 343423 987 6789)
models=(resnet34 resnet50 dino)
#wandb_logging=True

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
      python ~/Github/ALPBNN/scripts/feature_exp/tsne.py \
      --dataset "$dataset" \
      --model "$model" \
      &
  done
  wait
done


wait
