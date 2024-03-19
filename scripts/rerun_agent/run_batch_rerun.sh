#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="random_agent"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
###SBATCH --mem-per-cpu=16G
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


source_folder="small_scale_exp_new"
wandb_project_name="sr4_results"
#wandb_project_name="mnist_new_learned_agent_probs"
run_id_prefix="DQN"
dataset_task=("cifar")
#dataset_task=("kMNIST" "notMNIST" "MNIST" "notMNIST" "notMNIST")
#dataset_task=("notMNIST" "MNIST")
#dataset_task=("svhn")
#dataset_task=("MNIST")
#learning_rates=(0.0005 0.001 0.002)
#epochs=(15 25 30)
#run_id_suffix=(32 33 34 35 36 37)
#train_val_split=(40 40 60 60 80 80)
run_id_suffix=(42)
train_val_split=(10)
state_type="sr2_womargin"
random_seed=(7687 87465 4654658 1654)



process_params() {
    d_task=$1
    r_id_suffix=$2
    rseed=$3
    stype=$4
    itr=$5
    run_id="$run_id_prefix"_"$r_id_suffix"_"$rseed"_"$RANDOM"
    echo "Running dataset task = $d_task and run_id = $run_id"

    temp_yml="temp_config_${d_task}_${run_id}.yml"
    yq eval '
        .source_folder = "'$source_folder'" |
        .wandb_project_name = "'$wandb_project_name'" |
        .run_id = '$r_id_suffix' |
        .dataset_task = "'$d_task'" |
        .train_val_split = '${train_val_split[itr]}' |
        .seed = '$rseed' |
        .state_type = "'$stype'"
        ' agent_rerun.yml > "$temp_yml"

    python ~/Github/ALPBNN/scripts/rerun_agents_new/run_saved_agent.py --yaml_file "${temp_yml%.yml}"
}

for i in "${!run_id_suffix[@]}"
do
  for dataset in "${dataset_task[@]}"
  do
    for rseed in "${random_seed[@]}"
    do
      process_params "$dataset" "${run_id_suffix[i]}" "$rseed"  "${state_type}" "${i}"&
#      sleep 10
    done
    wait
  done
    wait
done

wait

rm temp_config_*
