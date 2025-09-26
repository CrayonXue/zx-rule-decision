#!/bin/bash
#SBATCH --job-name=rule_zx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # 确保这一行正确指定了 GPU 数量
#SBATCH --time=3-01:00:00
#SBATCH --partition=gpu  # 确保分区支持 GPU
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
module load miniforge3/24.11
module load cuda/12.4   cudnn/8.9.6.50_cuda12  # cudnn/9.6.0.74_cuda12 # cudnn/8.9.6.50_cuda12
#. /data/apps/miniforge3/etc/profile.d/conda.sh

source activate rule_env
export PYTHONUNBUFFERED=1
export http_proxy=http://10.244.6.36:8080
export https_proxy=http://10.244.6.36:8080

# SwanLab configuration
export SWANLAB_API_KEY="84RTqEhOn8jTqDLxYCR3F"

# ===============================================================
TRAIN=True # Set to True if you want to train the model, otherwise it will skip training.


# ===============================================================
# ----------------------------------------------------------

num_qubits_min=2
num_qubits_max=3
min_gates=3
max_gates=6
p_t=0.2
p_h=0.2

max_epochs=10000
ppo_updates_per_epoch=1
max_train_steps=10000
num_envs=8
env_max_steps=100
logger_type='swanlab'
if [ "$TRAIN" = "True" ]; then
    python run_train.py \
    --max_epochs $max_epochs \
    --ppo_updates_per_epoch $ppo_updates_per_epoch \
    --max_train_steps $max_train_steps \
    --num_envs $num_envs \
    --logger_type $logger_type \
    --env_max_steps $env_max_steps \
    --num_qubits_min $num_qubits_min \
    --num_qubits_max $num_qubits_max \
    --min_gates $min_gates \
    --max_gates $max_gates \
    --p_t $p_t \
    --p_h $p_h 
fi
