#!/bin/bash
#SBATCH --job-name=rule_zx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1

# ===============================================================
GENERATE=False
TRAIN=True # Set to True if you want to train the model, otherwise it will skip training.
EVAL=False  # Set to True if you want to evaluate the model after training.

# ===============================================================
# ----------------------------------------------------------

num_qubits_min=5
num_qubits_max=10
min_gates=10 
max_gates=20 
p_t=0.4 
p_h=0.1
p_s=0.1
p_cnot=0.3
p_not=0.1

max_epochs=100000
ppo_updates_per_epoch=1
max_train_steps=50000
emb_dim=1024
hid_dim=512
num_envs=8
rollout_steps=512
minibatch_size=256
env_max_steps=100
step_penalty=0.01
length_penalty=0.001
logger_type='swanlab'
precision='32-true'
data_length=10000  # Number of graphs in the graph bank pickle file.
if [ "$GENERATE" = "True" ]; then
    python zxreinforce/generate.py \
    --out_path ./data \
    --keep_limit $data_length \
    --num_qubits_min $num_qubits_min \
    --num_qubits_max $num_qubits_max \
    --min_gates $min_gates \
    --max_gates $max_gates \
    --p_t $p_t \
    --p_h $p_h \
    --p_s $p_s \
    --p_cnot $p_cnot \
    --p_not $p_not \
    --max_reward 0.95 
fi



if [ "$TRAIN" = "True" ]; then
    python run_train.py \
    --max_epochs $max_epochs \
    --ppo_updates_per_epoch $ppo_updates_per_epoch \
    --max_train_steps $max_train_steps \
    --emb_dim $emb_dim \
    --hid_dim $hid_dim \
    --num_envs $num_envs \
    --rollout_steps $rollout_steps \
    --minibatch_size $minibatch_size \
    --logger_type $logger_type \
    --precision $precision \
    --env_max_steps $env_max_steps \
    --step_penalty $step_penalty \
    --length_penalty $length_penalty \
    --data_dir ./data \
    --data_length $data_length \
    --num_qubits_min $num_qubits_min \
    --num_qubits_max $num_qubits_max \
    --min_gates $min_gates \
    --max_gates $max_gates \
    --p_t $p_t \
    --p_s $p_s \
    --p_cnot $p_cnot \
    --p_not $p_not \
    --p_h $p_h 
fi


ckpt="runs/ZX-PPO-GNN_nq[2-3]_gates[3-6]_envs8_T256_mb128_ppo4_lr0.0003_20250927-180908/checkpoints/last.ckpt"
ckpt="runs/ZX-PPO-GNN_nq[5-10]_gates[10-30]_envs8_T512_mb128_ppo4_lr0.0003_20250927-194912/checkpoints/last.ckpt"
ckpt="runs/ZX-PPO-GNN_nq[2-6]_gates[5-10]_envs8_T512_mb128_ppo4_lr0.0003_20250929-120817/checkpoints/last.ckpt"
ckpt="runs/ZX-PPO-GNN_nq[2-3]_gates[3-6]_envs8_T512_mb128_ppo4_lr0.0003_20250930-214313/checkpoints/last.ckpt"
if [ "$EVAL" = "True" ]; then
    python eval.py \
    --ckpt_path $ckpt \
    --num_qubits_min $num_qubits_min \
    --num_qubits_max $num_qubits_max \
    --min_gates $min_gates \
    --max_gates $max_gates \
    --data_dir ./data \
    --data_length $data_length \
    --p_t $p_t \
    --p_h $p_h \
    --env_max_steps $env_max_steps \
    --step_penalty $step_penalty \
    --length_penalty $length_penalty \
    --adapted_reward \
    --count_down_from 20
fi