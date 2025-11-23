#!/bin/bash
#SBATCH --job-name=g3_pure_jax_v0_cartpole_muax                   # Default job name (sweep overrides)
#SBATCH -c 2                               # CPU cores per task
#SBATCH -t 0-07:10                         # Runtime (D-HH:MM)
#SBATCH -p kempner_h100                    # Partition
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --mem=80G                          # RAM for the job
#SBATCH -o slurm-%x-%j_g3_pure_jax_v0_cartpole_muax.out                 # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e slurm-%x-%j_g3_pure_jax_v0_cartpole_muax.err                 # STDERR

# Load modules / env
module load python/3.10.9-fasrc01

# If you rely on conda commands:
# source ~/.bashrc || true
# conda activate torch || true
cd /n/home04/amuppidi/muax_control/muax_g3_jax
# Use explicit Python path from your torch env (as in your example)
~/.conda/envs/torch/bin/python train_cartpole_jax.py --support_size 10 --embedding_size 8 --discount 0.99 --num_actions 2 --num_simulations 250 --k_steps 10 --wandb_project muax_jax_cartpole_sims --wandb_mode online --max_episodes 2000
