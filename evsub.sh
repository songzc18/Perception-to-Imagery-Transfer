#!/bin/bash

#SBATCH --job-name=evsub
#SBATCH --output=log/%A_%a.log
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-719%20

# Classifiers
models=(lr xdawn eegnet eegconf atteca eegnex)

# Evaluation methods
evals=(ori tt csub in cin ll)

# Task configuration
task_id=$SLURM_ARRAY_TASK_ID

n_subs=20
n_evals=${#evals[@]}

model_id=$((task_id / (n_subs * n_evals)))
eval_id=$(((task_id / n_subs) % n_evals))
subnum=$((task_id % n_subs))

sclf=${models[$model_id]}
seval=${evals[$eval_id]}

# Job information
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${task_id}"
echo "Model: ${sclf}"
echo "Evaluation: ${seval}"
echo "Subject: ${subnum}"
echo "Node: $(hostname)"

nvidia-smi

# Run evaluation
python evsub.py \
    --sub "${subnum}" \
    --clf "${sclf}" \
    --eval "${seval}"