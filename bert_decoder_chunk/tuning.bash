#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

python hyperparameter_tuning.py \
    --language Tsez \
    --model morph \
    --track 1 \
    --trials 1
