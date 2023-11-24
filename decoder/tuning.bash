#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

python hyperparameter_tuning.py \
    --language Lezgi \
    --model morph \
    --track 1 \
    --trials 1
