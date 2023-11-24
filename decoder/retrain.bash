#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

nohup python -u retrain_best_hyperparameters.py --language Lezgi --model morph --track 1 --trial 1 > retrain.log 2>&1 &
