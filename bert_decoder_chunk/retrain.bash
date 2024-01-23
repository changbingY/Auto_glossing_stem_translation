#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

python  retrain_best_hyperparameters.py --language Tsez --model morph --track 1 --trial 1
