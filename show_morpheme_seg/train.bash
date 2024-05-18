#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

python main.py --language Statimcets --model morph --track 1
