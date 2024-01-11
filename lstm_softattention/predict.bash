#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

#module load gcc python/3.9 arrow/4

#source ENV/bin/activate

python predict_from_model.py
