#!/bin/bash
  
#SBATCH --account=def-msilfver
#SBATCH --time 80:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-user=changbing.yang@colorado.edu
#SBATCH --job-name=train_fairseq_5Lang
#SBATCH --nodes=1   

module load gcc python/3.9 arrow/4
source /home/ycblena/scratch/Hard/ENV/bin/activate

python main.py --language Lezgi --model morph --track 1
