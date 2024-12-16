#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=28
#SBATCH --job-name=posEEG2TEXT
#SBATCH --output=posEEG2TEXT.out
#SBATCH --account=ucb-general
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load python
module load anaconda

echo "Starting Script..."

conda activate EEG2TEXT
python POS_train.py

echo "Complete!"
