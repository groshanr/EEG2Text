#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=20
#SBATCH --job-name=fullEEG2TEXT
#SBATCH --output=fullEEG2TEXT.out
#SBATCH --account=ucb-general
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load python
module load anaconda

echo "Starting Script..."

conda activate EEG2TEXT
python full_train.py

echo "Complete!"