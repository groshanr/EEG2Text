#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=22
#SBATCH --job-name=freqEEG2TEXT
#SBATCH --output=freqEEG2TEXT.out
#SBATCH --account=ucb-general
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load python
module load anaconda

echo "Starting Script..."

conda activate EEG2TEXT
python frequency_train.py

echo "Complete!"
