#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=07:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=32
#SBATCH --job-name=semEEG2TEXT
#SBATCH --output=semEEG2TEXT.out
#SBATCH --account=ucb-general
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load python
module load anaconda

echo "Starting Script..."

conda activate EEG2TEXT
python sem_mod_train.py

echo "Complete!"
