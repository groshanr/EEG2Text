#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --partition=amilan
#SBATCH --ntasks=4
#SBATCH --mem=45G
#SBATCH --job-name=EEG2TEXT3D
#SBATCH --output=EEG2TEXT3D.out
#SBATCH --account=ucb-general
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load python
module load anaconda

echo "Starting Script..."

conda activate EEG2TEXT
python 3D_data_creator.py

echo "Complete!"
