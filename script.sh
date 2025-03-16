#!/bin/bash

#SBATCH --account=ucb289_asc3
#SBATCH --partition=aa100
#SBATCH --job-name=example-job
#SBATCH --output=example-job.%j.out
#SBATCH --time=00:30:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kuad8709@colorado.edu

module purge
module load anaconda
conda activate QST-UGD

python main.py --POVM "Tetra4" \
               --n_qubits 6 \
               --na_state "W_P" \
               --P_state 0.9 \
               --ty_state "mixed" \
               --noise "noise" \
               --r_path "/scratch/alpine/kuad8709/"
               # --n_samples $((100 * 4**10)) 
