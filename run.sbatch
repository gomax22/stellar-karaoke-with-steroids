#!/bin/bash
#SBATCH --job-name=stellar
#SBATCH --output=output_gpu.txt
#SBATCH --error=errors_gpu.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu                                              
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000                                                     

python run.py --dataset preprocessed/split/1/ --output recons/split/1/
