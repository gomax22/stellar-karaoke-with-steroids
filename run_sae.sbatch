#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --output=output_sae.txt
#SBATCH --error=errors_sae.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1      
#SBATCH --partition=xgpu                                              
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000                                                     
#SBATCH --nodelist=gnode03                                                         

python run_sae.py --dataset preprocessed/split/1/ --output recons/split/1/
