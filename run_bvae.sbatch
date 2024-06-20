#!/bin/bash
#SBATCH --job-name=bvae_h
#SBATCH --output=output_bvae_h_second.txt
#SBATCH --error=errors_bvae_h_second.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu                                              
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000                                                     

python run_bvae.py --dataset preprocessed/split/1/ --output recons/split/1/
