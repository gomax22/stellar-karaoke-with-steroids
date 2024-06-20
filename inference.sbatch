#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=output_test_inference.txt
#SBATCH --error=errors_test_inference.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu                                              
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000    
#SBATCH --nodelist=gnode01                                        

python inference.py
