#!/bin/bash
#SBATCH --job-name=forwarding
#SBATCH --output=output_cpu.txt
#SBATCH --error=errors_cpu.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xhicpu
#SBATCH --cpus-per-task=32                                              
#SBATCH --mem=44000                                                     

python parallel_forwarding.py --input preprocessed/
