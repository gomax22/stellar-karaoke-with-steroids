#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --output=output_cpu.txt
#SBATCH --error=errors_cpu.txt
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xhicpu
#SBATCH --cpus-per-task=32                                              
#SBATCH --mem=44000                                                     

python download_dataset.py --csv public_harps_idps_berv.csv --output fits/
