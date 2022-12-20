#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=s1s2_nyc
#SBATCH --output=slurm_s1s2_nyc.out

conda activate /scratch/ab9738/traffic/env/;
export PATH=/scratch/ab9738/traffic/env/bin:$PATH;
cd /scratch/ab9738/traffic/traffic-jams-uber/scripts
python compute_s1s2.py
