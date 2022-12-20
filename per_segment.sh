#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=per_segment
#SBATCH --output=per_segment%j.out

conda activate /scratch/ab9738/traffic/env/;
export PATH=/scratch/ab9738/traffic/env/bin:$PATH;
cd /scratch/ab9738/traffic/traffic-jams-uber/
python per_segment.py
