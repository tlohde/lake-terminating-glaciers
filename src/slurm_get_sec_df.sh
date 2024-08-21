#!/usr/bin/env bash

#SBATCH --ntasks=16
#SBATCH --mem=32G

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/s1759665/micromamba/envs/paper2

/usr/bin/env time python src/dem_make_sec_df.py --directory $1
echo "done"