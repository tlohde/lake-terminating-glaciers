#!/usr/bin/env bash

#SBATCH --ntasks=8
#SBATCH --mem=32G

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/s1759665/micromamba/envs/paper2

printf "\nworking on: $1\n"

python src/velocity_robust_spatial_trends.py \
--centrelines data/streams_v3.geojson \
--index $1 \
--get_robust_trend --export_trend
