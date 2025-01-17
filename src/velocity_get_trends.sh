#!/usr/bin/env bash

#SBATCH --ntasks=8
#SBATCH --mem=32G

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/s1759665/micromamba/envs/paper2

python src/velocity_robust_spatial_trends.py \
&& --centrelines '../data/streams_v3.geojson' \
&& --filter_cube False \
&& --sample_centreline False \
&& --get_robust_trend True \
&& --get_rgb \
&& --get_annual_median \
&& --export_trend \