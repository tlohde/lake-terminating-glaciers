# !/usr/bin/env bash

echo "working on: $1"
echo "downloading tiles"
python src/dem_download_tiles.py --directory $1 --months 6 --buffer 5000

echo "getting masks"
python src/dem_get_masks.py --directory $1

echo "coregistering"
python src/dem_coregister.py --directory $1

echo "stacking"
python src/dem_stacking.py --directory $1

echo "computing trends"
python src/dem_trends.py --directory $1

echo "tidying up"
python src/dem_cleanup.py --directory $1
