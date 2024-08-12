h# !/usr/bin/env bash

echo "working on: $1"
echo "downloading tiles"
python src/dem_download_tiles.py --directory $1 --months 6 7 8 9 --buffer 5000

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


# python src/make_dirs.py --centrelines data/streams_v3.geojson

# glacier=$(find data/ -type d -name "id*")

# for g in $glacier; do

#     echo "working on: $g"
#     echo "downloading tiles"
#     python src/dem_download_tiles.py --directory $g --months 6 7 8 9 --buffer 5000

#     echo "getting masks"
#     python src/dem_get_masks.py --directory $g

#     echo "coregistering"
#     python src/dem_coregister.py --directory $g

#     echo "stacking"
#     python src/dem_stacking.py --directory $g

#     echo "computing trends"
#     python src/dem_trends.py --directory $g

#     echo "tidying up"
#     python src/dem_cleanup.py --directory $g

# done


