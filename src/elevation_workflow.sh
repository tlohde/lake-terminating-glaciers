#!/usr/bin/env bash

python src/make_dirs.py --centrelines data/streams_v3.geojson

# if you're brave enough to trust the code to whip through everything in one hit:
glacier=$(find data/ -type d -name "id*")

for g in $glacier; do

    echo "working on: $g"
    echo "downloading tiles"
    python src/dem_download_tiles.py --directory $g --months 4 5 6 7 8 9 10 --buffer 5000

    echo "getting masks"
    python src/dem_get_masks.py --directory $g

    echo "coregistering"
    python src/dem_coregister.py --directory $g

    echo "stacking"
    python src/dem_stacking.py --directory $g

    echo "computing trends"
    python src/dem_trends.py --directory $g

    echo "tidying up"
    python src/dem_cleanup.py --directory $g

done


