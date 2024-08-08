#!/usr/bin/env bash

# cd "$(dirname "$0")"
# cd ../../data

# for iiml data (how et al., (2021) wiesman et al., (2021))
mkdir -p data/iiml
wget -e robots=off --mirror --no-parent -r -P data/iiml/ https://dap.ceda.ac.uk/neodc/esacci/glaciers/data/IIML/Greenland/v1/2017// 

# for arctic dem shapefile catalogue
mkdir -p arcticDEM
wget --mirror --no-parent -r -P data/arcticDEM/ https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Strip_Index_latest_gpqt.zip


# unzip any zip files created - in place
find . -name '*.zip' -type f -execdir unzip -n '{}' ';'


# run download script for getting arctic DEM tiles
# python ../src/download_data/download_arcticDEM_tiles.py

# list_of_dirs=$(ls /arcticDEM/ | grep id)
# parallel python arctic_dem_plotter.py --directory {1} ::: $list_of_dirs
# parallel python arctic_dem_padder.py --directory {1} ::: $list_of_dirs

