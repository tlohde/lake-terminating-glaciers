#!/usr/bin/env bash

# for iiml data (how et al., (2021) wiesman et al., (2021))
mkdir -p data/iiml
wget -e robots=off --mirror --no-parent -r -P data/iiml/ https://dap.ceda.ac.uk/neodc/esacci/glaciers/data/IIML/Greenland/v1/2017// 

# for arctic dem shapefile catalogue
mkdir -p arcticDEM
wget --mirror --no-parent -r -P data/arcticDEM/ https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Strip_Index_latest_gpqt.zip


# aero dem shapefile
mkdir -p data/aeroDEM
mkdir -p data/aeroDEM/metadata
wget -e robots=off --mirror --no-parent -R "index.html*" -r -P data/aeroDEM/metadata https://www.ncei.noaa.gov/data/oceans/archive/arc0088/0145405/1.1/data/0-data/G150AERODEM/Metadata/

# unzip any zip files created - in place
find . -name '*.zip' -type f -execdir unzip -n '{}' ';'
