# !/usr/bin/env bash

datadirs=$(find data/ -type d -name "id*")
for d in $datadirs; do 
nfiles=$(ls $d | wc -l)
if [ $nfiles -eq 1 ]
then
time bash src/elevation_workflow.sh $d
fi
done

# python src/robust_spatial_trends.py --centrelines "data/streams_v2.geojson" --buffer 1000 --get_robust_trend

