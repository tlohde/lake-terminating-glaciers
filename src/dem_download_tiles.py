"""
for downloading arctic DEM strip tiles.

takes index of centreline geodataframe as input
(because that made running it with $parallel easier)

uses the geoparquet arcticDEM catalog to identify tiles
that intersect the centreline...lazily opens these COGs, 
clips to an area 5000 m around the centreline, applies the
bitmask, and exports to geotiff
"""

import argparse
from dem_utils import ArcticDEM
import geopandas as gpd
from glob import glob
import json
import numpy as np
import os
import pandas as pd
import rioxarray as rio
from tqdm import tqdm
import xarray as xr


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int)
args = parser.parse_args()
index = args.index

print(os.getcwd())

# read in centrelines
print(f'read in centrelines, and select index: {index}')
lines = gpd.read_file('../../data/streams_v2.geojson')
lines = lines.loc[[index]]

# read in arcticDEM catalog
print('reading in arctic DEM catalog')
catalog = ArcticDEM.get_catalog_gdf(
    d=glob('../../data/arcticDEM/**/*.parquet', recursive=True)[0],
    months=[6,7,8,9],
    crs=lines.crs
    )


# for each centreline: create output directory;
# find arctic DEMs that intserect it;
# lazily apply clip arcticDEM to 
# 5 km buffer around centreline; apply bitmask;
# export both bitmask and dem to output directory

for line in lines.itertuples():
    print(f'working on {line.Index}/{len(lines)}')
    # derive params from line geometry to use for file naming
    cntr = line.geometry.centroid
    outdir = f'../../data/arcticDEM/id{line.Index}_{cntr.x:.0f}x_{cntr.y:.0f}y'
    envelope = line.geometry.buffer(5000).envelope
    bounds = envelope.bounds
    
    # make directory, and store some meta data
    print('making / finding directory and exporting process meta-data')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

        params = {
            'centreline': line.geometry.wkt,
            'predicate': 'dem footprint intersects line',
            'clipping buffer': 5000,
            'clipped bounds': bounds,
            'bitmask': 'applied',
            'date_downloaded': pd.Timestamp.now().strftime("%Y/%m/%d_%H:%M")
        }
        
        with open(f'{outdir}/download_notes.txt', 'w') as notes:
            notes.write(json.dumps(params))
    
    # find arcticDEMs that intersect centreline:
    # selection = catalog.loc[catalog.intersects(line.geometry)]
    selection = catalog.loc[catalog.intersects(line)]
    
    print(f'getting, clipping, masking, downloading {len(selection)} DEMs')
    for dem in tqdm(selection.itertuples()):
        already_downloaded = glob(f'{outdir}/*.tif')
        if len([f for f in already_downloaded if dem.dem_id in f]) > 0:
            print(f'already got {dem.dem_id}')
            continue
        print('downloading...')
        ArcticDEM.get_dem(dem, bounds, outdir)
        print('done, next...')

    print('all done')
print('finished')
