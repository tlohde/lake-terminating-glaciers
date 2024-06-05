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

# downloader function
def get_dem(row, bnds, outdir, apply_bm='y', download_bm='y'):
    '''
    row: named tuples (from `df.itertuples()`, where `df` is the geopandas
    dataframe of the `ArcticDEM_Strip_Index_s2s041.shp`
    bnds - (minx,miny,max,maxy) for study area (epsg:3413) the
    DEM's will be clipped to this
    apply_bm: y/n - whether or not to apply the bitmask.
        if 'y'. all pixels not set as 'good' are set to np.nan.
        if 'n'. original gets returned
    download_bm: y/n - whether or not to download the bitmask
        if 'y'. it will download with prefix 'bitmask_'
        if 'n'. it won't be downloaded

    '''
    # lazily open and clip DEM COG to bnds
    _tmp = rio.open_rasterio(row.downloadurl, chunks='auto')
    _tmp_clip = _tmp.rio.clip_box(*bnds)

    # lazily open and clip bitmask to bnds
    _tmp_bm = rio.open_rasterio(row.bitmaskurl, chunks='auto')
    _tmp_bm_clip = _tmp_bm.rio.clip_box(*bnds)

    # apply bitmask (fill everything that is either the DEM fill value,
    # or where the bitmask > 0 with nan)
    if apply_bm == 'y':
        _for_export = (xr.where((_tmp_clip == _tmp.attrs['_FillValue'])
                                | (_tmp_bm_clip[:, :, :] > 0),
                                np.nan,
                                _tmp_clip)
                       .rename('z')
                       .squeeze()
                       .rio.write_crs(_tmp.rio.crs.to_epsg()))
    else:  # don't apply bitmask
        _for_export = (_tmp_clip
                       .rename('z')
                       .squeeze()
                       .rio.write_crs(_tmp.rio.crs.to_epsg()))

    # appends the bounding box to the dem_id when saving
    bnds_str = f'{int(bnds[0])}_{int(bnds[1])}_{int(bnds[2])}_{int(bnds[3])}'
    fname = f'{row.dem_id}_{bnds_str}'
    _for_export.rio.to_raster(f'{outdir}/{fname}.tif')
    # _for_export.rio.to_raster(f'{fname}.tif')

    # for optional downloading of bitmask
    if download_bm == 'y':
        (_tmp_bm_clip.rename('mask')
         .squeeze()
         .rio.write_crs(_tmp_bm.rio.crs.to_epsg())
         .rio.to_raster(f'{outdir}/bitmask_{fname}.tif')
        # .rio.to_raster(f'bitmask_{fname}.tif')
         )

# read in centrelines
print(f'read in centrelines, and select index: {index}')
lines = gpd.read_file('../../data/streams_v2.geojson')
lines = lines.loc[[index]]

# read in arcticDEM catalog
print('reading in arctic DEM catalog')
catalog = gpd.read_parquet(glob('../../data/arcticDEM/**/*.parquet', recursive=True)[0])

# make timestamps, datetimes, and sort (so downloads in date order)
catalog['acqdate1'] = catalog['acqdate1'].astype('datetime64[ns]')
catalog['acqdate2'] = catalog['acqdate2'].astype('datetime64[ns]')
catalog.sort_values(by='acqdate1', inplace=True)

# only select dems from 'summer (-ish)' months
# (june, july, aug, sept)
catalog = catalog.loc[catalog['acqdate1'].dt.month.isin([6,7,8,9])]

# fix download urls
text_to_replace = 'polargeospatialcenter.github.io/stac-browser/#/external/'
catalog['downloadurl'] = (catalog['s3url']
                           .apply(lambda x: (
                               x.replace(text_to_replace, "")
                               .replace('.json', '_dem.tif')))
                           )
catalog['bitmaskurl'] = (catalog['downloadurl']
                          .apply(lambda x: x.replace('_dem', '_bitmask')))

# reproject to same crs as centrelines
catalog = catalog.to_crs(lines.crs)

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
    bounds = line.geometry.buffer(5000).envelope.bounds
    
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
    selection = catalog.loc[catalog.intersects(line.geometry)]
    
    print(f'getting, clipping, masking, downloading {len(selection)} DEMs')
    print(f'starting at: {pd.Timestamp.now().strftime("%Y/%m/%d_%H:%M")}')
    for dem in tqdm(selection.itertuples()):
        print('downloading...')
        get_dem(dem, bounds, outdir)
        print('done, next...')

    print('all done')
print('finished')



