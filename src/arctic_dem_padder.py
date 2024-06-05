"""
individual arcticDEM strip tiles that intersect a centreline
were clipped to an envelope 5000 m around the centreline.

this clipping results in DEMs that do not go beyond the envelope
but might not fill the envelope....this script fixes that.

it ensures that all DEMs in a directory have
the same extent - and pads them as necessary
"""

import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
import rioxarray as rio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--directory")
args = parser.parse_args()
directory = args.directory

os.chdir('../data/arcticDEM')
files = glob(f'{directory}/*.tif')
print(f'there are {len(files)} in {directory}')
if len(files) == 0:
    print('nothing to see here')

if not os.path.exists(f'{directory}/padded'):
    os.mkdir(f'{directory}/padded')

# get DEM extents
bounds = []
for f in tqdm(files):
    with rio.open_rasterio(f, chunks='auto') as dem:
        bounds.append(dem.rio.bounds())

bounds_df = pd.DataFrame(
        bounds,
        columns=['l', 'b', 'r', 't']
        )
max_bounds = (
        int(bounds_df['l'].min()),
        int(bounds_df['b'].min()),
        int(bounds_df['r'].max()),
        int(bounds_df['t'].max())
        )

# pad DEMs
for f in tqdm(files):
    export_path = f'{directory}/padded/{os.path.basename(f)}'
    if os.path.exists(export_path):
        continue
    else:
        with rio.open_rasterio(f) as dem:
            padded = dem.rio.pad_box(*max_bounds, constant_values=np.nan)
            padded.rio.to_raster(export_path)

print(f'job done: padded {len(files)} DEMs to same extent')

