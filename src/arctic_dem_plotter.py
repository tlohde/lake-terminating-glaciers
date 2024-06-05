"""
plots all arcticDEMs in given directory and
saves each plot in a sub-directory `/plot/`.
each plot is given a title: <date> and is saved
as <date>.png.

the centreline of the glacier is plotted in red

doesn't both forcing all plots in same dir to
be on a common colorscale.

this is used to visually assess which DEM would
be a good candidate to use for the 'reference' DEM
"""
import argparse
from glob import glob
import os
import json
from shapely import wkt
import pandas as pd
import rioxarray as rio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--directory")
args = parser.parse_args()
directory = args.directory

os.chdir('../data/arcticDEM')

with open(f'{directory}/download_notes.txt', 'r') as notes:
    notes = json.load(notes)
    line = wkt.loads(notes['centreline'])

files = [f for f in glob(f'{directory}/padded/*.tif') if 'bitmask' not in f]
print(f'working on {directory}; and {len(files)} DEMs')

if not os.path.exists(f'{directory}/padded/plot'):
    os.mkdir(f'{directory}/padded/plot')

for f in files:
    date = pd.to_datetime(os.path.basename(f).split('_')[3],
                          format="%Y%m%d").strftime('%Y-%m-%d')
    
    if os.path.exists(f'{directory}/padded/plot/{date}.png'):
        continue
    else:
        with rio.open_rasterio(f, chunks='auto').squeeze() as ds:
            fig, ax = plt.subplots()
            ds.plot(ax=ax)
            ax.plot(*line.coords.xy, c='r')
            ax.set_title(date)
            fig.savefig(f'{directory}/padded/plot/{date}.png')
            plt.close(fig)
            del fig, ax

