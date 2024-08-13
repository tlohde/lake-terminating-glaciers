import argparse
from glob import glob
import os

# set directory
parser = argparse.ArgumentParser()
parser.add_argument('--directory')

args = parser.parse_args()
directory = args.directory

tiffs = glob('*.tif', root_dir=directory)
tiffs = [tiff for tiff in tiffs if tiff != 'stable_terrain_mask.tif']

for f in tiffs:
    os.remove(os.path.join(directory, f))
