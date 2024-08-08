import argparse
from glob import glob
import os

# set directory
parser = argparse.ArgumentParser()
parser.add_argument('--directory')

args = parser.parse_args()
directory = args.directory

tiffs = glob('*.tiff', root_dir=directory)

for f in tiffs:
    os.remove(os.path.join(directory, f))
