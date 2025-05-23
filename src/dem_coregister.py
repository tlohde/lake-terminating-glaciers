"""
co-registers all DEMs in directory to a single reference DEM
the reference DEM is identified by a text file (reference.txt)
within that direcotry. once coregistered, new DEMs are saved to
a parallel directory /coregistered

regions of 'stable terrain' are identified for each DEM from
either a landsat or sentinel-2 imagery within a fortnight (either side)
of the DEM acquisition, using a simple threshold of ndwi < 0.

statistics such as the distribution of of differences over stable terrain
before and after coregistration (NMAD and median) are computed and added
as meta data to each coregistered DEM prior to export
"""
import argparse
from dem_utils import ArcticDEM
from glob import glob
from itertools import product
import numpy as np
import os
from tqdm import tqdm

# parse args
parser = argparse.ArgumentParser(prog='arctic DEM coregistration',
                                description='''
                                coregistration of arcticDEMs
                                ''')

parser.add_argument("--directory",
                    help='''
                    supplied as relative path from whatever
                    dir script is called from.
                    e.g. 'data/id01_Xx_Yy/')
                    ''')

args = parser.parse_args()
directory = args.directory

# keep cwd as src/
# append directory to filepaths
dem_files = [os.path.join(directory, f) for f in glob('padded_*', root_dir=directory)]

# works with single mask file
mask_file = glob('*mask*', root_dir=directory)
assert len(mask_file)==1, 'not enough / too many mask files found'
mask_file = os.path.join(directory, mask_file[0])

## identify reference dem
count_dict = ArcticDEM.get_counts_and_reference(dem_files)
count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
reference = max(count_dict, key=lambda key: count_dict[key])
print(f'the reference DEM is: {reference}')

# list of dems to register
_to_register = [f for f in dem_files if f != reference]
assert len(_to_register) + 1 == len(dem_files), 'mismatch in number of dems'

# list of tuples (reference DEM, to_register DEM)
coreg_pairs = list(
    product(
        [reference], _to_register
    )
)

_ = [ArcticDEM.coreg(pair) for pair in tqdm(coreg_pairs)]

# copy reference DEM
_ = ArcticDEM.copy_reference(reference)
