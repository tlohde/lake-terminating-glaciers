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
from glob import glob
from dem_utils import ArcticDEM
# import geopandas as gpd
import os
# import rioxarray as rio
# import matplotlib.pyplot as plt
# import planetary_computer as pc
# import stackstac
# import pystac_client
# from shapely import wkt
# import pandas as pd
# import utils
# from shapely import box
# from dask.distributed import Client, LocalCluster
import json
# import xarray as xr
# import xdem
# import numpy as np
# import warnings
# import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

print(os.getcwd())
# set directory
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

os.chdir(directory)
print(f'now working from: {os.getcwd()}')


files = [f for f in glob('padded_*')]

count_dict = ArcticDEM.get_counts_and_reference(files)
count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
pretty = json.dumps(count_dict, indent=4)
print(f'count_dict:\n{pretty}')

reference = max(count_dict, key=lambda key: count_dict[key])
print(f'the reference DEM is: {reference}')



# # do everything
# # get date of reference image
# with open('padded/reference.txt', 'r') as ref:
#     d = ref.readlines()[0]

# # get list of all padded DEMs
# files = [f for f in glob('padded/*.tif') if 'bitmask' not in f]

# # get path to reference DEM
# reference = [f for f in files if d in f]
# assert len(reference)==1, 'too many / not enough DEMs found'
# reference = reference[0]

# # get paths of DEMs that are to be aligned
# dems_to_register = [f for f in files if f != reference]

# assert len(files) - len(dems_to_register) == 1, 'missing / double counting a DEM'

# # read in centreline
# with open('download_notes.txt', 'r') as notes:
#     notes = json.load(notes)
#     line = wkt.loads(notes['centreline'])

# with warnings.catch_warnings(action="ignore"):
#     the_reference = utils.prep_reference(reference)
#     for i, dem_to_reg in enumerate(dems_to_register):
#         print(f'now working on #{i}/{len(dems_to_register)}')
#         if os.path.exists(f'coregistered/{os.path.basename(dem_to_reg)}'):
#             print('already done: skipping')
#             continue
#         else:
#             try:
#                 utils.register(dem_to_reg, the_reference)
#             except Exception as e:
#                 print(e)
#                 print(f"can't do it. won't do it. skipping {dem_to_reg}")
# print('done')