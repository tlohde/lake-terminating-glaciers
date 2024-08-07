"""
get stable terrain masks for all *padded* DEMs in directory
"""
import argparse
import dask
from glob import glob
from dem_utils import ArcticDEM
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(prog='stable terrain mask maker',
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

# set directory
assert os.path.isdir(directory), 'path is not a directory. try again'
os.chdir(directory)

files = [f for f in glob('padded_*')]

ArcticDEM.get_all_masks(files)
