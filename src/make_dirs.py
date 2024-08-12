"""
run from paper2/

makes directory for each centreline in supplied `.geojson`
each directory is given name of format: `id#_Xx_Yy`
where: `#` is the row index in the supplied .geojson
       'X' is the longitude compnent of the line centroid 
       'Y' is the latitude component of the line centroid
"""

import argparse
import geopandas as gpd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    prog='make_dirs',
    description='''
    makes a directory for each line in supplied centrelines
    .geojson file, if directory does not already exist
    '''
)

parser.add_argument("--centrelines",
                    default='data/streams_v3.geojson',
                    help='''
                    supplied as relative path from whatever
                    dir script is called from.
                    default:'data/streams_v3.geojson'
                    ''')

args = parser.parse_args()
path_to_centrelines = args.centrelines

# check that supplied path is valid. if it is. read in file
assert os.path.exists(path_to_centrelines), 'no centrelines at that path. doing nothing. try again'
centrelines = gpd.read_file(path_to_centrelines)

# change working directory to `src/`
os.chdir(script_dir)

# for each line make directory if it doesn't exist
for line in centrelines.itertuples():
    cntr = line.geometry.centroid
    outdir = f'../data/id{line.Index}_{cntr.x:.0f}x_{cntr.y:.0f}y'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # and add a geojson to that directory with just the one centreline in it
    new_line_dir = f'{outdir}/line_{cntr.x:.0f}x_{cntr.y:.0f}y.geojson'
    if not os.path.exists(new_line_dir):
        centrelines.loc[[line.Index]].to_file(new_line_dir, driver='GeoJSON')
