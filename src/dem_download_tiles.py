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
import dask
from dem_utils import ArcticDEM
import geopandas as gpd
from glob import glob
import logging
import os


if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(prog='arctic DEM downloader',
        description='''
        downloads arctic DEM strip tiles for given month
        and area defined by centreline
        ''')

    parser.add_argument("--directory",
                        help='''
                        supplied as relative path from whatever
                        dir script is called from.
                        e.g. 'data/id01_Xx_Yy/')
                        ''')
    parser.add_argument("--months",
                        type=int, 
                        nargs='+',
                        default=[6,7,8,9],
                        help='''
                        months to include. eg `7 8 9` for
                        july august and september
                        ''')
    parser.add_argument("--buffer",
                        type=int,
                        default=5000,
                        help='''
                        distance (in metres) to buffer centreline
                        ''')

    args = parser.parse_args()
    directory = args.directory
    months = args.months
    buffer = args.buffer
    
    assert os.path.isdir(directory), 'supplied directory not a directory'

    # read in centreline and get shapely geometry instance of line
    # print('reading in centreline')
    geojsons = glob("*.geojson", root_dir=directory)
    assert len(geojsons)==1, 'too many (or not enough) centrelines found.'

    line_file = os.path.join(directory, geojsons[0])
    line = gpd.read_file(line_file)
    line_geom = line.loc[0, 'geometry']

    # aoi bounds
    bounds = line_geom.buffer(buffer).bounds

    # read in arcticDEM catalog
    # print('reading in arctic DEM catalog')
    catalog = ArcticDEM.get_catalog_gdf(
        f=glob('data/arcticDEM/**/*.parquet', recursive=True)[0],
        months=months,
        crs=line.crs
        )

    # intersect catalog with centreline geometry
    catalog = catalog.loc[catalog.intersects(line_geom)]
    print(f'catalog contains {len(catalog)} DEMs after intsersecting with centreline')
    
    # start dask cluster
    # lazily apply `get_dem` for bitmasking, clipping, padding, and downloading 
    # each DEMs in filtered catalog
    cluster = dask.distributed.LocalCluster(n_workers=4,
                                            threads_per_worker=2,
                                            memory_limit='8G',
                                            silence_logs=logging.ERROR)
    client = cluster.get_client()
        
    lazy_download = [ArcticDEM.get_dem(row,
                                        bounds,
                                        directory) for row in catalog.itertuples()]
    
    ArcticDEM.export_dems(lazy_download)
    
    client.shutdown()
    client.close()
