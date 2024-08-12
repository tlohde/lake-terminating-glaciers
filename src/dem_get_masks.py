"""
get stable terrain mask
"""
import argparse
import dask
from dem_utils import ArcticDEM
import logging
import os

if __name__ == '__main__':
    
    # initiate dask cluster
    cluster = dask.distributed.LocalCluster(silence_logs=logging.ERROR)
    client = cluster.get_client()

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

    parser.add_argument("--months",
                        help='''
                        list of month numbers. only use satellite images from
                        these months to construct mask
                        ''',
                        type=int, 
                        nargs='+',
                        default=[7,8])


    args = parser.parse_args()
    directory = args.directory
    months = args.months

    # set directory
    assert os.path.isdir(directory), 'path is not a directory. try again'

    lazy_output = ArcticDEM.mask_stable_terrain(directory, months)

    dask.compute(lazy_output)
    
    client.shutdown()
    cluster.close()
