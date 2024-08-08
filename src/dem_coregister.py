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
import dask
from dem_utils import ArcticDEM
from glob import glob
from itertools import product
import logging
import os


# script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    
    # initiate dask cluster
    cluster = dask.distributed.LocalCluster(silence_logs=logging.ERROR)
    client = cluster.get_client()

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
    
    # TODO FIX THIS TO ONLY HANDLE SINGLE MASK 
    mask_files = [os.path.join(directory, f) for f in glob('mask_*', root_dir=directory)]

    assert len(dem_files) == len(mask_files), 'unequal number of DEMs and masks'

    # pair DEMs with their masks
    dem_mask_pairs = {}
    for _d in dem_files:
        for _m in mask_files:
            if os.path.basename(_m).split('mask_')[1] in _d:
                dem_mask_pairs[_d] = _m
    
    ## identify reference dem
    count_dict = ArcticDEM.get_counts_and_reference(dem_files)
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
    reference = max(count_dict, key=lambda key: count_dict[key])
    # print(f'the reference DEM is: {reference}')

    # list of dems to register
    _to_register = [f for f in dem_files if f != reference]
    assert len(_to_register) + 1 == len(dem_files), 'mismatch in number of dems'

    # list of tuples (reference DEM, to_register DEM)
    coreg_pairs = list(
        product(
            [reference], _to_register
        )
    )

    # collect lazy coregistered objects
    lazy_outputs = [ArcticDEM.coreg(pair, dem_mask_pairs) for pair in coreg_pairs]
    
    # then compute
    _ = dask.compute(*lazy_outputs)
    
    # copy reference DEM
    _ = ArcticDEM.copy_reference(reference,
                                    dem_mask_pairs)
    
    client.shutdown()
    client.close()
