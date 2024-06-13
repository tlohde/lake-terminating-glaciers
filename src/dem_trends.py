"""
for computing dh/dt trends from stack of DEMs
using theilslopes as a robust estimator
relies on dask for lazy computation and export
"""
import argparse
from glob import glob
import os
import pandas as pd
import utils
import warnings
import xarray as xr


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = cluster.get_client()
    print('cluster set up')

    # set directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    args = parser.parse_args()
    directory = args.directory
    os.chdir(f'../data/arcticDEM/{directory}/coregistered')
    print(f'working here: {os.getcwd()}')
    
    f = glob('stacked_coregistered*.zarr')
    assert len(f)==1, 'not enough or too many input files'
    
    f = f[0]
    print(f'stacked file: {f}')
    print(f'the time is now:{pd.Timestamp.now().strftime("%H:%M:%S")}')
    with warnings.catch_warnings(action='ignore'):
        with xr.open_dataset(f, engine='zarr', chunks='auto') as ds:
            print('have opened it')
            print(ds)
            print(f'ds shape: {ds['z'].shape}\n now chunking to\ny: {len(ds.y) // 2}\nx: {len(ds.x) // 5}')
            ds = ds['z'].chunk(chunks={'y': len(ds.y) // 2,
                                       'x': len(ds.x) // 5})
            
            print('coarsen over 10x10 grid - i.e down sample to 20 m')
            median = ds.coarsen(dim={'x': 10, 'y':10},
                                boundary='trim').median()

            print('rechunking...')
            median = median.chunk({'time': -1, 'x':'auto', 'y':'auto'})

            print('lazily compute spatial trends...')
            trend = utils.make_robust_trend(median, inp_core_dim='time')
            print(f'trend dataset {trend}')
            print('computing and exporting....')
            trend.to_zarr('robust_spatial_trend_20x20.zarr', compute=True)
    print('shutting down client')
    print(f'the time is now:{pd.Timestamp.now().strftime("%H:%M:%S")}')
    client.shutdown()
    cluster.close()
    print('done')