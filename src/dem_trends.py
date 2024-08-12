"""
for computing dh/dt trends from stack of DEMs
using theilslopes as a robust estimator
relies on dask for lazy computation and export
"""
import dask
from dem_utils import ArcticDEM
import argparse
from glob import glob
import os
import dask.distributed
import logging
import pandas as pd
import utils
import warnings
import xarray as xr


if __name__ == "__main__":

    _starttime = pd.Timestamp.now()    
    cluster = dask.distributed.LocalCluster(silence_logs=logging.ERROR)
    client = cluster.get_client()

    # set directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    parser.add_argument('--nmad',
                        type=float,
                        help='nmad threshold. only use coregistered DEMs with nmad_after < specified nmad',
                        default=2.0)
    parser.add_argument('--median',
                        type=float,
                        help='median threshold. only use coregistered DEMs with median_after < specified median',
                        default=1.0)
    args = parser.parse_args()
    directory = args.directory
    nmad_threshold = args.nmad
    median_threshold = args.median

    # keep cwd as src/
    # append directory to filepaths
    files = glob('stacked_coregd*.zarr',
                 root_dir=directory)
    assert len(files)==1, 'not enough or too many input files'
    file = os.path.join(directory, files[0])

    with warnings.catch_warnings(action='ignore'):
        with xr.open_dataset(file,
                             engine='zarr', 
                             ) as ds:
            
            centreline = ds.attrs['centreline']
            # idx = ((ds['nmad_after'] < ds['nmad_before']) 
            #        & (ds['median_after'] < ds['median_before']))
            
            idx = ((ds['nmad_after'] < nmad_threshold) 
                   & (ds['median_after'] < median_threshold))
            
            print(f'using {idx.sum().item()} out of possible {len(ds.time)} DEMs')
            
            dem = ds['z'].sel(time=idx).chunk(chunks={'y': len(ds.y) // 4,
                                                     'x': len(ds.x) // 4})
            
            _timestamps = pd.to_datetime(
                dem['time'].data
                ).to_series().apply(
                    lambda t: t.strftime('%Y-%m-%d %H:%M:%S')
                    ).tolist()
            
            downsampled = ArcticDEM.downsample(dem, factor=10)
            
            downsampled = downsampled.chunk(
                {'time': -1,
                 'y': 'auto',
                 'x': 'auto'}
                )
            
            n = downsampled.count(dim='time').rename('n')
                        
            trend = utils.Trends.make_robust_trend(downsampled,
                                                   inp_core_dim='time')

            trend = trend.rename('sec')
            
            trend = xr.merge([trend, n])
            
            trend.attrs = {
                'description': '''
                theilslope estimates of surface elevation change (sec)
                high_slope and low_slope are the 0.95 confidence interval
                ''',
                'timestamps': _timestamps,
                'dem_ids': ds['to_reg_dem_id'].sel(time=idx).data.tolist(),
                'reference_dem': list(set(ds['ref_dem_id'].data.tolist()))[0],
                'median_threshold': median_threshold,
                'nmad_threshold': nmad_threshold,
                'centreline': centreline,
                }
            
            trend.rio.write_crs(input_crs=downsampled.rio.crs,
                                inplace=True)
            
            trend.to_zarr(
                os.path.join(directory, 'sec.zarr'),
                mode='w',
                compute=True,
                )

    _endtime = pd.Timestamp.now()
    # print(f'operation took: {_endtime - _starttime}')

    client.shutdown()
    cluster.close()
