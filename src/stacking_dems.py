import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
import rioxarray as rio
from tqdm import tqdm
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--directory")
args = parser.parse_args()
directory = args.directory

os.chdir(f'../data/arcticDEM/{directory}/coregistered')

files = glob('*.tif')
print(f'there are {len(files)} in {directory}')
# print(files)

dems = []
attrs = []

for f in tqdm(files):
    with rio.open_rasterio(f, chunks='auto') as ds:
        time_index = xr.DataArray(
            data=[pd.to_datetime(ds.attrs['to_register_date'], format="%Y-%m-%d")],
            dims=['band'],
            coords={'band':[1]})
        ds['time'] = time_index
        ds = (ds.swap_dims({'band': 'time'})
              .drop_vars('band')
              .rename('z')
             )
        
        if '_FillValue' in ds.attrs.keys():
            fill_value = ds.attrs['_FillValue']
            ds = xr.where(ds!=fill_value, ds, np.nan, keep_attrs=True)
            ds.attrs['_FillValue'] = np.nan
            dems.append(ds)
        else:
            ds.attrs['_FillValue'] = np.nan
            dems.append(ds)

        attrs.append(ds.attrs)

meta_df = pd.DataFrame(attrs)
meta_df.drop(columns=['AREA_OR_POINT', 'processed_by',
                      'scale_factor', 'add_offset',
                      '_FillValue', 'long_name'],
             errors='ignore',
             inplace=True)
meta_df['date_processed'] = pd.to_datetime(meta_df.date_processed, format="%Y-%m-%d_%H:%M")
meta_df['reference_date'] = pd.to_datetime(meta_df.reference_date, format="%Y-%m-%d")
meta_df['to_register_date'] = pd.to_datetime(meta_df.to_register_date, format="%Y-%m-%d")
meta_df.sort_values(by='to_register_date', inplace=True)
meta_df.to_csv(f'stacked_coregistered_{directory}_meta.csv')

demstack = (xr.concat(dems,
                      dim='time',
                      combine_attrs='drop')
            .sortby('time'))

demstack.to_zarr(f'stacked_coregistered_{directory}.zarr')
