import argparse
import dask
import dask.distributed
import geopandas as gpd
from glob import glob
import numpy as np
import os
import logging
import pandas as pd
import rioxarray as rio
import xarray as xr


if __name__ == '__main__':
    
    cluster = dask.distributed.LocalCluster(n_workers=4,
                                            threads_per_worker=2,
                                            memory_limit='8G',
                                            silence_logs=logging.ERROR)
    client = cluster.get_client()

    parser = argparse.ArgumentParser(
        prog='DEM stacker',
        description='''
        lazily stacks coregistered DEMs; appends *all* the meta-data
        to the exported .zarr. also exports meta data as a .parquet
        ''')

    parser.add_argument("--directory")
    args = parser.parse_args()
    directory = args.directory


    # keep cwd as src/
    # append directory to filepaths
    corgd_dem_files = [os.path.join(directory, f) for f in glob('coregd_*', root_dir=directory)]
    centreline = gpd.read_file(
        os.path.join(directory, glob('*.geojson', root_dir=directory)[0])
    )
    centreline_wkt = centreline.loc[0, 'geometry'].wkt
    centreline_type = centreline.loc[0, 'lake_land']
        
###########################
### helper for handling ###
## meta-data and merging ##
###########################

    def make_attr_xrda(attrs):
        df = pd.DataFrame(attrs)

        df.drop(columns=['AREA_OR_POINT', 'scale_factor', 'add_offset',
                            '_FillValue', 'long_name', 'to_reg_catalogid1',
                            'ref_catalogid1', 'to_reg_catalogid2', 'ref_catalogid2'],
                    errors='ignore',
                    inplace=True)

        # add boolean column for is reference
        # df['is_reference'] = df['to_reg_acqdate1'].isnull()
        if 'to_reg_acqdate1' in df.columns:
            df['is_reference'] = False
        else:
            df['is_reference'] = True

        # copy/duplicate ref_/to_reg data for reference DEM
        for col in df.columns:
            if 'ref' in col:
                new_col = col.replace('ref_', 'to_reg_')
                df.loc[df['is_reference'], new_col] = df.loc[df['is_reference'], col]

        df = df.set_index(
            pd.to_datetime(df['to_reg_acqdate1'].rename('time'))
            )
        
        return df

###############################################

    dems = []
    attrs = []
    merged = []
    ## 
    for f in corgd_dem_files:
        with rio.open_rasterio(f, chunks='auto') as ds:
            
            if 'to_reg_acqdate1' in ds.attrs.keys():
                _acqdate1 = pd.to_datetime(ds.attrs['to_reg_acqdate1'])
            else:
                _acqdate1 = pd.to_datetime(ds.attrs['ref_acqdate1'])
            
            time_index = xr.DataArray(
                data=_acqdate1, # [pd.to_datetime(ds.attrs['to_register_date'], format="%Y-%m-%d")],
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
            
            meta_ds = make_attr_xrda([ds.attrs])
            
            merged.append(xr.merge([ds, meta_ds.to_xarray()]))
            
            ds.close()

    #########################
    ###### stacking dems ####
    #########################
    
    demstack = (xr.concat(merged,
                          dim='time')
                .sortby('time'))

    # add metadata descriptions
    demstack['z'].attrs = {
        'description': 'time dependent coregistered elevations',
        'units': 'metres'}

    demstack['time'].attrs = {
        'description': 'acqdate1 time from arcticDEM catalog'}

    demstack['nmad_after'].attrs = {
        'description': '''
        normalized median absolute deviation of differences in elevation
        over `stable terrain` between the reference dem and to_register_dem
        _after_ the coregistration process. lower is better.
        ''',
        'units': 'metres'
    }

    demstack['nmad_before'].attrs = {
        'description': '''
        normalized median absolute deviation of differences in elevation over
        `stable terrain` between the reference dem and to_register_dem _before_
        the coregistration process. lower is better
        ''',
        'units': 'metres'
    }

    demstack['median_after'].attrs = {
        'description': '''
        median difference in elevations over stable terrain between reference
        and to_register_dem _after_ coregistration. closer to zero is better.
        ''',
        'units': 'metres'
    }

    demstack['median_before'].attrs = {
        'description': '''
        median difference in elevations over stable terrain between reference
        and to_register_dem _before_ coregistration. closer to zero is better.
        ''',
        'units': 'metres'
    }

    demstack['coregistration_mask'].attrs = {
        'description': '''
        ID of satellite image(s) used as stable terrain mask during coregistration
        '''
    }
    
    demstack.attrs = {
        'processed by': 'tlohde',
        'processed on': pd.Timestamp.now().strftime('%Y/%m/%d %H:%M'),
        'centreline': centreline_wkt,
        'lake_land': centreline_type
    }

    demstack.to_zarr(
        os.path.join(directory, 'stacked_coregd.zarr'),
        mode='w'
    )
    
    # export metadata dataframe
    meta_df = (demstack
               .drop_vars(['x', 'y', 'z'])
               .to_dataframe())

    meta_df.attrs = {
        'processed by': 'tlohde',
        'processed on': pd.Timestamp.now().strftime('%Y/%m/%d %H:%M'),
        'centreline': centreline_wkt
    }

    # export to parquet as this preserveds the .attrs
    meta_df.to_parquet(
        os.path.join(directory, 'coregistration_metadata.parquet'),
        engine='pyarrow')
    
    client.shutdown()
    client.close()
