
import argparse
import geopandas as gpd
import dask.dataframe as da
import xarray as xr
from glob import glob
from tqdm import tqdm
import rioxarray as rio
import numpy as np

import os
from rasterio.enums import Resampling
from rasterio.features import rasterize

parser = argparse.ArgumentParser()
parser.add_argument('--directory')
args = parser.parse_args()
directory = args.directory

def demote_coords_to_vars(ds: xr.Dataset,
                          coords: str,
                          var_name: str):
    '''
    messy onliner to for reorganizing dataset.
    e.g. dataset with two variables: a (dims: x, y, t) and b (dims: x, y)
    this function will convert it to a dataset with 
    dimensions x, y and add as many `a` variables as there dim `t` is long
    '''
    return xr.merge([
        ds.drop_vars([coords, var_name]),
        xr.merge(
            [ds[var_name].isel({coords:i}).rename(ds[coords][i].item())
             for i in range(len(ds[coords]))], compat='override').drop_vars(coords)]
                    )

def get_summary_df(site):
    id = os.path.basename(site).split('_')[0][2:]
    
    cl_path = os.path.join(site, glob('*.geojson', root_dir=site)[0])
    cl = gpd.read_file(cl_path)
    where = cl.loc[0, 'where']
    lake_land = cl.loc[0, 'lake_land']
    
    sec_path = os.path.join(site, 'sec.zarr')
    dem_path = os.path.join(site, 'stacked_coregd.zarr')
    mask_path = os.path.join(site, 'stable_terrain_mask.tif')
    
    with rio.open_rasterio(mask_path).squeeze().drop_vars('band') as _mask:
        with xr.open_zarr(dem_path) as _dem:
            with xr.open_zarr(sec_path) as _sec:
                _dem_median = _dem['z'].median(dim='time')
                _dem_median = _dem_median.rio.reproject_match(_sec, resampling=Resampling.bilinear, nodata=np.nan)
                mask_rprj = _mask.rio.reproject_match(_sec, resampling=Resampling.bilinear).rename('mask')
                _sec = demote_coords_to_vars(_sec, 'result', 'sec')
                # _dem_median_mask = xr.where(mask_rprj == 0, _dem_median, np.nan)
                # _sec_mask = xr.where(mask_rprj == 0, _sec, np.nan)
                merged = xr.merge([_dem_median.rename('z_median'), _sec, mask_rprj])

                # buffer geometry, with square ends
                buff_geom = cl.loc[0,'geometry'].buffer(200, cap_style=3)

                # empty array of same x, y dim shape as merged
                arr = np.zeros(merged['z_median'].shape)

                # rasterize
                burned = rasterize(shapes=[(buff_geom, 1)],
                                fill=0,
                                out=arr,
                                transform=merged.rio.transform())

                # merged rasterized with all other dataarrays
                merged = xr.merge([merged, xr.DataArray(data=burned,
                                                        dims=['y','x'],
                                                        coords={'y': merged.y,
                                                                'x': merged.x}).rename('buffer_aoi')])
                
                ## take logical and of stable terrain and centreline masks,
                cl_mask = ((merged['mask'] == 0) & (merged['buffer_aoi'] == 1))

                # set everything else to nan
                cl_masked = xr.where(cl_mask, merged, np.nan)
                
                # convert to dask dataframe
                df = (cl_masked
                      .to_dask_dataframe()
                      .dropna()
                      .drop(columns='spatial_ref')
                      .reset_index())

                # add additional meta
                df['where'] = where
                df['id'] = id
                df['lake_land'] = lake_land

                outpath = os.path.join(site, 'elevation_sample.parquet')
                print(f'exproting to {outpath}')
                da.to_parquet(df=df, path=outpath, compute=True)
                print('done')
                
get_summary_df(directory)
