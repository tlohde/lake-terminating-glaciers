import argparse
from glob import glob
import geopandas as gpd
import os
import rioxarray as rio
import matplotlib.pyplot as plt
import planetary_computer as pc
import stackstac
import pystac_client
from shapely import wkt
import pandas as pd
import utils
from shapely import box
from dask.distributed import Client, LocalCluster
from pystac.extensions.eo import EOExtension as eo
import json
import xarray as xr
import xdem
import numpy as np
import warnings
import sys

print(os.getcwd())
# set directory
parser = argparse.ArgumentParser()
parser.add_argument('--directory')
args = parser.parse_args()
directory = args.directory
os.chdir(f'../data/arcticDEM/{directory}')
print(os.getcwd())

if os.path.exists(f'{os.getcwd()}/coregistered'):
    print('already done')
    sys.exit()

## helpers
def get_date(fname):
    '''getting date of dem from its filename'''
    return pd.to_datetime(fname.split('_')[3], format='%Y%m%d')

def make_mask(bbox, date, drange='14d'):
    '''
    return (lazy) stable terrain mask for given DEM
    queries planetary computer stac catalog for landsat/sentinel images
    14 days either side of the given date, that intersect the 
    bounding box, bbox
    '''
    _catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    d1 = (date - pd.Timedelta(drange)).strftime('%Y-%m-%d')
    d2 = (date + pd.Timedelta(drange)).strftime('%Y-%m-%d')
    _search_period = f'{d1}/{d2}'
    _search = _catalog.search(collections=['sentinel-2-l2a',
                                           'landsat-c2-l2'],
                             bbox=bbox,
                             datetime=_search_period)
    _items = _search.item_collection()
    assert len(_items) > 0, 'did not find any images'
    
    least_cloudy_item = min(_items, key=lambda item: eo.ext(item).cloud_cover)

    # return least_cloudy_item

    _asset_dict = {'l':['green','nir08'],
                   'S':['B03', 'B08']}

    _assets = _asset_dict[least_cloudy_item.properties['platform'][0]]
    
    img = (stackstac.stack(
        least_cloudy_item, epsg=3413,assets=_assets
        ).squeeze()
           .rio.clip_box(*bbox, crs=4326)
           )
    
    # can use [] indexing here because the order
    # of assets in _asset dict is consistent 
    ndwi = ((img[0,:,:] - img[1,:,:]) /
            (img[0,:,:] + img[1,:,:]))
    
    return xr.where(ndwi < 0, 1, 0)

def prep_reference(reference):
    ref = xdem.DEM(reference)
    ref_date = get_date(reference)
    ref_bounds = utils.shapely_reprojector(box(*ref.bounds),
                                           ref.crs.to_epsg(),
                                           4326).bounds
    ref_mask = make_mask(ref_bounds, ref_date)
    return (ref, ref_date, ref_bounds, ref_mask)

def register(dem_to_reg, the_reference):
    ref, ref_date, ref_bounds, ref_mask = the_reference
    to_reg = xdem.DEM(dem_to_reg)
    to_reg_date = get_date(dem_to_reg)
    to_reg_mask = make_mask(ref_bounds, to_reg_date)

    with rio.open_rasterio(reference) as ds:
            combined_mask = ((ref_mask.rio.reproject_match(ds)
                              & to_reg_mask.rio.reproject_match(ds)) == 1).data

    pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Tilt()
    pipeline.fit(
        reference_dem=ref,
        dem_to_be_aligned=to_reg,
        inlier_mask=combined_mask
    )
    regd = pipeline.apply(to_reg)

    stable_diff_before = (ref - to_reg)[combined_mask]
    stable_diff_after = (ref - regd)[combined_mask]
    
    before_median = np.ma.median(stable_diff_before)
    after_median = np.ma.median(stable_diff_after)
    
    before_nmad = xdem.spatialstats.nmad(stable_diff_before)
    after_nmad = xdem.spatialstats.nmad(stable_diff_after)

    output = regd.to_xarray()

    output.attrs['to_register'] = dem_to_reg
    output.attrs['to_register_date'] = get_date(dem_to_reg).strftime('%Y-%m-%d')
    output.attrs['to_reg_mask'] = to_reg_mask['id'].values.item()
    
    output.attrs['reference'] = reference
    output.attrs['reference_date'] = get_date(reference).strftime('%Y-%m-%d')
    output.attrs['ref_mask'] = ref_mask['id'].values.item()
    
    output.attrs['before_nmad'] = before_nmad
    output.attrs['after_nmad'] = after_nmad
    output.attrs['before_median'] = before_median
    output.attrs['after_median'] = after_median

    output.attrs['processing_params'] = {
        'coreg method' : 'xdem.coreg.NuthKaab(), xdem.coreg.Tilt()',
        'mask' : '(NDWI(to_reg_mask) < 0) & (NDWI(ref_mask) < 0)'
    }
    output.attrs['date_processed'] = pd.Timestamp.now().strftime('%Y-%m-%d_%H:%M')
    output.attrs['processed_by'] = 'tlohde'

    if not os.path.exists(f'{os.getcwd()}/coregistered'):
        os.mkdir(f'{os.getcwd()}/coregistered')

    output.rio.to_raster(f'coregistered/{os.path.basename(dem_to_reg)}')

# do everything
# get date of reference image
with open('padded/reference.txt', 'r') as ref:
    d = ref.readlines()[0]

# get list of all padded DEMs
files = [f for f in glob('padded/*.tif') if 'bitmask' not in f]

# get path to reference DEM
reference = [f for f in files if d in f]
assert len(reference)==1, 'too many / not enough DEMs found'
reference = reference[0]

# get paths of DEMs that are to be aligned
dems_to_register = [f for f in files if f != reference]

assert len(files) - len(dems_to_register) == 1, 'missing / double counting a DEM'

# read in centreline
with open('download_notes.txt', 'r') as notes:
    notes = json.load(notes)
    line = wkt.loads(notes['centreline'])

with warnings.catch_warnings(action="ignore"):
    the_reference = prep_reference(reference)
    for i, dem_to_reg in enumerate(dems_to_register):
        print(f'now working on #{i}/{len(dems_to_register)}: {dem_to_reg}')
        try:
            register(dem_to_reg, the_reference)
        except:
            print(f"can't do it. won't do it. skipping {dem_to_reg}")
print('done')