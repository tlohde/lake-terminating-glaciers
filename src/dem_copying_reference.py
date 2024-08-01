"""
copies reference dem from /padded to /coregistered
and adds some metadata fields
"""
import argparse
from glob import glob
import utils
import os
import pandas as pd
import rioxarray as rio

print(os.getcwd())
# set directory
parser = argparse.ArgumentParser()
parser.add_argument('--directory')
args = parser.parse_args()
directory = args.directory
os.chdir(f'../data/arcticDEM/{directory}/padded')
print(os.getcwd())

attributes = {}

with open('reference.txt', 'r') as q:
    date = q.readline()
    print(date)

    reference = [f for f in glob('*.tif') if date in f][0]
    # print(f'reference is: {reference} and is of type {type(reference)}')
    _, ref_date, ref_bounds, ref_mask = utils.prep_reference(reference)
    
    attributes['to_register'] = f'padded/{os.path.basename(reference)}'
    attributes['to_register_date'] = ref_date.strftime('%Y-%m-%d')
    attributes['to_reg_mask'] = ref_mask['id'].values.item()
    attributes['reference'] = f'padded/{os.path.basename(reference)}'
    attributes['reference_date'] = ref_date.strftime('%Y-%m-%d')
    attributes['ref_mask'] = ref_mask['id'].values.item()
    attributes['before_nmad'] = 0.0
    attributes['after_nmad'] = 0.0
    attributes['before_median'] = 0.0
    attributes['after_median'] = 0.0
    attributes['processing_params'] = 'n/a'
    attributes['date_processed'] = pd.Timestamp.now().strftime('%Y-%m-%d_%H:%M')
    attributes['processed_by'] = 'tlohde'

    with rio.open_rasterio(reference, chunks='auto') as ds:
        if os.path.exists('../coregistered/reference'):
            print('already exists')
        else:
            print('adding attributes....')
            for k, v in attributes.items():
                ds.attrs[k] = v
            print(f'exporting to: ../coregistered/{reference}')
            ds.rio.to_raster(f'../coregistered/{reference}')
        
