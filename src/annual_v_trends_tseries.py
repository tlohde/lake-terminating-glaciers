import argparse
from dask.distributed import Client
# import pandas as pd
# import itslive
import geopandas as gpd
# from shapely.geometry import Polygon
# from shapely import box
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.dates import date2num, DateFormatter, YearLocator
# import seaborn as sns
# import xrspatial as xrs
# import numpy as np
# import xarray as xr
from tqdm import tqdm
# import imagery
# import utils
import velocity_helpers

parser = argparse.ArgumentParser()
parser.add_argument('--centrelines')
parser.add_argument('--buffer', default=3000)
parser.add_argument('--filter_cube', choices=[True, False], default=True)
parser.add_argument('--ddt1', default='335d')
parser.add_argument('--ddt2', default='395d')
parser.add_argument('--mad', default=3)

args = parser.parse_args()
f = args.centrelines
buff = args.buffer
filter_cube = args.filter_cube
ddt_range = (args.ddt1, args.ddt2)
mad = args.mad

# set dask cluster running
# if __name__ == "__main__":
#     client = Client()
#     print(f'dask dashboard link: {client.dashboard_link}')

# read in centrelines
lines = gpd.read_file(f)

# iterate through centrelines, and get velocity cube
V = {}
failed = []
for row in tqdm(lines.sample(2).itertuples()):
    print(f'working on #{row.Index}')
    try:
        V[row.Index] = velocity_helpers.CentreLiner(
            geo=row.geometry,
            buff_dist=buff,
            index=row.Index,
            filter_cube=filter_cube,
            get_annual_trends=True,
            get_rgb=False,
            **{'ddt_range': ddt_range,
               'n': mad}
            )

    except Exception as e:
        failed.append((row.Index, e))
        print(f'#{row.Index} did not work because\n{e}')
        continue

print('finished')