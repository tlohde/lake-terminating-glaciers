"""
loops through every line in
data/streams_v3.geojson
makes a 4 panel plot:
UL: % velocity change along centre line
for each year relative to first year
LL: SEC along centreline
UR: velocity trend field (this is computed as linear trend
from annual median fields)
LR: SEC trend field
saves output figure to `results/figures/*id*_v_sec_plot.png`
"""

import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from glob import glob
from shapely import wkt
import numpy as np
import pandas as pd
from utils import misc

centrelines = gpd.read_file('data/streams_v3.geojson')

basins = (gpd.read_file('data/basins/Greenland_Basins_PS_v1.4.2.shp')
          .dissolve('SUBREGION1'))
centrelines = centrelines.sjoin(basins.drop(columns=['NAME', 'GL_TYPE']),
                                how='left'
                                ).rename(columns={'index_right': 'region'})



def change_plot(id):
    
    where = centrelines.loc[id, 'where']
    region = centrelines.loc[id, 'region']
    lake_land = centrelines.loc[id, 'lake_land']
    
    vaf_f = [f for f in glob('results/velocity/annual_fields/*') if f'id{id}_' in f]
    secf_f = [f for f in glob(f'data/id{id}_*/sec.zarr') if f'id{id}_' in f]
    
    assert len(vaf_f) == len(secf_f) == 1, 'found too many'
    
    vds = xr.open_zarr(vaf_f[0])
    zds = xr.open_zarr(secf_f[0])
    vcl = wkt.loads(vds.attrs['centreline'])
    zcl = wkt.loads(zds.attrs['centreline'])

    assert vcl.equals(zcl), 'centrelines not quite right'

    fig, axs = plt.subplot_mosaic([['v_cl', 'v'],
                                   ['sec_cl', 'sec']])
    
    axs['v_cl'].set_prop_cycle(color=plt.get_cmap('viridis', len(vds.year)).colors)
    
    (misc.sample_along_line(
        (100 * vds.sel(quantile=0.5)['v'] / vds.sel(quantile=0.5).isel(year=0)['v']), vcl, 250)
     .rename('delta V (%)')
     .plot(hue='year', add_legend=False, ax=axs['v_cl'])
    )
        
    (misc.sample_along_line(zds.sel(result='slope')['sec'], vcl, 250)
     .rename('centreline sec (m/yr)')
     .plot(hue='year', ax=axs['sec_cl'])
     )
    
    axs['sec_cl'].axhline(0, c='k', lw=0.5)
    

    (vds.sel(quantile=0.5)['v']
    .polyfit(dim='year', deg=1)['polyfit_coefficients']
    .sel(degree=1)
    .plot(robust=True,
        ax=axs['v'],
        cmap='PiYG',
        cbar_kwargs={'label':'v trend (m yr^-2)'})
    )

    axs['v'].plot(*vcl.coords.xy, c='k', ls=':')

    (zds['sec'].sel(result='slope').plot(robust=True,
                                         ax=axs['sec'],
                                         cmap='RdBu',
                                         cbar_kwargs={'label': 'sec trend (m yr^-1)'})
    )
    
    axs['sec'].plot(*vcl.coords.xy, c='k', ls=':')
    
    ################### FORMATTING ################
    axs['v_cl'].set(title=f'#{id}: {region} {where} / {lake_land}',
                    xlabel=None)
    
    axs['sec_cl'].set(title=None)
    
    axs['v'].set(title=None,
                 xlabel=None,
                 ylabel=None,
                 xticks=[],
                 yticks=[])
    
    axs['sec'].set(title=None,
                   xlabel=None,
                   ylabel=None,
                   xticks=[],
                   yticks=[])
    
    # fig.savefig(f'results/figures/id{id}_v_sec_plot.png')
    
failed = []
for i in range(centrelines.index.max() + 2):
    try:
        change_plot(i)
    
    except Exception as e:
        failed.append((i, e))
        print(f'#{i} did not work because\n{e}')
        continue

_ = [print(f) for f in failed]
print('finished')