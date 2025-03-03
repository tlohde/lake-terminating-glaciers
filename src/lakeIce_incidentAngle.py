from dask.distributed import LocalCluster
import logging
import argparse
import geopandas as gpd
import pandas as pd
import os
import lakeIce_utils as liU
import warnings

if __name__ == "__main__":
    
    # set directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',
                        type=int,
                        help='index into lakes',
                        )
    parser.add_argument('--sample',
                        action=argparse.BooleanOptionalAction,
                        default=False)
    
    args = parser.parse_args()
    index = args.index
    sample = args.sample
    
       
    print(f'index: {index} and cwd {os.getcwd()}')
    
    centrelines = gpd.read_file('data/streams_v3.geojson')

    basins = (gpd.read_file('data/basins/Greenland_Basins_PS_v1.4.2.shp')
            .dissolve('SUBREGION1'))
    centrelines = centrelines.sjoin(basins.drop(columns=['NAME', 'GL_TYPE']),
                                    how='left'
                                    ).rename(columns={'index_right': 'region'})

    lakes = gpd.read_file('data/lake_areas.geojson')

    lakes = lakes.sjoin_nearest(centrelines)
    
    lakes = (lakes
             .sort_values(by='id')
             .drop(columns=['index_right', 'name', 'lake_land'])
             .to_crs(4326)
    )
    
    row = lakes.loc[index]
    
    os.chdir('src/')
    
    print(f'changed cwd to (src hopefully?): {os.getcwd()}')
    
    print(f'starting cluster at: {pd.Timestamp.now()}')
    
    with LocalCluster(n_workers=8,
                      threads_per_worker=2,
                      memory_limit='4G',
                      silence_logs=logging.ERROR) as cluster:
        
        client = cluster.get_client()
        print(client.dashboard_link)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            li = liU.Sentinel1(
                lakes.loc[index],
                export=True,
                sample=sample
                )
    print('all done')