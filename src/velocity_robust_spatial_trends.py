import argparse
import dask
import dask.distributed
import geopandas as gpd
from glob import glob
import logging
from tqdm import tqdm
import velocity_utils
import os

# set dask cluster running
if __name__ == "__main__":
    cluster = dask.distributed.LocalCluster(n_workers=4,
                                            threads_per_worker=2,
                                            memory_limit='8G',
                                            silence_logs=logging.ERROR)
    
    client = cluster.get_client()
    print(f'dask dashboard link: {client.dashboard_link}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--centrelines')
    parser.add_argument('--index',
                        default=-9999,
                        type=int)
    parser.add_argument('--buffer',
                        default=3000,
                        type=int)
    parser.add_argument('--ddt1',
                        default=335,
                        type=int)
    parser.add_argument('--ddt2',
                        default=395,
                        type=int)
    parser.add_argument('--mad',
                        default=3,
                        type=int)
    parser.add_argument('--filter_cube',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--sample_centreline',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--get_robust_trend',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--get_rgb',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--get_annual_median',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--export_trend',
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    f = args.centrelines
    idx = args.index
    buff = args.buffer
    ddt_range = (f'{args.ddt1}d', f'{args.ddt2}d')
    mad = args.mad
    filter_cube = args.filter_cube
    sample_centreline = args.sample_centreline
    get_robust_trend = args.get_robust_trend
    export_trend = args.export_trend
    get_annual_median = args.get_annual_median
    get_rgb = args.get_rgb

    # read in centrelines
    lines = gpd.read_file(f)

    # iterate through centrelines, and get velocity cube
    V = {}
    failed = []
    already_done = glob('results/intermediate/velocity/robust_annual_trends/*id*',
                root_dir=os.getcwd())
    already_done = [os.path.basename(zarr).split('_')[0].split('id')[-1] for zarr in already_done]
    already_done = [int(id) for id in already_done]
    
    if idx != -9999:
        lines = lines.loc[[idx]]
        
    print(f'there are now {len(lines)} in centrelines')
    
    print(lines)
    
    for row in tqdm(lines.itertuples()):
        if row.Index in already_done:
            print(f'already done #{row.Index}, onwards')
            continue
        else:
            print(f'working on #{row.Index}')
        
            try:
                V[row.Index] = velocity_utils.CentreLiner(
                    geo=row.geometry,
                    buff_dist=buff,
                    index=row.Index,
                    filter_cube=filter_cube,
                    get_robust_trend=get_robust_trend,
                    get_annual_median=get_annual_median,
                    sample_centreline=sample_centreline,
                    get_rgb=get_rgb,
                    **{'ddt_range': ddt_range,
                    'n': mad,
                    'robust_trend_export': export_trend}
                    )

            except Exception as e:
                failed.append((row.Index, e))
                print(f'#{row.Index} did not work because\n{e}')
                continue

    print('finished')
