import argparse
from dask.distributed import Client
import geopandas as gpd
from tqdm import tqdm
import velocity_utils

parser = argparse.ArgumentParser()
parser.add_argument('--centrelines')
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
print(lines)
# set dask cluster running
if __name__ == "__main__":
    client = Client()
    print(f'dask dashboard link: {client.dashboard_link}')

    # iterate through centrelines, and get velocity cube
    V = {}
    failed = []
    for row in tqdm(lines.itertuples()):
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
