import geopandas as gpd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--centrelines')


print(f'working from {os.getcwd()}')

args = parser.parse_args()
f = args.centrelines

print(f'centreline input: {f}')

lines = gpd.read_file(f)
print(lines)