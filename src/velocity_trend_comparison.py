import argparse
from utils import misc, Site
import matplotlib.pyplot as plt
import os
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--index')
args = parser.parse_args()
# index = args.index

comparisons = []
fig, ax = plt.subplots()

for index in range(5):
    
    site = Site(index)

    pf = (site
        .v_field['v']
        .sel(quantile=0.5)
        .polyfit(dim='year', deg=1)
        .sel(degree=1)['polyfit_coefficients'])

    pf_line = misc.sample_along_line(pf, site.cl, freq=250)

    pf_line = pf_line.to_dataframe()

    compare = pf_line.merge(site.v_cl, 
                            left_index=True,
                            right_index=True)

    compare['diff'] = compare['slope'] - compare['polyfit_coefficients']

    comparisons.append(compare['diff'])

comparisons = pd.concat(comparisons).reset_index(drop=True)

fig, ax = plt.subplots()
comparisons.plot.hist(ax=ax)

fig.savefig(f'results/figures/v_compare_robustSUBTlinear_dist.png')

