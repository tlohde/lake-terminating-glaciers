import argparse
from utils import misc, Site
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--index')
args = parser.parse_args()
# index = args.index

comparisons = []
failed = []
fig, ax = plt.subplots()

for index in range(36):
    
    try:
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
    except Exception as e:
        print(f'#{index} did not work because\n{e}')
        failed.append((index, e))
        continue


comparisons = pd.concat(comparisons).reset_index(drop=True)

fig, ax = plt.subplots()
comparisons.plot.hist(bins=np.arange(-100, 100, 0.5), ax=ax)

ax.set(
    xlabel='v trend difference (m yr^-2)',
    title='theil slope subtract linear fit on annual medians',
    xlim=(-15, 15)
)

fig.savefig(f'results/figures/v_compare_robustSUBTlinear_dist.png')

print(f'these failed:\n{failed}')