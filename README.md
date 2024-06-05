# paper 2

## contents

### code
#### `imagery.py`
contains following functions
- `get_annual_median_mosaic()`
    - takes points from [potential_study_sites_v1](data/potential_study_sites_v1.geojson) 
    - gets all collection-2 level 2 landsat for july-sept (inc) with eo:cloud_cover < 20 %
    - apply bit mask for cirrus, cloud, cloud shadow
    - group by year and get median
- `animate_rgb()`
    - animate annual medians and save .gif [here](results/intermediate/study_site_animations/)

#### `velocity_helpers.py`
contains classes for handling and processing its_live velocity data
##### `CentreLiner()`
- takes either point, or linestring input
- gets appropriate itslive velocity cube(s)
- crops it to fit buffer around centreline

it includes helper functions that...
- can filter along the time axis on `date_dt` or median absolute deviation (mad) of velocity values
- construct median annual composites
- generate flow line from a point
    - this was how the centrelines were originally constructed
- convenience plotting functions
- calculate robust trends using the [Theil-Sen estimator](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator) as implemented in `scipy.stats.mstats.theilslopes`.

#### `robust_spatial_trends.py`
computes and exports velocity trend field around each centreline using the Theil-Sen estimator. and saves output as .zarr (because that eases the memory burden and lets dask take the strain)

this leans heavily on the `CentreLiner()` class in `velocity_helpers.py`

#### `download_data/download_arcticDEM_tiles.py`
- `get_dem()` : convenience function for lazily clipping and masking COG instance of arctic DEM strip tile.

### data
see [data/README.md](data/data_README.md) for details on individual data files used/created
mosaic, and surface/bed topography.

## data sources

### ice marginal lakes

#### paper
how, p., et al., (2021) greenland-wide inventory of ice marginal lakes using a multi-method approach. *sci rep* **11**, 4481 https://doi.org/10.1038/s41598-021-83509-1

#### dataset
wiesmann, a., et al., (2021) esa glaciers climate change initiative (glaciers_cci): 2017 inventory of ice marginal lakes in greenland (IIML), v1. centre for environmental data analysis, 19 february 2021. doi https://dx.doi.org/10.5285/7ea7540135f441369716ef867d217519

### ice velocity

- ice velocity is taken from the latest version of itslive
- velocities derived using autoRift on pairs of landsat / sentinel acquisitions.
- data cubes of velocity are stored on aws and programmatically accessed using the [`itslive`](https://github.com/nasa-jpl/itslive-py/tree/main) python package

#### data sources
see [here](https://its-live.jpl.nasa.gov/#how-to-cite) for list of appropriate references

### arctic DEM
see [here](https://www.pgc.umn.edu/data/arcticdem/)
Porter, Claire, et al., 2022, "ArcticDEM - Strips, Version 4.1", https://doi.org/10.7910/DVN/C98DVS, Harvard Dataverse, V1, Accessed: 4th June 2024. 


## analyses
### ice velocity
robust spatial trends computed using the Theil-Sen estimator

### elevation
DEM

