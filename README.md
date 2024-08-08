# paper 2

## contents

### workflow
#### elevation
run these scripts, in this order...
- `make_dirs.py`
    - makes a directory for in `data/` for each centreline in `data/streams_v2.geojson`
    - and puts copy of centreline (*singular*) in each directory
- `dem_download_tiles.py`
    - usage: `python dem_download_tiles.py --directory data/id#_Xx_Yy --months 6 7 8 9 --buffer 5000`
    - inputs: `--directory`, `--months`, `--buffer`
    - for given directory, downloads all arctic DEM strips that intersect with the the centreline in that directory
    - clips and pads each DEM to the bounds of the centreline + buffer (default=5000 m)
    - only includes DEMs captured during specified months
- `dem_get_masks.py`
    - usage: `python dem_get_masks.py --directory data/id#_Xx-Yy`
    - inputs: `--directory`
    - returns/outputs `.tif`
    - for given directory (`--directory`) take all DEMs with file name `padded_*` and get binary stable terrain mask (where 1==stable terrain; 0==snow/ice/water/unstable terrain) from landsat/sentinel
    - mask is re-projected to same extent & resolution as DEM
- `dem_coregister.py`
    - usage: `python dem_coregister.py --directory data/id#_Xx_Yy`
    - inputs: `--directory`
    - outputs: coregistered DEMs
    - for given directory containing several DEMs (all padded to same extent, with filenames `padded_*`) and their stable terrain masks (`masks_*`) auto-magically decide which DEM to use as the *reference* on the basis of number of valid pixels
    - coregister all DEMs to the reference, renaming to `coregd_*`
    - stable terrain mask used for coregistration is the logical and of both `masks_*`, except when there are no overlapping valid pixels, in which case, revert to reference mask.
    - all meta-data from reference, and to_register DEM are added to the output `coregd_`
    - reference DEM is copied / renamed.
- `dem_stacking.py`
    - usage: `python src/dem_stacking.py --directory data/id1_6685x_-3188046y/`
    - inputs: `--directory`
    - outputs: `stacked_coregd.zarr` folder/file
    - for given directory containing multiple DEMs that have been coregistered (filename: `coregd_*`), read in and stack in time dimension.
    - retain *all* metadata and append to stack. export as zarr
    - also export a `coregistration_metadata.parquet` containing the metadata
- `dem_trends.py`
    - usage: `python src/dem_trends.py --directory data/id#_Xx_Yy --nmad 2.5 --median 2`
    - inputs: `--directory`, `--nmad` (default 2.0), `--median` (default 1.0)
    - outputs `sec.zarr`
    - for given directory
        - open the stack of co-registered DEMs (`stacked_coregd.zarr`)
        - filter stack by `nmad_after` and `median_after` to only include DEMs whose values we 'trust'
        - down sample (by factor of 10 to 20x20 m using bilinear)
        - compute robust trends using `scipy.stats.theilslopes()`
        - export trends to `sec.zarr`
        - output has dimensions: `x`, `y` and `result`. where `result` has length four, and includes the slope estimate (`slope`) along with the 0.95 confidence intervals (`high_slope` and `low_slope`) as well as `intercept`
        - output has variables `sec` (dim: `y`, `x`, `result`) and `n` (dim: `y`, `x`) which counts the number of not null observations
- `dem_cleanup.py`
    - usage: `python src/dem_cleanup.py --directory data/id#_Xx_Yy/`
    - inputs: `--directory`
    - deletes all `.tiff` files in directory

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




