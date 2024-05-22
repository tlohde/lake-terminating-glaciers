# paper 2

## contents

### code
#### imagery.py
contains following functions
- `get_annual_median_mosaic()`
    - takes points from [potential_study_sites_v1](data/potential_study_sites_v1.geojson) 
    - gets all collection-2 level 2 landsat for july-sept (inc) with eo:cloud_cover < 20 %
    - apply bit mask for cirrus, cloud, cloud shadow
    - group by year and get median
- `animate_rgb()`
    - animate annual medians and save .gif [here](results/intermediate/study_site_animations/)

#### get_velocity.py
contains classes for handling and processing its_live velocity data
##### `CentreLiner()`
- takes either point, or linestring input
- gets ppropriate itslive veloicty cube 

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

## analyses

