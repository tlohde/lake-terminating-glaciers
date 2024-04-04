# paper 2

## contents

### code
#### animate_potential_study_sites.py

- takes points from [potential_study_sites_v1](data/potential_study_sites_v1.geojson) 
- gets all collection-2 level 2 landsat for july-sept (inc) with eo:cloud_cover < 20 %
- apply bit mask for cirrus, cloud, cloud shadow
- group by year and get median
- animate annual medians and save .gif [here](results/intermediate/study_site_animations/)

### data



## data sources

### ice marginal lakes

#### paper
how, p., et al., (2021) greenland-wide inventory of ice marginal lakes using a multi-method approach. *sci rep* **11**, 4481 https://doi.org/10.1038/s41598-021-83509-1

#### dataset
wiesmann, a., et al., (2021) esa glaciers climate change initiative (glaciers_cci): 2017 inventory of ice marginal lakes in greenland (IIML), v1. centre for environmental data analysis, 19 february 2021. doi https://dx.doi.org/10.5285/7ea7540135f441369716ef867d217519


