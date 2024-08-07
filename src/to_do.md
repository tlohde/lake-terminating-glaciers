## to do

### re-organise data workflow...
- want to have single `.geojson` in `data/` (at the moment this is [`data/streams_v2.geojson`](data/streams_v2.geojson))

- need script that takes this `.geojson` and makes a new directory in `data/` with name `id#_xxxxx_yyyyy` where `id#` is the index in the `.geojson` and `XXXXx_YYYYy` is the coordinates of the centroid of the point in the projected crs (although this could be as a lat-lon)

- into each of these directories put a `.geojson` of that one centreline

- all subsequent scripts should rely only on that `.geojson`, such that script inputs can simply be the directory name / id.

### arctic dem flow structure

|script|inputs|what it does|what needs changing|
|:-:|:-|:-|:-|
|`dem_download_tiles.py`|index of `.geojson`|uses centreline to get arctic dem strip tiles|- [x] change input from index to single centreline<br>- [x] currently includes the creation of directories routine which needs moving outside/before this|
|`dem_padder.py`|`directory`|pads all DEMs in `directory` to same extent|- [x] this can be removed, and this routine put inside `dem_download_tiles.py`<br>- see [this](https://github.com/tlohde/isortuarsuupSermia_2/blob/0dee85a72c1ade2b22d32d2d7888a4072e44aa09/src/utils.py#L268) for how|
|`dem_coregister.py`|`directory`|- coregisters all DEMs in directory to reference<br> - NOTE: requires one DEM to be called 'reference|- [x] auto-magically select reference (see [this](https://github.com/tlohde/isortuarsuupSermia_2/blob/0dee85a72c1ade2b22d32d2d7888a4072e44aa09/src/utils.py#L314) for how to do so lazily) and copy across<br>- [ ] tidy up/delete original DEMs (for space savings)|
|`dem_copying_reference.py`|`directory`|copies reference DEM into /coregistered directory|- [x] remove this and include within `dem_coregister.py`|
|`dem_stacking.py`|`directory`|stacks all coregistered DEMs, adds time dimension|<br>- [ ] fix so uses `acqdate1` as time<br>- [ ] add more complete metadata|
|`dem_trends.py`|`directory`|reads in zarr stack; coarsens; computes thiel-sen estimator|- [ ] use `rasterio` for downsampling *not `.coarsen()`*|