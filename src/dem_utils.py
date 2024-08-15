"""
contains helper functions for the downloading
and co-registering of arctic dem strips.
"""
import geopandas as gpd
from glob import glob
import numpy as np
import os
import pandas as pd
import planetary_computer as pc
import pystac_client
from pystac.extensions.eo import EOExtension as eo
from rasterio.enums import Resampling
import rioxarray as rio
import shapely
import shapely.wkt
from utils import misc, Trends
import utils
import stackstac
import xarray as xr
import xdem


class ArcticDEM():

    @staticmethod
    def get_catalog_gdf(f: str,
                        months: list=[6,7,8,9],
                        crs: int=3413) -> gpd.GeoDataFrame:
        '''
        open arctic dem catalog parquet file and location 'f'
        filter by months and reproject to crs.

        inputs:
            f: str. path to file
            months: list. list of month numbers to _include_
            crs: epsg code to project catalog to

        additionally this function:
        - changes datetime to be of datetime type, and sorts by date
        - changes download links - so that they work

        returns: geodataframe
        '''
        catalog = gpd.read_parquet(f)
        # make timestamps, datetimes, and sort (so downloads in date order)
        catalog['acqdate1'] = (catalog['acqdate1']
                               .dt.tz_localize(None)
                               .astype('datetime64[ns]')
                               )
        catalog['acqdate2'] = (catalog['acqdate2']
                               .dt.tz_localize(None)
                               .astype('datetime64[ns]')
                               )
        catalog.sort_values(by='acqdate1', inplace=True)

        # only select dems from 'summer (-ish)' months
        # (june, july, aug, sept)
        catalog = catalog.loc[catalog['acqdate1'].dt.month.isin(months)]

        # fix download urls
        text_to_replace = 'polargeospatialcenter.github.io/stac-browser/#/external/'
        catalog['downloadurl'] = (catalog['s3url']
                                  .apply(lambda x: (
                                      x.replace(text_to_replace, "")
                                      .replace('.json', '_dem.tif')))
                                  )
        catalog['bitmaskurl'] = (catalog['downloadurl']
                                .apply(lambda x: x.replace('_dem', '_bitmask')))

        # reproject to same crs as centrelines
        catalog = catalog.to_crs(crs)
        return catalog


    @staticmethod
    def get_dem(row: pd.core.frame,
                bounds: tuple,
                outdir: str):
        '''
        lazy function for getting delayed object for
        downloading single arcticDEM strip
        bitmask is applied to DEM, and DEM is clipped and padded
        to supplied bounds

        inputs:
            - row: named tuples (from `df.itertuples()`, where `df` is
        a GeoDataFrame of the `ArcticDEM_Strip_Index_s2s041`
            - bounds: tuple (minx,miny,max,maxy) _MUST BE SAME CRS AS CATALOG_
        the DEM's will be clipped and padded to this
            - outdir: str - directory to output to. does not need '/'
        at the end

        returns dask delayed object that can save the DEM with filename
        given by the dem_id in the arcticDEM catalog; other metadata from
        catalog is added as dictionary of attributes

        to download the lazy object call `dask.compute()`
        '''
        _output_fname = f'padded_{row.dem_id}.tif'
        _output_path = os.path.join(outdir, _output_fname)
        
        if os.path.exists(_output_path):
            print(f'already got {_output_path}')
        else:
            # open and clip DEM and bitmask COGs to bounds
            with rio.open_rasterio(row.downloadurl,
                                chunks='auto') as _dem,\
                rio.open_rasterio(row.bitmaskurl,
                                chunks='auto') as _bitmask:

                    _fill_value = _dem.attrs['_FillValue']
                    _dem_crs = _dem.rio.crs.to_epsg()

                    _dem_clip = _dem.rio.clip_box(*bounds)
                    _bitmask_clip = _bitmask.rio.clip_box(*bounds)

                    # apply bit mask
                    _masked = (xr.where(
                        (_dem_clip == _fill_value)
                        | (_bitmask_clip[:, :, :] > 0),
                        _fill_value,
                        _dem_clip)
                            .rename('z')
                            .squeeze()
                            .rio.write_crs(_dem_crs)
                            )

                    _padded = _masked.rio.pad_box(*bounds,
                                                constant_values=_fill_value)
                    _padded.rio.write_nodata(_fill_value, inplace=True)

                    attributes = row._asdict()
                    attributes['geom'] = attributes['geom'].wkt
                    _padded.attrs = attributes

                    _padded.rio.to_raster(_output_path,
                                          compute=True)
                

    # @staticmethod
    # def export_dems(list_of_dems: list):
    #     '''
    #     download dems from list of lazy clipped/padded bitmasked dems
    #     '''
    #     dask.compute(*list_of_dems)

######### for coregistration
###### making masks

    @staticmethod
    def mask_stable_terrain(directory: str,
                            months: list=[7,8]):
        
        line_file = glob('*.geojson', root_dir=directory)
        line = gpd.read_file(
            os.path.join(directory, line_file[0])
        ).loc[0, 'geometry']
        bounds = line.buffer(5000).bounds
        aoi_4326 = utils.misc.shapely_reprojector(shapely.box(*bounds),
                                                  3413, 4326)
        
        _catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
            )
        
        _search = _catalog.search(collections=['sentinel-2-l2a'],
                                  intersects=aoi_4326,
                                  datetime='2012-01-01/2025-01-01',
                                  query={"eo:cloud_cover": {"lt": 10}})
        
        _items = _search.item_collection()
        _items = [i for i in _items if pd.to_datetime(i.properties['datetime']).month in months]
        
        imgstack = stackstac.stack(_items,
                           assets=['B03', 'B08'],
                           epsg=3413).rio.clip_box(*bounds)
        
        img_ids = imgstack['id'].data.tolist()
        
        with np.errstate(invalid='ignore', divide='ignore'):
        
            median_ndwi = (
                (imgstack[:,0,:,:] - imgstack[:,1,:,:]) / 
                (imgstack[:,0,:,:] + imgstack[:,1,:,:])
                ).median(dim='time')
            
            # open random DEM for reprojecting mask to same extent
            dem_file = glob('padded_*', root_dir=directory)
            if len (dem_file) > 0:
                dem_file = dem_file[0]
                _ds = rio.open_rasterio(os.path.join(directory, dem_file),
                                        chunks='auto')
            # if already cleaned up and deleted padded dems
            # use the stacked coreg'd dataset's projection
            else:
                print('no padded DEMs, using the stack instead')
                dem_file = glob('stack*', root_dir=directory)[0]
                _ds = xr.open_dataset(os.path.join(directory, dem_file),
                                      engine='zarr')['z']

            _mask = xr.where(median_ndwi < 0, 1, 0).rio.reproject_match(_ds)

            _mask.attrs['ids'] = img_ids
            
            _mask.rio.to_raster(os.path.join(directory,
                                             'stable_terrain_mask.tif'),
                                compute=True)

###### picking reference DEM
    @staticmethod
    def get_count(filepath: str):
        '''
        lazily count all valid pixels in DEM
        '''
        with rio.open_rasterio(filepath, chunks='auto') as _dem:
            _fill_value = _dem.attrs['_FillValue']
            _total = (_dem != _fill_value).sum().compute()
            return _total.data.item()

    @staticmethod
    def get_counts_and_reference(filepaths: list):
        counts = [ArcticDEM.get_count(f) for f in filepaths]
        # _lazy_counts = [ArcticDEM.get_lazy_count(f) for f in filepaths]
        # _result = dask.compute(*_lazy_counts)
        # return dict(zip(filepaths, _result))
        return dict(zip(filepaths, counts))

####### doing the coregistration

    @staticmethod
    def copy_reference(reference):
        with rio.open_rasterio(reference, chunks='auto') as ref_dem:
            output_name = reference.replace('padded', 'coregd')
            if os.path.exists(output_name):
                return None
            else:
                _reference_attrs = ref_dem.attrs

                attrs = {}

                attrs['nmad_before'] = 0.0
                attrs['nmad_after'] = 0.0
                attrs['median_before'] = 0.0
                attrs['median_after'] = 0.0
                attrs['coregistration_mask'] = 'n/a'

                for k, v in _reference_attrs.items():
                    new_k = 'ref_' + k
                    attrs[new_k] = v

                ref_dem.attrs = attrs

                ref_dem.rio.to_raster(output_name)

    @staticmethod
    def coreg(pair):
        ref_dem_path, to_reg_dem_path = pair
        directory = os.path.dirname(ref_dem_path)
        mask_path = glob('*mask*', root_dir=directory)
        assert len(mask_path)==1, 'too many / not enough files that could be the mask'
        mask_path = os.path.join(directory, mask_path[0])
        
        output_name = to_reg_dem_path.replace('padded', 'coregd')
        if os.path.exists(output_name):
            print(f'already done: {output_name}\nexiting...')
            return None

        with rio.open_rasterio(ref_dem_path, chunks='auto') as _dem:
            _reference_attrs = _dem.attrs
        with rio.open_rasterio(to_reg_dem_path, chunks='auto') as _dem:
            _to_reg_attrs = _dem.attrs

        _ref = xdem.DEM(ref_dem_path)
        _to_reg = xdem.DEM(to_reg_dem_path)
        with rio.open_rasterio(mask_path) as _mask:
            _mask_ids = _mask.attrs['ids']
            _mask = _mask.squeeze().data == 1
        
        _pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Tilt()

        try:
            _pipeline.fit(
                reference_dem=_ref,
                dem_to_be_aligned=_to_reg,
                inlier_mask=_mask
            )

            coregistered = _pipeline.apply(_to_reg)

            stable_diff_before = (_ref - _to_reg)[_mask]
            stable_diff_after = (_ref - coregistered)[_mask]

            median_before = np.ma.median(stable_diff_before)
            median_after = np.ma.median(stable_diff_after)

            nmad_before = xdem.spatialstats.nmad(stable_diff_before)
            nmad_after = xdem.spatialstats.nmad(stable_diff_after)

            output = coregistered.to_xarray()

            output.attrs['nmad_before'] = nmad_before
            output.attrs['nmad_after'] = nmad_after
            output.attrs['median_before'] = median_before
            output.attrs['median_after'] = median_after

            output.attrs['coregistration_mask'] = _mask_ids

            for k, v in _reference_attrs.items():
                new_k = 'ref_' + k
                output.attrs[new_k] = v

            for k, v in _to_reg_attrs.items():
                new_k = 'to_reg_' + k
                output.attrs[new_k] = v
                
            output.rio.to_raster(output_name,
                                 compute=True)
            
            # return output.rio.to_raster(output_name,
            #                 compute=True,
            #                 lock=dask.distributed.Lock(output_name))

        except Exception as e:
            print(e)
            print(f'failed: {to_reg_dem_path}')


    def downsample(dem, factor=10):
        new_width = int(dem.rio.width / factor)
        new_height = int(dem.rio.height / factor)

        downsampled = dem.rio.reproject(
            dem.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear
        )

        downsampled.rio.write_nodata(np.nan)

        downsampled.attrs['description'] = f'''
        time dependent coregistered elvations. bilinearly downsampled by
        factor of {factor} from {dem.rio.resolution()} m to
        {np.round(downsampled.rio.resolution(),4).tolist()} m.
        '''
        downsampled = downsampled.rio.write_crs('epsg:3413')

        return downsampled

######## centreline sampling

    def sample_along_line(fp: str,
                          geom=False,
                          var=False):
        '''
        reads in .zarr from file path
        (i.e. the output of `dem_trends.py`, or output of `dem_stacking.py`)
        
        takes the centreline stored in .attrs['centreline']
        densifies it (to a total of 100 vertices)
        
        and samples the dataset
        (using .interp(), as opposed to .sel(method='nearest')
        at those points.
        
        option to specify `var` for only selecting certain variables
        from the dataset. e.g. `var='z'` when reading `stacked_coreg.zarr`
        as this contains lots variables that only have a time component
        
        returning Dataset with dims `time`, `cumulative_distance`
        
        inputs: sec_file_path (str) 
        outputs xr.Dataset()
        '''
        
        assert(os.path.isdir(fp)), 'invalid path'
        with xr.open_dataset(fp, engine='zarr') as ds:
            
            if geom:
                centreline = geom
            else:
                centreline = shapely.wkt.loads(ds.attrs['centreline'])
            
            # densify line
            points = [centreline.interpolate(i/100, normalized=True)
                      for i in range(0, 100)]
            cumulative_distance = [centreline.project(p)/1000 for p in points]
            gdf_points = (gpd.GeoDataFrame(geometry=list(points),
                                           index=cumulative_distance,
                                           crs=3413
                                           ).rename_axis('cumulative_distance'))
            
            gdf_points['x'] = gdf_points['geometry'].x
            gdf_points['y'] = gdf_points['geometry'].y
            
            # sample dataarrays
            sampled = ds.interp(x=gdf_points['x'].to_xarray(),
                                y=gdf_points['y'].to_xarray())
            
            return sampled
