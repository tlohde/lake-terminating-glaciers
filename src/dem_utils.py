
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import planetary_computer as pc
import pystac_client
from pystac.extensions.eo import EOExtension as eo
import rioxarray as rio
import shapely
from src.utils import misc, Trends
# import utils
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
        catalog['acqdate1'] = catalog['acqdate1'].astype('datetime64[ns]')
        catalog['acqdate2'] = catalog['acqdate2'].astype('datetime64[ns]')
        catalog.sort_values(by='acqdate1', inplace=True)

        # only select dems from 'summer (-ish)' months
        # (june, july, aug, sept)
        catalog = catalog.loc[catalog['acqdate1'].dt.month.isin([6,7,8,9])]

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
    def get_dems(geom: shapely.geometry,
                 outdir: str,
                 catalog_dir: str,
                 months: list=[6,7,8,9],
                 crs: int=3413,
                 apply_bm: bool=True,
                 download_bm: bool=False,
                 ):
        '''
        for iteratively getting all arcticDEM strips
        that intersect with a given geometry.
        geometry can be (multi)Point, (multi)LineString,
        or (multi)Polygon
        
        calls `ArcticDEM.get_dem()` iteratively by using .itertuples()
        on the geodataframe created by intersecting `geom` with `catalog`
        
        inputs:
            - geom: shapely geometry. geometry in crs, crs
            - outdir: str. directory to save outputs to (no '/' needed at end)
            - catalog_dir: str. where is the arcticDEM strip catalog stored
            - months: list of months. (1-->12). for filtering dem catalog
        and only getting DEMs generated in those months
            - crs: what projection is the `geom` in. catalog will be reprojected
        to this prior to intersection
            - apply_bm: boolean (default=True) whether to apply bitmask or not
            - download_bm: boolean (default=True) whether to download_bm
        '''
        
        catalog = ArcticDEM.get_catalog_gdf(catalog_dir,
                                            months,
                                            crs)
        
        dem_selection = catalog.loc[catalog.intersects(geom)]
        print(f'found {len(dem_selection)} DEMs')
        
        for row in dem_selection.itertuples():
            ArcticDEM.get_dem(row=row,
                              bounds=geom.bounds,
                              outdir=outdir,
                              apply_bm=apply_bm,
                              download_bm=download_bm)


    @staticmethod
    def get_dem(row: pd.core.frame,
                bounds: tuple,
                outdir: str,
                apply_bm: bool=True,
                download_bm: bool=False):
        '''
        downloads single arcticDEM strip
        optionally applying bitmask; optionally download bitmask
        
        inputs:
            - row: named tuples (from `df.itertuples()`, where `df` is
        a GeoDataFrame of the `ArcticDEM_Strip_Index_s2s041`
            - bounds: tuple (minx,miny,max,maxy) _MUST BE SAME CRS AS CATALOG_
        the DEM's will be clipped to this
            - outdir: str - directory to output to. does not need '/'
        at the end
            - apply_bm: boolean - whether or not to apply the bitmask.
        default == True. if True, all pixels 'good' are set to np.nan.
            - download_bm: boolean - whether or not to download the bitmask
        default == False. if True, will be download with prefix 'bitmask_'

        saves dem with file name given by the dem_id in the arcticDEM
        catalog with a suffix of the bounds clipped to (_minx_miny_maxx_maxy)
        additional meta data is added from the 
        '''
        # lazily open and clip DEM COG to bounds
        _tmp = rio.open_rasterio(row.downloadurl, chunks='auto')
        _tmp_clip = _tmp.rio.clip_box(*bounds)

        # lazily open and clip bitmask to bounds
        _tmp_bm = rio.open_rasterio(row.bitmaskurl, chunks='auto')
        _tmp_bm_clip = _tmp_bm.rio.clip_box(*bounds)

        # apply bitmask (fill everything that is either the DEM fill value,
        # or where the bitmask > 0 with nan)
        if apply_bm:
            _for_export = (xr.where((_tmp_clip == _tmp.attrs['_FillValue'])
                                    | (_tmp_bm_clip[:, :, :] > 0),
                                    np.nan,
                                    _tmp_clip)
                        .rename('z')
                        .squeeze()
                        .rio.write_crs(_tmp.rio.crs.to_epsg()))
        else:  # don't apply bitmask
            _for_export = (_tmp_clip
                        .rename('z')
                        .squeeze()
                        .rio.write_crs(_tmp.rio.crs.to_epsg()))

        # appends the bounding box to the dem_id when saving
        bounds_str = '_'.join([str(int(b)) for b in bounds])
        fname = f'{row.dem_id}_{bounds_str}'
        
        # add meta data from catalog df
        _for_export.attrs['bounds'] = bounds
        _for_export.attrs['downloaded_on'] = (pd.Timestamp.now()
                                              .strftime('%Y-%m-%d_%H:%M'))
        _for_export.attrs['acqdate1'] = row.acqdate1.strftime("%Y-%m-%d")
        _for_export.attrs['acqdate2'] = row.acqdate2.strftime("%Y-%m-%d")
        _for_export.attrs['rmse'] = row.rmse
        _for_export.attrs['avg_expected_height_accuracy'] = row.avg_expected_height_accuracy
        _for_export.attrs['full_dem_geom'] = shapely.wkt.dumps(row.geom)

        _for_export.rio.to_raster(f'{outdir}/{fname}.tif')

        # for optional downloading of bitmask
        if download_bm:
            (_tmp_bm_clip.rename('mask')
            .squeeze()
            .rio.write_crs(_tmp_bm.rio.crs.to_epsg())
            .rio.to_raster(f'{outdir}/bitmask_{fname}.tif')
            )


    @staticmethod
    def get_date(fname):
        '''getting date of dem from its filename'''
        return pd.to_datetime(fname.split('_')[3], format='%Y%m%d')

    @staticmethod
    def make_mask(bbox: tuple,
                  date: pd._libs.tslibs.timestamps.Timestamp,
                  crs: int,
                  **kwargs):
        '''
        return (lazy) stable terrain mask for given DEM
        queries planetary computer stac catalog for landsat &
        sentinel images `drange` days either side of the given date,
        that intersect the bounding box, bbox
        
        inputs:
            - bbox: (tuple) min_lon, min_lat, max_lon, max_lat
            - date: (pandas timestamp)
            - crs: to project outputs
            - kwargs: `drange` pandas timedelta string for +/-
        either side of date. default=14d
        
        uses the least cloudy scene identified within the search period
        uses ndwi and threshold of 0 to decide what is stable terrain.
        
        returns: lazy/chunked dataarray. binary mask. 1=stable, 0=not
        
        '''
        # default lookup either side of date
        drange = kwargs.get('drange', '14d')
        
        _catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        d1 = (date - pd.Timedelta(drange)).strftime('%Y-%m-%d')
        d2 = (date + pd.Timedelta(drange)).strftime('%Y-%m-%d')
        _search_period = f'{d1}/{d2}'
        _search = _catalog.search(collections=['sentinel-2-l2a',
                                            'landsat-c2-l2'],
                                bbox=bbox,
                                datetime=_search_period)
        _items = _search.item_collection()
        assert len(_items) > 0, 'did not find any images'
        
        least_cloudy_item = min(_items, key=lambda item: eo.ext(item).cloud_cover)

        _asset_dict = {'l':['green','nir08'],
                       'S':['B03', 'B08']}

        _assets = _asset_dict[least_cloudy_item.properties['platform'][0]]
        
        img = (stackstac.stack(
            least_cloudy_item, epsg=crs, assets=_assets
            ).squeeze()
            .rio.clip_box(*bbox, crs=4326) # because bbox is in lat/lon
            )
        
        # can use [] indexing here because the order
        # of assets in _asset dict is consistent 
        ndwi = ((img[0,:,:] - img[1,:,:]) /
                (img[0,:,:] + img[1,:,:]))
        
        return xr.where(ndwi < 0, 1, 0)

    @staticmethod
    def prep_reference(reference: str) -> tuple:
        '''
        helper for prearing the reference DEM when doing 
        lots of co-registrations. (so no need to keep opening it)
        inputs:
            - reference: str. path to reference dem
        returns:
            tuple:
                reference : filepath,
                ref : xdem.DEM instance of DEM,
                ref_date : date of reference DEM
                ref_bounds : tuple (min_lon, min_lat, max_lon, max_lat)
                ref_mask : stable terrain mask (output of `make_mask()`)
        '''
        ref = xdem.DEM(reference)
        ref_date = ArcticDEM.get_date(reference)
        ref_bounds = misc.shapely_reprojector(shapely.box(*ref.bounds),
                                        ref.crs.to_epsg(),
                                        4326).bounds
        ref_mask = ArcticDEM.make_mask(ref_bounds, ref_date)
        return (reference, ref, ref_date, ref_bounds, ref_mask)

    @staticmethod
    def register(dem_to_reg: str,
                 the_reference: tuple):
        '''
        coregistering `dem_to_reg` to `the_reference`
        inputs:
            - dem_to_reg: str. file path to dem to register
            - the_reference. tuple. output of `prep_reference()`
        
        gets stable terrain mask for `dem_to_reg` and & it with
        the reference mask for a unique stable terrain mask for 
        this coregistration pair.
        
        apply NuthKaab() and Tilt() in pipeline
        get nmad and median difference over stable terrain
        before and after and add these stats to the output
        dataarray
        
        returns: nothing. just exports the coregistered dem
        to path: /coregistered/dem_to_reg
        '''
        # unpack 'the_reference'
        reference, ref, ref_date, ref_bounds, ref_mask = the_reference
        
        to_reg = xdem.DEM(dem_to_reg)
        to_reg_date = ArcticDEM.get_date(dem_to_reg)
        
        # can use ref_bounds here - because they've already 
        # been padded to same extent
        to_reg_mask = ArcticDEM.make_mask(ref_bounds, to_reg_date)

        # AND of stable terrain masks
        # re-open the reference DEM for the sake of using
        # rio.reproject_match to get final mask to have same shape
        # as both DEMs
        with rio.open_rasterio(reference) as ds:
                combined_mask = ((ref_mask.rio.reproject_match(ds)
                                & to_reg_mask.rio.reproject_match(ds)) == 1).data

        # coregistraion
        pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Tilt()
        pipeline.fit(
            reference_dem=ref,
            dem_to_be_aligned=to_reg,
            inlier_mask=combined_mask
        )
        regd = pipeline.apply(to_reg)

        # statistics
        stable_diff_before = (ref - to_reg)[combined_mask]
        stable_diff_after = (ref - regd)[combined_mask]
        
        before_median = np.ma.median(stable_diff_before)
        after_median = np.ma.median(stable_diff_after)
        
        before_nmad = xdem.spatialstats.nmad(stable_diff_before)
        after_nmad = xdem.spatialstats.nmad(stable_diff_after)

        output = regd.to_xarray()

        # add meta data to output
        output.attrs['to_register'] = dem_to_reg
        output.attrs['to_register_date'] = ArcticDEM.get_date(dem_to_reg).strftime('%Y-%m-%d')
        output.attrs['to_reg_mask'] = to_reg_mask['id'].values.item()
        
        output.attrs['reference'] = reference
        output.attrs['reference_date'] = ArcticDEM.get_date(reference).strftime('%Y-%m-%d')
        output.attrs['ref_mask'] = ref_mask['id'].values.item()
        
        output.attrs['before_nmad'] = before_nmad
        output.attrs['after_nmad'] = after_nmad
        output.attrs['before_median'] = before_median
        output.attrs['after_median'] = after_median

        output.attrs['processing_params'] = {
            'coreg method' : 'xdem.coreg.NuthKaab(), xdem.coreg.Tilt()',
            'mask' : '(NDWI(to_reg_mask) < 0) & (NDWI(ref_mask) < 0)'
        }
        output.attrs['date_processed'] = pd.Timestamp.now().strftime('%Y-%m-%d_%H:%M')
        output.attrs['processed_by'] = 'tlohde'

        # export
        if not os.path.exists(f'{os.getcwd()}/coregistered'):
            os.mkdir(f'{os.getcwd()}/coregistered')

        output.rio.to_raster(f'coregistered/{os.path.basename(dem_to_reg)}')
