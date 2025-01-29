"""
functions for handling itslive velocity cube
- getting flow centrelines
- filtering (on date_dt or mid_date)
"""
import cmcrameri.cm as cmc
import dask
import dask.delayed
import pandas as pd
import imagery
from functools import partial
import itertools
import itslive
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.dates import date2num
import os
import shapely
from shapely import LineString, Point, box
from shapely.ops import unary_union
import scipy.stats as stats
import xarray as xr
from xrspatial.multispectral import true_color
import utils
import warnings


class Tools():

    @staticmethod
    def filter(ds_df, var, range, axis):
        '''
        convenience method for filtering itslive xr.Dataset(DataArray)
        by 'var' along 'axis' where 'var' is within range (tuple)
        ds_df: xr.Dataset / DataArray OR pandas dataframe
        ddt_range: tuble of datestrings ('yyyy/mm/dd') for setting
        upper and lower range of filter values
        returns index
        '''
        lower, upper = range
        idx = ((ds_df[var] >= lower) & (ds_df[var] < upper))
        if isinstance(ds_df, (xr.core.dataset.Dataset,
                              xr.core.dataarray.DataArray)):
            return idx
        if isinstance(ds_df, pd.core.frame.DataFrame):
            return idx
    
    @staticmethod
    def filter_middate_datedt(ds, middate_range, ddt_range):
        '''
        wraps the above `filter()` for when you want to filter either
        by middate or ddt, or both, or neither
        returns index
        '''
        idxs = []
        if middate_range:
            middate_idx = Tools.filter(ds, 'mid_date', middate_range, 'mid_date')
            idxs.append(middate_idx)
            
        if ddt_range:
            ddt_idx = Tools.filter(ds, 'date_dt', ddt_range, 'mid_date')
            idxs.append(ddt_idx)
        
        if len(idxs) == 1:
            return idxs[0]
        elif len(idxs) > 1:
            return np.logical_and(*idxs)
        else:
            return [True] * len(ds.mid_date)
        

    @staticmethod
    def filter_mad(ds_df,
                   var,
                   axis=None,
                   n=5):
        '''
        for filtering / removing outliers
        values that are `n` * median absolute deviation (mad)
        away from the median are swapped with removed

        ds_df: input dataset (can be either xarray dataset/array)
        or pandas dataframe
        var: which variable in dataset is to be filtered
        axis: along which axis (*singular*) is median to be calculated
        can be either int or named coordinate axis
        n: number of mads considered outlying.
        returns: xr.dataarray of same dims as inputs but with outliers
        replaced with nans

        note: outliers are filled with nans rather than removed
        because in case where filtering vx and vy components separately,
        then computing resultant velocity, need to ensure indices still
        line up. *also* dropping nans from 3d array is messy
        '''

        if isinstance(ds_df, (xr.core.dataset.Dataset,
                              xr.core.dataarray.DataArray)):
            if isinstance(axis, int):
                dim = ds_df[var].dims[axis]
            elif isinstance(axis, str):
                dim = axis
                axis = [i for i, k in enumerate(ds_df[var].dims)
                        if k == dim][0]

            mad = stats.median_abs_deviation(ds_df[var].data,
                                             axis=axis,
                                             nan_policy='omit')
            modified_z = (ds_df[var] - ds_df[var].median(dim=dim)) / mad
            # print(f'MAD: {mad}')
            return xr.where(modified_z < n, ds_df[var], np.nan)

        elif isinstance(ds_df, pd.core.frame.DataFrame):
            mad = stats.median_abs_deviation(ds_df[var],
                                             nan_policy='omit')
            modified_z = (ds_df[var] - ds_df[var].median()) / mad
            # print(f'MAD: {mad}')
            return ds_df.where(modified_z < n)

    @staticmethod
    # think this is fine now
    # currently onlu used in Plotters()
    def filter_line_df(self, middate_range, ddt_range, **kwargs):
        '''
        bundles both filter_ddt with filter_mad
        for a given distance along centreline
        filters centreline velocity df (self.line_df) by date_dt
        and then by MAD
        if no distance (_val) is supplied it function auto-magically
        picks distance along centreline that has the highest velocity
        and uses that
        returns filtered dataframe. that will contain nans
        '''
        
        idx = Tools.filter_middate_datedt(self.line_df,
                                          middate_range=middate_range,
                                          ddt_range=ddt_range
                                          )
        
        _df = self.line_df[idx]

        _var = kwargs.get('var', 'v')
        _vals = kwargs.get('vals', None)
        _col = kwargs.get('col', 'cumul_dist')
        _mad = kwargs.get('mad', 5)

        if _vals is None:
            print('no val supplied - finding distance where v is peak')
            _df = _df.loc[_df[_col] == (_df
                                        .set_index(_col)
                                        .groupby('mid_date')[_var]
                                        .idxmax()
                                        .value_counts()
                                        .idxmax())
                          ].sort_values(by='mid_date')

            if _mad:
                _df = Tools.filter_mad(_df,
                                       _var,
                                       n=_mad).dropna()
                _df = _df.sort_values(by='mid_date')

            return _df

        else:
            if isinstance(_vals, int):
                _vals = [_vals]
            _dfs = []
            for _val in _vals:
                # print(f'now gather {_val}')
                _df_sub = (utils.misc.nearest(_df,
                                              _col,
                                              _val)
                           .sort_values(by='mid_date')
                           )
                if _mad:
                    _df_sub = (Tools.filter_mad(_df_sub,
                                                _var,
                                                n=_mad)
                               .dropna()
                               .sort_values(by='mid_date'))
                _dfs.append(_df_sub)

            return (pd.concat(_dfs)
                    .sort_values(by=['cumul_dist', 'mid_date'])
                    )

    def get_year_counts(list_of_ds):
        _counts_per_year = [ds.mid_date.groupby('mid_date.year').count() for ds in list_of_ds]
        _years = set(itertools.chain.from_iterable([ds.year.values for ds in _counts_per_year]))
        count_dict = {}
        for _y in _years:
            vals = ()
            for ds in _counts_per_year:
                vals += ds.sel(year=_y).values.item(),
            count_dict[_y] = vals
        return count_dict   

class CentreLiner():
    '''
    class for handling velocity data from itslive
    '''
    def __init__(self,
                 geo,
                 buff_dist,
                 index,
                 sample_centreline=False,
                 filter=False,
                 get_robust_trend=False,
                 get_annual_quantiles=False,
                 get_rgb=False,
                 **kwargs):
        '''
        geo: shapely point of area of interest or shapely linestring
        index: int. useful index (for matching class instance with
        geojson/geodataframe consisting of multiple aois)
        buff_dist: area around point/linestring to get velocity and imagery
        '''
        self.index = index
        self.buff_dist = buff_dist
        if isinstance(geo, shapely.geometry.point.Point):
            self.point = geo
            self.geo = geo.buffer(buff_dist,
                                  cap_style=3)
        elif isinstance(geo, shapely.geometry.linestring.LineString):
            self.tidy_stream = geo
            self.point = geo.boundary.geoms[0]
            self.geo = geo.buffer(buff_dist).envelope

        # reproject input geo (epsg:3413) to epsg4326
        # and getting coordinate pairs of polygon exterior
        # for interfacing with itslive api
        self.geo4326 = utils.misc.shapely_reprojector(self.geo,
                                                      3413,
                                                      4326)

        self.coords = list(zip(*self.geo4326.exterior.coords.xy))
        # use itslive api to get list of dictionaries of zarr velocity cubes
        self.cubes = itslive.velocity_cubes.find_by_polygon(self.coords)
        print('getting cubes from itslive')
        self.get_cubes()
        
        # resolution (in m) of velocity dataset
        self.res = np.mean(np.abs(self.dss[0].rio.resolution()))       
        
        # default kwargs for filtering cube
        # only used if filter=True
        self.ddt_range = kwargs.get('ddt_range',
                                    ('335d', '395d'))
        
        self.middate_range = kwargs.get('middate_range',
                                        self.get_mid_date_range())
        
        self.n = kwargs.get('n', False)
        
        # sample along centrelines
        if sample_centreline:
            print('sampling along centreline')
            self.sample_along_line()

        if filter:
            print(f'filtering velocity cube')
            self.filter_v_components(middate_range=self.middate_range,
                                     ddt_range=self.ddt_range,
                                     n=self.n)

        if get_robust_trend:
            assert get_robust_trend in ['line', 'cube', 'both'], 'oops'
            if get_robust_trend in ['cube', 'both']:
                print('computing spatial trends')
                self.robust_spatial_trends(middate_range=self.middate_range,
                                           ddt_range=self.ddt_range)
            if get_robust_trend in ['line', 'both']:
                print('computing trend along centreline')
                self.robust_centreline_trend()

        if get_annual_quantiles:
            print('generating annual median field')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                self.get_annual_quantiles()

        if get_rgb:
            print('getting rgb mosaic')
            self.get_rgb_mosaic()

        if isinstance(geo, shapely.geometry.point.Point):
            print('generating stream line')
            self.get_annual_quantiles()
            self.clean_median()
            self.get_stream()


    def get_cubes(self):
        '''
        for each returned cube (because input geometry might
        overlap multiple zarr cubes) clip to aoi and append to list
        of xr.datasets
        '''
        self.dss = []
        for cube in self.cubes:
            _ds = (xr.open_dataset(cube['properties']['zarr_url'],
                                   engine='zarr',
                                   chunks='auto')
                   .rio.write_crs(3413)
                   .rio.clip_box(*self.geo.bounds)
                   )
            _ds.rio.write_transform(
                _ds.rio.transform(recalc=True),
                inplace=True
            )
            self.dss.append(_ds)

    def get_mid_date_range(self):
        min_dates = np.asarray([ds.mid_date.min().values for ds in self.dss])
        max_dates = np.asarray([ds.mid_date.max().values for ds in self.dss])
        min_date = min_dates.min()
        max_date = max_dates.max()
        self.middate_range = (min_date, max_date)

    def filter_v_components(self,
                            middate_range=False,
                            ddt_range=False,
                            n=False):
        '''
        filter velocity cubes along time dimension
        if `middate_range` is supplied, filter by mid_date.
        if `ddt_range` is supplied, filter by ddt_range
        if `n` is supplied replace outlier values with nan. where
        outliers are based on those that are `n`
        median absolute deviations from the median.
        MAD filtering done independently on x and y compnents
        and then resultant velocity is computed.

        this is done for each returned cube in cases where aoi
        covers more than one.

        constructs two new lists: filtered_v and filtered_v_idx
        filtered_v houses the filtered velocity datasets
        filtered_v_idx the mid_date / date_ddt boolean indexer
        
        if centreline has been sampled and `self.line_df` exists
        this will be filtered by ddt and middate (NOT BY MAD)
        '''
        self.filtered_v = []
        self.filtered_v_idx = []
        for _ds in self.dss:
            idxs = []
            
            _f_idx = Tools.filter_middate_datedt(_ds,
                                                 middate_range,
                                                 ddt_range)
            if n:            
                _vx = Tools.filter_mad(_ds.sel(mid_date=_f_idx), 'vx', 'mid_date', n)
                _vy = Tools.filter_mad(_ds.sel(mid_date=_f_idx), 'vy', 'mid_date', n)
                _v = np.hypot(_vx, _vy).rename('v')
                self.filtered_v.append(
                    xr.Dataset(data_vars=dict(zip(['vx', 'vy', 'v'],
                                                  [_vx, _vy, _v])))
                    )
            else:
                self.filtered_v.append(
                    _ds.sel(mid_date=_f_idx)
                )    
            
            self.filtered_v_idx.append(_f_idx)

            attrs = {
                'middate_range': ' - '.join(
                    [t.strftime('%Y/%m%d') for t in self.middate_range]
                ),
                'ddt_range': ' - '.join(
                    [str(dt.components.days ) + dt.resolution_string for dt in self.ddt_range]
                    ),
                'MAD_n': n,
                'date_processed': pd.Timestamp.now().strftime('%Y%m%d_%H%M'),
                'centreline': self.tidy_stream.wkt,
                'centreline_id': self.index
                }
            for ds in self.filtered_v:
                ds.attrs = attrs                
            
        # filter centreline df
        if hasattr(self, 'line_df'):
            _df_idx = Tools.filter_middate_datedt(self.line_df,
                                                  middate_range,
                                                  ddt_range)
            _nan_idx = ~self.line_df['v'].isna()
            self.filtered_line_df = self.line_df[_df_idx & _nan_idx]
        
    def robust_centreline_trend(self):
        @dask.delayed
        def trend_df(df, y='v', t='mid_date'):
            return utils.Trends.robust_slope(y=df[y], t=df[t])
        
        _grps = self.filtered_line_df.grou
            
        self.centreline_trend_df = (self.filtered_line_df.groupby('cumul_dist')
                                    .apply(trend_df)
                                    .apply(pd.Series)
                                    .rename(columns={
                                        0: 'slope',
                                        1: 'intercept',
                                        2: 'low_slope',
                                        3: 'high_slope'
                                        })
                                    )
        
        attrs = {
        'middate_range': ' - '.join(
            [t.strftime('%Y/%m%d') for t in self.middate_range]
        ),
        'ddt_range': ' - '.join(
            [str(dt.components.days ) + dt.resolution_string for dt in self.ddt_range]
            ),
        'year_counts': str(self.filtered_line_df
                        .groupby('cumul_dist')['year']
                        .value_counts()
                        .to_dict()),
        'date_processed': pd.Timestamp.now().strftime('%Y%m%d_%H%M'),
        'centreline': self.tidy_stream.wkt,
        'centreline_id': self.index
        }
        self.centreline_trend_df.attrs = attrs
        
        _path = f'results/velocity/centreline_trend/id{self.index}_trend.parquet'
        if os.path.exists(os.path.dirname(_path)):
            self.centreline_trend_df.to_parquet(_path)
            print(f'written to {_path}')
        else:
            print('it is done, but not saved')

    def robust_spatial_trends(self,
                              middate_range=False,
                              ddt_range=False,
                              _var='v'):
        _trends = []
        for ds in self.dss:
            _list_of_filtered_dss = []
            # filter on date_dt and mid_date
            _f_idx = Tools.filter_middate_datedt(ds,
                                                 middate_range,
                                                 ddt_range)
            
            _filtered_ds = ds.sel(mid_date=_f_idx)
            _list_of_filtered_dss.append(_filtered_ds)
            # compute trend
            _trends.append(
                (utils.Trends.make_robust_trend(_filtered_ds[_var])
                 .rename(f'{_var}_trend'))
                )
        self.robust_trend = xr.merge(_trends)
        
        year_counts = Tools.get_year_counts(_list_of_filtered_dss)
        
        # add some meta data        
        _now = pd.Timestamp.now().strftime('%y%m%d_%H%M')
        print(f'trends lazily computed, adding metadata: {_now}')

        self.robust_trend['v_trend'].attrs = {
            'crs': 3413,
            'buffer': self.buff_dist,
            'middate_range': middate_range,
            'ddt_range': ddt_range,
            'year_counts': year_counts,
            'date_processed': _now,
            'centreline': self.tidy_stream.wkt,
            'centreline_id': self.index
        }
    
        _path = 'results/intermediate/velocity/robust_annual_trends/'
        _file = f'id{self.index}_{_now}.zarr'
        print(f'now exporting and computing trend output\n{_path}{_file}')
        (self.robust_trend['v_trend']
            .chunk(dict(zip(self.robust_trend['v_trend'].dims,
                            self.robust_trend['v_trend'].shape)))
            .to_zarr(_path+_file,
                    mode='w'))
        print('finished writing')

    @dask.delayed
    def export_trend(self):
        _now = self.robust_trend['v_trend'].attrs['date_processed']
        _path = 'results/intermediate/velocity/robust_annual_trends/'
        _file = f'id{self.index}_{_now}.zarr'
        print(f'now exporting and computing trend output\n{_path}{_file}')
        (self.robust_trend['v_trend']
         .chunk(dict(zip(self.robust_trend['v_trend'].dims,
                         self.robust_trend['v_trend'].shape)))
         .to_zarr(_path+_file,
                  mode='w'))
    
    def get_annual_quantiles(self,
                             vars=['v', 'vx', 'vy'],
                             qs=[0.25, 0.5, 0.75]):
        '''
        groups by year of mid_date and computes lower, median, and upper
        quantiles of velocity field.
        gets lq, median, uq  of **filtered** velocities.
        
        lq,median,uq calculated for each [filtered] velocity cube
        this gives them a common `year` index, along which they can
        be aligned with xr.merge so this returns a single dataset
        
        by default the variables included in self.filtered_v are
        v, vx, and vy.

        inputs: qs: list of quantiles of calculated
            defaults to lower (0.25), median (0.5) and
            upper (0.75)
        
        returns dataset dims: x, y, year, q (self.q_v)
        '''
        q_dss = []
        count_dss = []
        for ds in self.filtered_v:
            
            _q_ds = (ds[vars].chunk({'mid_date': -1})
                     .groupby(ds.mid_date.dt.year)
                     .quantile(qs, dim='mid_date')
                     )
            
            _c_ds = (ds['v'].chunk({'mid_date': -1})
                     .groupby(ds.mid_date.dt.year)
                     .count()
                     .rename('count')
                     )

            q_dss.append(xr.merge([_q_ds, _c_ds]))
            
        self.q_v = xr.merge(q_dss)
        
        attrs = {
            'middate_range': ' - '.join(
                [t.strftime('%Y/%m%d') for t in self.middate_range]
            ),
            'ddt_range': ' - '.join(
                [str(dt.components.days ) + dt.resolution_string for dt in self.ddt_range]
                ),
            'MAD_n': self.n,
            'date_processed': pd.Timestamp.now().strftime('%Y%m%d_%H%M'),
            'centreline': self.tidy_stream.wkt,
            'centreline_id': self.index
        }
        
        self.q_v.attrs = attrs
        
        _path = f'results/velocity/annual_fields/id{self.index}_vField.zarr'
        if os.path.exists(os.path.dirname(_path)):
            self.q_v.chunk('auto').to_zarr(_path, mode='w')
            print(f'written to {_path}')
        else:
            print('it is done, but not saved')

    # stream / centreline work fucntions
    def clean_median(self):
        '''
        for tidy stream delineation want to get interpolate nans
        this takes most recent year of vx/y components and does
        2d interpolation
        '''
        self.clean_vx = self.median['vx'].isel(year=-1).copy()
        self.clean_vy = self.median['vy'].isel(year=-1).copy()

        self.clean_vx.data = utils.misc.twoD_interp(self.clean_vx.compute())
        self.clean_vy.data = utils.misc.twoD_interp(self.clean_vy.compute())

    def _detectLoop(self, xVals, yVals):
        """ Detect closed loops and nodes in a streamline. """
        x = xVals[-1]
        y = yVals[-1]
        D = np.array([np.hypot(x-xj, y-yj)
                      for xj, yj in zip(xVals[:-1],
                                        yVals[:-1])])
        return (D < 0.9 * self.res).any()

    def _half_stream(self, x0, y0, sign, max_counts):
        '''
        for following flow field up/down from point
        x0, y0 - starting coords of field
        sign - -1/+1 for following the velocity field
        against or with the flow
        max_counts - how many steps to take
        from http://web.mit.edu/speth/Public/streamlines.py
        '''
        xmin = self.clean_vx.x.values.min()
        xmax = self.clean_vx.x.values.max()
        ymin = self.clean_vy.y.values.min()
        ymax = self.clean_vy.y.values.max()

        sx = []
        sy = []

        x = x0
        y = y0

        i = 0
        while xmin < x < xmax and ymin < y < ymax:
            if i == max_counts:
                break
            u = self.clean_vx.interp(
                x=x,
                y=y,
                ).values
            v = self.clean_vy.interp(
                x=x,
                y=y,
                ).values

            theta = np.arctan2(v, u)

            x += sign * self.res * np.cos(theta)  # * dr *
            y += sign * self.res * np.sin(theta)  # * dr *

            if np.isnan(x) | np.isnan(y):
                break
            else:
                sx.append(x)
                sy.append(y)
                i += 1

            if i % 10 == 0 and self._detectLoop(sx, sy):
                break
        return sx, sy

    def _fix_stream(self, threshold=None):
        '''
        this isn't perfect, but does a reasonable job
        of ditching verticies at either end of the line where
        there is a series of back-n-forth zig-zags between
        nieghboring cells. this looks at distance between every
        other vertex and wherever that distance is less than threshold
        ditch those vertices
        '''
        if not threshold:
            threshold = (2*(self.res**2))**0.5
        _points = [Point(p) for p in zip(*self.stream_coords)]
        self.tidy_stream = LineString(
            [_points[i] for i, p
             in enumerate(zip(_points, _points[2:])) if
             p[0].distance(p[1]) > threshold]
             )

    def get_stream(self,
                   x0=None,
                   y0=None):
        '''
        get flow line extending up/down from point x0, y0
        from http://web.mit.edu/speth/Public/streamlines.py
        '''
        self.clean_median()

        if not x0:
            x0 = self.point.x
        if not y0:
            y0 = self.point.y

        rx, ry = self._half_stream(x0, y0, 1, 100)  # forwards
        sx, sy = self._half_stream(x0, y0, -1, 100)  # backwards

        rx.reverse()
        ry.reverse()

        self.stream_coords = (rx+[x0]+sx, ry+[y0]+sy)
        self.stream = LineString(zip(*self.stream_coords))
        self._fix_stream()

    def pair_points_with_box(self):
        '''
        in some instances the centreline will cross several
        itslive cubes. this pairs each centreline point
        with the appropriate cube for sampling
        returns self.grouped which is a list of geodataframes
        with index `cumul_dist` and the point
        samples every 250 m along centreline
        `index_right` can be used to index self.dss.
        '''
        self.bboxes = [box(*ds.rio.bounds()) for ds in self.dss]
        self.points = [self.tidy_stream.interpolate(i, normalized=True)
                       for i in np.arange(0, 1, 0.01)]
        self.points = [
            self.tidy_stream.interpolate(x)
            for x in np.arange(0,
                               self.tidy_stream.length + 250,
                               250)
            ]
        self.cumul_dist = [
            self.tidy_stream.project(p)/1000 for p in self.points
            ]

        # geodataframe of centreline vertices
        _gdf_points = (gpd.GeoDataFrame(geometry=list(self.points),
                                        index=self.cumul_dist)
                       .rename_axis('cumul_dist'))

        # geodataframe of bounding boxes of cubes (that have been
        # clipped to aoi)
        _gdf_bboxes = gpd.GeoDataFrame(geometry=list(self.bboxes))

        self.pointInBox = gpd.sjoin(_gdf_points,
                                    _gdf_bboxes,
                                    how='left')

        self.pointInBox['x'] = self.pointInBox['geometry'].x
        self.pointInBox['y'] = self.pointInBox['geometry'].y
        _grouped = self.pointInBox.groupby('index_right')
        _grouped = [q[1] for q in _grouped]
        self.grouped = _grouped

    def sample_along_line(self):
        '''
        sample (unfiltered) cube along centreline
        _quicker_ to do the filtering (both date_dt & outlier)
        after sampling along the line that way fewer
        medians to calculate rather than doing so for
        the whole domain
        returns pandas dataframe
        
        ################ if there are two cubes in the domain, but the line only crosses the second
        cube - this falls over, because len(self.grouped) == 1 whereas self.dss == 2.
        
        '''
        self.pair_points_with_box()
        _dfs = []
        for _ps in self.grouped:
            _dssIdx = _ps['index_right'].unique().item()
            # consider using .coarsen() here as way of sampling around
            _sampled_ds = (self.dss[_dssIdx][['v', 'vx', 'vy',
                                'v_error', 'vx_error', 'vy_error',
                                'acquisition_date_img1',
                                'acquisition_date_img2',
                                'date_dt']]
                           .interp(x=_ps['x'].to_xarray(),
                                   y=_ps['y'].to_xarray())
                           .to_dataframe()
                           .drop(columns='mapping')
                           )
            _sampled_ds.reset_index(inplace=True)
            _sampled_ds['year'] = _sampled_ds.mid_date.dt.year
            _sampled_ds['cumul_dist'] = _sampled_ds['cumul_dist'].round(2)
            _dfs.append(_sampled_ds)
        self.line_df = pd.concat(_dfs).sort_values(by=['mid_date',
                                                       'cumul_dist'])

    def get_rgb_mosaic(self):
        _minx, _miny, _maxx, _maxy = unary_union(self.bboxes).exterior.bounds
        _width = _maxx - _minx
        _height = _maxy - _miny
        _buff_dist = max(_width, _height)
        _centroid = unary_union(self.bboxes).centroid
        _composite = imagery.get_annual_quantiles_mosaic(
            geo=_centroid,
            buffer_dist=_buff_dist,
            src_crs=3413,
            timeperiod="2023-01-01/2023-12-31",
            months=[7, 8, 9]
            )

        self.composite = (_composite
                          .isel(year=-1)
                          .transpose('band', 'y', 'x')
                          .rio.reproject(3413, nodata=np.nan)
                          )

        self.rgb_img = true_color(
            r=self.composite.sel(band='B04'),
            g=self.composite.sel(band='B03'),
            b=self.composite.sel(band='B02')
            )


class Plotters():

    @staticmethod
    def annual_v_profile(self, ax, **kwargs):
        '''
        convenience method for plotting velocities
        along centreline coloured by year
        '''
        cmap = kwargs.get('cmap', cmc.batlow_r)

        _annual_v = (self.line_df.groupby(
            ['cumul_dist', 'year']
            )['v'].agg(['median',
                        partial(stats.median_abs_deviation,
                                nan_policy='omit')])
            .reset_index())

        _annual_v['y1'] = (_annual_v['median']
                           + (1.4826 * _annual_v['median_abs_deviation'])
                           )
        _annual_v['y2'] = (_annual_v['median']
                           - (1.4826 * _annual_v['median_abs_deviation'])
                           )
        norm = Normalize(*_annual_v['year'].agg(['min', 'max']))

        for year in _annual_v['year'].unique():
            _idx = _annual_v['year'] == year
            ax.plot(_annual_v.loc[_idx, 'cumul_dist'],
                    _annual_v.loc[_idx, 'median'],
                    c=cmap(norm(year)))

            ax.fill_between(_annual_v.loc[_idx, 'cumul_dist'],
                            _annual_v.loc[_idx, 'y1'],
                            _annual_v.loc[_idx, 'y2'],
                            alpha=0.2,
                            color=cmap(norm(year)),
                            label=year)
        plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    @staticmethod
    def plotter(self):
        '''
        plot median velocity field from most recent year
        rgb image from recent year & annual velocity profiles
        '''
        self.fig, axs = plt.subplot_mosaic([['v_map', 'rgb'],
                                            ['profile', 'profile']],
                                           figsize=[8, 8])

        self.median.isel(year=-1)['v'].plot(ax=axs['v_map'])
        axs['v_map'].plot(*self.tidy_stream.coords.xy, c='r')
        axs['v_map'].set_title(f'index:{self.index}')

        p = utils.misc.shapely_reprojector(self.point, 3413, 4326)
        axs['v_map'].set_title(f'{p.y:.2f} N, {-1*p.x:.2f} W')

        self.rgb_img.plot.imshow(rgb='band', ax=axs['rgb'])
        axs['rgb'].plot(
            *box(*self.median.isel(year=-1).rio.bounds()).exterior.coords.xy,
            c='tab:orange', lw=0.5
            )

        for ax in ['v_map', 'rgb']:
            axs[ax].set_aspect('equal')
            axs[ax].set_axis_off()

        Plotters.annual_v_profile(self, ax=axs['profile'])

    @staticmethod
    def rolling_median(self, ax, **kwargs):
        '''
        adds rolling median to axes

        this function has ability to first filter the cube using date_dt
        and then basic outlier detection with MAD - does this by making
        a call to `get_velocity.filter_line_df()`

        can specify:
            variable to plot (defaults to `var='v'`)
            how strict the outlier detection `mad=5`
            window size of rolling median (in days) `window='30d'`
            minimum number of observations that must fall
        within window in order to return a result `min_periods=5`
            line color `c='r'`
            line width `lw=2`
        '''
        _ddt_range = kwargs.get('ddt_range', ('335d', '395d'))
        _ddt_bars = kwargs.get('ddt_bars', False)
        _c = kwargs.get('c', 'tab:blue')
        _lw = kwargs.get('lw', 1)
        _vals = kwargs.get('vals', None)
        _col = kwargs.get('col', 'cumul_dist')
        _var = kwargs.get('var', 'v')
        _mad = kwargs.get('mad', 5)
        _window = kwargs.get('window', '21d')
        _min_periods = kwargs.get('min_periods', 5)
        # print(f'inupt vals: {_vals}')
        _df = Tools.filter_line_df(self,
                                   _ddt_range,
                                   var=_var,
                                   **{'col': _col,
                                      'vals': _vals,
                                      'mad': _mad})

        _df = _df.loc[~_df[_var].isna()]

        if isinstance(_vals, list):
            _for_plotting = (
                _df
                .groupby(_col)
                .rolling(_window,
                         on='mid_date',
                         center=True,
                         min_periods=_min_periods)[_var]
                .median()
                .reset_index()
                .groupby(_col)
                )

            for _grp in _for_plotting:
                _grp[1].plot(x='mid_date',
                             y=_var,
                             lw=_lw,
                             ax=ax,
                             label=f'{_var}: {int(_grp[0])} m')

            if _ddt_bars:
                _for_plotting = _df.groupby(_col)
                for _grp in _for_plotting:
                    _v_collection = LineCollection(list(zip(
                        list(zip(date2num(_grp[1]['acquisition_date_img1']),
                                 _grp[1][_var])),
                        list(zip(date2num(_grp[1]['acquisition_date_img2']),
                                 _grp[1][_var]))
                        )))

                    ax.add_collection(_v_collection, autolim=False)

        else:
            _val = _df[_col].unique()[0]
            (_df.rolling(_window,
                         on='mid_date',
                         min_periods=_min_periods)[_var, 'mid_date']
                .median()
                .plot(x='mid_date',
                      y=_var,
                      c=_c,
                      lw=_lw,
                      ax=ax,
                      label=f'{_var}: {int(_val)} m'))

            if _ddt_bars:
                _v_collection = LineCollection(list(zip(
                    list(zip(date2num(_df['acquisition_date_img1']),
                             _df[_var])),
                    list(zip(date2num(_df['acquisition_date_img2']),
                             _df[_var]))
                    )))

                ax.add_collection(_v_collection, autolim=False)
