import pandas as pd
import imagery
from functools import partial
import itslive
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import shapely
from shapely import LineString, Point, box
from shapely.ops import unary_union
import scipy.stats as stats
import xarray as xr
from xrspatial.multispectral import true_color
import utils


class Tools():

    @staticmethod
    def filter_ddt(ds, ddt_range):
        '''
        convenience method for filtering itslive xr.Dataset(DataArray)
        by 'date_dt' along mid_date dimension
        ds: xr.Dataset / DataArray
        ddt_range: tuble of pandas timestrings for setting
        upper and lower range of filter values
        returns tuple (filtered dataset, and index for safe keeping)
        '''
        lower, upper = [pd.Timedelta(dt) for dt in ddt_range]
        idx = ((ds['date_dt'] >= lower) & (ds['date_dt'] < upper))
        return ds.sel(mid_date=(idx)), idx

    @staticmethod
    def filter_mad(ds, var, axis, n):
        '''
        for filtering / removing outliers
        values that are `n` * median absolute deviation (mad)
        away from the median are swapped with removed

        ds: input dataset
        var: which variable in dataset is to be filtered
        axis: along which axis (*singular*) is median to be calculated
        can be either int or named coordinate axis
        n: number of mads considered outlying.
        returns: xr.dataarray of same dims as inputs but with outliers
        replaced with nans

        note: outliers are filled with nans rather than removed
        because in case where filtering vx and vy components separately,
        then computing resultant velocity, need to ensure indices still
        line up. *also* dropping nans from 3d array is messy.
        '''
        if isinstance(axis, int):
            dim = ds[var].dims[axis]
        elif isinstance(axis, str):
            dim = axis
            axis = [i for i, k in enumerate(ds[var].dims)
                    if k == dim][0]

        mad = stats.median_abs_deviation(ds[var].data,
                                         axis=axis,
                                         nan_policy='omit')
        modified_z = (ds[var] - ds[var].median(dim=dim)) / mad

        return xr.where(modified_z < n, ds[var], np.nan)


class CentreLiner():
    '''
    class for handling velocity data from itslive
    '''
    def __init__(self, geo, buff_dist, index, **kwargs):
        '''
        geo: shapely polygon of area of interest
        index: int. useful index (for matching class instance with
        geojson/geodataframe consisting of multiple aois)
        '''

        self.index = index

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
        self.geo4326 = utils.shapely_reprojector(self.geo,
                                                 3413,
                                                 4326)

        self.coords = list(zip(*self.geo4326.exterior.coords.xy))
        # use itslive api to get list of dictionaries of zarr velocity cubes
        self.cubes = itslive.velocity_cubes.find_by_polygon(self.coords)

        self.get_cubes()
        # resolution (in m) of velocity dataset
        self.res = np.mean(np.abs(self.dss[0].rio.resolution()))

        self.ddt_range = kwargs.get('ddt_range',
                                    ('335d', '395d'))
        self.n = kwargs.get('n', 5)

        # run class functions
        self.filter_v_components(ddt_range=self.ddt_range, n=self.n)
        self.get_annual_median(['v', 'vx', 'vy'])
        self.clean_median()

        if isinstance(geo, shapely.geometry.point.Point):
            self.get_stream()

        self.pair_points_with_box()
        self.sample_along_line()
        self.get_rgb_mosaic()

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

    def filter_v_components(self, ddt_range, n):
        '''
        filter velocity cubes along time dimension
        by first filtering by date_dt with ddt_range
        then replacing outlier values with nan. where
        outliers are based on those that are `n`
        median absolute deviations from the median.
        MAD filtering done independently on x and y compnents
        and then resultant velocity is computed.

        this is done for each returned cube in cases where aoi
        covers more than one.

        constructs two new lists: filtered_v and filtered_v_idx
        filtered_v houses the filtered velocity datasets
        filtered_v_idx the mid_date / date_ddt boolean indexer
        '''
        self.filtered_v = []
        self.filtered_v_idx = []
        for _ds in self.dss:
            _f_ddt, _f_ddt_idx = Tools.filter_ddt(_ds, ddt_range)
            _vx = Tools.filter_mad(_f_ddt, 'vx', 'mid_date', n)
            _vy = Tools.filter_mad(_f_ddt, 'vy', 'mid_date', n)
            _v = np.hypot(_vx, _vy).rename('v')
            self.filtered_v.append(
                xr.Dataset(data_vars=dict(zip(['vx', 'vy', 'v'],
                                              [_vx, _vy, _v])))
                )
            self.filtered_v_idx.append(_f_ddt_idx)

    def get_annual_median(self, vars=['v', 'vx', 'vy']):
        '''
        groups by year of mid_date and computes median
        of velocity field.
        gets median of **filtered** velocities. which
        means that this median is not equal to the same median
        that is used when doing the outlier detections
        median calculated for each [filtered] velocity cube
        this gives them a common `year` index, along which they can
        be aligned with xr.merge so this returns a single dataset

        inputs: vars - list of variables to apply median to
        defaults to ['v', 'vx', 'vy'] as the component medians
        are used for constructing flow lines

        returns median dataset (self.median)
        '''
        medians = []
        for ds in self.filtered_v:
            medians.append(ds[vars]
                           .groupby(ds.mid_date.dt.year)
                           .median()
                           .compute())
        self.median = xr.merge(medians)

    def clean_median(self):
        '''
        for tidy stream delineation want to get interpolate nans
        this takes most recent year of vx/y components and does
        2d interpolation
        '''
        self.clean_vx = self.median['vx'].isel(year=-1).copy()
        self.clean_vy = self.median['vy'].isel(year=-1).copy()

        self.clean_vx.data = utils.twoD_interp(self.clean_vx.data)
        self.clean_vy.data = utils.twoD_interp(self.clean_vy.data)

    # stream / centreline work fucntions
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
        '''
        self.bboxes = [box(*ds.rio.bounds()) for ds in self.filtered_v]
        self.points = [self.tidy_stream.interpolate(i, normalized=True)
                       for i in np.arange(0, 1, 0.01)]
        self.cumul_dist = [self.tidy_stream.project(p) for p in self.points]

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
        _dfs = []
        for _ps, _ds in zip(self.grouped, self.filtered_v):
            # consider using .coarsen() here as way of sampling around
            # centreline vertices
            _sampled_ds = (_ds['v']
                           .sel(x=_ps['x'].to_xarray(),
                                y=_ps['y'].to_xarray(),
                                method='nearest')
                           .to_dataframe()
                           .drop(columns='mapping')
                           )
            _sampled_ds.reset_index(inplace=True)
            _sampled_ds['year'] = _sampled_ds.mid_date.dt.year
            _dfs.append(_sampled_ds)
        self.v_line_df = pd.concat(_dfs)

    def get_rgb_mosaic(self):
        _minx, _miny, _maxx, _maxy = unary_union(self.bboxes).exterior.bounds
        _width = _maxx - _minx
        _height = _maxy - _miny
        _buff_dist = max(_width, _height)
        _centroid = unary_union(self.bboxes).centroid
        _composite = imagery.get_annual_median_mosaic(
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

    def v_profile_plotter(self, ax):
        _annual_v = (self.v_line_df.groupby(
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
        cmap = plt.get_cmap('viridis')

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

    def plotter(self):

        fig, axs = plt.subplot_mosaic([['v_map', 'rgb'],
                                       ['profile', 'profile']],
                                      figsize=[8, 8])

        self.median.isel(year=-1)['v'].plot(ax=axs['v_map'])
        axs['v_map'].plot(*self.tidy_stream.coords.xy, c='r')
        axs['v_map'].set_title(f'index:{self.index}')

        p = utils.shapely_reprojector(self.point, 3413, 4326)
        axs['v_map'].set_title(f'{p.y:.2f} N, {-1*p.x:.2f} W')

        self.rgb_img.plot.imshow(rgb='band', ax=axs['rgb'])
        axs['rgb'].plot(
            *box(*self.median.isel(year=-1).rio.bounds()).exterior.coords.xy,
            c='tab:orange', lw=0.5
            )

        for ax in ['v_map', 'rgb']:
            axs[ax].set_aspect('equal')
            axs[ax].set_axis_off()

        self.v_profile_plotter(self, ax=axs['profile'])
