import pandas as pd
import itslive
from shapely import LineString
import numpy as np
import xarray as xr
import utils


class VelocityBox():
    def __init__(self, geo, index):

        self.geo = geo
        self.index = index

        self.geo4326 = utils.shapely_reprojector(
            self.geo, 3413, 4326
        )
        self.coords = list(zip(*self.geo4326.exterior.coords.xy))

        self.cubes = itslive.velocity_cubes.find_by_polygon(
            self.coords
        )

        self.get_cubes()

    def get_cubes(self):
        self.dss = []
        for cube in self.cubes:
            _ds = (xr.open_dataset(cube['properties']['zarr_url'],
                                   engine='zarr',
                                   chunks='auto')
                   .rio.write_crs(3413)
                   .rio.clip_box(*self.geo.bounds)
                   )
            self.dss.append(_ds)

    def get_median(self, ddt_range, vars):
        lower, upper = [pd.Timedelta(dt) for dt in ddt_range]
        medians = []

        for ds in self.dss:
            ds_filtered = (ds.sel(
                mid_date=(
                    (ds['date_dt'] > lower)
                    & (ds['date_dt'] < upper)
                    )
                ))

            medians.append(ds_filtered[vars]
                           .groupby(ds_filtered.mid_date.dt.year)
                           .median()
                           .compute())

        self.median = xr.merge(medians)

    def get_flow_dir(self):
        self.flow_dir = np.rad2deg(
            np.arctan2(
                self.median['vx'][-1, :, :],
                self.median['vy'][-1, :, :]
            )
        )

    def _half_stream(self, x0, y0, sign):
        '''
        from http://web.mit.edu/speth/Public/streamlines.py
        '''
        xmin = self.median.x.values.min()
        xmax = self.median.x.values.max()
        ymin = self.median.y.values.min()
        ymax = self.median.y.values.max()

        sx = []
        sy = []

        x = x0
        y = y0

        while xmin < x < xmax and ymin < y < ymax:
            u = self.median['vx'][-1, :, :].interp(
                x=x,
                y=y,
                ).values
            v = self.median['vy'][-1, :, :].interp(
                x=x,
                y=y,
                ).values

            theta = np.arctan2(v, u)

            x += sign * 120 * np.cos(theta)  # * dr *
            y += sign * 120 * np.sin(theta)  # * dr *
            sx.append(x)
            sy.append(y)

        return sx, sy

    def get_stream(self, x0, y0):
        '''
        from http://web.mit.edu/speth/Public/streamlines.py
        '''
        sx, sy = self._half_stream(x0, y0, 1)  # forwards
        rx, ry = self._half_stream(x0, y0, -1)  # backwards

        rx.reverse()
        ry.reverse()

        stream_coords = (rx+[x0]+sx, ry+[y0]+sy)
        self.stream = LineString(zip(*stream_coords))
