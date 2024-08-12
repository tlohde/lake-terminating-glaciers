'''
- general helper functions in misc()
- functions for computing robust trends,
and piecewise trends in Trends()
'''
import numpy as np
import os
import pandas as pd
import pyproj
from scipy import interpolate, optimize
from scipy.stats.mstats import theilslopes
import shapely
import sys
import warnings
import xarray as xr


class misc():
    @staticmethod
    def get_script_path():
        return os.path.dirname(os.path.realpath(sys.argv[0]))
    
    @staticmethod
    def validate_type(func, locals):
        for var, var_type in func.__annotations__.items():
            if var == 'return':
                continue
            if not any([isinstance(locals[var], vt) for vt in [var_type]]):
                raise TypeError(
                    f'{var} must be (/be one of): {var_type} not a {locals[var]}'
                    )

    @staticmethod
    def twoD_interp(img: np.ndarray) -> np.ndarray:
        misc.validate_type(misc.twoD_interp, locals=locals())
        h, w = img.shape[:2]
        mask = np.isnan(img)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        known_x = xx[~mask]
        known_y = yy[~mask]
        known_z = img[~mask]
        missing_x = xx[mask]
        missing_y = yy[mask]

        interp_vals = interpolate.griddata((known_x, known_y),
                                        known_z,
                                        (missing_x, missing_y),
                                        method='cubic',
                                        fill_value=np.nan)
        interpolated = img.copy()
        interpolated[missing_y, missing_x] = interp_vals
        return interpolated

    @staticmethod
    def shapely_reprojector(geo: shapely.geometry,
                            src_crs: int=3413,
                            target_crs: int=4326):
        """
        reproject shapely point (geo) from src_crs to target_crs
        avoids having to create geopandas series to handle crs transformations
        """

        assert isinstance(geo,
                        (shapely.geometry.polygon.Polygon,
                        shapely.geometry.linestring.LineString,
                        shapely.geometry.point.Point)
                        ), 'geo must be shapely geometry'

        transformer = pyproj.Transformer.from_crs(
            src_crs,
            target_crs,
            always_xy=True
        )
        if isinstance(geo, shapely.geometry.point.Point):
            _x, _y = geo.coords.xy
            return shapely.Point(*transformer.transform(_x, _y))
        elif isinstance(geo, shapely.geometry.linestring.LineString):
            _x, _y = geo.coords.xy
            return shapely.LineString(zip(*transformer.transform(_x, _y)))
        elif isinstance(geo, shapely.geometry.polygon.Polygon):
            _x, _y = geo.exterior.coords.xy
            return shapely.Polygon(zip(*transformer.transform(_x, _y)))

    @staticmethod
    def nearest(df, col, val):
        '''
        finds value closest to `val` from column `col` in dataframe `df`
        and returns dataframe that only contains rows where col==val
        '''
        return df.loc[df[col] == min(df[col], key=lambda x: abs(x - val))]
   

class Trends():
    @staticmethod
    def robust_slope(y, t):
        '''
        for robust trends using theilslopes
        y - input array of variable of concern
        t - array of corresponding timestamps
            converts timestamps to years since first observation
            identify nan values in `y`, return theilslopes for non-nan values
        '''
        x = (t-t.min()) / pd.Timedelta('365.25D')
        idx = np.isnan(y)  # .compute()

        if len(idx) == idx.sum():
            return np.stack((np.nan, np.nan, np.nan, np.nan),
                            axis=-1)
        else:
            with warnings.catch_warnings(action='ignore'):
                slope, intercept, low, high = theilslopes(y[~idx], x[~idx])
            return np.stack((slope, intercept, low, high),
                            axis=-1)

    @staticmethod
    def make_robust_trend(ds, inp_core_dim='mid_date'):
        '''
        robust_slope as ufunc to dask array, dss
        this is a lazy operation
        --> very helpful SO
        https://stackoverflow.com/questions/58719696/
        how-to-apply-a-xarray-u-function-over-netcdf-and-return-a-2d-array-multiple-new
        /62012973#62012973
        --> also helpful:https://stackoverflow.com/questions/71413808/
        understanding-xarray-apply-ufunc
        --> and this:
        https://docs.xarray.dev/en/stable/examples/
        apply_ufunc_vectorize_1d.html#apply_ufunc
        '''
        output = xr.apply_ufunc(Trends.robust_slope,
                                ds,
                                ds[inp_core_dim],
                                input_core_dims=[[inp_core_dim],
                                                [inp_core_dim]],
                                output_core_dims=[['result']],
                                exclude_dims=set([inp_core_dim]),
                                vectorize=True,
                                dask='parallelized',
                                output_dtypes=[float],
                                dask_gufunc_kwargs={
                                    'allow_rechunk': True,
                                    'output_sizes': {'result': 4}
                                    }
                                )
        
        output['result'] = xr.DataArray(['slope',
                                        'intercept',
                                        'low_slope',
                                        'high_slope'],
                                        dims=['result'])
        
        return output
    
    @staticmethod
    def piecewise_fit(X, Y, maxcount):
        '''
        ref: https://discovery.ucl.ac.uk/id/eprint/10070516/1/AIC_BIC_Paper.pdf
        ref: https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
        piecewise linear fit
        does not require specifying number of segments
            '''
        xmin = X.min()
        xmax = X.max()

        n = len(X)

        AIC_ = float('inf')
        BIC_ = float('inf')
        r_ = None

        for count in range(1, maxcount+1):

            seg = np.full(count - 1, (xmax - xmin) / count)

            px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean()
                                for x in px_init])

            def func(p):
                seg = p[:count - 1]
                py = p[count - 1:]
                px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
                return px, py

            def err(p):  # This is RSS / n
                px, py = func(p)
                Y2 = np.interp(X, px, py)
                return np.mean((Y - Y2)**2)

            r = optimize.minimize(err,
                                x0=np.r_[seg, py_init],
                                method='Nelder-Mead')

            # Compute AIC/ BIC.
            AIC = n * np.log10(err(r.x)) + 4 * count
            BIC = n * np.log10(err(r.x)) + 2 * count * np.log(n)

            if ((BIC < BIC_) & (AIC < AIC_)):  # Continue adding complexity.
                r_ = r
                AIC_ = AIC
                BIC_ = BIC
            else:  # Stop.
                count = count - 1
                break

        return func(r_.x)  # Return the last (n-1)
