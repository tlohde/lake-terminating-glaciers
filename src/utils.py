import numpy as np
import os
import pandas as pd
import planetary_computer as pc
import pystac_client
from pystac.extensions.eo import EOExtension as eo
import pyproj
import rioxarray as rio
from scipy import interpolate, optimize
from scipy.stats.mstats import theilslopes
import shapely
from shapely import wkt
import stackstac
import xarray as xr
import xdem


def validate_type(func, locals):
    for var, var_type in func.__annotations__.items():
        if var == 'return':
            continue
        if not any([isinstance(locals[var], vt) for vt in [var_type]]):
            raise TypeError(
                f'{var} must be (/be one of): {var_type} not a {locals[var]}'
                )


def twoD_interp(img: np.ndarray) -> np.ndarray:
    validate_type(twoD_interp, locals=locals())
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


def shapely_reprojector(geo,
                        src_crs=3413,
                        target_crs=4326):
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
    elif isinstance(geo, shapely.geometry.polygon.Polygon):
        _x, _y = geo.exterior.coords.xy
        return shapely.Polygon(zip(*transformer.transform(_x, _y)))


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


def nearest(df, col, val):
    '''
    finds value closest to `val` from column `col` in dataframe `df`
    and returns dataframe that only contains rows where col==val
    '''
    return df.loc[df[col] == min(df[col], key=lambda x: abs(x - val))]


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
    # print(idx.shape)
    if len(idx) == idx.sum():
        return np.stack((np.nan, np.nan, np.nan, np.nan),
                        axis=-1)
    else:
        slope, intercept, low, high = theilslopes(y[~idx], x[~idx])
        return np.stack((slope, intercept, low, high),
                        axis=-1)


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
    output = xr.apply_ufunc(robust_slope,
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
    
def get_date(fname):
    '''getting date of dem from its filename'''
    return pd.to_datetime(fname.split('_')[3], format='%Y%m%d')


def make_mask(bbox, date, drange='14d'):
    '''
    return (lazy) stable terrain mask for given DEM
    queries planetary computer stac catalog for landsat/sentinel images
    14 days either side of the given date, that intersect the 
    bounding box, bbox
    '''
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

    # return least_cloudy_item

    _asset_dict = {'l':['green','nir08'],
                   'S':['B03', 'B08']}

    _assets = _asset_dict[least_cloudy_item.properties['platform'][0]]
    
    img = (stackstac.stack(
        least_cloudy_item, epsg=3413,assets=_assets
        ).squeeze()
           .rio.clip_box(*bbox, crs=4326)
           )
    
    # can use [] indexing here because the order
    # of assets in _asset dict is consistent 
    ndwi = ((img[0,:,:] - img[1,:,:]) /
            (img[0,:,:] + img[1,:,:]))
    
    return xr.where(ndwi < 0, 1, 0)

def prep_reference(reference):
    ref = xdem.DEM(reference)
    ref_date = get_date(reference)
    ref_bounds = shapely_reprojector(shapely.box(*ref.bounds),
                                     ref.crs.to_epsg(),
                                     4326).bounds
    ref_mask = make_mask(ref_bounds, ref_date)
    return (reference, ref, ref_date, ref_bounds, ref_mask)


def register(dem_to_reg, the_reference):
    reference, ref, ref_date, ref_bounds, ref_mask = the_reference
    to_reg = xdem.DEM(dem_to_reg)
    to_reg_date = get_date(dem_to_reg)
    to_reg_mask = make_mask(ref_bounds, to_reg_date)

    with rio.open_rasterio(reference) as ds:
            combined_mask = ((ref_mask.rio.reproject_match(ds)
                              & to_reg_mask.rio.reproject_match(ds)) == 1).data

    pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Tilt()
    pipeline.fit(
        reference_dem=ref,
        dem_to_be_aligned=to_reg,
        inlier_mask=combined_mask
    )
    regd = pipeline.apply(to_reg)

    stable_diff_before = (ref - to_reg)[combined_mask]
    stable_diff_after = (ref - regd)[combined_mask]
    
    before_median = np.ma.median(stable_diff_before)
    after_median = np.ma.median(stable_diff_after)
    
    before_nmad = xdem.spatialstats.nmad(stable_diff_before)
    after_nmad = xdem.spatialstats.nmad(stable_diff_after)

    output = regd.to_xarray()

    output.attrs['to_register'] = dem_to_reg
    output.attrs['to_register_date'] = get_date(dem_to_reg).strftime('%Y-%m-%d')
    output.attrs['to_reg_mask'] = to_reg_mask['id'].values.item()
    
    output.attrs['reference'] = reference
    output.attrs['reference_date'] = get_date(reference).strftime('%Y-%m-%d')
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

    if not os.path.exists(f'{os.getcwd()}/coregistered'):
        os.mkdir(f'{os.getcwd()}/coregistered')

    output.rio.to_raster(f'coregistered/{os.path.basename(dem_to_reg)}')
