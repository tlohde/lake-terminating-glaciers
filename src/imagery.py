# import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pystac_client
import planetary_computer
import pyproj
import rioxarray as rio  # noqa # pylint: disable=unused-import
import shapely
import stackstac
from utils import shapely_reprojector
# from tqdm import tqdm
import warnings
import xrspatial as xrs
warnings.filterwarnings("ignore")


def get_annual_median_mosaic(geo,
                             buffer_dist=None,
                             src_crs=None,
                             timeperiod="1975-01-01/2030-12-31",
                             months=[7, 8, 9],
                             ):

    """
    takes shapely point geometry (geo), buffers it by distance
    (buffer_dist) (in metres, whilst ensuring that buffering is done
    in a projected crs).
    uses buffer to query stac catalog of sentinel-2-l2a (can be reworked
    to get landsat-c2-l2 scenes)
    """

    assert isinstance(geo, shapely.geometry.point.Point), 'geo must be a point'

    # desire point to be in a projected crs for buffering
    # but in geographic crs for intersecting with stac catalog
    poi = shapely_reprojector(
        geo,
        src_crs,
        4326
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)

    # query landsat collection on planetary computer
    # with cloud cover % filter
    _search = catalog.search(
        collections=["sentinel-2-l2a"],  # "landsat-c2-l2"
        intersects=poi,
        datetime=timeperiod,
        query={"eo:cloud_cover": {"lt": 20}})

    _items = _search.item_collection()

    # get CRS for each returned item, and chose most freq occurring
    # as the 'target' crs
    vals, cnts = np.unique(
        [itm.properties['proj:epsg'] for itm in _items], return_counts=True)
    target_epsg = pyproj.CRS.from_epsg(vals[cnts.argmax()])

    # get daskarrays of landsat images in a projected crs
    try:
        _ds = stackstac.stack(_items,
                              assets=['B04', 'B03', 'B02'],
                              epsg=target_epsg.to_epsg())
        # return _ds
    except ValueError:
        print('no can do')
        return None

    # because median mosiac - only want summer images - to
    # minimize seasonal variations
    # clip image to bounding box of circle created by buffer_dist
    poi = shapely_reprojector(poi, 4326, target_epsg)

    _ds = (_ds
           .sel(time=_ds.time.dt.month.isin(months))
           .rio.clip_box(*poi.buffer(buffer_dist).bounds))

    # # construct annual median composites
    median_composite = (_ds
                        # .sel(band=['red', 'green', 'blue'])
                        .sel(band=['B04', 'B03', 'B02'])
                        .groupby(_ds.time.dt.year)
                        .median(skipna=True)
                        .transpose('year', 'y', 'x', 'band')
                        .rio.write_transform(grid_mapping_name='spatial_ref')
                        .rio.write_crs(target_epsg.to_epsg(),
                                       grid_mapping_name='spatial_ref')
                        )    

    return median_composite.compute()


def animate_rgb(ds):
    """
    saves .gif animation of annual rgbs
    """

    # for nice plot and file titles
    # get lat lon coords from centroid of dataarray
    cntr = shapely.box(*ds.rio.bounds()).centroid
    cntr_4326 = shapely_reprojector(cntr, ds.rio.crs, 4326)
    lon, lat = cntr_4326.coords.xy
    lon, lat = lon[0], lat[0]
    years = ds.year.values
    title = f'{lat:.1f}N_{-1*lon:.1f}W_{min(years)}-{max(years)}'

    fig, ax = plt.subplots()

    # multispectral.true_color gives better contrast than imshow
    img = (xrs.multispectral.true_color(
        r=ds[0, :, :, 0],
        g=ds[0, :, :, 1],
        b=ds[0, :, :, 2],
        nodata=np.nan
    ).plot.imshow(rgb='band',
                      ax=ax))

    ax.set_title(years[0])

    # grab data from next year in dataarray and update title
    def animate(frame):
        _tc = xrs.multispectral.true_color(
            r=ds[frame, :, :, 0],
            g=ds[frame, :, :, 1],
            b=ds[frame, :, :, 2],
            nodata=np.nan
        )
        img.set_data(_tc)
        ax.set_title(f'{lat:.1f} N,  {-1*lon:.1f} W : {years[frame]}')

    ani = FuncAnimation(
        fig,  animate,  frames=len(years), interval=1000
    )
    ani.save(f'results/intermediate/study_site_animations/{title}.gif')
