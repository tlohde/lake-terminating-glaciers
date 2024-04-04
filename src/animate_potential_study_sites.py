import geopandas as gpd
import numpy as np
import pystac_client
import planetary_computer
import stackstac
import shapely
import pyproj
import rioxarray as rio
import xrspatial as xrs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def shapely_reprojector(geo,
                        src_crs=3413,
                        target_crs=4326):
    """
    reproject shapely point (geo) from src_crs to target_crs
    avoids having to create geopandas series to handle crs transformations
    """

    assert isinstance(
        geo, shapely.geometry.point.Point), 'input geometry must be a point'
    transformer = pyproj.Transformer.from_crs(
        src_crs,
        target_crs,
        always_xy=True
    )
    _x, _y = geo.coords.xy
    return shapely.Point(*transformer.transform(_x, _y))


def get_annual_median_mosaic(geo,
                             buffer_dist=None,
                             src_crs=None,
                             target_crs=None):

    poi = shapely_reprojector(
        geo,
        src_crs,
        target_crs
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)

    # query landsat collection
    _search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects=poi,
        query={"eo:cloud_cover": {"lt": 20}})

    _items = _search.item_collection()

    # get CRS for each returned item, and chose mode as the 'target'
    vals, cnts = np.unique(
        [itm.properties['proj:epsg'] for itm in _items], return_counts=True)
    target_epsg = pyproj.CRS.from_epsg(vals[cnts.argmax()])

    # get daskarrays of landsat images, and clip to geometry
    try:
        _ds_landsat = stackstac.stack(_items, epsg=target_epsg.to_epsg())
    except ValueError:
        print('no can do')
        return None

    poi = shapely_reprojector(poi, target_crs, target_epsg)

    _ds_landsat = (_ds_landsat
                   .sel(time=_ds_landsat.time.dt.month.isin([7, 8, 9]))
                   .rio.clip_box(*poi.buffer(buffer_dist).bounds))

    # dilated cloud, cirrus, cloud, cloud shadow
    _mask_bitfields = [1, 2, 3, 4]
    _bitmask = 0
    for _field in _mask_bitfields:
        _bitmask |= 1 << _field
    try:
        _qa = _ds_landsat.sel(band="qa_pixel").astype("uint")
        _bad = _qa & _bitmask  # just look at those 4 bits
        _ds_landsat = _ds_landsat.where(_bad == 0)
    except KeyError:
        # ds_landsat.band.values)
        print('failed at line 74')
        return None

    median_composite = (_ds_landsat
                        .sel(band=['red', 'green', 'blue'])
                        .groupby(_ds_landsat.time.dt.year)
                        .median(skipna=True))

    return median_composite.transpose('year', 'y', 'x', 'band').compute()


def animate_rgb(ds):
    cntr = shapely.box(*ds.rio.bounds()).centroid
    cntr_4326 = shapely_reprojector(cntr, ds.rio.crs, 4326)
    lon, lat = cntr_4326.coords.xy
    lon, lat = lon[0], lat[0]
    years = ds.year.values
    title = f'{lat:.1f}N_{-1*lon:.1f}W_{min(years)}-{max(years)}'

    fig, ax = plt.subplots()

    img = (xrs.multispectral.true_color(
        r=ds[0, :, :, 0],
        g=ds[0, :, :, 1],
        b=ds[0, :, :, 2],
        nodata=np.nan
    ).plot.imshow(rgb='band',
                      ax=ax))

    ax.set_title(years[0])

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


# load in manually selected study site points of interest
study_sites = gpd.read_file('data/potential_study_sites_v1.geojson')

# run get cheap and cheerful annual summer cloud free mosaic
# in 10 km buffer around each one
# make animate and save
for row in tqdm(study_sites.itertuples()):
    try:
        landsat_stack = get_annual_median_mosaic(row.geometry,
                                                 buffer_dist=10_000,
                                                 src_crs=3413,
                                                 target_crs=4326)
        if landsat_stack is None:
            continue
        animate_rgb(landsat_stack)
    except:
        continue
