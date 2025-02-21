# use `pro` env
from dask.distributed import LocalCluster
import dask
import dask.dataframe as dd
import geopandas as gpd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pystac
import pystac_client
import planetary_computer
import rioxarray as rio
import seaborn as sns
from shapely.geometry import LineString, Point, box, Polygon
import stackstac
from tqdm import tqdm
import xarray as xr
import xarray_sentinel as xrs
import utils
from geocube.api.core import make_geocube

import os
import sarsen
import adlfs
import requests


def mirror_folder(fs, bucket, folder):
    # from https://github.com/bopen/sarsen/blob/main/notebooks/gamma_wrt_incidence_angle-S1-GRD-IW-RTC-South-of-Redmond.ipynb
    for path, folders, files in fs.walk(f"{bucket}/{folder}"):
        os.makedirs(path[len(bucket) + 1 :], exist_ok=True)
        for f in files:
            file_path = os.path.join(path, f)
            lfile_path = file_path[len(bucket) + 1 :]
            if not os.path.isfile(lfile_path):
                fs.download(file_path, lfile_path + "~")
                os.rename(lfile_path + "~", lfile_path)

class Sentinel11():
    def __init__(self, row):
        self.geometry = row.geometry  # must be in epsg:4326
        self.id = row.id


        self.gdf = gpd.GeoDataFrame(geometry=[self.geometry],
                                    data={'id': [self.id]},
                                    crs=4326)


        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
            )

        self.get_lazy_staccstack()
        self.make_mask()
        self.mask_to_dB()
        self.get_dem()
        self.get_unique_orbits()
        self.download_s1_safes()
        self.get_incident_angles()

    def get_dem(self, res=30):

        search = self.catalog.search(collections=[f'cop-dem-glo-{res}'],
                                     intersects=self.geometry.envelope)

        items = search.item_collection()
        if len(items) > 0:
            dem = (stackstac.stack(
                items,
                epsg=self.epsg)
            )

            self.dem = (dem
                        .mean(dim='time', skipna=True)
                        .squeeze()
                        .rio.write_crs(self.epsg)
                        .rio.reproject_match(self.s1_ds, nodata=np.nan)
                        )

            ## prep for sarsen
            if self.dem.y.diff('y').values[0] < 0:
                self.dem = self.dem.isel(y=slice(None, None, -1))
            self.dem.attrs['long_name'] = 'elevation'
            self.dem.attrs['units'] = 'm'
            self.dem = self.dem.rename('dem').squeeze(drop=True)

            self.dem_ecef = sarsen.scene.convert_to_dem_ecef(self.dem,
                                                             source_crs=self.dem.rio.crs)

    def get_lazy_staccstack(self):

        search = self.catalog.search(collections=['sentinel-1-rtc'],
                            intersects=self.geometry)

        self.s1items = search.item_collection()

        # get most common projection
        vals, cnts = np.unique([item.properties['proj:epsg'] for item in self.s1items] ,return_counts=True)
        epsg = int(vals[np.argmax(cnts)])

        # stack & clip
        self.s1_ds = stackstac.stack(
            self.s1items,
            epsg=epsg,
            bounds_latlon=self.geometry.bounds
            )

        self.epsg = int(
            self.s1_ds.attrs['crs'].split(':')[-1]
        )

    def make_mask(self):

        self.mask = (make_geocube(self.gdf,
                                  fill=np.nan,
                                  like=self.s1_ds)['id']
                     .rename('mask'))

    def mask_to_dB(self):
        self.dB = xr.where(self.mask==self.id,
                           10 * np.log10(self.s1_ds),
                           np.nan)

    def get_unique_orbits(self):
        unique_relative_orbits = np.unique(self.s1_ds['sat:relative_orbit'])
        folders = []
        for orb in unique_relative_orbits:

            ids = self.s1_ds[self.s1_ds['sat:relative_orbit']==orb].id.values

            for id in ids:

                try:
                    rtc_item = self.catalog.get_collection('sentinel-1-rtc').get_item(id)
                    grd_item = pystac.read_file(rtc_item.get_single_link("derived_from").target)

                except Exception as e:
                    print(f'an error of type {type(e)}: {e}')

                else:
                    grd_band = [grd_item.assets.get(band, None)
                                for band in ['hh', 'hv', 'vh', 'vv']
                                if band in grd_item.assets.keys()][0]

                    folders.append(grd_band.href[53:-23])
                    break

        self.orb_dirs = dict(zip(unique_relative_orbits, folders))

    def download_s1_safes(self):

        account = 'sentinel1euwest'
        bucket = "s1-grd"

        grd_fs = planetary_computer.get_adlfs_filesystem(account_name=account,
                                                         container_name=bucket)

        for orb, folder in self.orb_dirs.items():
            mirror_folder(grd_fs, bucket, folder)

    def get_incident_angles(self):

        def slant_to_ground(a, b):
            xrs.slant_range_time_to_ground_range(a,
                                                 b,
                                                 coordinate_conversion=coord_conversion)

        for orb, folder in self.orb_dirs.items():

            angles = []

            orbit_ecef, _ = sarsen.sentinel1.open_dataset_autodetect(folder, group='IW/HH/orbit')

            coord_conv = [
                grp for grp in sarsen.sentinel1.open_dataset_autodetect(folder)[0].attrs['subgroups']
                if 'coordinate_conversion' in grp
                ][0]

            coord_conversion, _ = sarsen.sentinel1.open_dataset_autodetect(folder,
                                                                           group=coord_conv)

            acquisition = sarsen.apps.simulate_acquisition(self.dem_ecef,
                                                           orbit_ecef.position,
                                                           slant_range_time_to_ground_range=slant_to_ground,
                                                           correct_radiometry=True)

            oriented_area = sarsen.scene.compute_dem_oriented_area(self.dem_ecef)

            dem_normal = -oriented_area / np.sqrt(xr.dot(oriented_area,
                                                         oriented_area,
                                                         dims="axis"))

            orbit_interpolator = sarsen.orbit.OrbitPolyfitIterpolator.from_position(orbit_ecef.position)

            position_ecef = orbit_interpolator.position()

            velocity_ecef = orbit_interpolator.velocity()

            acquisition = sarsen.geocoding.backward_geocode(self.dem_ecef,
                                                            orbit_ecef.position,
                                                            velocity_ecef)

            slant_range = np.sqrt((acquisition.dem_distance**2).sum(dim="axis"))

            dem_direction = acquisition.dem_distance / slant_range

            angle = np.arccos(xr.dot(dem_normal,
                                     dem_direction,
                                     dims="axis"))

            angles.append(angle)

        self.angle_ds = xr.concat(angles, dim='orbit')
        self.angle_ds['orbit'] = list(self.orb_dirs.keys())
