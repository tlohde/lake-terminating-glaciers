# use `pro` env
import dask.delayed
from dask.distributed import LocalCluster
import dask
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pystac_client
import planetary_computer
import rioxarray as rio
from shapely import wkt
import stackstac
import xarray as xr
import xarray_sentinel as xrs
from geocube.api.core import make_geocube
import os
import sarsen


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

class Sentinel1():
    def __init__(self, row, export=True):
        self.geometry = row.geometry  # must be in epsg:4326
        self.id = row.id
        self.region = row.SUBREGION1
        self.export_angle = export
        self.angle_export_path = f'../results/lakeIce/id{self.id}_s1_incident_angles.zarr'
        self.sample_export_path = f'../results/lakeIce/id{self.id}_sample.parquet'


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
        self.get_median()
        
        if os.path.exists(self.angle_export_path):
            print('already got incident angle ds...reading from .zarr')
            self.angle_ds = xr.open_zarr(self.angle_export_path)['incident_angle']
        else:
            print('need to get orbit info for calculating incident angles')
            self.get_dem()
            self.get_unique_orbits()
            self.download_s1_safes()
            self.get_incident_angles()
        
        self.angle_sample()


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

        self.s1_ds = self.s1_ds.rename('s1')
        
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
        
        self.dB = self.dB.rename('dB')
    
    def get_median(self):
        self.median = (self.dB
                       .median(dim=['y', 'x'], skipna=True)
                       .rename('dB')
                       )

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
                self.flipped = True
                self.dem = self.dem.isel(y=slice(None, None, -1))
            else:
                self.flipped = False
            self.dem.attrs['long_name'] = 'elevation'
            self.dem.attrs['units'] = 'm'
            self.dem = self.dem.rename('dem').squeeze(drop=True)

            self.dem_ecef = sarsen.scene.convert_to_dem_ecef(self.dem,
                                                             source_crs=self.dem.rio.crs)

    def get_unique_orbits(self):
        self.unique_relative_orbits = np.unique(self.s1_ds['sat:relative_orbit'])
        folders = []
        for orb in self.unique_relative_orbits:

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

        self.orb_dirs = dict(zip(self.unique_relative_orbits, folders))

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

        self.angles = []
        
        for orb, folder in self.orb_dirs.items():

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

            self.angles.append(angle)

        self.angle_ds = xr.concat(self.angles, dim='sat:relative_orbit')
        self.angle_ds['sat:relative_orbit'] = list(self.orb_dirs.keys())
        
        if self.flipped:
            # need to flip y axis back to align with dB and mask
            self.angle_ds = self.angle_ds.isel(y=slice(None, None, -1))
        
        self.angle_ds = self.angle_ds.drop_vars(
            ['band', 'platform', 'proj:shape', 'proj:epsg',
             'spatial_ref', 'gsd', 'proj:transform'],
            errors='ignore'
            )
    
        # apply mask
        self.angle_ds = xr.where(self.mask==self.id,
                                 self.angle_ds,
                                 np.nan)
        
        self.angle_ds = self.angle_ds.rio.write_crs(self.s1_ds.rio.crs)
        
        ## add in some additinoal data
        self.angle_ds['grd'] = xr.DataArray(
            list(self.orb_dirs.values()),
            coords={'sat:relative_orbit' :self.angle_ds['sat:relative_orbit']}
            )
        
        self.angle_ds = self.angle_ds.rename('incident_angle')
        
        
        if self.export_angle:
            attrs = {
                'geometry': wkt.dumps(self.geometry),
                'id': self.id,
                'region': self.region,
                'date_processed': pd.Timestamp.now().strftime('%y%m%d_%H%M')
            }

            self.angle_ds.attrs = attrs

            self.angle_ds.to_zarr(
                self.angle_export_path,
                mode='w'
                )

    def angle_sample(self, N=200):
        
        print(f'sampling at backscatter stack at {N} (x,y) points')
        print('pairing observations with the following relative\
              orbit numbers:\
              {self.angle_ds["sat:relative_orbit"].values.tolist()}')
    
        _x = xr.DataArray(np.random.choice(self.angle_ds.x, N), dims=('xy'))
        _y = xr.DataArray(np.random.choice(self.angle_ds.y, N), dims=('xy'))

        dbdf = [(self.dB
                .sel(x=_x, y=_y, band=b)
                .to_pandas()
                .stack()
                .reset_index()
                .rename(columns={0: f'dB_{b}'})
        ) for b in self.dB.band.values]

        dbdf = pd.merge(dbdf[0], dbdf[1],
                        left_on=['xy', 'time'],
                        right_on=['xy', 'time'])

        orbdf = (self.dB
                .sel(x=_x, y=_y, band=self.dB.band.values[0])['sat:relative_orbit']
                .to_pandas()
                .rename('sat:relative_orbit')
        )

        angledf = (self.angle_ds
                   .sel(x=_x, y=_y)
                   .to_pandas()
                   .stack()
                   .reset_index()
                   .rename(columns={0: 'angle'})
                   )
        
        df = dbdf.merge(
                orbdf,
                left_on='time',
                right_index=True
        )
        
        df = df.merge(
                angledf,
                left_on=['xy', 'sat:relative_orbit'],
                right_on=['xy', 'sat:relative_orbit']
        )
        
        df['month'] = df['time'].dt.month
        
        self.sampled = df
        self.sampled['id'] = self.id
        self.sampled['region'] = self.region
        
        if self.export_angle:
            attrs = {
                'geometry': wkt.dumps(self.geometry),
                'id': str(self.id),
                'region': self.region,
                'date_processed': pd.Timestamp.now().strftime('%y%m%d_%H%M'),
                'N': f'{N} samples over x,y domain'
            }
            
            self.sampled.attrs = attrs
            self.sampled.to_parquet(self.sample_export_path)
    
    def tidy_up(self):
        2+2
        # remove everything downloaded during mirror_folder
        