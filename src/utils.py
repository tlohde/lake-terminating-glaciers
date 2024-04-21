import shapely
import pyproj


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
