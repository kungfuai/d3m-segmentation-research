## Bootstrapped from https://github.com/cfld/locusts

import os
import random

import backoff
import rasterio
import ee
ee.Initialize()
from polygon_geohasher.polygon_geohasher import geohash_to_polygon, polygon_to_geohashes
from shapely import geometry
import urllib
from urllib.request import urlretrieve
import numpy as np
from scipy.spatial import ConvexHull

sentinel_channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60']

def geohash2cell(geohash):
    polygon = geohash_to_polygon(geohash)
    cell = ee.Geometry(geometry.mapping(polygon))
    return cell

def maskS2clouds(image):
    qa = image.select('QA60')

    cloudBitMask = 1 << 10 
    cirrusBitMask = 1 << 11 

    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask)

@backoff.on_exception(backoff.constant, urllib.error.HTTPError, max_tries=4, interval=2)
def safe_urlretrieve(url, outpath):
    _ = urlretrieve(url, outpath)

def get_one_sentinel(date_start, date_end, geohash, outpath, transform):
    
    cell = geohash2cell(geohash)
    collection  = (
        ee.ImageCollection('COPERNICUS/S2')
            .select(sentinel_channels)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2clouds)  # Apply cloud mask
    )
    image = collection.sort('system:index', opt_ascending=False).median()

    try:
        url = image.clip(cell).getDownloadURL(
            params={
                "name": geohash, 
                "crs": "EPSG:4326", 
                "crs_transform": transform
            }
        )
        _ = safe_urlretrieve(url, outpath)
    except:
        pass

def geotiff_to_geohashes(geotiff, max_pts = 260000000):
    img = rasterio.open(geotiff)
    transform = list(img.transform)[:6]
    nz = np.nonzero(img.read(1))

    coords = np.empty((len(nz[0]), 2))
    for k, (i,j) in enumerate(zip(nz[0], nz[1])):
        coords[k] = np.array(img.transform*(i,j))

    random_pts = coords[np.random.choice(coords.shape[0], max_pts, replace=False), :]
    hull = ConvexHull(random_pts)
    polygon = geometry.Polygon(random_pts[hull.vertices, :])
    # hull_vertices = np.load('hull-vertices.npy')
    # polygon = geometry.Polygon(hull_vertices)
    geohashes = polygon_to_geohashes(polygon, precision=5, inner=True)
    return geohashes, transform

def main(
    geotiff,
    out_dir='data/sentinel_2_download',
    date_start="2016-01-01",
    date_end="2016-12-31",
):
    os.makedirs(out_dir, exist_ok=True)

    geohashes, transform = geotiff_to_geohashes(geotiff)

    for geohash in geohashes:
        outpath = os.path.join(out_dir, f'{geohash}.zip')
        get_one_sentinel(date_start, date_end, geohash, outpath, transform)

if __name__ == '__main__':
    main('data/land-cover-10m.tiff')

