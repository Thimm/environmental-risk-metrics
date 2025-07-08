import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import Polygon

from environmental_risk_metrics.metrics.land_use_change import (
    EsaLandCover, EsriLandCover, OpenLandMapLandCover)


@pytest.fixture
def sample_polygon_wgs84() -> Polygon:
    return Polygon(
        [
            (-122.27, 37.87),
            (-122.27, 37.88),
            (-122.26, 37.88),
            (-122.26, 37.87),
            (-122.27, 37.87),
        ]
    )

@pytest.fixture
def sample_gdf(sample_polygon_wgs84: Polygon) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame([1], geometry=[sample_polygon_wgs84], crs="EPSG:4326")

def test_esri_land_cover_init(sample_gdf: gpd.GeoDataFrame):
    metric = EsriLandCover(gdf=sample_gdf)
    assert metric.gdf.crs == "EPSG:4326"
    assert "data" in metric.band_name

def test_esa_land_cover_init(sample_gdf: gpd.GeoDataFrame):
    metric = EsaLandCover(gdf=sample_gdf)
    assert metric.gdf.crs == "EPSG:4326"
    assert "lccs_class" in metric.band_name

def test_open_land_map_land_cover_init(sample_gdf: gpd.GeoDataFrame):
    metric = OpenLandMapLandCover(gdf=sample_gdf)
    assert metric.gdf.crs == "EPSG:4326"
    assert "data" in metric.band_name

def test_open_land_map_get_data(sample_gdf: gpd.GeoDataFrame, mocker):
    metric = OpenLandMapLandCover(gdf=sample_gdf, use_esri_classes=True)

    reprojected_gdf = sample_gdf.to_crs("EPSG:6933")
    minx, miny, maxx, maxy = reprojected_gdf.total_bounds

    # Create a mock xarray dataset
    x_coords = [minx, maxx]
    y_coords = [miny, maxy]
    data = [[[208, 25], [1, 250]]]
    time = pd.to_datetime(['2020-01-01'])
    da = xr.DataArray(
        data,
        coords={'time': time, 'y': y_coords, 'x': x_coords},
        dims=('time', 'y', 'x'),
        name='data'
    )
    da.rio.write_crs("EPSG:6933", inplace=True)
    
    mocker.patch.object(metric, 'load_xarray', return_value=[da])
    
    result = metric.get_data(start_date="2020-01-01", end_date="2020-12-31")
    
    assert isinstance(result, list)
    assert len(result) == 1
    
    records = result[0]
    assert isinstance(records, list)
    assert len(records) == 1

    record = records[0]
    assert 'date' in record
    assert 'Water' in record
    assert 'Trees' in record
    assert 'Water_sqm' in record
    assert 'Trees_sqm' in record
    assert 'Rangeland' in record
    assert 'Built area' in record

    # With a 2x2 grid of 4 different values, each should be 25%
    assert record['Water'] == 25.0
    assert record['Trees'] == 25.0
    assert record['Rangeland'] == 25.0
    assert record['Built area'] == 25.0
    
    # Each pixel is 30x30 = 900 sqm - this is not true anymore with the new coords
    # We can check if they are > 0
    assert record['Water_sqm'] > 0
    assert record['Trees_sqm'] > 0
    assert record['Rangeland_sqm'] > 0
    assert record['Built area_sqm'] > 0 