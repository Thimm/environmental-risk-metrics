from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from environmental_risk_metrics.sentinel2 import Sentinel2, interpolate_ndvi


@pytest.fixture
def sentinel2():
    return Sentinel2()

@pytest.fixture
def sample_polygon():
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
        }
    }

@pytest.fixture
def sample_ndvi_data():
    # Create sample xarray DataArray with NDVI values
    times = pd.date_range('2023-01-01', periods=3)
    lats = np.linspace(0, 1, 5)
    lons = np.linspace(0, 1, 5)
    
    data = np.random.uniform(-0.2, 0.8, size=(3, 5, 5))
    return xr.DataArray(
        data,
        coords=[times, lats, lons],
        dims=['time', 'y', 'x']
    ).assign_coords(spatial_ref=32632)  # UTM zone 32N

def test_init(sentinel2):
    assert sentinel2.catalog is not None

@patch('pystac_client.Client.search')
def test_get_items(mock_search, sentinel2, sample_polygon):
    mock_search.return_value.item_collection.return_value = ['item1', 'item2']
    
    items = sentinel2.get_items(
        start_date='2023-01-01',
        end_date='2023-01-31',
        polygon=sample_polygon,
        entire_image_cloud_cover_threshold=20
    )
    
    assert len(items) == 2
    mock_search.assert_called_once()

@patch('odc.stac.load')
def test_load_xarray(mock_load, sentinel2, sample_polygon):
    mock_load.return_value = xr.Dataset()
    
    with patch.object(sentinel2, 'get_items') as mock_get_items:
        mock_get_items.return_value = ['item1', 'item2']
        
        ds = sentinel2.load_xarray(
            start_date='2023-01-01',
            end_date='2023-01-31',
            polygon=sample_polygon
        )
        
        assert isinstance(ds, xr.Dataset)
        mock_get_items.assert_called_once()
        mock_load.assert_called_once()

def test_interpolate_ndvi():
    # Create sample data with gaps
    dates = pd.date_range('2023-01-01', '2023-01-05')
    values = [0.1, np.nan, 0.3, np.nan, 0.5]
    df = pd.DataFrame({'mean_ndvi': values}, index=dates)
    
    # Test interpolation
    result = interpolate_ndvi(df, '2023-01-01', '2023-01-05')
    
    assert len(result) == 5
    assert not result['mean_ndvi'].isna().any()
    assert result.index[0].date() == pd.Timestamp('2023-01-01').date()
    assert result.index[-1].date() == pd.Timestamp('2023-01-05').date()

def test_calculate_mean_ndvi(sentinel2, sample_polygon, sample_ndvi_data):
    with patch('xarray.DataArray.rio.clip') as mock_clip:
        # Mock the clipped data
        mock_clip.return_value = sample_ndvi_data
        
        result = sentinel2.calculate_mean_ndvi(
            sample_ndvi_data,
            sample_polygon,
            polygon_crs='EPSG:4326'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'mean_ndvi' in result.columns
        assert len(result) == 3  # Number of time steps in sample data

def test_ndvi_thumbnails(sentinel2, sample_polygon, sample_ndvi_data):
    images = sentinel2.ndvi_thumbnails(
        ndvi=sample_ndvi_data,
        polygon=sample_polygon,
        polygon_crs='EPSG:4326'
    )
    
    assert isinstance(images, dict)
    assert len(images) == 3  # Number of time steps in sample data
    for timestamp, image_bytes in images.items():
        assert isinstance(timestamp, str)
        assert isinstance(image_bytes, bytes)

def test_load_ndvi_images(sentinel2, sample_polygon):
    with patch.object(sentinel2, 'load_xarray') as mock_load_xarray:
        # Create proper coordinates
        times = pd.date_range('2023-01-01', periods=2)
        y = np.linspace(0, 1, 3)
        x = np.linspace(0, 1, 3)
        
        # Create mock dataset with B08, B04, and SCL bands
        mock_ds = xr.Dataset(
            data_vars={
                'B08': (('time', 'y', 'x'), np.random.rand(2, 3, 3)),
                'B04': (('time', 'y', 'x'), np.random.rand(2, 3, 3)),
                'SCL': (('time', 'y', 'x'), np.random.randint(0, 11, size=(2, 3, 3)))
            },
            coords={
                'time': times,
                'y': y,
                'x': x
            }
        )
        mock_load_xarray.return_value = mock_ds
        
        ndvi = sentinel2.load_ndvi_images(
            start_date='2023-01-01',
            end_date='2023-01-31',
            polygon=sample_polygon
        )
        
        assert isinstance(ndvi, xr.DataArray)
        mock_load_xarray.assert_called_once()

def test_get_items_no_results(sentinel2, sample_polygon):
    with patch('pystac_client.Client.search') as mock_search:
        mock_search.return_value.item_collection.return_value = []
        
        items = sentinel2.get_items(
            start_date='2023-01-01',
            end_date='2023-01-31',
            polygon=sample_polygon
        )
        
        assert len(items) == 0

def test_load_xarray_no_items(sentinel2, sample_polygon):
    with patch.object(sentinel2, 'get_items') as mock_get_items:
        mock_get_items.return_value = []
        
        with pytest.raises(ValueError):
            sentinel2.load_xarray(
                start_date='2023-01-01',
                end_date='2023-01-31',
                polygon=sample_polygon
            )