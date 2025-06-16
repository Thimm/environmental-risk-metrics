from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from environmental_risk_metrics.metrics.ndvi import HarmonizedNDVI, Sentinel2


@pytest.fixture
def sample_geometry():
    """Create a sample geometry for testing"""
    bbox = [-117.8, 33.65, -117.65, 33.8]  # Orange County, California area
    geometry = box(*bbox)
    return gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")


@pytest.fixture
def ndvi_client(sample_geometry):
    """Create a HarmonizedNDVI client for testing"""
    return HarmonizedNDVI(
        start_date="2023-01-01",
        end_date="2023-12-31",
        gdf=sample_geometry,
        collections=["sentinel-2-l2a", "hls2-s30", "hls2-l30"],
        resolution=30,
        entire_image_cloud_cover_threshold=10,
        cropped_image_cloud_cover_threshold=50,
        max_workers=4,
    )


def test_init(ndvi_client):
    """Test initialization of HarmonizedNDVI client"""
    assert ndvi_client.start_date == "2023-01-01"
    assert ndvi_client.end_date == "2023-12-31"
    assert len(ndvi_client.collections) == 3
    assert "sentinel-2-l2a" in ndvi_client.collections
    assert "hls2-s30" in ndvi_client.collections
    assert "hls2-l30" in ndvi_client.collections


@patch('environmental_risk_metrics.metrics.ndvi.HarmonizedNDVI.calculate_mean_ndvi')
def test_multiple_collections(mock_calculate_mean_ndvi, ndvi_client):
    """Test working with multiple satellite collections"""
    # Mock the mean NDVI data
    mock_data = {
        "sentinel-2-l2a": [pd.DataFrame({
            'ndvi': [0.1, 0.2, 0.3],
            'timestamp': pd.date_range('2023-01-01', periods=3)
        })],
        "hls2-s30": [pd.DataFrame({
            'ndvi': [0.2, 0.3, 0.4],
            'timestamp': pd.date_range('2023-01-01', periods=3)
        })],
        "hls2-l30": [pd.DataFrame({
            'ndvi': [0.3, 0.4, 0.5],
            'timestamp': pd.date_range('2023-01-01', periods=3)
        })]
    }
    mock_calculate_mean_ndvi.return_value = mock_data

    mean_ndvi_data = ndvi_client.calculate_mean_ndvi()

    assert isinstance(mean_ndvi_data, dict)
    assert len(mean_ndvi_data) == 3
    for collection in ["sentinel-2-l2a", "hls2-s30", "hls2-l30"]:
        assert collection in mean_ndvi_data
        assert len(mean_ndvi_data[collection]) == 1
        assert isinstance(mean_ndvi_data[collection][0], pd.DataFrame)


@patch('environmental_risk_metrics.metrics.ndvi.HarmonizedNDVI.rgb_ndvi_images')
def test_rgb_ndvi_images(mock_rgb_ndvi_images, ndvi_client):
    """Test generation of RGB+NDVI side-by-side images"""
    # Mock the RGB+NDVI images
    mock_images = {
        "sentinel-2-l2a": [{
            "2023-06-01": b"mock_image_data",
            "2023-06-15": b"mock_image_data"
        }]
    }
    mock_rgb_ndvi_images.return_value = mock_images

    rgb_ndvi_images = ndvi_client.rgb_ndvi_images()

    assert isinstance(rgb_ndvi_images, dict)
    assert "sentinel-2-l2a" in rgb_ndvi_images
    assert len(rgb_ndvi_images["sentinel-2-l2a"]) == 1
    assert len(rgb_ndvi_images["sentinel-2-l2a"][0]) == 2


@patch('environmental_risk_metrics.metrics.ndvi.HarmonizedNDVI.generate_ndvi_gif')
def test_gif_generation(mock_generate_gif, ndvi_client):
    """Test generation of animated GIF"""
    mock_gif_data = b"mock_gif_data"
    mock_generate_gif.return_value = mock_gif_data

    gif_bytes = ndvi_client.generate_ndvi_gif(
        collection="hls2-s30",
        geometry_index=0,
        duration=0.8,
        vmin=-0.2,
        vmax=0.8
    )

    assert isinstance(gif_bytes, bytes)
    mock_generate_gif.assert_called_once_with(
        collection="hls2-s30",
        geometry_index=0,
        duration=0.8,
        vmin=-0.2,
        vmax=0.8
    )


def test_backward_compatibility(sample_geometry):
    """Test backward compatibility with Sentinel2 class"""
    sentinel2_client = Sentinel2(
        start_date="2023-07-01",
        end_date="2023-07-31",
        gdf=sample_geometry,
        resolution=10,
    )

    assert isinstance(sentinel2_client, Sentinel2)
    assert sentinel2_client.start_date == "2023-07-01"
    assert sentinel2_client.end_date == "2023-07-31"
    assert sentinel2_client.resolution == 10


@patch('environmental_risk_metrics.metrics.ndvi.HarmonizedNDVI.ndvi_thumbnails')
def test_hls2_specific(mock_thumbnails, ndvi_client):
    """Test HLS2-specific functionality"""
    # Mock the thumbnails
    mock_thumb_data = {
        "hls2-s30": [{"2023-01-01": b"mock_thumb"}],
        "hls2-l30": [{"2023-01-01": b"mock_thumb"}]
    }
    mock_thumbnails.return_value = mock_thumb_data

    thumbnails = ndvi_client.ndvi_thumbnails(image_format="png")

    assert isinstance(thumbnails, dict)
    assert "hls2-s30" in thumbnails
    assert "hls2-l30" in thumbnails
    assert len(thumbnails["hls2-s30"]) == 1
    assert len(thumbnails["hls2-l30"]) == 1 