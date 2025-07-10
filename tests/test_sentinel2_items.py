import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from environmental_risk_metrics.metrics.sentinel2_items import Sentinel2Items


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    polygon = Polygon(
        [
            (-74.0060, 40.7128),
            (-73.9354, 40.7128),
            (-73.9354, 40.7831),
            (-74.0060, 40.7831),
            (-74.0060, 40.7128),
        ]
    )
    return gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")


def test_sentinel2_items_initialization(sample_gdf):
    """Test the initialization of the Sentinel2Items class."""
    s2_items = Sentinel2Items(
        gdf=sample_gdf, start_date="2023-01-01", end_date="2023-01-31"
    )
    assert s2_items is not None
    assert s2_items.collections == ["sentinel-2-l2a"]
    assert s2_items.gdf.crs == "EPSG:4326"


def test_get_items_with_real_data(sample_gdf):
    """Test get_items with a real call to Planetary Computer."""
    s2_items = Sentinel2Items(
        gdf=sample_gdf,
        start_date="2023-01-01",
        end_date="2023-01-31",
        entire_image_cloud_cover_threshold=90,
    )
    items_dict = s2_items.get_items()

    assert "sentinel-2-l2a" in items_dict
    assert len(items_dict["sentinel-2-l2a"]) > 0
    assert len(items_dict["sentinel-2-l2a"][0]) > 0
    assert "pystac.item.Item" in str(type(items_dict["sentinel-2-l2a"][0][0]))


def test_load_xarray_with_real_data(sample_gdf):
    """Test load_xarray with a real call to Planetary Computer."""
    s2_items = Sentinel2Items(
        gdf=sample_gdf,
        start_date="2023-01-01",
        end_date="2023-01-31",
        entire_image_cloud_cover_threshold=90,
        cropped_image_cloud_cover_threshold=100,
    )
    xarray_data = s2_items.load_xarray(include_rgb=True, filter_cloud_cover=False)

    assert "sentinel-2-l2a" in xarray_data
    assert len(xarray_data["sentinel-2-l2a"]) > 0
    
    dataset = xarray_data["sentinel-2-l2a"][0]
    assert dataset is not None
    assert "xarray.core.dataset.Dataset" in str(type(dataset))
    assert "B02" in dataset.data_vars
    assert "B03" in dataset.data_vars
    assert "B04" in dataset.data_vars
    assert "B08" in dataset.data_vars
    assert len(dataset.time) > 0 