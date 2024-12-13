import logging
from datetime import datetime
from unittest.mock import Mock, patch

import pystac
import pytest
import xarray as xr

from risks.data.fetch.sentinel2 import Sentinel2

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_polygon():
    logger.debug("Creating sample polygon fixture")
    return {
        "type": "Polygon",
        "coordinates": [[
            [13.0, 52.0],
            [13.1, 52.0],
            [13.1, 52.1],
            [13.0, 52.1],
            [13.0, 52.0]
        ]]
    }

@pytest.fixture
def mock_stac_item():
    logger.debug("Creating mock STAC item fixture")
    item = Mock(spec=pystac.Item)
    item.id = "test_item"
    return item

@pytest.fixture
def mock_dataset():
    logger.debug("Creating mock xarray dataset fixture")
    # Create a simple xarray dataset for testing
    return xr.Dataset(
        data_vars={
            "B02": (["time", "y", "x"], [[[1, 2], [3, 4]]]),
            "B03": (["time", "y", "x"], [[[5, 6], [7, 8]]]),
        },
        coords={
            "time": [datetime(2023, 1, 1)],
            "y": [0, 1],
            "x": [0, 1],
        }
    )

class TestSentinel2:
    @patch("odc.stac.load")
    @patch("planetary_computer.sign")
    def test_load_xarray_success(
        self, 
        mock_pc_sign, 
        mock_odc_load,
        sample_polygon,
        mock_stac_item,
        mock_dataset
    ):
        logger.info("Testing successful xarray loading")
        # Setup
        sentinel = Sentinel2()
        sentinel.get_items = Mock(return_value=[mock_stac_item])
        mock_pc_sign.return_value = mock_stac_item
        mock_odc_load.return_value = mock_dataset

        logger.debug("Executing load_xarray with test parameters")
        # Execute
        result = sentinel.load_xarray(
            start_date="2023-01-01",
            end_date="2023-01-02",
            polygon=sample_polygon,
            bands=["B02", "B03"],
            resolution=10
        )

        # Assert
        logger.debug("Verifying test results")
        assert isinstance(result, xr.Dataset)
        assert list(result.data_vars.keys()) == ["B02", "B03"]
        sentinel.get_items.assert_called_once_with(
            "2023-01-01", 
            "2023-01-02", 
            sample_polygon
        )
        mock_pc_sign.assert_called_once_with(mock_stac_item)
        mock_odc_load.assert_called_once()
        logger.info("Successfully completed xarray loading test")

    @patch("odc.stac.load")
    def test_load_xarray_no_items(
        self,
        mock_odc_load,
        sample_polygon
    ):
        logger.info("Testing xarray loading with no items")
        # Setup
        sentinel = Sentinel2()
        sentinel.get_items = Mock(return_value=[])

        # Execute & Assert
        logger.debug("Verifying exception is raised when no items found")
        with pytest.raises(Exception):
            sentinel.load_xarray(
                start_date="2023-01-01",
                end_date="2023-01-02",
                polygon=sample_polygon
            ) 
        logger.info("Successfully verified no items exception")

