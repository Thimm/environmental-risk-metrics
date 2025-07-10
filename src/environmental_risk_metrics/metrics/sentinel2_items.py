import logging
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import odc.stac
import planetary_computer
import pystac
import xarray as xr
from tqdm.auto import tqdm

from environmental_risk_metrics.base import BaseEnvironmentalMetric
from environmental_risk_metrics.utils.planetary_computer import \
    get_planetary_computer_items

logger = logging.getLogger(__name__)


class Sentinel2Items(BaseEnvironmentalMetric):
    """
    A class for accessing Sentinel-2 items from Planetary Computer.
    This class is focused on retrieving items and loading them as xarray Datasets.
    """

    BAND_MAPPINGS = {
        "sentinel-2-l2a": {
            "red": "B04",
            "green": "B03",
            "blue": "B02",
            "nir": "B08",
            "cloud_mask": "SCL",
            "cloud_clear_values": [4, 5],  # SCL values for clear pixels
        },
    }

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        start_date: str,
        end_date: str,
        resolution: int = 10,
        entire_image_cloud_cover_threshold: int = 100,
        cropped_image_cloud_cover_threshold: int = 80,
        max_workers: int = 10,
    ):
        sources = ["https://planetarycomputer.microsoft.com/api/stac/v1"]
        description = "Sentinel-2 L2A items from Planetary Computer."

        super().__init__(sources=sources, description=description)

        self.collections = ["sentinel-2-l2a"]
        self.resolution = resolution
        self.entire_image_cloud_cover_threshold = entire_image_cloud_cover_threshold
        self.cropped_image_cloud_cover_threshold = cropped_image_cloud_cover_threshold
        self.max_workers = max_workers
        self.start_date = start_date
        self.end_date = end_date
        self.gdf = gdf.to_crs(epsg=4326)
        self.items = None
        self.xarray_data = None
        logger.debug("Initializing Sentinel2Items client")

    def get_items(
        self,
        entire_image_cloud_cover_threshold: int = None,
    ) -> dict[str, list[pystac.Item]]:
        """
        Search for items from Sentinel-2 within a given date range and polygon.
        Args:
            entire_image_cloud_cover_threshold: Maximum cloud cover percentage for search
        Returns:
            Dictionary with collection name as key and list of pystac.Item objects as values
        """
        if self.items is not None:
            return self.items

        if entire_image_cloud_cover_threshold is None:
            entire_image_cloud_cover_threshold = self.entire_image_cloud_cover_threshold

        gdf = self.gdf.to_crs(epsg=4326)
        items_by_collection = {"sentinel-2-l2a": []}
        collection = "sentinel-2-l2a"

        def fetch_items_for_collection(collection_polygon_pair):
            collection, polygon = collection_polygon_pair
            return collection, get_planetary_computer_items(
                collections=[collection],
                start_date=self.start_date,
                end_date=self.end_date,
                polygon=polygon,
                entire_image_cloud_cover_threshold=entire_image_cloud_cover_threshold,
            )

        collection_polygon_pairs = [(collection, geom) for geom in gdf.geometry]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(fetch_items_for_collection, collection_polygon_pairs),
                    total=len(collection_polygon_pairs),
                    desc=f"Fetching {collection} items",
                )
            )

        for i, (_, items_list) in enumerate(results):
            if i >= len(items_by_collection[collection]):
                items_by_collection[collection].extend(
                    [[] for _ in range(i + 1 - len(items_by_collection[collection]))]
                )
            items_by_collection[collection][i] = items_list

        self.items = items_by_collection
        return self.items

    def _get_bands_for_collection(
        self, collection: str, include_rgb: bool = False
    ) -> list[str]:
        """Get the required bands for a collection"""
        mapping = self.BAND_MAPPINGS[collection]
        bands = [mapping["nir"], mapping["red"]]
        if include_rgb:
            bands.extend([mapping["green"], mapping["blue"]])
        if mapping["cloud_mask"]:
            bands.append(mapping["cloud_mask"])
        return bands

    def load_xarray(
        self,
        bands: dict[str, list[str]] = None,
        show_progress: bool = True,
        filter_cloud_cover: bool = True,
        include_rgb: bool = False,
    ) -> dict[str, list[xr.Dataset]]:
        """Load data for all collections into xarray Datasets.
        Args:
            bands: Dictionary with collection names as keys and band lists as values
            show_progress: Whether to show a progress bar
            filter_cloud_cover: Whether to filter the data based on cloud cover
            include_rgb: Whether to include RGB bands for visualization
        Returns:
            Dictionary with collection names as keys and lists of xarray Datasets as values
        """
        if self.xarray_data is not None:
            return self.xarray_data

        logger.debug(
            f"Loading data for collections {self.collections} at {self.resolution}m resolution"
        )

        items_dict = self.get_items()

        if not any(items_dict.values()):
            logger.error(
                "No items found for any collection in the given date range and polygon"
            )
            raise ValueError(
                "No items found for any collection in the given date range and polygon"
            )

        self.xarray_data = {}

        for collection in self.collections:
            self.xarray_data[collection] = []
            items_list = items_dict.get(collection, [])

            if not items_list:
                logger.warning(f"No items found for collection {collection}")
                continue

            if bands and collection in bands:
                collection_bands = bands[collection]
            else:
                collection_bands = self._get_bands_for_collection(
                    collection, include_rgb
                )

            mapping = self.BAND_MAPPINGS[collection]

            for geometry_idx, (items, geometry) in enumerate(
                zip(items_list, self.gdf.geometry)
            ):
                if not items:
                    logger.warning(
                        f"No items for geometry {geometry_idx} in collection {collection}"
                    )
                    self.xarray_data[collection].append(None)
                    continue

                logger.debug(f"Signing {len(items)} items for collection {collection}")
                signed_items = [planetary_computer.sign(i) for i in items]

                thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

                logger.debug(f"Loading data into xarray Dataset for {collection}")
                progress = tqdm if show_progress else None

                try:
                    ds = odc.stac.load(
                        items=signed_items,
                        bands=collection_bands,
                        resolution=self.resolution,
                        pool=thread_pool,
                        geopolygon=geometry,
                        progress=progress,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load data for {collection}, geometry {geometry_idx}: {e}"
                    )
                    self.xarray_data[collection].append(None)
                    continue

                if (
                    self.cropped_image_cloud_cover_threshold
                    and filter_cloud_cover
                    and mapping["cloud_mask"]
                ):
                    logger.debug(
                        f"Filtering data based on cloud cover using {mapping['cloud_mask']} band"
                    )

                    if collection == "sentinel-2-l2a":
                        cloud_clear_mask = (ds[mapping["cloud_mask"]] == 4) | (
                            ds[mapping["cloud_mask"]] == 5
                        )
                    else:
                        cloud_clear_mask = ds[mapping["cloud_mask"]] == 0

                    cloud_cover_pct = (1 - cloud_clear_mask.mean(dim=["x", "y"])) * 100
                    logger.debug(
                        f"Cloud cover percentage: {cloud_cover_pct}. Filtering based on {self.cropped_image_cloud_cover_threshold}% threshold"
                    )

                    logger.debug(f"Dataset time steps before filtering: {len(ds.time)}")
                    ds = ds.sel(
                        time=cloud_cover_pct <= self.cropped_image_cloud_cover_threshold
                    )
                    logger.debug(f"Filtered dataset to {len(ds.time)} time steps")

                    if filter_cloud_cover:
                        ds = ds.where(cloud_clear_mask, drop=False)

                logger.debug(f"Successfully loaded {collection} data")
                self.xarray_data[collection].append(ds)

        return self.xarray_data 