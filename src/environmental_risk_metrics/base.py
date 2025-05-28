from abc import abstractmethod
from typing import Dict

import geopandas as gpd

from environmental_risk_metrics.legends import (
    convert_legend_to_value_color_dict,
    convert_legend_to_value_label_dict,
)
from environmental_risk_metrics.utils import (
    ensure_geometry_crs,
    get_centroid_of_geometry,
)


class BaseEnvironmentalMetric:
    """Base class for environmental metrics"""

    # Default CRS for all metrics unless overridden
    DEFAULT_CRS = "EPSG:4326"

    def __init__(
        self,
        sources: list[str],
        description: str,
        target_crs: str = None,
        legend: dict = None,
        **kwargs,
    ):
        """
        Initialize the metric with optional CRS override and legend

        Args:
            target_crs: Override default CRS for this instance
            legend: Legend for the metric
            sources: List of data source URLs or descriptions
            description: Description of what this metric measures
            **kwargs: Additional initialization parameters
        """
        self.target_crs = target_crs or self.DEFAULT_CRS
        self.legend = legend or {}
        if sources is None:
            raise ValueError("sources must be provided")
        self.sources = sources
        if description is None:
            raise ValueError("description must be provided")
        self.description = description
        super().__init__(**kwargs)

    def _preprocess_geometry(
        self, geometry: dict, source_crs: str, target_crs: str = None
    ) -> dict:
        """
        Preprocess geometry to ensure consistent format and CRS

        Args:
            geometry: Input geometry
            target_crs: Optional CRS override for this specific operation
            source_crs: Optional source CRS override for this specific operation
        """
        return ensure_geometry_crs(
            geometry=geometry,
            source_crs=source_crs,
            target_crs=target_crs or self.target_crs,
        )

    @abstractmethod
    def get_data(self, geometry: dict, geometry_crs: str, **kwargs) -> Dict:
        """Get data for a given geometry"""
        geometry = self._preprocess_geometry(geometry, source_crs=geometry_crs)
        pass

    def get_centroid(self, geometry: dict, source_crs: str, **kwargs) -> tuple:
        """Get the centroid of a geometry"""
        return get_centroid_of_geometry(geometry, source_crs)

    def create_map(self, polygons: dict, polygons_crs: str, **kwargs) -> None:
        """
        Create a visualization map for the metric data. Optional method that can be
        implemented by child classes.

        Args:
            polygons: Input polygons to visualize
            polygons_crs: CRS of the input polygons
            **kwargs: Additional visualization parameters

        Raises:
            NotImplementedError: If the child class doesn't implement this method
        """
        raise NotImplementedError(
            f"create_map() is not implemented for {self.__class__.__name__}"
        )

    def get_legend(self, **kwargs) -> Dict:
        """Get the legend for the metric"""
        if not self.legend:
            raise ValueError(f"Legend is not set for {self.__class__.__name__}")

    def get_legend_labels_dict(self, **kwargs) -> Dict:
        """Get the legend labels for the metric"""
        return convert_legend_to_value_label_dict(self.legend)

    def get_legend_colors(self, **kwargs) -> Dict:
        """Get the legend colors for the metric"""
        return convert_legend_to_value_color_dict(self.legend)

    def get_data_for_polygons(
        self,
        polygons: list,
        polygons_crs: str,
        concurrent=None,
        max_workers: int = 10,
        **kwargs,
    ) -> list:
        """Get data for multiple geometries

        Args:
            geometries: List of geometry dictionaries
            geometry_crs: CRS of the input geometries
            concurrent: If None, runs sequentially. If True, uses all available cores. If int, uses that many cores.
            **kwargs: Additional parameters to pass to get_data

        Returns:
            List of dictionaries containing data for each geometry
        """
        if not polygons:
            raise ValueError("No polygons provided")
        if not polygons_crs:
            raise ValueError("No polygons CRS provided")
        if isinstance(polygons, gpd.GeoDataFrame):
            polygons = [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": x.geometry.__geo_interface__,
                }
                for x in polygons.geometry
            ]
        if not concurrent:
            return [
                self.get_data(polygon=polygon, polygon_crs=polygons_crs, **kwargs)
                for polygon in polygons
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.get_data,
                        polygon=polygon,
                        polygon_crs=polygons_crs,
                        **kwargs,
                    )
                    for polygon in polygons
                ]
                return [future.result() for future in futures]
