import json
import os
from typing import Dict, List

import pandas as pd
import rasterstats

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class SoilOrganicCarbon(BaseEnvironmentalMetric):
    """Class for analyzing soil organic carbon data"""

    def __init__(self):
        super().__init__()
        self.source = [
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b0cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "0cm",
            },
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b10cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "10cm",
            },
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b30cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "30cm",
            },
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b60cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "60cm",
            },
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b100cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "100cm",
            },
            {
                "url": "https://s3.openlandmap.org/arco/organic.carbon_usda.6a1c_m_250m_b200cm_19500101_20171231_go_epsg.4326_v0.2.tif",
                "depth": "200cm",
            },
        ]

    def get_carbon_stats(
        self, polygon: dict, all_touched: bool = True, polygon_crs: str = None
    ) -> Dict[str, float]:
        """
        Get soil organic carbon statistics for a given geometry at different depths

        Args:
            polygon: GeoJSON polygon to analyze
            all_touched: Include all pixels touched by geometry (default True)
            polygon_crs: CRS of the polygon (default None)

        Returns:
            Dictionary containing soil organic carbon percentages at different depths
        """
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        results = {}
        for soc_cog in self.source:
            stats = rasterstats.zonal_stats(
                polygon, soc_cog["url"], stats=["mean"], all_touched=all_touched
            )[0]

            results[soc_cog["depth"]] = round(stats["mean"] / 2, 2)

        return results

    def get_data(
        self, polygon: dict, polygon_crs: str = None, all_touched: bool = True, **kwargs
    ) -> Dict[str, float]:
        """Get soil organic carbon statistics for a given geometry"""
        polygon = self._preprocess_geometry(geometry=polygon, source_crs=polygon_crs)
        return self.get_carbon_stats(polygon=polygon, all_touched=all_touched)
