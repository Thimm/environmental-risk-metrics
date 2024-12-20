import json
import os
from typing import Dict, List

import rasterstats
import rioxarray
import xarray as xr

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class SoilTypes(BaseEnvironmentalMetric):
    """Class for analyzing USDA soil type data"""

    def __init__(self):
        sources = [
            "https://s3.openlandmap.org/arco/grtgroup_usda.soiltax_c_250m_s_19500101_20171231_go_espg.4326_v0.2.tif",
        ]
        description = "USDA Soil Types from OpenLandMap"
        super().__init__(sources=sources, description=description)
        self.cog_url = "https://s3.openlandmap.org/arco/grtgroup_usda.soiltax_c_250m_s_19500101_20171231_go_espg.4326_v0.2.tif"
        soil_types_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resources", "soil_types.json"
        )
        with open(soil_types_path) as f:
            self.legend = json.load(f)

    def get_soil_type_stats(
        self, polygon: dict, polygon_crs: str, all_touched: bool = True
    ) -> Dict[str, str]:
        """
        Get soil type statistics for a given polygon

        Args:
            polygon: GeoJSON polygon to analyze
            all_touched: Include all pixels touched by geometry (default True)

        Returns:
            Dictionary containing the majority soil type and its description
        """
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        # Calculate zonal statistics to get majority soil type value
        soil_stats = rasterstats.zonal_stats(
            polygon, self.cog_url, stats=["majority"], all_touched=all_touched
        )[0]  # Take first result since we just want majority

        majority_value = str(int(soil_stats["majority"]))

        # Find matching soil type in legend
        soil_type = next(
            (item for item in self.legend if item["value"] == majority_value),
            {"label": "NODATA", "description": "NODATA"},
        )

        return {
            "Soil Type": soil_type["label"],
            "Description": soil_type["description"],
        }

    def get_data(
        self, polygon: dict, polygon_crs: str, all_touched: bool = True, **kwargs
    ) -> Dict[str, str]:
        """Get soil type statistics for a given geometry"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return self.get_soil_type_stats(polygon=polygon, polygon_crs=polygon_crs, all_touched=all_touched)
