import json
import os
from typing import Dict, List

import leafmap
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
        self, polygon: dict, polygon_crs: str = None, all_touched: bool = True
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

    def create_map(self, polygon: dict, polygon_crs: str, **kwargs) -> None:
        """Create a map for the soil organic carbon data"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        center = self.get_centroid(polygon, polygon_crs)
        m = leafmap.Map(
            center=(center[1], center[0]),
            zoom=12,
            draw_control=False,
            measure_control=False,
            fullscreen_control=False,
            attribution_control=False,
            search_control=False,
            layers_control=True,
            scale_control=False,
            toolbar_control=True,
        )
        colormap = [
            ((0, 1), "#ffffa0"),
            ((1, 2), "#f7fcb9"),
            ((2, 4), "#d9f0a3"),
            ((4, 6), "#addd8e"),
            ((6, 10), "#78c679"),
            ((10, 15), "#41ab5d"),
            ((15, 25), "#238443"),
            ((25, 40), "#005b29"),
            ((40, 60), "#004b29"),
            ((60, 120), "#012b13"),
            ((120, 256), "#00120b"),
        ]
        # divide the colormap by 2 to get the percentage
        for i in range(len(colormap)):
            colormap[i] = (
                (colormap[i][0][0] / 2, colormap[i][0][1] / 2),
                colormap[i][1],
            )

        for soc_cog in self.source:
            m.add_cog_layer(
                soc_cog["url"],
                colormap=json.dumps(colormap),
                name=soc_cog["depth"],
                attribution="OpenLandMap",
            )
        m.add_geojson(
            in_geojson=json.dumps(polygon.__geo_interface__),
            layer_name="Your Parcels",
            zoom_to_layer=True,
        )
        m.add_colorbar(
            colors=[x[1] for x in colormap],
            index=[x[0][0] for x in colormap],
            vmin=colormap[0][0][0],
            vmax=colormap[-1][0][0],
            caption="Soil Organic Carbon (%)",
        )
        return m


class SoilOrganicCarbonPotential(BaseEnvironmentalMetric):
    """Class for analyzing soil organic carbon potential using FAO GSOCseq data"""

    def __init__(self):
        super().__init__()
        self.units = "kg/ha"
        self.source = "http://54.229.242.119/GSOCseqv1.1/GSOCseq_T0_Map030.tif"

    def get_carbon_potential(self, polygon: dict, polygon_crs: str, all_touched: bool = True) -> Dict[str, float]:
        """Get soil organic carbon potential for a given geometry"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        stats = rasterstats.zonal_stats(polygon, self.source, stats=["mean"], all_touched=all_touched)
        return stats[0]["mean"]
    
    def get_data(self, polygon: dict, polygon_crs: str, all_touched: bool = True, **kwargs) -> Dict[str, float]:
        """Get soil organic carbon potential for a given geometry"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return self.get_carbon_potential(polygon=polygon, polygon_crs=polygon_crs, all_touched=all_touched)

