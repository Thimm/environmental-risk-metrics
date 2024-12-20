import os
from typing import Dict

import geopandas as gpd
import pandas as pd

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class GlobalWitness(BaseEnvironmentalMetric):
    """Class for analyzing Global Witness environmental defender data"""

    def __init__(self):
        sources = [ 
            "https://globalwitness.org/wp-content/uploads/2024/10/Global-Witness-Led-10-10-24.csv",
            "https://globalwitness.org/wp-content/uploads/2024/10/Global-Witness-Led-10-10-24.csv",
        ]
        description = "Global Witness data"
        super().__init__(sources=sources, description=description)
        # Load country boundaries
        self.countries = gpd.read_file(
            filename=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "world_countries_geojson.gpkg",
            )
        )
        self.countries["un_a3"] = self.countries["un_a3"].astype(int)
        
        # Load Global Witness data
        self.global_witness = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "global_witness_led_10-10-24.csv",
            )
        )

    def get_global_witness_data(self, polygon: dict, polygon_crs: str) -> Dict:
        """
        Get Global Witness data for the countries containing or intersecting the given geometry
        
        Args:
            polygon: GeoJSON polygon to analyze
            polygon_crs: CRS of the polygon
            
        Returns:
            Dictionary containing Global Witness statistics for the countries
        """
        # Convert geometry to GeoDataFrame
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        gdf = gpd.GeoDataFrame([polygon], crs=polygon_crs, columns=["geometry"])
        
        # Find which countries intersect with this geometry
        countries = gpd.sjoin(
            gdf,
            self.countries,
            how="inner",
            predicate="intersects"
        )
        
        if countries.empty:
            return {}
            
        country_un_a3s = countries["un_a3"].unique()
        country_names = countries["name"].unique().tolist()
        
        # Get data for these countries
        country_data = self.global_witness[
            self.global_witness["country_numeric"].isin(country_un_a3s)
        ]
        
        return {
            "total_incidents": len(country_data),
            "years": sorted(country_data["year"].unique().tolist()) if not country_data.empty else [],
            "countries": country_names
        }

    def get_data(self, polygon: dict, polygon_crs: str, **kwargs) -> Dict:
        """Get Global Witness data for a given geometry"""
        polygon = self._preprocess_geometry(geometry=polygon, source_crs=polygon_crs)
        return self.get_global_witness_data(polygon=polygon, polygon_crs=polygon_crs)
