from typing import Dict

import geopandas as gpd
import pandas as pd
from pygbif import occurrences
from shapely.geometry import Point
from shapely.geometry.polygon import orient

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class EndangeredSpecies(BaseEnvironmentalMetric):
    """Class for analyzing endangered species data from GBIF"""

    def __init__(self):
        super().__init__()
        self.iucn_categories = {
            "EX": "Extinct",
            "EW": "Extinct in the Wild", 
            "CR": "Critically Endangered",
            "EN": "Endangered",
            "VU": "Vulnerable",
            "NT": "Near Threatened",
            "LC": "Least Concern",
            "DD": "Data Deficient",
            "NE": "Not Evaluated"
        }

    def get_species_stats(self, polygon: dict, polygon_crs: str, buffer_meters: int) -> pd.DataFrame:
        """
        Get endangered species statistics for a given geometry
        
        Args:
            geometry: GeoJSON geometry to analyze
            
        Returns:
            DataFrame containing unique species counts by kingdom, class and IUCN category
        """
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=self.target_crs, columns=["geometry"])
        # Convert to projected CRS for buffering in meters
        gdf = gdf.to_crs("EPSG:3857")

        gdf["geometry"] = gdf.buffer(buffer_meters).to_crs(self.target_crs)
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: orient(geom, sign=1.0))

        
        geometry_wkt = gdf["geometry"].to_wkt()[0]

        # Query GBIF for species occurrences
        results = occurrences.search(
            limit=10000,
            geometry=geometry_wkt
        )

        # Convert to GeoDataFrame
        gdbf_df = pd.DataFrame(results["results"])
        gdbf_df["geometry"] = gdbf_df.apply(
            lambda x: Point(x["decimalLongitude"], x["decimalLatitude"]), 
            axis=1
        )
        gdbf_df = gpd.GeoDataFrame(gdbf_df, geometry="geometry", crs=self.target_crs)

        # Map IUCN categories
        gdbf_df["iucnRedListCategory"] = gdbf_df["iucnRedListCategory"].replace(
            self.iucn_categories
        )

        # Get unique species counts
        species_counts = gdbf_df[["kingdom", "class", "species", "iucnRedListCategory"]].drop_duplicates()
        species_counts = species_counts.dropna()

        return species_counts

    def get_data(self, polygon: dict, polygon_crs: str, buffer_meters: int = 30000, **kwargs) -> Dict:
        """Get endangered species statistics for a given geometry"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return self.get_species_stats(polygon=polygon, polygon_crs=polygon_crs, buffer_meters=buffer_meters)
