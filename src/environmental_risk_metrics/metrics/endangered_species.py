from typing import Dict

import geopandas as gpd
import pandas as pd
from pygbif import occurrences
from shapely.geometry import Point
from shapely.geometry.polygon import orient

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class EndangeredSpecies(BaseEnvironmentalMetric):
    """Class for analyzing endangered species data from GBIF"""

    def __init__(self, gdf: gpd.GeoDataFrame):
        sources = [
            "https://api.gbif.org/v1",
            "https://api.gbif.org/v1",
        ]
        self.gdf = gdf.to_crs(epsg=4326)
        description = "Endangered species data from GBIF"
        super().__init__(sources=sources, description=description)
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

    def get_species_stats(self, buffer_meters: int) -> pd.DataFrame:
        """
        Get endangered species statistics for a given geometry
        
        Args:
            geometry: GeoJSON geometry to analyze
            
        Returns:
            DataFrame containing unique species counts by kingdom, class and IUCN category
        """
        # Convert to projected CRS for buffering in meters
        gdf = self.gdf.to_crs("EPSG:3857")

        gdf["geometry"] = gdf.buffer(buffer_meters).to_crs(self.gdf.crs)
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: orient(geom, sign=1.0))
        species_counts_list = []
        for geometry in gdf["geometry"].to_wkt():
            geometry_wkt = geometry

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
            species_counts_list.append(species_counts)

        return species_counts_list

    def get_data(self, polygon: dict, polygon_crs: str, buffer_meters: int = 30000, **kwargs) -> Dict:
        """Get endangered species statistics for a given geometry"""
        df =  self.get_species_stats(buffer_meters=buffer_meters)
        records = df.to_dict(orient="records")
        return records