import os
from typing import Dict, List

import geopandas as gpd
import requests
from bs4 import BeautifulSoup
from shapely.geometry import Point

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class RamsarProtectedAreas(BaseEnvironmentalMetric):
    """Class for analyzing protected areas data from Ramsar sites"""

    def __init__(self):
        super().__init__()
        self.ramsar_sites = gpd.read_parquet(
            path=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "ramsar_sites.parquet",
            )
        ).to_crs("EPSG:3857")

    def get_nearest_ramsar_sites(
        self, polygon: dict, polygon_crs: str, limit: int = 5
    ) -> List[Dict]:
        """
        Get nearest Ramsar protected sites for a given geometry

        Args:
            geometry: GeoJSON geometry to analyze
            limit: Number of nearest sites to return (default 5)

        Returns:
            List of dictionaries containing nearest Ramsar sites with distances and descriptions
        """
        # Convert geometry to GeoDataFrame and get centroid
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        gdf = gpd.GeoDataFrame([polygon], crs="EPSG:4326", columns=["geometry"])
        gdf = gdf.to_crs("EPSG:3857")
        center_point = gdf.centroid
        center_point_df = gpd.GeoDataFrame(geometry=[center_point[0]], crs="EPSG:3857")

        # Find nearest sites
        nearest_sites = gpd.sjoin_nearest(
            self.ramsar_sites,
            center_point_df,
            how="inner",
            max_distance=None,
            distance_col="distance",
        ).nsmallest(limit, "distance")

        results = []
        for _, site in nearest_sites.iterrows():
            description = self._get_site_description(site["ramsarid"])
            results.append(
                {
                    "name": site["officialna"],
                    "distance_km": round(site["distance"] / 1000, 2),
                    "description": description,
                    "ramsar_id": site["ramsarid"],
                }
            )

        return results

    def _get_site_description(self, ramsar_id: int) -> str:
        """
        Get description for a Ramsar site from its webpage

        Args:
            ramsar_id: Ramsar site ID

        Returns:
            Site description or None if not found
        """
        url = f"https://rsis.ramsar.org/ris/{ramsar_id}"
        response = requests.get(url)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        summary_div = soup.find("div", {"class": "field-name-asummary"})

        if summary_div:
            return summary_div.get_text(strip=True)
        return None

    def get_data(
        self, polygon: dict, polygon_crs: str, limit: int = 5, **kwargs
    ) -> List[Dict]:
        """Get nearest Ramsar protected sites for a given geometry"""
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return self.get_nearest_ramsar_sites(
            polygon=polygon, polygon_crs=polygon_crs, limit=limit
        )
