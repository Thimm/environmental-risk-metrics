import os
from typing import Dict, List

import geopandas as gpd
import leafmap
import requests
from bs4 import BeautifulSoup

from environmental_risk_metrics.base import BaseEnvironmentalMetric


class RamsarProtectedAreas(BaseEnvironmentalMetric):
    """Class for analyzing protected areas data from Ramsar sites"""

    def __init__(self, gdf: gpd.GeoDataFrame):
        sources = [
            "https://rsis.ramsar.org/geoserver/wms",
            "https://rsis.ramsar.org/geoserver/wms",
        ]
        description = "Ramsar protected areas data" 
        super().__init__(sources=sources, description=description)

        self.ramsar_sites = gpd.read_parquet(
            path=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "ramsar_sites.parquet",
            )
        ).to_crs("EPSG:3857")
        self.gdf = gdf.to_crs("EPSG:3857")
    def get_nearest_ramsar_sites(
        self, limit: int = 5
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
        self.gdf["center_point"] = self.gdf.centroid
        results_list = []
        for center_point in self.gdf["center_point"]:
            # Find nearest sites
            nearest_sites = gpd.sjoin_nearest(
                left_df=self.ramsar_sites,
                right_df=gpd.GeoDataFrame(geometry=[center_point], crs="EPSG:3857"),
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
            results_list.append(results)
        return results_list

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
        self, limit: int = 5, **kwargs
    ) -> List[Dict]:
        """Get nearest Ramsar protected sites for a given geometry"""
        return self.get_nearest_ramsar_sites(limit=limit)

    def create_map(self, polygons: dict | list, polygon_crs: str, **kwargs) -> None:
        """Create a map for the Ramsar protected areas data
        
        Args:
            polygons: Single GeoJSON polygon or list of polygons
            polygon_crs: CRS of the input polygon(s)
        """
        # Convert single polygon to list for consistent handling
        if isinstance(polygons, dict):
            polygons = [polygons]
            
        # Preprocess all polygons
        processed_polygons = [
            self._preprocess_geometry(polygon, source_crs=polygon_crs)
            for polygon in polygons
        ]
        
        # Get center from first polygon
        gdf = gpd.GeoDataFrame(geometry=processed_polygons, crs=self.target_crs)
        bounds = gdf.total_bounds
        center = ((bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2)

        m = leafmap.Map(
            center=(center[1], center[0]),
            zoom=9,
            draw_control=False,
            measure_control=False,
            fullscreen_control=False,
            attribution_control=False,
            search_control=False,
            layers_control=False,
            scale_control=False,
            toolbar_control=False,
            basemap="Esri.WorldGrayCanvas",
        )

        ramsar_url = "https://rsis.ramsar.org/geoserver/wms?"
        m.add_wms_layer(
            url=ramsar_url,
            layers="ramsar_sdi:features",
            name="NAIP Imagery",
            format="image/png",
            shown=True,
            srs="EPSG:4326",
            zoom_to_layer=False,
        )
        m.add_wms_layer(
            url=ramsar_url,
            layers="ramsar_sdi:features_centroid",
            name="NAIP Imagery",
            format="image/png",
            shown=True,
            srs="EPSG:4326",
            zoom_to_layer=False,
        )

        # Create GeoDataFrame from processed polygons
        gdf = gpd.GeoDataFrame(geometry=processed_polygons, crs=self.target_crs)
        m.add_gdf(gdf, layer_name="Your Parcels", zoom_to_layer=True)

        m.add_tile_layer(
            url="https://data-gis.unep-wcmc.org/server/rest/services/ProtectedSites/The_World_Database_of_Protected_Areas/MapServer/tile/{z}/{y}/{x}",
            attribution="The World Database of Protected Areas",
            name="The World Database of Protected Areas",
        )
        return m
