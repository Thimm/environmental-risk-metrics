import json
import logging
from typing import Optional, Union

import geopandas as gpd
import shapely
from shapely.geometry import shape

from .metric_calculator import calculate_metrics
from .planetary_computer import get_planetary_computer_items

__all__ = ["calculate_metrics", "get_planetary_computer_items", "ensure_geometry_crs", "get_centroid_of_geometry"]

logger = logging.getLogger(__name__)


def ensure_geometry_crs(
    geometry: Union[dict, str, gpd.GeoDataFrame, "shapely.geometry.base.BaseGeometry"],
    source_crs: Optional[str],
    target_crs: str = "EPSG:4326",
) -> "shapely.geometry.base.BaseGeometry":
    """
    Ensures a geometry is in the specified CRS, converting if necessary.

    Args:
        geometry: Input geometry in various formats (GeoJSON dict/string, GeoDataFrame, or Shapely geometry)
        target_crs: Target CRS as string (default: "EPSG:4326")
        source_crs: Source CRS of the input geometry as string (optional)

    Returns:
        Shapely geometry in target CRS

    Raises:
        ValueError: If geometry format is not recognized or if CRS information is missing when required
    """
    logger.debug(
        f"Ensuring geometry CRS. Target CRS: {target_crs}, Source CRS: {source_crs}"
    )

    # Handle GeoJSON string
    if isinstance(geometry, str):
        try:
            logger.debug("Converting GeoJSON string to dict")
            geometry = json.loads(geometry)
        except json.JSONDecodeError:
            logger.error("Failed to parse GeoJSON string")
            raise ValueError("Invalid GeoJSON string")

    # Handle GeoDataFrame
    if isinstance(geometry, gpd.GeoDataFrame):
        logger.debug("Processing GeoDataFrame")
        if geometry.crs != target_crs:
            logger.debug(f"Converting GeoDataFrame from {geometry.crs} to {target_crs}")
            geometry = geometry.to_crs(target_crs)
        return geometry.geometry.iloc[0]

    # Handle Shapely geometry
    if hasattr(geometry, "geom_type"):  # Check if it's a Shapely geometry
        logger.debug("Processing Shapely geometry")
        # Convert to GeoDataFrame to handle CRS transformation
        gdf = gpd.GeoDataFrame(geometry=[geometry])
        if source_crs:
            logger.debug(f"Setting source CRS: {source_crs}")
            gdf.set_crs(source_crs, inplace=True)
        else:
            logger.debug(f"No source CRS provided, assuming {target_crs}")
            gdf.set_crs(
                target_crs, inplace=True
            )  # Assume target_crs if source_crs not provided
        if gdf.crs != target_crs:
            logger.debug(f"Converting from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
        return gdf.geometry.iloc[0]

    # Handle GeoJSON dict
    if isinstance(geometry, dict):
        logger.debug("Processing GeoJSON dict")
        # Convert to GeoDataFrame to ensure proper CRS
        try:
            gdf = gpd.GeoDataFrame.from_features([geometry])
            if source_crs:
                logger.debug(f"Setting source CRS: {source_crs}")
                gdf.set_crs(source_crs, inplace=True)
            else:
                logger.debug(f"No source CRS provided, assuming {target_crs}")
                gdf.set_crs(
                    target_crs, inplace=True
                )  # Assume target_crs if source_crs not provided
            if gdf.crs != target_crs:
                logger.debug(f"Converting from {gdf.crs} to {target_crs}")
                gdf = gdf.to_crs(target_crs)
            return gdf.geometry.iloc[0]
        except Exception as e:
            logger.error(f"Failed to process GeoJSON dict: {str(e)}")
            raise ValueError("Invalid GeoJSON dictionary")

    logger.error("Unsupported geometry format")
    raise ValueError("Unsupported geometry format")


def get_centroid_of_geometry(
    geometry: Union[dict, gpd.GeoDataFrame, "shapely.geometry.base.BaseGeometry"],
    source_crs: Optional[str] = None,
    target_crs: str = "EPSG:4326",
) -> tuple:
    """Get the center of a geometry"""
    geometry = ensure_geometry_crs(geometry, source_crs=source_crs, target_crs=target_crs)
    return geometry.centroid.x, geometry.centroid.y
