import logging
import typing as t

import planetary_computer
import pystac
import pystac_client

logger = logging.getLogger(__name__)


def get_planetary_computer_items(
    collections: list[str],
    start_date: str,
    end_date: str,
    polygon: dict,
    entire_image_cloud_cover_threshold: t.Optional[int] = None,
) -> list[pystac.Item]:
    """
    Search for Sentinel-2 items within a given date range and polygon.

    Args:
        start_date: Start date for the search (YYYY-MM-DD)
        end_date: End date for the search (YYYY-MM-DD)
        polygon: GeoJSON polygon to intersect with the search
        entire_image_cloud_cover_threshold: Maximum cloud cover percentage to include in the search
        cropped_image_cloud_cover_threshold: Maximum cloud cover within a cropped image to include in the search

    Returns:
        List of STAC items matching the search criteria
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,  # noqa: F821
    )

    logger.debug(
        f"Searching for Sentinel-2 items between {start_date} and {end_date}"
    )
    search = catalog.search(
        collections=collections,
        datetime=f"{start_date}/{end_date}",
        intersects=polygon,
        query={"eo:cloud_cover": {"lt": entire_image_cloud_cover_threshold}} if entire_image_cloud_cover_threshold else None,
    )
    items = search.item_collection()
    logger.debug(f"Found {len(items)} items")
    return items