import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import leafmap
import pandas as pd
import rasterio
import xarray as xr
from IPython.display import display
from pystac.item import Item
from shapely.geometry import box

from environmental_risk_metrics.base import BaseEnvironmentalMetric
from environmental_risk_metrics.legends.land_use_change import (
    ESA_LAND_COVER_LEGEND, ESRI_LAND_COVER_LEGEND, OPENLANDMAP_LC_LEGEND)
from environmental_risk_metrics.utils.planetary_computer import \
    get_planetary_computer_items

logger = logging.getLogger(name=__name__)


OPENLANDMAP_LC = {
    "2000": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20000101_20001231_go_epsg.4326_v20230901.tif",
    "2005": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20050101_20051231_go_epsg.4326_v20230901.tif",
    "2010": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20100101_20101231_go_epsg.4326_v20230901.tif",
    "2015": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20150101_20151231_go_epsg.4326_v20230901.tif",
    "2020": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20200101_20201231_go_epsg.4326_v20230901.tif",
}


def map_esa_to_esri_classes() -> Optional[int]:
    """Maps ESA land cover classes to ESRI land cover classes"""
    mapping = {
        # ESA 'No data' -> ESRI 'No Data'
        0: 0,
        # ESA 'Cropland, rainfed' -> ESRI 'Crops'
        10: 5,
        # ESA 'Cropland, rainfed, herbaceous cover' -> ESRI 'Crops'
        11: 5,
        # ESA 'Cropland, rainfed, tree, or shrub cover' -> ESRI 'Crops'
        12: 5,
        # ESA 'Cropland, irrigated or post-flooding' -> ESRI 'Crops'
        20: 5,
        # ESA 'Mosaic cropland/natural vegetation' -> ESRI 'Crops'
        30: 5,
        # ESA 'Mosaic natural vegetation/cropland' -> ESRI 'Rangeland'
        40: 11,
        # ESA 'Tree cover, broadleaved, evergreen' -> ESRI 'Trees'
        50: 2,
        # ESA 'Tree cover, broadleaved, deciduous' -> ESRI 'Trees'
        60: 2,
        # ESA 'Tree cover, broadleaved, deciduous, closed' -> ESRI 'Trees'
        61: 2,
        # ESA 'Tree cover, broadleaved, deciduous, open' -> ESRI 'Trees'
        62: 2,
        # ESA 'Tree cover, needleleaved, evergreen' -> ESRI 'Trees'
        70: 2,
        # ESA 'Tree cover, needleleaved, evergreen, closed' -> ESRI 'Trees'
        71: 2,
        # ESA 'Tree cover, needleleaved, evergreen, open' -> ESRI 'Trees'
        72: 2,
        # ESA 'Tree cover, needleleaved, deciduous' -> ESRI 'Trees'
        80: 2,
        # ESA 'Tree cover, needleleaved, deciduous, closed' -> ESRI 'Trees'
        81: 2,
        # ESA 'Tree cover, needleleaved, deciduous, open' -> ESRI 'Trees'
        82: 2,
        # ESA 'Tree cover, mixed leaf type' -> ESRI 'Trees'
        90: 2,
        # ESA 'Mosaic tree and shrub/herbaceous cover' -> ESRI 'Rangeland'
        100: 11,
        # ESA 'Mosaic herbaceous cover/tree and shrub' -> ESRI 'Rangeland'
        110: 11,
        # ESA 'Shrubland' -> ESRI 'Rangeland'
        120: 11,
        # ESA 'Evergreen shrubland' -> ESRI 'Rangeland'
        121: 11,
        # ESA 'Deciduous shrubland' -> ESRI 'Rangeland'
        122: 11,
        # ESA 'Grassland' -> ESRI 'Rangeland'
        130: 11,
        # ESA 'Lichens and mosses' -> ESRI 'Rangeland'
        140: 11,
        # ESA 'Sparse vegetation' -> ESRI 'Rangeland'
        150: 11,
        # ESA 'Sparse tree' -> ESRI 'Rangeland'
        151: 11,
        # ESA 'Sparse shrub' -> ESRI 'Rangeland'
        152: 11,
        # ESA 'Sparse herbaceous cover' -> ESRI 'Rangeland'
        153: 11,
        # ESA 'Tree cover, flooded, fresh/brackish' -> ESRI 'Flooded vegetation'
        160: 4,
        # ESA 'Tree cover, flooded, saline water' -> ESRI 'Flooded vegetation'
        170: 4,
        # ESA 'Shrub or herbaceous cover, flooded' -> ESRI 'Flooded vegetation'
        180: 4,
        # ESA 'Urban areas' -> ESRI 'Built area'
        190: 7,
        # ESA 'Bare areas' -> ESRI 'Bare ground'
        200: 8,
        # ESA 'Consolidated bare areas' -> ESRI 'Bare ground'
        201: 8,
        # ESA 'Unconsolidated bare areas' -> ESRI 'Bare ground'
        202: 8,
        # ESA 'Water bodies' -> ESRI 'Water'
        210: 1,
        # ESA 'Permanent snow and ice' -> ESRI 'Snow/ice'
        220: 9,
    }
    NEW_ESA_CLASS_MAPPING = {}
    for key, value in mapping.items():
        if value is not None:
            NEW_ESA_CLASS_MAPPING[key] = {
                "value": key,
                "color": ESRI_LAND_COVER_LEGEND[value]["color"],
                "label": ESRI_LAND_COVER_LEGEND[value]["label"],
            }
    return NEW_ESA_CLASS_MAPPING


def map_openlandmap_to_esri_classes() -> Optional[int]:
    """Maps GLAD land cover classes to ESRI land cover classes"""
    GLAD_TO_CLASSES = {
        # Terra Firma short vegetation (1-24)
        **{i: 11 for i in range(1, 25)},
        # Terra Firma stable tree cover (25-48)
        **{i: 2 for i in range(25, 49)},
        # Terra Firma tree cover with prev. disturb. (49-72)
        **{i: 2 for i in range(49, 73)},
        # Terra Firma tree height gain (73-96)
        **{i: 2 for i in range(73, 97)},
        # Wetland short vegetation (100-124)
        **{i: 4 for i in range(100, 125)},
        # Wetland stable tree cover (125-148)
        **{i: 4 for i in range(125, 149)},
        # Wetland tree cover with prev. disturb. (149-172)
        **{i: 4 for i in range(149, 173)},
        # Wetland tree height gain (173-196)
        **{i: 4 for i in range(173, 197)},
        # Open surface water (208-211)
        **{i: 1 for i in range(208, 212)},
        # Short veg. after tree loss (240)
        240: 11,
        # Snow/ice stable/gain/loss (241-243)
        **{i: 9 for i in range(241, 244)},
        # Cropland stable/gain/loss (244-249)
        **{i: 5 for i in range(244, 250)},
        # Built-up stable/gain/loss (250-253)
        **{i: 7 for i in range(250, 254)},
        # Ocean (254)
        254: 1,
        # No data (255)
        255: 0,
    }
    NEW_GLAD_CLASS_MAPPING = {}
    for key, value in GLAD_TO_CLASSES.items():
        if value is not None:
            NEW_GLAD_CLASS_MAPPING[key] = {
                "value": key,
                "color": ESRI_LAND_COVER_LEGEND[value]["color"],
                "label": ESRI_LAND_COVER_LEGEND[value]["label"],
            }
    return NEW_GLAD_CLASS_MAPPING


class BaseLandCover(BaseEnvironmentalMetric):
    def __init__(
        self,
        collections: List[str],
        band_name: str,
        name: str,
        legend: Dict[int, str],
        sources: List[str],
        description: str,
        max_workers: int = 10,
        show_progress: bool = True,
    ) -> None:
        """Initializes the BaseLandCover class.

        Args:
            collections: List of Planetary Computer collections to use.
            band_name: Name of the band to use from the collections.
            name: Name of the land cover metric.
            legend: Legend for the land cover classes.
            sources: List of data sources.
            description: Description of the metric.
            max_workers: Maximum number of workers for parallel processing.
            show_progress: Whether to show a progress bar.
        """
        super().__init__(sources=sources, description=description)
        self.collections = collections
        self.band_name = band_name
        self.name = name
        self.legend = legend
        self.sources = sources
        self.max_workers = max_workers
        self.show_progress = show_progress

    def get_items(
        self, start_date: str, end_date: str, polygon: dict, polygon_crs: str
    ) -> List[Item]:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return get_planetary_computer_items(
            collections=self.collections,
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
        )

    def getlegend(self) -> Dict[int, str]:
        return self.legend



    def get_land_use_metrics(
        self,
        start_date: str,
        end_date: str,
        all_touched: bool = True,
        expected_total_ha: float = None,
    ) -> List[Dict[str, Any]]:
        """Calculate land use class areas in hectares using rasterstats zonal_stats.

        Args:
            start_date: Start date for the analysis.
            end_date: End date for the analysis.
            all_touched: Whether to include all pixels touching the geometry.
            expected_total_ha: Optional, if provided, scales the output to match this area.

        Returns:
            A list of dictionaries, one for each polygon, containing years as keys and land cover areas in hectares.
        """
        from rasterstats import zonal_stats

        # Get all items in the date range
        items = self.get_items(
            start_date=start_date,
            end_date=end_date,
            polygon=self.gdf,
            polygon_crs=self.gdf.crs,
        )
        if not items:
            raise ValueError("No items found for the given date range and polygon")

        # Get legend mapping
        label_map = self.get_legend_labels_dict()

        # Calculate total areas in hectares
        total_areas_ha = self.gdf.to_crs("EPSG:6933").area / 10000

        # Initialize list to store all data for pandas processing
        all_data = []

        # Process each item
        for item in items:
            with rasterio.open(item.assets[self.band_name].href) as src:
                raster_crs = src.crs
            gdf_raster_crs = self.gdf.to_crs(raster_crs)
            
            # Extract year from item properties
            year = str(pd.to_datetime(item.properties["start_datetime"]).year)

            raster_path = item.assets[self.band_name].href

            # Count pixels per land cover class within each polygon
            stats_dict = zonal_stats(
                gdf_raster_crs,
                raster_path,
                categorical=True,
                all_touched=all_touched,
            )

            # Process each polygon's stats
            for polygon_idx, (polygon_stats, total_area_ha) in enumerate(zip(stats_dict, total_areas_ha)):
                if not polygon_stats:  # Skip empty stats
                    continue
                    
                total_pixels = sum(polygon_stats.values())
                
                # Calculate areas for each class
                for class_id, count in polygon_stats.items():
                    area_ha = round(count / total_pixels * total_area_ha, 2)
                    label = label_map.get(class_id, str(class_id))
                    
                    # Store data for pandas processing
                    all_data.append({
                        'polygon_idx': polygon_idx,
                        'year': year,
                        'class_id': class_id,
                        'label': label,
                        'pixel_count': count,
                        'area_ha': area_ha
                    })

        # Convert to DataFrame for easier processing
        if not all_data:
            return [{} for _ in range(len(self.gdf))]
            
        df = pd.DataFrame(all_data)
        
        # Group by polygon, year, and label, then sum the areas
        grouped_df = df.groupby(['polygon_idx', 'year', 'label'])['area_ha'].sum().reset_index()
        
        # Pivot to get years as columns and labels as rows
        pivot_df = grouped_df.pivot_table(
            index=['polygon_idx', 'label'], 
            columns='year', 
            values='area_ha', 
            fill_value=0
        ).reset_index()
        
        # Convert to the expected format
        polygon_results = [{} for _ in range(len(self.gdf))]
        
        for _, row in pivot_df.iterrows():
            polygon_idx = int(row['polygon_idx'])
            label = row['label']
            
            # Add each year's data to the polygon
            for year in df['year'].unique():
                year_str = str(year)
                if year_str in row.index:
                    area = row[year_str]
                    if area > 0:  # Only add non-zero areas
                        if year_str not in polygon_results[polygon_idx]:
                            polygon_results[polygon_idx][year_str] = {}
                        polygon_results[polygon_idx][year_str][label] = area

        return polygon_results

    def get_data(
        self,
        start_date: str,
        end_date: str,
        all_touched: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Get land use class areas for a given geometry.

        Args:
            start_date: Start date for the analysis.
            end_date: End date for the analysis.
            all_touched: Whether to include all pixels touching the geometry.

        Returns:
            A list of lists of dictionaries containing the land use class
            areas in square meters for each polygon and time step.
        """
        area_dicts = self.get_land_use_metrics(
            start_date=start_date,
            end_date=end_date,
            all_touched=all_touched,
        )

        # Get ESRI class labels (not the raw numbers)
        esri_labels = list(ESRI_LAND_COVER_LEGEND.values())
        esri_label_names = [label["label"] for label in esri_labels]

        # Process each polygon's data
        for polygon_data in area_dicts:
            # Process each year's data within the polygon
            for year_data in polygon_data.values():
                # Ensure all ESRI classes are present with 0 values if missing
                for label_name in esri_label_names:
                    if label_name not in year_data:
                        year_data[label_name] = 0

        return area_dicts


class EsaLandCover(BaseLandCover):
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> None:
        """Initializes the EsaLandCover class.

        Args:
            gdf: GeoDataFrame with the geometry to analyze.
        """
        sources = [
            "https://planetarycomputer.microsoft.com/dataset/esa-cci-lc",
            "https://doi.org/10.24381/cds.006f2c9a",
        ]
        description = "ESA Climate Change Initiative (CCI) Land Cover"
        super().__init__(
            collections=["esa-cci-lc"],
            sources=sources,
            description=description,
            name="ESA Climate Change Initiative (CCI) Land Cover",
            band_name="lccs_class",
            legend=map_esa_to_esri_classes(),
        )
        self.gdf = gdf


class EsriLandCover(BaseLandCover):
    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        """Initializes the EsriLandCover class.

        Args:
            gdf: GeoDataFrame with the geometry to analyze.
        """
        sources = [
            "https://planetarycomputer.microsoft.com/dataset/io-lulc-annual-v02",
            "https://planetarycomputer.microsoft.com/dataset/io-lulc-annual-v02",
        ]
        description = "Esri Land Use"
        super().__init__(
            collections=["io-lulc-annual-v02"],
            sources=sources,
            description=description,
            name="Esri Land Use",
            band_name="data",
            legend=ESRI_LAND_COVER_LEGEND,
        )
        self.gdf = gdf


class OpenLandMapLandCover(BaseLandCover):
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> None:
        """Initializes the OpenLandMapLandCover class.

        Args:
            gdf: GeoDataFrame with the geometry to analyze.
        """
        sources = [
            "https://glad.umd.edu/dataset/GLCLUC",
            "https://glad.umd.edu/dataset/GLCLUC",
        ]
        description = "GLAD Land Use/Cover"
        super().__init__(
            collections=None,
            sources=sources,
            description=description,
            name="GLAD Land Use/Cover",
            band_name="data",
            legend=map_openlandmap_to_esri_classes(),
        )
        self.gdf = gdf.to_crs(epsg=4326)

    def create_map(self, polygons: dict | list, polygon_crs: str, **kwargs) -> None:
        """Create a map for the land use change data

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
        center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)

        m = leafmap.Map(
            center=(center[1], center[0]),
            zoom=14,
            draw_control=False,
            measure_control=False,
            fullscreen_control=False,
            attribution_control=False,
            search_control=False,
            layers_control=True,
            scale_control=False,
            toolbar_control=True,
        )

        colormap = self.get_legend_colors()
        for year, cog in OPENLANDMAP_LC.items():
            m.add_cog_layer(
                cog,
                colormap=json.dumps(colormap),
                name=year,
                attribution="UMD GLAD",
                shown=True,
            )

        # Create GeoDataFrame from processed polygons
        gdf = gpd.GeoDataFrame(geometry=processed_polygons, crs=self.target_crs)
        m.add_gdf(gdf, layer_name="Your Parcels", zoom_to_layer=True)

        return m
        # Create GeoDataFrame from processed polygons
        gdf = gpd.GeoDataFrame(geometry=processed_polygons, crs=self.target_crs)
        m.add_gdf(gdf, layer_name="Your Parcels", zoom_to_layer=True)

        return m
