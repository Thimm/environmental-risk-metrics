import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import geopandas as gpd
import leafmap
import odc.stac
import pandas as pd
import planetary_computer
import rioxarray
import xarray as xr
from pystac.item import Item
from tqdm import tqdm

from environmental_risk_metrics.base import BaseEnvironmentalMetric
from environmental_risk_metrics.legends.land_use_change import (
    ESA_LAND_COVER_LEGEND,
    ESRI_LAND_COVER_LEGEND,
    OPENLANDMAP_LC_LEGEND,
)
from environmental_risk_metrics.utils.planetary_computer import (
    get_planetary_computer_items,
)

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
        resolution: int,
        max_workers: int = 10,
        show_progress: bool = True,
    ) -> None:
        super().__init__(sources=sources, description=description)
        self.collections = collections
        self.band_name = band_name
        self.name = name
        self.legend = legend
        self.sources = sources
        self.resolution = resolution
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

    def load_xarray(
        self,
        start_date: str,
        end_date: str,
     
    ) -> xr.Dataset:
        logger.debug(
            f"Loading {self.collections} data at {self.resolution}m resolution"
        )
        ds_list = []
        for polygon in self.gdf["geometry"]:
            items = self.get_items(
                start_date=start_date,
                end_date=end_date,
                polygon=polygon,
                polygon_crs=self.gdf.crs,
            )

            if not items:
                raise ValueError(
                    f"No {self.name} items found for the given date range and polygon"
                )

            signed_items = [planetary_computer.sign(item) for item in items]
            thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

            ds = odc.stac.load(
                signed_items,
                bands=[self.band_name],
                resolution=self.resolution,
                pool=thread_pool,
                geopolygon=polygon,
                progress=tqdm if self.show_progress else None,
            )
            ds_list.append(ds)
        return ds_list

    def getlegend(self) -> Dict[int, str]:
        return self.legend

    def get_xarray_with_class_names(
        self,
        start_date: str,
        end_date: str,
    ):
        ds_list = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
        )
        for ds in ds_list:
            ds = ds.assign_coords(
                **{self.band_name: ds[self.band_name].map(self.legend)}
            )
        return ds_list

    def get_land_use_class_percentages(
        self,
        start_date: str,
        end_date: str,
      
        all_touched: bool = True,
    ) -> pd.DataFrame:
        ds_list = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
        )
        percentage_list = []
        for ds, polygon in zip(ds_list, self.gdf["geometry"]):
            crs = ds.coords["spatial_ref"].values.item()
            clipped_data = ds.rio.write_crs(crs).rio.clip(
                [polygon], self.gdf.crs, all_touched=all_touched
            )
            clipped_data = clipped_data.where(clipped_data[self.band_name] != 0)

            clipped_data_df = clipped_data.to_dataframe()
            clipped_data_df[self.band_name] = clipped_data_df[self.band_name].map(
                self.get_legend_labels_dict()
            )
            grouped = clipped_data_df.groupby("time")
            value_counts = grouped[self.band_name].value_counts()
            total_counts = grouped[self.band_name].count()

            percentage = (value_counts / total_counts).unstack(level=1)
            percentage_list.append(round(percentage * 100, 2))
        return percentage_list

    def get_data(
        self,
        start_date: str,
        end_date: str,
        all_touched: bool = True,
    ) -> Dict:
        """Get land use class percentages for a given geometry"""
        df_list =  self.get_land_use_class_percentages(
            start_date=start_date,
            end_date=end_date,
            all_touched=all_touched,
        )   
        df_list = [df.fillna(0) for df in df_list]
        df_list = [df.reset_index(names="date") for df in df_list]
        all_records = [df.to_dict(orient="records") for df in df_list]

        # Add all legend values with 0 for any missing classes
        all_legend_values = self.get_legend_labels_dict().values()
        for records_list in all_records:
            for record in records_list:
                for label in all_legend_values:
                    if label not in record:
                        record[label] = 0
        return all_records


class EsaLandCover(BaseLandCover):
    def __init__(self, gdf: gpd.GeoDataFrame, use_esri_classes: bool = False) -> None:
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
            resolution=0.00009,
            legend=ESA_LAND_COVER_LEGEND
            if not use_esri_classes
            else map_esa_to_esri_classes(),
        )
        self.gdf = gdf.to_crs(epsg=4326)

class EsriLandCover(BaseLandCover):
    def __init__(self, gdf: gpd.GeoDataFrame, use_esri_classes: bool = False) -> None:
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
            resolution=10,
        )
        self.gdf = gdf.to_crs(epsg=4326)

class OpenLandMapLandCover(BaseLandCover):
    def __init__(self, gdf: gpd.GeoDataFrame, use_esri_classes: bool = False) -> None:
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
            resolution=10,
            legend=map_openlandmap_to_esri_classes()
            if use_esri_classes
            else OPENLANDMAP_LC_LEGEND,
        )
        self.gdf = gdf.to_crs(epsg=4326)
    def load_xarray(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Item]:
        """Override get_items to use GLAD land cover data instead of Planetary Computer"""
        # Convert dates to years
        start_year = str(pd.to_datetime(start_date).year)
        end_year = str(pd.to_datetime(end_date).year)

        # Get available years within range
        available_years = [
            y for y in OPENLANDMAP_LC.keys() if start_year <= y <= end_year
        ]

        if not available_years:
            raise ValueError(
                f"No GLAD data available between {start_year} and {end_year}"
            )
        
        ds_list = []

        # Load and merge data for all available years
        for geometry in self.gdf["geometry"]:
            data_arrays = []
            minx, miny, maxx, maxy = (
                gpd.GeoDataFrame([geometry], columns=["geometry"])
                .set_geometry("geometry")
                .bounds.iloc[0]
            )
            for year in available_years:
                url = OPENLANDMAP_LC[year]
                da = rioxarray.open_rasterio(url)
                da = da.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
                da = da.rio.clip_box(
                    minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=self.gdf.crs
                )
                data_arrays.append(da)
            ds = xr.concat(data_arrays, dim="time")
            ds = ds.squeeze()
            ds_list.append(ds)
        return ds_list

    def get_land_use_class_percentages(
        self,
        start_date: str,
        end_date: str,
        all_touched: bool = True,
    ) -> pd.DataFrame:
        percentage_list = []
        ds_list = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
        )
        for ds, geometry in zip(ds_list, self.gdf["geometry"]):
            clipped_data = ds.rio.clip(geometry, self.gdf.crs, all_touched=all_touched)
            clipped_data_df = clipped_data.to_dataframe("class").reset_index()
            clipped_data_df = clipped_data_df[clipped_data_df["class"] != 0]
            clipped_data_df["class"] = clipped_data_df["class"].map(
                self.get_legend_labels_dict()
            )
            grouped = clipped_data_df.groupby("time")
            value_counts = grouped["class"].value_counts()
            total_counts = grouped["class"].count()
            percentage = (value_counts / total_counts).unstack(level=1)
            percentage = percentage.fillna(0)
            percentage = round(percentage * 100, 2)
            percentage_list.append(percentage)
        return percentage_list

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
        center = ((bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2)
        
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
                shown=True
            )
            
        # Create GeoDataFrame from processed polygons
        gdf = gpd.GeoDataFrame(geometry=processed_polygons, crs=self.target_crs)
        m.add_gdf(gdf, layer_name="Your Parcels", zoom_to_layer=True)
        
        return m


        
        
