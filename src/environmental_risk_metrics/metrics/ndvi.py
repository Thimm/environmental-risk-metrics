import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import odc
import odc.stac
import pandas as pd
import planetary_computer
import pystac
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from environmental_risk_metrics.base import BaseEnvironmentalMetric
from environmental_risk_metrics.utils.planetary_computer import (
    get_planetary_computer_items,
)

matplotlib.use(backend="Agg")

logger = logging.getLogger(__name__)


class HarmonizedNDVI(BaseEnvironmentalMetric):
    """
    A class for accessing and analyzing NDVI data from multiple harmonized satellite collections
    including Sentinel-2, HLS2-S30, and HLS2-L30 from Planetary Computer.
    """
    
    # Band mappings for different collections
    BAND_MAPPINGS = {
        "sentinel-2-l2a": {
            "red": "B04",
            "green": "B03", 
            "blue": "B02",
            "nir": "B08",
            "cloud_mask": "SCL",
            "cloud_clear_values": [4, 5]  # SCL values for clear pixels
        },
        "hls2-s30": {
            "red": "B04",
            "green": "B03",
            "blue": "B02", 
            "nir": "B08",
            "cloud_mask": "Fmask",
            "cloud_clear_values": [0]  # Fmask value for clear pixels
        },
        "hls2-l30": {
            "red": "B04",
            "green": "B03",
            "blue": "B02",
            "nir": "B05",  # Different NIR band for Landsat
            "cloud_mask": "Fmask", 
            "cloud_clear_values": [0]  # Fmask value for clear pixels
        }
    }
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        gdf: gpd.GeoDataFrame,
        collections: list[str] = ["sentinel-2-l2a"],
        resolution: int = 30,
        entire_image_cloud_cover_threshold: int = 10,
        cropped_image_cloud_cover_threshold: int = 80,
        max_workers: int = 10,
        is_bare_soil_threshold: float = 0.25,
    ):
        sources = [
            "https://planetarycomputer.microsoft.com/api/stac/v1",
        ]
        description = f"Harmonized NDVI data from collections: {', '.join(collections)}"

        super().__init__(sources=sources, description=description)
        
        # Validate collections
        invalid_collections = set(collections) - set(self.BAND_MAPPINGS.keys())
        if invalid_collections:
            raise ValueError(f"Unsupported collections: {invalid_collections}. "
                           f"Supported collections: {list(self.BAND_MAPPINGS.keys())}")
        
        self.collections = collections
        self.resolution = resolution
        self.entire_image_cloud_cover_threshold = entire_image_cloud_cover_threshold
        self.cropped_image_cloud_cover_threshold = cropped_image_cloud_cover_threshold
        self.max_workers = max_workers
        self.start_date = start_date
        self.end_date = end_date
        self.gdf = gdf.to_crs(epsg=4326)
        self.items = None
        self.xarray_data = None
        self.ndvi_data = None
        self.ndvi_thumbnails_data = None
        self.rgb_ndvi_images_data = None
        self.gif_data = None
        self.mean_ndvi_data = None
        self.is_bare_soil_threshold = is_bare_soil_threshold
        logger.debug(f"Initializing HarmonizedNDVI client with collections: {collections}")

    def get_items(
        self,
        entire_image_cloud_cover_threshold: int = None,
    ) -> dict[str, list[pystac.Item]]:
        """
        Search for items from all collections within a given date range and polygon.

        Args:
            entire_image_cloud_cover_threshold: Maximum cloud cover percentage to include in the search

        Returns:
            Dictionary with collection names as keys and list of pystac.Item objects as values
        """
        if self.items is not None:
            return self.items

        if entire_image_cloud_cover_threshold is None:
            entire_image_cloud_cover_threshold = self.entire_image_cloud_cover_threshold
            
        gdf = self.gdf.to_crs(epsg=4326)
        items_by_collection = {}

        def fetch_items_for_collection(collection_polygon_pair):
            collection, polygon = collection_polygon_pair
            return collection, get_planetary_computer_items(
                collections=[collection],
                start_date=self.start_date,
                end_date=self.end_date,
                polygon=polygon,
                entire_image_cloud_cover_threshold=entire_image_cloud_cover_threshold,
            )

        # Fetch items for each collection and geometry combination
        for collection in self.collections:
            items_by_collection[collection] = []
            
            collection_polygon_pairs = [(collection, geom) for geom in gdf.geometry]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(
                    executor.map(fetch_items_for_collection, collection_polygon_pairs), 
                    total=len(collection_polygon_pairs),
                    desc=f"Fetching {collection} items"
                ))
                
            # Organize results by geometry index
            for i, (_, items_list) in enumerate(results):
                if i >= len(items_by_collection[collection]):
                    items_by_collection[collection].extend([[] for _ in range(i + 1 - len(items_by_collection[collection]))])
                items_by_collection[collection][i] = items_list

        self.items = items_by_collection
        return self.items

    def _get_bands_for_collection(self, collection: str, include_rgb: bool = False) -> list[str]:
        """Get the required bands for a collection"""
        mapping = self.BAND_MAPPINGS[collection]
        bands = [mapping["nir"], mapping["red"]]
        if include_rgb:
            bands.extend([mapping["green"], mapping["blue"]])
        if mapping["cloud_mask"]:
            bands.append(mapping["cloud_mask"])
        return bands

    def load_xarray(
        self,
        bands: dict[str, list[str]] = None,
        show_progress: bool = True,
        filter_cloud_cover: bool = True,
        include_rgb: bool = False,
    ) -> dict[str, list[xr.Dataset]]:
        """Load data for all collections into xarray Datasets.

        Args:
            bands: Dictionary with collection names as keys and band lists as values
            show_progress: Whether to show a progress bar
            filter_cloud_cover: Whether to filter the data based on cloud cover
            include_rgb: Whether to include RGB bands for visualization

        Returns:
            Dictionary with collection names as keys and lists of xarray Datasets as values
        """
        if self.xarray_data is not None:
            return self.xarray_data
            
        logger.debug(f"Loading data for collections {self.collections} at {self.resolution}m resolution")
        
        items_dict = self.get_items()
        
        if not any(items_dict.values()):
            logger.error("No items found for any collection in the given date range and polygon")
            raise ValueError("No items found for any collection in the given date range and polygon")

        self.xarray_data = {}
        
        for collection in self.collections:
            self.xarray_data[collection] = []
            items_list = items_dict.get(collection, [])
            
            if not items_list:
                logger.warning(f"No items found for collection {collection}")
                continue
                
            # Get bands for this collection
            if bands and collection in bands:
                collection_bands = bands[collection]
            else:
                collection_bands = self._get_bands_for_collection(collection, include_rgb)
            
            mapping = self.BAND_MAPPINGS[collection]
            
            for geometry_idx, (items, geometry) in enumerate(zip(items_list, self.gdf.geometry)):
                if not items:
                    logger.warning(f"No items for geometry {geometry_idx} in collection {collection}")
                    self.xarray_data[collection].append(None)
                    continue
                    
                # Sign the items to get access
                logger.debug(f"Signing {len(items)} items for collection {collection}")
                signed_items = [planetary_computer.sign(i) for i in items]

                thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

                # Load the data into an xarray Dataset
                logger.debug(f"Loading data into xarray Dataset for {collection}")
                progress = tqdm if show_progress else None
                
                try:
                    ds = odc.stac.load(
                        items=signed_items,
                        bands=collection_bands,
                        resolution=self.resolution,
                        pool=thread_pool,
                        geopolygon=geometry,
                        progress=progress,
                    )
                except Exception as e:
                    logger.error(f"Failed to load data for {collection}, geometry {geometry_idx}: {e}")
                    self.xarray_data[collection].append(None)
                    continue

                # Apply cloud filtering if requested and cloud mask is available
                if self.cropped_image_cloud_cover_threshold and filter_cloud_cover and mapping["cloud_mask"]:
                    logger.debug(f"Filtering data based on cloud cover using {mapping['cloud_mask']} band")
                    
                    if collection == "sentinel-2-l2a":
                        # For Sentinel-2, use SCL band
                        cloud_clear_mask = (ds[mapping["cloud_mask"]] == 4) | (ds[mapping["cloud_mask"]] == 5)
                    else:
                        # For HLS collections, use Fmask band
                        cloud_clear_mask = ds[mapping["cloud_mask"]] == 0
                    
                    cloud_cover_pct = (1 - cloud_clear_mask.mean(dim=["x", "y"])) * 100
                    logger.debug(f"Cloud cover percentage: {cloud_cover_pct}. Filtering based on {self.cropped_image_cloud_cover_threshold}% threshold")
                    
                    logger.debug(f"Dataset time steps before filtering: {len(ds.time)}")
                    ds = ds.sel(time=cloud_cover_pct <= self.cropped_image_cloud_cover_threshold)
                    logger.debug(f"Filtered dataset to {len(ds.time)} time steps")

                    # Apply spatial cloud masking
                    if filter_cloud_cover:
                        ds = ds.where(cloud_clear_mask, drop=False)

                logger.debug(f"Successfully loaded {collection} data")
                self.xarray_data[collection].append(ds)
                
        return self.xarray_data

    def load_ndvi_images(
        self,
        filter_cloud_cover: bool = True,
    ) -> dict[str, list[xr.DataArray]]:
        """Load NDVI data for all collections.

        Args:
            filter_cloud_cover: Whether to filter the data based on cloud cover

        Returns:
            Dictionary with collection names as keys and lists of NDVI DataArrays as values
        """
        if self.ndvi_data is not None:
            return self.ndvi_data
            
        logger.debug("Loading NDVI data for all collections")
        self.ndvi_data = {}
        
        xarray_data = self.load_xarray(filter_cloud_cover=filter_cloud_cover)
        
        for collection in self.collections:
            self.ndvi_data[collection] = []
            datasets = xarray_data.get(collection, [])
            mapping = self.BAND_MAPPINGS[collection]
            
            for ds in datasets:
                if ds is None:
                    self.ndvi_data[collection].append(None)
                    continue
                    
                logger.debug(f"Calculating NDVI for {collection} using bands {mapping['nir']} and {mapping['red']}")
                nir_band = ds[mapping["nir"]]
                red_band = ds[mapping["red"]]
                
                # Calculate NDVI
                ndvi = (nir_band - red_band) / (nir_band + red_band)
                # Mask out invalid values where denominator is zero
                ndvi = ndvi.where((nir_band + red_band) != 0)
                
                logger.debug(f"Successfully calculated NDVI for {collection}")
                self.ndvi_data[collection].append(ndvi)
                
        return self.ndvi_data

    def rgb_ndvi_images(
        self,
        vmin: float = -0.2,
        vmax: float = 0.8,
        boundary_color: str = "red",
        boundary_linewidth: float = 2,
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
        image_format: str = "png",
        timestamp_format: str = "%Y-%m-%d",
        figsize: tuple = (12, 6),
    ) -> dict[str, list[dict]]:
        """
        Generate side-by-side RGB and NDVI images for all collections and geometries.

        Args:
            vmin: Minimum value for NDVI color scale
            vmax: Maximum value for NDVI color scale
            boundary_color: Color of the polygon boundary
            boundary_linewidth: Line width of the polygon boundary
            bbox_inches: Bounding box setting for saving figure
            pad_inches: Padding when saving figure
            image_format: Format to save images in
            timestamp_format: Format string for timestamp keys
            figsize: Figure size as (width, height) tuple

        Returns:
            Dictionary with collection names as keys and lists of image dictionaries as values
        """
        if self.rgb_ndvi_images_data is not None:
            return self.rgb_ndvi_images_data
            
        logger.debug("Generating RGB+NDVI images for all collections")
        
        # Load data with RGB bands included
        xarray_data = self.load_xarray(include_rgb=True, filter_cloud_cover=True)
        ndvi_data = self.load_ndvi_images()
        
        self.rgb_ndvi_images_data = {}
        
        for collection in self.collections:
            self.rgb_ndvi_images_data[collection] = []
            datasets = xarray_data.get(collection, [])
            ndvi_datasets = ndvi_data.get(collection, [])
            mapping = self.BAND_MAPPINGS[collection]
            
            for geom_idx, (ds, ndvi, geometry) in enumerate(zip(datasets, ndvi_datasets, self.gdf.geometry)):
                images = {}
                
                if ds is None or ndvi is None:
                    self.rgb_ndvi_images_data[collection].append(images)
                    continue
                
                crs = ds.coords["spatial_ref"].values.item()
                
                for time in ds.time:
                    try:
                        # Create side-by-side plot
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                        
                        # RGB plot
                        rgb_data = ds.sel(time=time)[[mapping["red"], mapping["green"], mapping["blue"]]]
                        rgb_array = rgb_data.to_array().values
                        
                        # Normalize RGB values to 0-1 range for display
                        rgb_array = np.transpose(rgb_array, (1, 2, 0))
                        rgb_array = np.clip(rgb_array / np.percentile(rgb_array, 98), 0, 1)
                        
                        ax1.imshow(rgb_array)
                        ax1.set_title(f"RGB\n{pd.Timestamp(time.values).strftime(timestamp_format)}, {collection}")
                        ax1.set_axis_off()
                        
                        # Add boundary to RGB plot
                        gdf_crs = gpd.GeoDataFrame(
                            {"geometry": [geometry]}, geometry="geometry", crs=self.gdf.crs
                        ).to_crs(crs)
                        gdf_crs.boundary.plot(ax=ax1, color=boundary_color, linewidth=boundary_linewidth)
                        
                        # NDVI plot
                        ndvi_time = ndvi.sel(time=time)
                        ndvi_plot = ndvi_time.plot.imshow(
                            ax=ax2, cmap="RdYlGn", vmin=vmin, vmax=vmax, add_colorbar=False
                        )
                        ax2.set_title(f"NDVI\n{pd.Timestamp(time.values).strftime(timestamp_format)}, {collection}")
                        ax2.set_axis_off()
                        
                        # Add boundary to NDVI plot
                        gdf_crs.boundary.plot(ax=ax2, color=boundary_color, linewidth=boundary_linewidth)
                        
                        # Add colorbar for NDVI
                        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
                        fig.colorbar(ndvi_plot, cax=cbar_ax, label="NDVI")
                        
                        # Format axes
                        for ax in (ax1, ax2):
                            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                        
                        plt.tight_layout()
                        
                        # Save to bytes buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format=image_format, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=150)
                        buf.seek(0)
                        
                        # Add to dictionary with timestamp as key
                        timestamp = pd.Timestamp(time.values).strftime(timestamp_format)
                        images[timestamp] = buf.getvalue()
                        
                        plt.close(fig)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate image for {collection}, time {time}: {e}")
                        continue
                
                self.rgb_ndvi_images_data[collection].append(images)
                
        return self.rgb_ndvi_images_data

    def generate_ndvi_gif(
        self,
        collection: str = None,
        geometry_index: int = 0,
        duration: float = 0.5,
        loop: int = 0,
        vmin: float = -0.2,  
        vmax: float = 0.8,
        figsize: tuple = (12, 6),
    ) -> bytes:
        """
        Generate a GIF showing the time series of RGB and NDVI images.

        Args:
            collection: Collection to use for GIF generation. If None, uses first available collection
            geometry_index: Index of geometry to use for GIF generation
            duration: Duration of each frame in seconds
            loop: Number of loops (0 for infinite)
            vmin: Minimum value for NDVI color scale
            vmax: Maximum value for NDVI color scale
            figsize: Figure size as (width, height) tuple

        Returns:
            GIF as bytes
        """
        if collection is None:
            collection = self.collections[0]
        
        if collection not in self.collections:
            raise ValueError(f"Collection {collection} not in available collections: {self.collections}")
        
        logger.debug(f"Generating GIF for collection {collection}, geometry {geometry_index}")
        
        # Get RGB+NDVI images
        rgb_ndvi_images = self.rgb_ndvi_images(vmin=vmin, vmax=vmax, figsize=figsize)
        
        if collection not in rgb_ndvi_images:
            raise ValueError(f"No data available for collection {collection}")
        
        if geometry_index >= len(rgb_ndvi_images[collection]):
            raise ValueError(f"Geometry index {geometry_index} out of range")
        
        images_dict = rgb_ndvi_images[collection][geometry_index]
        
        if not images_dict:
            raise ValueError(f"No images available for collection {collection}, geometry {geometry_index}")
        
        # Sort images by timestamp
        sorted_timestamps = sorted(images_dict.keys())
        frames = []
        
        for timestamp in sorted_timestamps:
            image_bytes = images_dict[timestamp]
            frame = imageio.imread(io.BytesIO(image_bytes))
            frames.append(frame)
        
        # Create GIF in memory
        gif_buffer = io.BytesIO()
        imageio.mimsave(gif_buffer, frames, format="GIF", duration=duration, loop=loop)
        gif_buffer.seek(0)
        
        logger.debug(f"Successfully generated GIF with {len(frames)} frames")
        return gif_buffer.getvalue()

    def ndvi_thumbnails(
        self,
        vmin: float = -0.2,
        vmax: float = 0.8,
        boundary_color: str = "red",
        boundary_linewidth: float = 2,
        add_colorbar: bool = False,
        add_labels: bool = False,
        bbox_inches: str = "tight",
        pad_inches: float = 0,
        image_format: str = "jpg",
        timestamp_format: str = "%Y-%m-%d",
    ) -> dict[str, list[dict]]:
        """
        Plot NDVI images and return them as jpgs in a dictionary

        Args:
            vmin: Minimum value for NDVI color scale
            vmax: Maximum value for NDVI color scale
            boundary_color: Color of the polygon boundary
            boundary_linewidth: Line width of the polygon boundary
            add_colorbar: Whether to add a colorbar to the plot
            add_labels: Whether to add labels to the plot
            bbox_inches: Bounding box setting for saving figure
            pad_inches: Padding when saving figure
            image_format: Format to save images in
            timestamp_format: Format string for timestamp keys

        Returns:
            Dictionary with collection names as keys and lists of image dictionaries as values
        """
        if self.ndvi_thumbnails_data is not None:
            return self.ndvi_thumbnails_data
            
        ndvi_data = self.load_ndvi_images()
        self.ndvi_thumbnails_data = {}

        for collection in self.collections:
            self.ndvi_thumbnails_data[collection] = []
            ndvi_list = ndvi_data.get(collection, [])
            
            for ndvi, geometry in zip(ndvi_list, self.gdf.geometry):
                images = {}
                
                if ndvi is None:
                    self.ndvi_thumbnails_data[collection].append(images)
                    continue
                
                crs = ndvi.coords["spatial_ref"].values.item()

                for time in ndvi.time:
                    try:
                        # Create new figure for each timestamp
                        fig, ax = plt.subplots()

                        # Plot NDVI data and polygon boundary
                        ndvi.sel(time=time).plot(
                            ax=ax,
                            vmin=vmin,
                            vmax=vmax,
                            add_colorbar=add_colorbar,
                            add_labels=add_labels,
                        )
                        gpd.GeoDataFrame(
                            {"geometry": [geometry]}, geometry="geometry", crs=self.gdf.crs
                        ).to_crs(crs).boundary.plot(
                            ax=ax, color=boundary_color, linewidth=boundary_linewidth
                        )
                        ax.set_axis_off()

                        # Save plot to bytes buffer
                        buf = io.BytesIO()
                        plt.savefig(
                            buf,
                            format=image_format,
                            bbox_inches=bbox_inches,
                            pad_inches=pad_inches,
                        )
                        buf.seek(0)

                        # Add to dictionary with timestamp as key
                        timestamp = pd.Timestamp(time.values).strftime(timestamp_format)
                        images[timestamp] = buf.getvalue()

                        # Close figure to free memory
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Failed to generate thumbnail for {collection}, time {time}: {e}")
                        continue
                        
                self.ndvi_thumbnails_data[collection].append(images)
                
        return self.ndvi_thumbnails_data

    def calculate_mean_ndvi(
        self,
        interpolate: bool = True,
        all_touched: bool = True,
    ) -> dict[str, list[pd.DataFrame]]:
        """
        Calculate mean NDVI value for the given polygon at each timestamp

        Args:
            interpolate (bool): Whether to interpolate missing values
            all_touched (bool): Whether to use all touched for clipping
        Returns:
            Dictionary with collection names as keys and lists of DataFrames with mean NDVI values as values
        """
        if self.mean_ndvi_data is not None:
            return self.mean_ndvi_data
            
        logger.debug("Calculating mean NDVI values for all collections")

        ndvi_data = self.load_ndvi_images()
        self.mean_ndvi_data = {}

        for collection in self.collections:
            self.mean_ndvi_data[collection] = []
            ndvi_images_list = ndvi_data.get(collection, [])
            
            for ndvi_images, geometry in zip(ndvi_images_list, self.gdf.geometry):
                if ndvi_images is None:
                    # Create empty dataframe for missing data
                    empty_df = pd.DataFrame(columns=["ndvi"])
                    if self.is_bare_soil_threshold:
                        empty_df["is_bare_soil"] = []
                    self.mean_ndvi_data[collection].append(empty_df)
                    continue
                    
                # Convert to rioxarray and clip once for all timestamps
                crs = ndvi_images.coords["spatial_ref"].values.item()
                ndvi_images = ndvi_images.rio.write_crs(crs)
                clipped_data = ndvi_images.rio.clip(
                    [geometry], self.gdf.crs, all_touched=all_touched
                )

                # Calculate means for all timestamps at once
                mean_values = clipped_data.mean(dim=["x", "y"]).values

                # Create dictionary mapping timestamps to means
                mean_ndvi = pd.DataFrame(
                    mean_values, columns=["ndvi"], index=clipped_data.time.values
                )
                mean_ndvi.index = pd.to_datetime(mean_ndvi.index).date
                if interpolate:
                    mean_ndvi = interpolate_ndvi(mean_ndvi, self.start_date, self.end_date)

                if self.is_bare_soil_threshold:
                    mean_ndvi["is_bare_soil"] = mean_ndvi["ndvi"] < self.is_bare_soil_threshold

                logger.debug(f"Calculated mean NDVI for {collection} with {len(mean_ndvi)} timestamps")
                self.mean_ndvi_data[collection].append(mean_ndvi)
                
        return self.mean_ndvi_data

    def get_data(
        self,
        all_touched: bool = True,
        interpolate: bool = True,
    ) -> dict[str, list[list[dict]]]:
        """Get mean NDVI values for all collections and geometries"""
        mean_ndvi_data = self.calculate_mean_ndvi(
            interpolate=interpolate,
            all_touched=all_touched,
        )
        output = {}
        
        for collection in self.collections:
            output[collection] = []
            mean_ndvi_df_list = mean_ndvi_data.get(collection, [])
            
            for mean_ndvi_df in mean_ndvi_df_list:
                if mean_ndvi_df.empty:
                    output[collection].append([])
                    continue
                    
                mean_ndvi_df_copy = mean_ndvi_df.reset_index(names="date").copy()
                mean_ndvi_dict = mean_ndvi_df_copy.to_dict(orient="records")
                
                for record in mean_ndvi_dict:
                    if 'ndvi' in record:
                        if pd.isna(record['ndvi']):
                            record.pop("ndvi")
                        else:
                            record['ndvi'] = round(record['ndvi'], 2)
                    if 'interpolated_ndvi' in record:
                        record['interpolated_ndvi'] = round(record['interpolated_ndvi'], 2)
                        
                output[collection].append(mean_ndvi_dict)
                
        return output


def interpolate_ndvi(df: pd.DataFrame, start_date: str, end_date: str):
    """
    Create a DataFrame from NDVI values, interpolate missing dates, and plot the results.

    Args:
        mean_ndvi_values (dict): Dictionary of dates and NDVI values
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame with interpolated daily NDVI values
    """
    date_range = pd.date_range(
        start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="D"
    )
    df = df.reindex(date_range)
    df["interpolated_ndvi"] = df["ndvi"].interpolate(method="linear", limit_direction="both")
    return df


# Backward compatibility alias
class Sentinel2(HarmonizedNDVI):
    """
    Backward compatibility class for Sentinel2. 
    This is now an alias for HarmonizedNDVI with Sentinel-2 as the default collection.
    """
    def __init__(
        self,
        start_date: str,
        end_date: str,
        gdf: gpd.GeoDataFrame,
        resolution: int = 10,
        entire_image_cloud_cover_threshold: int = 10,
        cropped_image_cloud_cover_threshold: int = 80,
        max_workers: int = 10,
        is_bare_soil_threshold: float = 0.25,
    ):
        # Call parent with Sentinel-2 collection only for backward compatibility
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            gdf=gdf,
            collections=["sentinel-2-l2a"],
            resolution=resolution,
            entire_image_cloud_cover_threshold=entire_image_cloud_cover_threshold,
            cropped_image_cloud_cover_threshold=cropped_image_cloud_cover_threshold,
            max_workers=max_workers,
            is_bare_soil_threshold=is_bare_soil_threshold,
        )
        
    def get_data(self, all_touched: bool = True, interpolate: bool = True) -> list[list[dict]]:
        """Get mean NDVI values for Sentinel-2 data (backward compatibility method)"""
        data = super().get_data(all_touched=all_touched, interpolate=interpolate)
        # Return just the Sentinel-2 data for backward compatibility
        return data.get("sentinel-2-l2a", [])
