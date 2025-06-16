# HarmonizedNDVI - Multi-Collection NDVI Analysis

The `HarmonizedNDVI` class provides comprehensive NDVI analysis capabilities across multiple satellite collections including Sentinel-2, HLS2-S30, and HLS2-L30 from Microsoft's Planetary Computer.

## Features

### Multi-Collection Support
- **Sentinel-2**: High resolution optical imagery (10-60m)
- **HLS2-S30**: Harmonized Landsat Sentinel-2 derived from Sentinel-2 (30m)
- **HLS2-L30**: Harmonized Landsat Sentinel-2 derived from Landsat 8/9 (30m)

### Visualization Capabilities
- **NDVI Thumbnails**: Generate individual NDVI images for each timestamp
- **RGB+NDVI Images**: Side-by-side RGB and NDVI visualizations
- **Animated GIFs**: Time series animations showing vegetation changes

### Data Analysis
- **Mean NDVI Calculation**: Compute spatially averaged NDVI values
- **Time Series Interpolation**: Fill gaps in temporal data
- **Cloud Filtering**: Automatic cloud masking using collection-specific methods

## Quick Start

```python
import geopandas as gpd
from environmental_risk_metrics.metrics.ndvi import HarmonizedNDVI

# Create your area of interest
gdf = gpd.read_file("your_polygons.geojson")

# Initialize with multiple collections
ndvi = HarmonizedNDVI(
    start_date="2023-01-01",
    end_date="2023-12-31",
    gdf=gdf,
    collections=["sentinel-2-l2a", "hls2-s30", "hls2-l30"],
    resolution=30
)

# Get mean NDVI data
mean_ndvi = ndvi.calculate_mean_ndvi()

# Generate RGB+NDVI images
rgb_ndvi_images = ndvi.rgb_ndvi_images()

# Create animated GIF
gif_bytes = ndvi.generate_ndvi_gif(collection="hls2-s30")
with open("ndvi_animation.gif", "wb") as f:
    f.write(gif_bytes)
```

## Collection-Specific Band Mappings

Each collection uses different band names for the same spectral regions:

| Collection | Red | Green | Blue | NIR | Cloud Mask |
|------------|-----|-------|------|-----|------------|
| sentinel-2-l2a | B04 | B03 | B02 | B08 | SCL |
| hls2-s30 | B04 | B03 | B02 | B08 | Fmask |
| hls2-l30 | B04 | B03 | B02 | B05 | Fmask |

## Advanced Usage

### Custom Band Selection
```python
# Specify custom bands for each collection
custom_bands = {
    "sentinel-2-l2a": ["B04", "B08", "B02", "B03", "SCL"],
    "hls2-s30": ["B04", "B08", "B02", "B03", "Fmask"]
}

xarray_data = ndvi.load_xarray(bands=custom_bands, include_rgb=True)
```

### Cloud Filtering Options
```python
ndvi = HarmonizedNDVI(
    # ... other parameters ...
    entire_image_cloud_cover_threshold=10,  # Skip scenes with >10% cloud cover
    cropped_image_cloud_cover_threshold=50,  # Filter time steps with >50% clouds in AOI
)
```

### GIF Customization
```python
gif_bytes = ndvi.generate_ndvi_gif(
    collection="hls2-s30",
    geometry_index=0,
    duration=0.5,  # 0.5 seconds per frame
    loop=0,        # Infinite loop
    vmin=-0.2,     # NDVI color scale minimum
    vmax=0.8,      # NDVI color scale maximum
    figsize=(14, 8)  # Figure size
)
```

## Backward Compatibility

The original `Sentinel2` class is maintained for backward compatibility:

```python
from environmental_risk_metrics.metrics.ndvi import Sentinel2

# Old code continues to work
sentinel2 = Sentinel2(
    start_date="2023-01-01",
    end_date="2023-12-31", 
    gdf=gdf
)
data = sentinel2.get_data()
```

## Error Handling

The class gracefully handles:
- Missing data for specific collections/geometries
- Cloud-covered scenes
- Network timeouts
- Invalid geometries

Failed operations are logged and return empty results rather than crashing.

## Performance Tips

1. **Use appropriate resolution**: Higher resolutions (10m) require more memory and processing time
2. **Limit time ranges**: Shorter time periods reduce data volume and processing time
3. **Adjust worker count**: Set `max_workers` based on your system capabilities
4. **Filter cloud cover**: Use cloud cover thresholds to reduce processing of poor-quality data

## Dependencies

- `imageio` (for GIF generation)
- `matplotlib` (for visualization)
- `xarray` (for data handling)
- `odc-stac` (for data loading)
- `planetary-computer` (for data access)
- `geopandas` (for geometry handling) 