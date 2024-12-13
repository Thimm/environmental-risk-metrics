# Environmental Risk Metrics

This project is a Python package that calculates environmental risk metrics for a given polygon.

## Installation

```bash
pip install environmental-risk-metrics
```

## Usage

```python
from environmental_risk_metrics import Sentinel2

# Example usage
polygon = {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              -38.73370947430459,
              -6.143308439590044
            ],
            [
              -38.73370947430459,
              -6.144138579435889
            ],
            [
              -38.7327974681142,
              -6.144138579435889
            ],
            [
              -38.7327974681142,
              -6.143308439590044
            ],
            [
              -38.73370947430459,
              -6.143308439590044
            ]
          ]
        ],
        "type": "Polygon"
    }
}

sentinel2 = Sentinel2()
sentinel2.load_ndvi_images(
    start_date="2024-06-27",
    end_date="2024-12-31",
    polygon=polygon,
    cropped_image_cloud_cover_threshold=80,
    entire_image_cloud_cover_threshold=20,
)
```
