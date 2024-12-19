from abc import abstractmethod
from typing import Dict

from environmental_risk_metrics.utils import ensure_geometry_crs


class BaseEnvironmentalMetric:
    """Base class for environmental metrics"""
    
    # Default CRS for all metrics unless overridden
    DEFAULT_CRS = "EPSG:4326"
    
    def __init__(self, target_crs: str = None, **kwargs):
        """
        Initialize the metric with optional CRS override
        
        Args:
            target_crs: Override default CRS for this instance
            **kwargs: Additional initialization parameters
        """
        self.target_crs = target_crs or self.DEFAULT_CRS
        super().__init__(**kwargs)

    def _preprocess_geometry(self, geometry: dict, source_crs: str, target_crs: str = None) -> dict:
        """
        Preprocess geometry to ensure consistent format and CRS
        
        Args:
            geometry: Input geometry
            target_crs: Optional CRS override for this specific operation
            source_crs: Optional source CRS override for this specific operation
        """
        return ensure_geometry_crs(geometry=geometry, source_crs=source_crs, target_crs=target_crs or self.target_crs)
    
    @abstractmethod
    def get_data(self, geometry: dict, **kwargs) -> Dict:
        """Get data for a given geometry"""
        geometry = self._preprocess_geometry(geometry)
        pass 