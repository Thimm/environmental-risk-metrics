import logging

from .metrics.endangered_species import EndangeredSpecies
from .metrics.land_use_change import EsaLandCover, EsriLandCover, OpenLandMapLandCover
from .metrics.ndvi import Sentinel2
from .metrics.protected_areas import RamsarProtectedAreas
from .metrics.social_indices import GlobalWitness
from .metrics.soil_organic_carbon import SoilOrganicCarbon, SoilOrganicCarbonPotential
from .metrics.soil_types import SoilTypes
from .utils.metric_calculator import calculate_metrics

# Create a null handler to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Sentinel2",
    "EsaLandCover",
    "EsriLandCover",
    "OpenLandMapLandCover",
    "SoilOrganicCarbon",
    "SoilTypes",
    "EndangeredSpecies",
    "RamsarProtectedAreas",
    "GlobalWitness",
    "calculate_metrics",
    "SoilOrganicCarbonPotential",
]
    