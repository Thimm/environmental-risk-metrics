from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Type


def calculate_metrics(
    metric_classes: List[Type],
    polygon: dict,
    start_date: str,
    end_date: str,
    polygon_crs: str = "EPSG:4326",
    timeout: float = None
) -> Dict[str, Any]:
    """
    Calculate multiple environmental metrics in parallel.
    
    Args:
        metric_classes: List of metric classes to calculate
        polygon: GeoJSON polygon
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        polygon_crs: CRS of the input polygon
        timeout: Maximum time in seconds to wait for each metric
    
    Returns:
        Dictionary with metric names as keys and their results as values
    """
    def calculate_single_metric(metric_class: Type) -> tuple[str, Any]:
        try:
            metric = metric_class()
            result = metric.get_data(
                polygon=polygon,
                start_date=start_date,
                end_date=end_date,
                polygon_crs=polygon_crs
            )
            return metric_class.__name__, result
        except Exception as e:
            return metric_class.__name__, {"error": str(e)}

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_single_metric, metric_class)
            for metric_class in metric_classes
        ]
        
        return dict(
            future.result(timeout=timeout)
            for future in futures
        )