class EnvironmentalRiskMetricsError(Exception):
    """Base exception class for environmental risk metrics"""
    pass

class DataNotFoundError(EnvironmentalRiskMetricsError):
    """Raised when required data is not found"""
    pass

class ValidationError(EnvironmentalRiskMetricsError):
    """Raised when input validation fails"""
    pass 

class NotImplementedError(EnvironmentalRiskMetricsError):
    """Raised when a method is not implemented"""
    pass
