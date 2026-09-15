"""fedmed/privacy package initialization."""

try:
    from .tenseal_engine import TenSEALEngine, check_tenseal
except ImportError:
    TenSEALEngine = None  # Fallback when C++ bindings are not compiled locally

    def check_tenseal() -> bool:
        return False


from .secagg_config import SecAggPlusConfig
from .secure_aggregation import SecureAggregationManager

__all__ = [
    "TenSEALEngine",
    "check_tenseal",
    "SecAggPlusConfig",
    "SecureAggregationManager",
]