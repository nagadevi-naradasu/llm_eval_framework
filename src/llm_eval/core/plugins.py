import importlib
from typing import Any, Dict
from ..metrics.base import Metric

def load_custom_metric(class_path: str, args: Dict[str, Any]) -> Metric:
    """
    Dynamically loads a metric class from a string path.
    e.g. "my_package.metrics.MyMetric"
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**args)
    except Exception as e:
        raise ValueError(f"Failed to load custom metric '{class_path}': {e}")
