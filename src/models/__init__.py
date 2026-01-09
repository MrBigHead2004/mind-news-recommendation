from typing import Dict, Type
from .base import BaseNewsRecommender

# Model registry
_REGISTRY: Dict[str, Type[BaseNewsRecommender]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_model(name: str, config: dict) -> BaseNewsRecommender:
    """Factory function to create a model by name."""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](config)


def list_models() -> list:
    """Return list of available model names."""
    return list(_REGISTRY.keys())


# Import models to trigger registration
from .nrms import NRMS
from .miner import MINER
