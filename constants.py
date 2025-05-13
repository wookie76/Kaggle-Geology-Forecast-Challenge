"""
Global constants, Enums, and static data structures.
"""

from enum import Enum, auto
from typing import Dict, List, Any


# --- Enums ---
class ModelType(Enum):
    """Enumeration for model types used in Stage 1."""

    XGBOOST = "xgb"
    RANDOM_FOREST = "rf"


class TrendType(Enum):
    """Defines types of trends that can be applied to realizations."""

    NONE = auto()
    RISING = auto()
    FALLING = auto()
    RISING_LATE = auto()
    FALLING_LATE = auto()
    OSCILLATE_SHORT = auto()
    OSCILLATE_LONG = auto()


class CovarianceRegion(Enum):
    """Defines regions for idealized covariance curve fitting."""

    EXPONENTIAL_GROWTH = 0
    STABLE_ERRORS = 1
    FURTHER_GROWTH = 2


class InputSignalType(Enum):
    """Characterization of input signal complexity for realization strategies."""

    SMOOTH = auto()
    COMPLEX = auto()
    POTENTIALLY_DISCONTINUOUS = auto()


# --- Model Parameters & Data Structures ---

# This was OptimizedModelParams dataclass. Using Dict for simplicity if it's just a container
# passed around. If it had methods or more complex validation, a Pydantic model or dataclass is good.
# For now, assuming its primary use is as a structured dictionary.
OptimizedModelParams = Dict[
    str, Dict[str, Any]
]  # e.g., {"xgb_params": {...}, "rf_params": {...}}


# Realization Strategy Configurations
# This list would typically be defined where it's used or in a config if highly dynamic.
# For modularity, it's fine here. Can be imported by the realization generation module.
REALIZATION_STRATEGIES_CONFIG_LIST: List[Dict[str, Any]] = [
    {
        "name": "hifi",
        "scale": 0.8,
        "corr": 0.8,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": True,
    },
    {
        "name": "lowvar",
        "scale": 0.6,
        "corr": 1.2,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
    },
    {
        "name": "highvar",
        "scale": 1.0,
        "corr": 0.6,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
    },
    {
        "name": "rise",
        "scale": 0.8,
        "corr": 1.0,
        "trend": TrendType.RISING,
        "t_amp": 1.5,
        "smooth": False,
    },
    {
        "name": "fall",
        "scale": 0.8,
        "corr": 1.0,
        "trend": TrendType.FALLING,
        "t_amp": 1.5,
        "smooth": False,
    },
    {
        "name": "osc_short",
        "scale": 0.7,
        "corr": 0.5,
        "trend": TrendType.OSCILLATE_SHORT,
        "t_amp": 1.0,
        "smooth": False,
    },
    {
        "name": "osc_long",
        "scale": 0.7,
        "corr": 1.5,
        "trend": TrendType.OSCILLATE_LONG,
        "t_amp": 1.2,
        "smooth": True,
    },
    {
        "name": "fault1",
        "scale": 0.7,
        "corr": 0.7,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
        "is_fault": True,
        "f_min": 0.5,
        "f_max": 2.0,
    },
    {
        "name": "v_short_corr",
        "scale": 0.9,
        "corr": 0.15,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
    },
    {
        "name": "v_long_corr",
        "scale": 0.7,
        "corr": 4.0,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": True,
    },
    {
        "name": "laplace_noise",
        "scale": 0.8,
        "corr": 0.7,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
        "use_laplacian": True,
    },
    {
        "name": "median_filter",
        "scale": 0.8,
        "corr": 0.0,
        "trend": TrendType.NONE,
        "t_amp": 0.0,
        "smooth": False,
        "use_median": True,
        "median_kernel_factor": 0.05,
    },
]
