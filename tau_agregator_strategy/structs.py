from dataclasses import dataclass, field
from fractal.core.base import (BaseStrategyParams)
from typing import List

@dataclass
class ModelPrediction:
    """Prediction vector from the model"""
    r: float  # probability of rebalancing necessity [0,1]
    c: float  # range center (price)
    w: float  # range width (tau value)


@dataclass
class ModelWeight:
    """Model weight in the ensemble"""
    alpha: float = 1.0  # Model weight
    history: List[float] = field(default_factory=list)  # Weight history


@dataclass
class TauEnsembleParams(BaseStrategyParams):
    """
    Parameters for ensemble strategy:
    - TAU: Base tau value for the main strategy
    - INITIAL_BALANCE: Initial balance for liquidity placement
    """
    TAU: float
    INITIAL_BALANCE: float