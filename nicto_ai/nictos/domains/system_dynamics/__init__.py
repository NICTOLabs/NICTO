"""System-dynamics domain — populations, epidemiology, ecology, markets.

Entities are *stocks* (no position, no velocity).  Rules compute
attribute-value changes each step via :class:`Effect` with *attr* + *delta*.
"""
from nicto_ai.nictos.domains.system_dynamics.types import Stock
from nicto_ai.nictos.domains.system_dynamics.rules import LotkaVolterraRule, SIRRule, ExponentialGrowthRule

__all__ = [
    "Stock",
    "LotkaVolterraRule",
    "SIRRule",
    "ExponentialGrowthRule",
]
