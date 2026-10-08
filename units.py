from __future__ import annotations
from dataclasses import dataclass
from .dimensions import Dimension


@dataclass(frozen=True)
class Unit:
    name: str
    symbol: str
    dimension: Dimension
    scale_to_si: float = 1.0

    def compatible_with(self, other: "Unit") -> bool:
        return self.dimension == other.dimension


METER = Unit("meter", "m", Dimension(length=1))
SECOND = Unit("second", "s", Dimension(time=1))
KILOGRAM = Unit("kilogram", "kg", Dimension(mass=1))
AMPERE = Unit("ampere", "A", Dimension(current=1))
KELVIN = Unit("kelvin", "K", Dimension(temperature=1))
NEWTON = Unit("newton", "N", Dimension(mass=1, length=1, time=-2))
JOULE = Unit("joule", "J", Dimension(mass=1, length=2, time=-2))
WATT = Unit("watt", "W", Dimension(mass=1, length=2, time=-3))
