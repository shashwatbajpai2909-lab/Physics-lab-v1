from __future__ import annotations
from dataclasses import dataclass
from .dimensions import Dimension, DIMENSIONLESS
from .units import Unit


class DimensionError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def to(self, target: Unit) -> "Quantity":
        if self.dimension != target.dimension:
            raise DimensionError(
                f"Cannot convert {self.unit.symbol} to {target.symbol}: incompatible dimensions"
            )
        si_value = self.value * self.unit.scale_to_si
        return Quantity(si_value / target.scale_to_si, target)

    def __add__(self, other: "Quantity") -> "Quantity":
        if self.dimension != other.dimension:
            raise DimensionError("Addition requires equal dimensions")
        converted = other.to(self.unit)
        return Quantity(self.value + converted.value, self.unit)

    def __sub__(self, other: "Quantity") -> "Quantity":
        if self.dimension != other.dimension:
            raise DimensionError("Subtraction requires equal dimensions")
        converted = other.to(self.unit)
        return Quantity(self.value - converted.value, self.unit)

    def __mul__(self, other):
        if isinstance(other, Quantity):
            dimension = self.dimension * other.dimension
            unit = Unit(
                f"{self.unit.name}*{other.unit.name}",
                f"{self.unit.symbol}*{other.unit.symbol}",
                dimension,
                self.unit.scale_to_si * other.unit.scale_to_si,
            )
            return Quantity(self.value * other.value, unit)
        return Quantity(self.value * other, self.unit)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            dimension = self.dimension / other.dimension
            unit = Unit(
                f"{self.unit.name}/{other.unit.name}",
                f"{self.unit.symbol}/{other.unit.symbol}",
                dimension,
                self.unit.scale_to_si / other.unit.scale_to_si,
            )
            return Quantity(self.value / other.value, unit)
        return Quantity(self.value / other, self.unit)

    @staticmethod
    def dimensionless(value: float) -> "Quantity":
        unit = Unit("dimensionless", "1", DIMENSIONLESS)
        return Quantity(value, unit)
