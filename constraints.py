from __future__ import annotations
from dataclasses import dataclass
from .dimensions import Dimension, DIMENSIONLESS
from .symbolic.expr import Expr, Symbol, simplify


@dataclass(frozen=True)
class Constraint:
    """A holonomic symbolic constraint represented by g(q, parameters) = 0."""
    name: str
    expression: Expr

    def residual(self) -> Expr:
        return simplify(self.expression)


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """Constraint requiring expression == 0."""
    pass


@dataclass(frozen=True)
class CoordinateConstraint(Constraint):
    """Constraint attached to a named generalized coordinate."""
    coordinate: str = ""


@dataclass(frozen=True)
class LagrangeMultiplier:
    """Multiplier used to enforce a constraint in an augmented Lagrangian."""
    name: str
    symbol: Symbol
    dimension: Dimension = DIMENSIONLESS
    constraint_name: str = ""
