from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from .dimensions import Dimension, DIMENSIONLESS
from .symbolic.expr import Expr, Symbol, as_expr
from .constraints import Constraint


@dataclass(frozen=True)
class Parameter:
    name: str
    symbol: Symbol
    dimension: Dimension = DIMENSIONLESS


@dataclass(frozen=True)
class Coordinate:
    name: str
    symbol: Symbol
    dimension: Dimension = DIMENSIONLESS


@dataclass(frozen=True)
class LagrangianModel:
    name: str
    coordinates: Tuple[Coordinate, ...]
    lagrangian: Expr
    lagrangian_dimension: Dimension = DIMENSIONLESS
    constraints: Tuple[Constraint, ...] = ()
    parameters: Tuple[Parameter, ...] = ()

    @staticmethod
    def from_symbols(
        name: str,
        coordinates: Tuple[str, ...],
        lagrangian: Expr,
        dimensions: Tuple[Dimension, ...] | None = None,
        constraints: Tuple[Constraint, ...] = (),
        lagrangian_dimension: Dimension = DIMENSIONLESS,
        parameters: Tuple[tuple[str, Dimension], ...] = (),
    ):
        if dimensions is None:
            dimensions = (DIMENSIONLESS,) * len(coordinates)
        if len(dimensions) != len(coordinates):
            raise ValueError("dimensions must match coordinates")

        return LagrangianModel(
            name=name,
            coordinates=tuple(
                Coordinate(q, Symbol(q), d)
                for q, d in zip(coordinates, dimensions)
            ),
            lagrangian=as_expr(lagrangian),
            lagrangian_dimension=lagrangian_dimension,
            constraints=tuple(constraints),
            parameters=tuple(
                Parameter(name, Symbol(name), dimension)
                for name, dimension in parameters
            ),
        )
