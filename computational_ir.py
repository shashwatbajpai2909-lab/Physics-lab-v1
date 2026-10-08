from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from .dimensions import Dimension, DIMENSIONLESS
from .symbolic.expr import Expr


class EquationKind(str, Enum):
    DIFFERENTIAL = "differential"
    ALGEBRAIC = "algebraic"


@dataclass(frozen=True)
class StateVariable:
    name: str
    dimension: Dimension = DIMENSIONLESS
    derivative_order: int = 0


@dataclass(frozen=True)
class EquationOfMotion:
    coordinate: str
    acceleration: str
    residual: Expr
    kind: EquationKind = EquationKind.DIFFERENTIAL
    derivative_order: int = 2
    residual_dimension: Dimension = DIMENSIONLESS


@dataclass(frozen=True)
class AlgebraicEquation:
    name: str
    expression: Expr
    kind: EquationKind = EquationKind.ALGEBRAIC
    dimension: Dimension = DIMENSIONLESS


@dataclass(frozen=True)
class ComputationalIR:
    model_name: str
    equations: Tuple[EquationOfMotion, ...]
    algebraic_equations: Tuple[AlgebraicEquation, ...] = ()
    state_variables: Tuple[StateVariable, ...] = ()
