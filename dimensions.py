from __future__ import annotations
from dataclasses import dataclass
from ..dimensions import Dimension, DIMENSIONLESS


class SymbolicDimensionError(ValueError):
    pass


@dataclass(frozen=True)
class DimensionalContext:
    symbols: dict[str, Dimension]
    strict: bool = False

    def dimension_of(self, name: str) -> Dimension:
        if name in self.symbols:
            return self.symbols[name]
        if self.strict:
            raise SymbolicDimensionError(
                f"Unknown symbol {name!r} in strict dimensional context"
            )
        return DIMENSIONLESS


def infer_dimension(expr, context: DimensionalContext) -> Dimension:
    from .expr import Constant, Symbol, Add, Mul, Pow, Sin, Cos, Exp, Log

    if isinstance(expr, Constant):
        return DIMENSIONLESS

    if isinstance(expr, Symbol):
        return context.dimension_of(expr.name)

    if isinstance(expr, Add):
        left = infer_dimension(expr.left, context)
        right = infer_dimension(expr.right, context)
        if left != right:
            raise SymbolicDimensionError(
                f"Cannot add expressions with dimensions {left} and {right}"
            )
        return left

    if isinstance(expr, Mul):
        return infer_dimension(expr.left, context) * infer_dimension(expr.right, context)

    if isinstance(expr, Pow):
        base = infer_dimension(expr.base, context)
        exponent = expr.exponent
        if not isinstance(exponent, Constant):
            if base != DIMENSIONLESS:
                raise SymbolicDimensionError(
                    "A dimensional quantity cannot have a symbolic exponent"
                )
            return DIMENSIONLESS
        if not isinstance(exponent.value, int):
            raise SymbolicDimensionError(
                "Non-integer powers of dimensional quantities are not supported"
            )
        return base ** exponent.value

    if isinstance(expr, (Sin, Cos, Exp, Log)):
        arg_dim = infer_dimension(expr.arg, context)
        if arg_dim != DIMENSIONLESS:
            raise SymbolicDimensionError(
                f"{type(expr).__name__} requires a dimensionless argument, got {arg_dim}"
            )
        return DIMENSIONLESS

    raise TypeError(f"Unsupported symbolic node: {type(expr).__name__}")


def validate_dimension(expr, expected: Dimension, context: DimensionalContext) -> None:
    actual = infer_dimension(expr, context)
    if actual != expected:
        raise SymbolicDimensionError(
            f"Expected dimension {expected}, inferred {actual}"
        )


def derivative_dimension(expr, variable, context: DimensionalContext) -> Dimension:
    """Return the physical dimension of ∂expr/∂variable."""
    expr_dim = infer_dimension(expr, context)
    variable_dim = infer_dimension(variable, context)
    return expr_dim / variable_dim


def differentiate_with_dimension(expr, variable, context: DimensionalContext):
    """Differentiate an expression and verify the derivative's dimension."""
    from .expr import differentiate, simplify

    derivative = simplify(differentiate(expr, variable))
    expected = derivative_dimension(expr, variable, context)
    # Zero is dimension-polymorphic: the derivative of an expression that
    # does not depend on the variable is physically valid at any target dimension.
    from .expr import Constant
    if isinstance(derivative, Constant) and derivative.value == 0:
        return derivative, expected
    actual = infer_dimension(derivative, context)
    if actual != expected:
        raise SymbolicDimensionError(
            f"Derivative dimension mismatch for ∂{expr}/∂{variable}: "
            f"expected {expected}, inferred {actual}"
        )
    return derivative, expected
