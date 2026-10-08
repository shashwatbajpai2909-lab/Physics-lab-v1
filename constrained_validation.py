from __future__ import annotations

from .constraints import Constraint
from .constrained import build_multipliers, constrained_euler_lagrange
from .dimension_validation import coordinate_context
from .dimensions import Dimension
from .physics_ir import LagrangianModel
from .symbolic.dimensions import DimensionalContext, infer_dimension, validate_dimension


def constraint_dimensions(model: LagrangianModel, context: DimensionalContext) -> tuple[Dimension, ...]:
    return tuple(infer_dimension(c.residual(), context) for c in model.constraints)


def validate_constraints_dimensions(model: LagrangianModel) -> tuple[Dimension, ...]:
    """Validate every constraint and its multiplier contribution dimension."""
    context = coordinate_context(model)
    dimensions = constraint_dimensions(model, context)
    multipliers = build_multipliers(model, dimensions)

    for constraint, dimension, multiplier in zip(model.constraints, dimensions, multipliers):
        # g itself may carry units; only its additive terms must agree.
        validate_dimension(constraint.residual(), dimension, context)
        multiplier_context = DimensionalContext(
            {**context.symbols, multiplier.name: multiplier.dimension}, strict=True
        )
        infer_dimension(multiplier.symbol * constraint.residual(), multiplier_context)
        if infer_dimension(multiplier.symbol * constraint.residual(), multiplier_context) != model.lagrangian_dimension:
            raise ValueError(
                f"Multiplier contribution for constraint '{constraint.name}' does not match Lagrangian dimension"
            )
    return dimensions


def validate_constrained_euler_lagrange_dimensions(model: LagrangianModel):
    """Validate the augmented Lagrangian and every constrained E-L residual."""
    from .dimension_validation import validate_lagrangian_dimensions
    from .validation import assert_valid

    assert_valid(model)
    validate_lagrangian_dimensions(model)
    dimensions = validate_constraints_dimensions(model)
    multipliers = build_multipliers(model, dimensions)
    dynamics = constrained_euler_lagrange(model, multipliers)

    mapping = coordinate_context(model).symbols.copy()
    for multiplier in multipliers:
        mapping[multiplier.name] = multiplier.dimension
    context = DimensionalContext(mapping, strict=True)
    validate_dimension(dynamics.augmented_lagrangian, model.lagrangian_dimension, context)

    for coordinate in model.coordinates:
        expected = model.lagrangian_dimension / coordinate.dimension
        equation = next(e for e in dynamics.equations if e.coordinate == coordinate.name)
        actual = infer_dimension(equation.residual, context)
        if actual != expected:
            raise ValueError(
                f"Constrained Euler–Lagrange residual for '{coordinate.name}' has inconsistent physical dimension"
            )
    return dynamics
