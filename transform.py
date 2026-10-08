from .physics_ir import LagrangianModel
from .computational_ir import ComputationalIR, EquationOfMotion, AlgebraicEquation, StateVariable
from .symbolic.expr import Symbol, simplify
from .symbolic.lagrangian import euler_lagrange_residual
from .dimension_validation import (
    validate_lagrangian_dimensions,
    validate_euler_lagrange_dimensions,
    coordinate_context,
)
from .symbolic.dimensions import infer_dimension
from .dimensions import DIMENSIONLESS, TIME


def lagrangian_to_computational_ir(model: LagrangianModel) -> ComputationalIR:
    typed_model = (
        model.lagrangian_dimension != DIMENSIONLESS
        or any(q.dimension != DIMENSIONLESS for q in model.coordinates)
        or any(p.dimension != DIMENSIONLESS for p in model.parameters)
    )
    context = validate_euler_lagrange_dimensions(model) if typed_model else coordinate_context(model)
    equations = []
    states = []

    for coordinate in model.coordinates:
        q = coordinate.symbol
        qdot = Symbol(f"{q.name}_dot")
        qddot = Symbol(f"{q.name}_ddot")
        residual = simplify(euler_lagrange_residual(model.lagrangian, q, qdot, qddot))
        residual_dim = infer_dimension(residual, context) if typed_model else DIMENSIONLESS
        equations.append(EquationOfMotion(
            coordinate=q.name, acceleration=qddot.name, residual=residual,
            residual_dimension=residual_dim,
        ))
        states.extend((
            StateVariable(q.name, coordinate.dimension, 0),
            StateVariable(qdot.name, coordinate.dimension / TIME, 1),
        ))

    algebraic = tuple(
        AlgebraicEquation(c.name, c.residual(),
                          dimension=infer_dimension(c.residual(), context) if typed_model else DIMENSIONLESS)
        for c in model.constraints
    )

    return ComputationalIR(
        model_name=model.name,
        equations=tuple(equations),
        algebraic_equations=algebraic,
        state_variables=tuple(states),
    )



def lagrangian_to_constrained_dynamics(model: LagrangianModel):
    """Transform a constrained PhysicsIR model using Lagrange multipliers."""
    if not model.constraints:
        raise ValueError("Constrained transformation requires at least one constraint")
    from .constrained_validation import validate_constrained_euler_lagrange_dimensions
    from .constrained import build_multipliers, constrained_euler_lagrange
    from .dimension_validation import coordinate_context
    from .symbolic.dimensions import infer_dimension

    typed_model = (
        model.lagrangian_dimension != __import__("physics_lab").DIMENSIONLESS
        or any(q.dimension != __import__("physics_lab").DIMENSIONLESS for q in model.coordinates)
        or any(p.dimension != __import__("physics_lab").DIMENSIONLESS for p in model.parameters)
    )
    if typed_model:
        return validate_constrained_euler_lagrange_dimensions(model)

    context = coordinate_context(model)
    dimensions = tuple(infer_dimension(c.residual(), context) for c in model.constraints)
    multipliers = build_multipliers(model, dimensions)
    return constrained_euler_lagrange(model, multipliers)
