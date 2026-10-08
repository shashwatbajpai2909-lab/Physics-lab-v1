from __future__ import annotations
from .physics_ir import LagrangianModel
from .symbolic.dimensions import (
    DimensionalContext,
    validate_dimension,
)
from .dimensions import TIME


def coordinate_context(model: LagrangianModel) -> DimensionalContext:
    mapping = {p.name: p.dimension for p in model.parameters}
    mapping.update({q.name: q.dimension for q in model.coordinates})

    for q in model.coordinates:
        mapping[f"{q.name}_dot"] = q.dimension / TIME
        mapping[f"{q.name}_ddot"] = q.dimension / (TIME ** 2)

    return DimensionalContext(mapping, strict=True)


def validate_lagrangian_dimensions(model: LagrangianModel):
    from .validation import assert_valid
    assert_valid(model)
    context = coordinate_context(model)
    validate_dimension(model.lagrangian, model.lagrangian_dimension, context)
    return context


def validate_euler_lagrange_dimensions(model: LagrangianModel):
    """Validate every dimensional stage of the Euler–Lagrange transformation."""
    from .symbolic.dimensions import (
        differentiate_with_dimension,
        infer_dimension,
    )
    from .symbolic.expr import Symbol
    from .symbolic.lagrangian import euler_lagrange_residual

    context = validate_lagrangian_dimensions(model)
    L = model.lagrangian
    L_dim = model.lagrangian_dimension

    for coordinate in model.coordinates:
        q = coordinate.symbol
        qdot = Symbol(f"{q.name}_dot")
        qddot = Symbol(f"{q.name}_ddot")

        dL_dq, dL_dq_dim = differentiate_with_dimension(L, q, context)
        dL_dv, dL_dv_dim = differentiate_with_dimension(L, qdot, context)

        expected_dL_dq = L_dim / coordinate.dimension
        expected_dL_dv = L_dim / (coordinate.dimension / TIME)
        if dL_dq_dim != expected_dL_dq:
            raise ValueError("∂L/∂q has inconsistent physical dimension")
        if dL_dv_dim != expected_dL_dv:
            raise ValueError("∂L/∂q_dot has inconsistent physical dimension")

        # d/dt(∂L/∂q_dot) = (∂²L/∂q∂q_dot)q_dot
        #                    + (∂²L/∂q_dot²)q_ddot
        partial_q = differentiate_with_dimension(dL_dv, q, context)[0] * qdot
        partial_v = differentiate_with_dimension(dL_dv, qdot, context)[0] * qddot
        from .symbolic.expr import simplify
        time_derivative = simplify(partial_q + partial_v)
        expected_force = L_dim / coordinate.dimension
        if infer_dimension(time_derivative, context) != expected_force:
            raise ValueError("d/dt(∂L/∂q_dot) has inconsistent physical dimension")

        residual = euler_lagrange_residual(L, q, qdot, qddot)
        if infer_dimension(residual, context) != expected_force:
            raise ValueError("Euler–Lagrange residual has inconsistent physical dimension")

    return context
