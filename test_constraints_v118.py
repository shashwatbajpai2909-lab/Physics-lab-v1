from physics_lab import (
    DIMENSIONLESS, ENERGY, FORCE, LENGTH, MASS, TIME,
    LagrangianModel, EqualityConstraint, ValidationError,
)
from physics_lab.constrained import build_multipliers, constrained_euler_lagrange
from physics_lab.constrained_validation import validate_constrained_euler_lagrange_dimensions

from physics_lab.symbolic.expr import Symbol
from physics_lab.transform import lagrangian_to_constrained_dynamics


def test_dimensionless_constraint_multiplier_matches_lagrangian():
    x = Symbol("x")
    m = Symbol("m")
    v = Symbol("x_dot")
    model = LagrangianModel.from_symbols(
        "particle", ("x",), m*v*v/2,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(("m", MASS),),
        constraints=(EqualityConstraint("fixed", x),),
    )
    # x=0 has length dimension, hence lambda has energy/length = force.
    dynamics = validate_constrained_euler_lagrange_dimensions(model)
    assert dynamics.multipliers[0].dimension == FORCE


def test_constrained_harmonic_equation_contains_multiplier():
    x, v, m, k = Symbol("x"), Symbol("x_dot"), Symbol("m"), Symbol("k")
    L = m*v*v/2 - k*x*x/2
    constraint = EqualityConstraint("origin", x)
    model = LagrangianModel.from_symbols(
        "constrained_oscillator", ("x",), L,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("k", MASS/(TIME**2))),
        constraints=(constraint,),
    )
    dynamics = validate_constrained_euler_lagrange_dimensions(model)
    assert "lambda_origin" in str(dynamics.equations[0].residual)


def test_constraint_equation_is_preserved():
    x = Symbol("x")
    m, v = Symbol("m"), Symbol("x_dot")
    model = LagrangianModel.from_symbols(
        "fixed", ("x",), m*v*v/2,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(("m", MASS),),
        constraints=(EqualityConstraint("origin", x),),
    )
    dynamics = lagrangian_to_constrained_dynamics(model)
    assert str(dynamics.constraint_equations[0]) == "x"


def test_duplicate_constraints_rejected():
    x = Symbol("x")
    model = LagrangianModel.from_symbols(
        "bad", ("x",), x,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        constraints=(EqualityConstraint("c", x), EqualityConstraint("c", x*x)),
    )
    try:
        validate_constrained_euler_lagrange_dimensions(model)
    except ValidationError as exc:
        assert "Constraint names must be unique" in str(exc)
    else:
        raise AssertionError("duplicate constraints were not rejected")


def test_multiplier_collision_rejected():
    x = Symbol("x")
    model = LagrangianModel.from_symbols(
        "bad", ("x", "lambda_c"), x,
        dimensions=(LENGTH, DIMENSIONLESS), lagrangian_dimension=ENERGY,
        constraints=(EqualityConstraint("c", x),),
    )
    try:
        validate_constrained_euler_lagrange_dimensions(model)
    except ValidationError as exc:
        assert "lambda_c" in str(exc)
    else:
        raise AssertionError("multiplier collision was not rejected")


def test_circular_holonomic_constraint_has_consistent_multiplier_dimension():
    x, y = Symbol("x"), Symbol("y")
    vx, vy = Symbol("x_dot"), Symbol("y_dot")
    m, R = Symbol("m"), Symbol("R")
    L = m * (vx*vx + vy*vy) / 2
    g = x*x + y*y - R*R
    model = LagrangianModel.from_symbols(
        "circular_particle", ("x", "y"), L,
        dimensions=(LENGTH, LENGTH), lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("R", LENGTH)),
        constraints=(EqualityConstraint("circle", g),),
    )
    dynamics = validate_constrained_euler_lagrange_dimensions(model)
    # [lambda] = [energy]/[length^2] = M/T^2
    assert dynamics.multipliers[0].dimension == MASS/(TIME**2)
    assert "lambda_circle" in str(dynamics.equations[0].residual)
    assert "lambda_circle" in str(dynamics.equations[1].residual)
