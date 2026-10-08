import pytest

from physics_lab import (
    LENGTH, TIME, MASS, ENERGY,
    LagrangianModel, validate_lagrangian_dimensions,
)
from physics_lab.symbolic.expr import Symbol, Sin
from physics_lab.symbolic.dimensions import (
    DimensionalContext, SymbolicDimensionError, infer_dimension,
)
from physics_lab.transform import lagrangian_to_computational_ir


def test_symbolic_addition_requires_equal_dimensions():
    x = Symbol("x")
    v = Symbol("v")
    ctx = DimensionalContext({"x": LENGTH, "v": LENGTH / TIME})

    with pytest.raises(SymbolicDimensionError):
        infer_dimension(x + v, ctx)


def test_symbolic_multiplication_infers_dimension():
    m = Symbol("m")
    a = Symbol("a")
    ctx = DimensionalContext({"m": MASS, "a": LENGTH / (TIME ** 2)})
    assert infer_dimension(m * a, ctx) == MASS * LENGTH / (TIME ** 2)


def test_symbolic_function_requires_dimensionless_argument():
    x = Symbol("x")
    ctx = DimensionalContext({"x": LENGTH})

    with pytest.raises(SymbolicDimensionError):
        infer_dimension(Sin(x), ctx)


def test_dimensionally_valid_harmonic_oscillator_pipeline():
    x = Symbol("x")
    xd = Symbol("x_dot")
    m = Symbol("m")
    k = Symbol("k")

    # [m]=M, [x]=L, [k]=M/T², so Lagrangian has ML²/T².
    k_dim = MASS / (TIME ** 2)
    L = (m * xd**2) / 2 - (k * x**2) / 2

    model = LagrangianModel.from_symbols(
        "dimensional_ho",
        ("x",),
        L,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("k", MASS / (TIME ** 2))),
    )

    validate_lagrangian_dimensions(model)
    computational = lagrangian_to_computational_ir(model)
    assert str(computational.equations[0].residual) == "k*x + m*x_ddot"


def test_dimensionally_invalid_lagrangian_is_rejected():
    x = Symbol("x")
    bad = x + Symbol("x_dot")

    model = LagrangianModel.from_symbols(
        "invalid",
        ("x",),
        bad,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
    )

    with pytest.raises(SymbolicDimensionError):
        validate_lagrangian_dimensions(model)


def test_dimensionless_trig_argument_is_allowed():
    theta = Symbol("theta")
    ctx = DimensionalContext({"theta": __import__("physics_lab").DIMENSIONLESS})
    assert infer_dimension(Sin(theta), ctx) == __import__("physics_lab").DIMENSIONLESS


def test_parameter_dimensions_are_part_of_physicsir():
    x = Symbol("x")
    m = Symbol("m")
    model = LagrangianModel.from_symbols(
        "parameterized",
        ("x",),
        m * x,
        dimensions=(LENGTH,),
        parameters=(("m", MASS),),
    )
    assert model.parameters[0].dimension == MASS


def test_typed_harmonic_oscillator_residual_dimension():
    from physics_lab.symbolic.dimensions import infer_dimension
    x = Symbol("x")
    xd = Symbol("x_dot")
    m = Symbol("m")
    k = Symbol("k")
    L = (m * xd**2) / 2 - (k * x**2) / 2

    model = LagrangianModel.from_symbols(
        "typed_residual",
        ("x",),
        L,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("k", MASS / (TIME ** 2))),
    )
    ir = lagrangian_to_computational_ir(model)
    context = __import__("physics_lab").dimension_validation.coordinate_context(model)
    assert infer_dimension(ir.equations[0].residual, context) == ENERGY / LENGTH
