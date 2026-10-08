import pytest

from physics_lab import LENGTH, TIME, MASS, ENERGY, DIMENSIONLESS
from physics_lab.physics_ir import LagrangianModel
from physics_lab.symbolic.expr import Symbol, Sin
from physics_lab.symbolic.dimensions import (
    DimensionalContext,
    SymbolicDimensionError,
    infer_dimension,
    derivative_dimension,
    differentiate_with_dimension,
)
from physics_lab.dimension_validation import (
    validate_euler_lagrange_dimensions,
    coordinate_context,
)
from physics_lab.validation import validate_model
from physics_lab.transform import lagrangian_to_computational_ir


def test_derivative_dimension_is_expression_over_variable():
    x = Symbol("x")
    ctx = DimensionalContext({"x": LENGTH})
    assert derivative_dimension(x**2, x, ctx) == LENGTH


def test_parameter_derivative_dimension():
    m = Symbol("m")
    a = Symbol("a")
    ctx = DimensionalContext({"m": MASS, "a": LENGTH / (TIME**2)})
    assert derivative_dimension(m * a, a, ctx) == MASS


def test_differentiate_with_dimension_validates_actual_tree():
    x = Symbol("x")
    ctx = DimensionalContext({"x": LENGTH})
    derivative, dim = differentiate_with_dimension(x**2, x, ctx)
    assert str(derivative) == "2*x"
    assert dim == LENGTH


def test_unknown_symbol_rejected_in_strict_typed_context():
    x = Symbol("x")
    y = Symbol("y")
    ctx = DimensionalContext({"x": LENGTH}, strict=True)
    with pytest.raises(SymbolicDimensionError):
        infer_dimension(x + y, ctx)


def test_duplicate_names_rejected_by_validation():
    x = Symbol("x")
    duplicate_coords = LagrangianModel.from_symbols(
        "duplicate_coords", ("x", "x"), x, dimensions=(LENGTH, LENGTH)
    )
    duplicate_params = LagrangianModel.from_symbols(
        "duplicate_params", ("x",), x,
        dimensions=(LENGTH,), parameters=(("m", MASS), ("m", MASS))
    )
    collision = LagrangianModel.from_symbols(
        "collision", ("x",), x,
        dimensions=(LENGTH,), parameters=(("x", MASS),)
    )
    assert any(i.code == "DUPLICATE_COORDINATE" for i in validate_model(duplicate_coords))
    assert any(i.code == "DUPLICATE_PARAMETER" for i in validate_model(duplicate_params))
    assert any(i.code == "PARAMETER_COORDINATE_COLLISION" for i in validate_model(collision))


def test_euler_lagrange_derivative_dimensions_for_harmonic_oscillator():
    x = Symbol("x")
    xd = Symbol("x_dot")
    m = Symbol("m")
    k = Symbol("k")
    L = (m * xd**2) / 2 - (k * x**2) / 2
    model = LagrangianModel.from_symbols(
        "typed_ho", ("x",), L,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("k", MASS / (TIME**2))),
    )
    validate_euler_lagrange_dimensions(model)


def test_typed_pipeline_remains_valid_and_produces_force_residual():
    x = Symbol("x")
    xd = Symbol("x_dot")
    m = Symbol("m")
    k = Symbol("k")
    L = (m * xd**2) / 2 - (k * x**2) / 2
    model = LagrangianModel.from_symbols(
        "typed_pipeline", ("x",), L,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
        parameters=(("m", MASS), ("k", MASS / (TIME**2))),
    )
    ir = lagrangian_to_computational_ir(model)
    assert str(ir.equations[0].residual) == "k*x + m*x_ddot"
    assert infer_dimension(ir.equations[0].residual, coordinate_context(model)) == ENERGY / LENGTH


def test_legacy_dimensionless_models_still_work():
    x = Symbol("x")
    L = x**2 + Sin(x)
    model = LagrangianModel.from_symbols("legacy", ("x",), L)
    ir = lagrangian_to_computational_ir(model)
    assert len(ir.equations) == 1
    assert infer_dimension(model.lagrangian, coordinate_context(model)) == DIMENSIONLESS
