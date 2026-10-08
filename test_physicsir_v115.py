import pytest

from physics_lab import (
    LENGTH, TIME, MASS, ENERGY, FORCE, VELOCITY,
    Quantity, METER, SECOND, KILOGRAM, NEWTON,
    DimensionError, LagrangianModel, validate_model, EqualityConstraint,
)
from physics_lab.symbolic.expr import Symbol


def test_dimension_algebra():
    assert VELOCITY == LENGTH / TIME
    assert FORCE == MASS * LENGTH / (TIME ** 2)
    assert ENERGY == FORCE * LENGTH


def test_quantity_addition_and_conversion():
    q1 = Quantity(2, METER)
    q2 = Quantity(30, METER)
    assert (q1 + q2).value == 32
    assert (Quantity(1, METER) / Quantity(1, SECOND)).dimension == VELOCITY


def test_dimension_mismatch_is_rejected():
    with pytest.raises(DimensionError):
        Quantity(1, METER) + Quantity(1, SECOND)


def test_quantity_product():
    force = Quantity(10, NEWTON)
    distance = Quantity(2, METER)
    energy = force * distance
    assert energy.dimension == ENERGY
    assert energy.value == 20


def test_typed_physicsir():
    x = Symbol("x")
    L = (Symbol("m") * Symbol("x_dot")**2) / 2
    model = LagrangianModel.from_symbols(
        "typed_model",
        ("x",),
        L,
        dimensions=(LENGTH,),
        lagrangian_dimension=ENERGY,
    )
    assert model.coordinates[0].dimension == LENGTH
    assert model.lagrangian_dimension == ENERGY


def test_constraint_and_validation():
    x = Symbol("x")
    constraint = EqualityConstraint("fixed_origin", x)
    model = LagrangianModel.from_symbols(
        "constrained",
        ("x",),
        x,
        dimensions=(LENGTH,),
        constraints=(constraint,),
    )
    assert validate_model(model) == ()
    assert model.constraints[0].name == "fixed_origin"


def test_duplicate_coordinates_rejected_by_validation():
    x = Symbol("x")
    model = LagrangianModel.from_symbols(
        "bad",
        ("x", "x"),
        x,
    )
    issues = validate_model(model)
    assert any(i.code == "DUPLICATE_COORDINATE" for i in issues)
