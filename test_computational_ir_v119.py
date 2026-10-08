from physics_lab import LagrangianModel, EqualityConstraint, LENGTH, MASS, TIME, ENERGY
from physics_lab.computational_ir import EquationKind
from physics_lab.symbolic.expr import Symbol
from physics_lab.transform import lagrangian_to_computational_ir


def test_computational_ir_classifies_differential_equations():
    x, v, m, k = Symbol('x'), Symbol('x_dot'), Symbol('m'), Symbol('k')
    model = LagrangianModel.from_symbols(
        'ho', ('x',), m*v*v/2-k*x*x/2,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(('m', MASS), ('k', MASS/(TIME**2))),
    )
    ir = lagrangian_to_computational_ir(model)
    eq = ir.equations[0]
    assert eq.kind == EquationKind.DIFFERENTIAL
    assert eq.derivative_order == 2
    assert eq.coordinate == 'x'


def test_state_variables_include_coordinate_and_velocity():
    x, v, m = Symbol('x'), Symbol('x_dot'), Symbol('m')
    model = LagrangianModel.from_symbols(
        'free', ('x',), m*v*v/2,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(('m', MASS),),
    )
    ir = lagrangian_to_computational_ir(model)
    assert [s.name for s in ir.state_variables] == ['x', 'x_dot']
    assert ir.state_variables[0].derivative_order == 0
    assert ir.state_variables[1].derivative_order == 1
    assert ir.state_variables[1].dimension == LENGTH/TIME


def test_constraints_become_algebraic_computational_equations():
    x, v, m = Symbol('x'), Symbol('x_dot'), Symbol('m')
    model = LagrangianModel.from_symbols(
        'fixed', ('x',), m*v*v/2,
        dimensions=(LENGTH,), lagrangian_dimension=ENERGY,
        parameters=(('m', MASS),),
        constraints=(EqualityConstraint('fixed', x),),
    )
    ir = lagrangian_to_computational_ir(model)
    assert len(ir.algebraic_equations) == 1
    assert ir.algebraic_equations[0].kind == EquationKind.ALGEBRAIC
    assert str(ir.algebraic_equations[0].expression) == 'x'
    assert ir.algebraic_equations[0].dimension == LENGTH


def test_legacy_dimensionless_model_remains_compatible():
    x, v, m = Symbol('x'), Symbol('x_dot'), Symbol('m')
    model = LagrangianModel.from_symbols('legacy', ('x',), m*v*v/2)
    ir = lagrangian_to_computational_ir(model)
    assert len(ir.equations) == 1
    assert ir.equations[0].residual_dimension == model.lagrangian_dimension
