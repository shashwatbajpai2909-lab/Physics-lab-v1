from physics_lab import LagrangianModel, lagrangian_to_computational_ir
from physics_lab.symbolic.expr import Symbol


def test_harmonic_oscillator_physicsir_to_computationalir():
    x = Symbol("x")
    xdot = Symbol("x_dot")
    m = Symbol("m")
    k = Symbol("k")

    L = (m * xdot**2) / 2 - (k * x**2) / 2
    model = LagrangianModel.from_symbols(
        "harmonic_oscillator",
        ("x",),
        L,
    )

    computational = lagrangian_to_computational_ir(model)

    assert computational.model_name == "harmonic_oscillator"
    assert len(computational.equations) == 1

    eq = computational.equations[0]
    assert eq.coordinate == "x"
    assert eq.acceleration == "x_ddot"
    assert str(eq.residual) == "k*x + m*x_ddot"


def test_multi_dof_pipeline():
    x = Symbol("x")
    y = Symbol("y")
    xd = Symbol("x_dot")
    yd = Symbol("y_dot")
    mx = Symbol("mx")
    my = Symbol("my")

    L = (mx * xd**2) / 2 + (my * yd**2) / 2
    model = LagrangianModel.from_symbols(
        "free_particle_2d",
        ("x", "y"),
        L,
    )

    computational = lagrangian_to_computational_ir(model)

    assert len(computational.equations) == 2
    assert str(computational.equations[0].residual) == "mx*x_ddot"
    assert str(computational.equations[1].residual) == "my*y_ddot"
