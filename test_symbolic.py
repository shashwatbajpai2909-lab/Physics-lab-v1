from physics_lab.symbolic.expr import (
    Symbol, Constant, Sin, Cos, Exp, Log,
    differentiate, simplify
)
from physics_lab.symbolic.lagrangian import euler_lagrange_residual


def test_basic_derivatives():
    x = Symbol("x")
    assert differentiate(Constant(7), x) == Constant(0)
    assert differentiate(x, x) == Constant(1)
    assert differentiate(x**3, x) == Constant(3) * x**2


def test_product_rule():
    x = Symbol("x")
    expr = x * x
    assert simplify(differentiate(expr, x)) == simplify(2 * x)


def test_chain_rules():
    x = Symbol("x")
    assert differentiate(Sin(x), x) == Cos(x)
    assert simplify(differentiate(Cos(x), x)) == simplify(-Sin(x))
    assert differentiate(Exp(x), x) == Exp(x)
    assert simplify(differentiate(Log(x), x)) == simplify(1 / x)


def test_harmonic_oscillator_lagrangian():
    x = Symbol("x")
    v = Symbol("xdot")
    m = Symbol("m")
    k = Symbol("k")

    L = Constant(0.5) * m * v**2 - Constant(0.5) * k * x**2

    assert simplify(differentiate(L, x)) == simplify(-k * x)
    assert simplify(differentiate(L, v)) == simplify(m * v)


def test_euler_lagrange_harmonic_oscillator():
    x = Symbol("x")
    v = Symbol("xdot")
    a = Symbol("xddot")
    m = Symbol("m")
    k = Symbol("k")

    L = Constant(0.5) * m * v**2 - Constant(0.5) * k * x**2

    residual = euler_lagrange_residual(L, x, v, a)

    assert simplify(residual) == simplify(m * a + k * x)


def test_canonical_multiplication_and_powers():
    x=Symbol("x")
    assert str(simplify(x*x)) == "x^2"
    assert str(simplify((x*2)*(3*x))) == "6*x^2"


def test_like_terms():
    x=Symbol("x")
    assert str(simplify(2*x+3*x)) == "5*x"


def test_nested_add_canonicalization():
    x=Symbol("x"); y=Symbol("y")
    assert str(simplify((x+(2*x+3))+(y+y))) == "3 + 3*x + 2*y"


def test_zero_and_one_identities():
    x=Symbol("x")
    assert simplify(x+0)==x
    assert simplify(x*1)==x
    assert simplify(x*0)==Constant(0)
    assert simplify(x**0)==Constant(1)
