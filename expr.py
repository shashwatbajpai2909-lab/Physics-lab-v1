from __future__ import annotations
from dataclasses import dataclass
from math import sin, cos, exp, log
from typing import Union

Number = Union[int, float]


class Expr:
    """Immutable symbolic expression."""

    def __add__(self, other):
        return Add(self, as_expr(other))

    def __radd__(self, other):
        return Add(as_expr(other), self)

    def __sub__(self, other):
        return Add(self, Mul(Constant(-1), as_expr(other)))

    def __rsub__(self, other):
        return Add(as_expr(other), Mul(Constant(-1), self))

    def __mul__(self, other):
        return Mul(self, as_expr(other))

    def __rmul__(self, other):
        return Mul(as_expr(other), self)

    def __truediv__(self, other):
        return Mul(self, Pow(as_expr(other), Constant(-1)))

    def __rtruediv__(self, other):
        return Mul(as_expr(other), Pow(self, Constant(-1)))

    def __pow__(self, other):
        return Pow(self, as_expr(other))

    def __neg__(self):
        return Mul(Constant(-1), self)

    def diff(self, variable):
        raise NotImplementedError

    def simplify(self):
        return self


@dataclass(frozen=True)
class Constant(Expr):
    value: Number

    def diff(self, variable):
        return Constant(0)

    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class Symbol(Expr):
    name: str

    def diff(self, variable):
        return Constant(1 if self == variable else 0)

    def __str__(self):
        return self.name


def as_expr(value):
    return value if isinstance(value, Expr) else Constant(value)


def _sort_key(expr):
    return str(expr)


def _split_power(expr):
    if isinstance(expr, Pow):
        return expr.base, expr.exponent
    return expr, Constant(1)


def _split_coefficient(expr):
    if isinstance(expr, Mul):
        factors = _flatten_mul(expr)
        if factors and isinstance(factors[0], Constant):
            coefficient = factors[0].value
            rest = factors[1:]
            if not rest:
                return coefficient, Constant(1)
            base = rest[0] if len(rest) == 1 else _build_mul(rest)
            return coefficient, base
    if isinstance(expr, Constant):
        return expr.value, Constant(1)
    return 1, expr


def _flatten_add(expr):
    if isinstance(expr, Add):
        return _flatten_add(expr.left) + _flatten_add(expr.right)
    return [expr]


def _flatten_mul(expr):
    if isinstance(expr, Mul):
        return _flatten_mul(expr.left) + _flatten_mul(expr.right)
    return [expr]


def _build_add(terms):
    if not terms:
        return Constant(0)
    result = terms[0]
    for term in terms[1:]:
        result = Add(result, term)
    return result


def _build_mul(factors):
    if not factors:
        return Constant(1)
    result = factors[0]
    for factor in factors[1:]:
        result = Mul(result, factor)
    return result


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr

    def diff(self, variable):
        return simplify(Add(self.left.diff(variable), self.right.diff(variable)))

    def simplify(self):
        raw = [simplify(t) for t in _flatten_add(self)]
        constant = 0
        groups = {}

        for term in raw:
            if isinstance(term, Constant):
                constant += term.value
                continue
            coeff, base = _split_coefficient(term)
            key = _sort_key(base)
            if key in groups:
                old_base, old_coeff = groups[key]
                groups[key] = (old_base, old_coeff + coeff)
            else:
                groups[key] = (base, coeff)

        result = []
        if constant != 0:
            result.append(Constant(constant))

        for key in sorted(groups):
            base, coeff = groups[key]
            if coeff == 0:
                continue
            if coeff == 1:
                result.append(base)
            else:
                result.append(simplify(Mul(Constant(coeff), base)))

        return _build_add(result)

    def __str__(self):
        terms = _flatten_add(self)
        rendered = []
        for term in terms:
            if isinstance(term, Mul):
                factors = _flatten_mul(term)
                if factors and isinstance(factors[0], Constant) and factors[0].value < 0:
                    c = factors[0].value
                    rest = factors[1:]
                    magnitude = -c
                    body = str(_build_mul(rest)) if rest else ""
                    rendered.append(("-", f"{magnitude if magnitude != 1 else ''}{body}"))
                    continue
            rendered.append(("+", str(term)))

        if not rendered:
            return "0"

        first_sign, first_body = rendered[0]
        out = ("-" if first_sign == "-" else "") + first_body
        for sign, body in rendered[1:]:
            out += (" - " if sign == "-" else " + ") + body
        return out


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def diff(self, variable):
        return simplify(
            Add(
                Mul(self.left.diff(variable), self.right),
                Mul(self.left, self.right.diff(variable)),
            )
        )

    def simplify(self):
        factors = [simplify(f) for f in _flatten_mul(self)]
        coefficient = 1
        powers = {}

        for factor in factors:
            if isinstance(factor, Constant):
                coefficient *= factor.value
                continue

            base, exponent = _split_power(factor)
            key = _sort_key(base)
            old = powers.get(key)
            if old is None:
                powers[key] = (base, exponent)
            else:
                powers[key] = (old[0], simplify(old[1] + exponent))

        if coefficient == 0:
            return Constant(0)

        result = []
        if coefficient != 1 or not powers:
            result.append(Constant(coefficient))

        for key in sorted(powers):
            base, exponent = powers[key]
            exponent = simplify(exponent)
            if isinstance(exponent, Constant) and exponent.value == 0:
                continue
            if isinstance(exponent, Constant) and exponent.value == 1:
                result.append(base)
            else:
                result.append(Pow(base, exponent))

        return _build_mul(result)

    def __str__(self):
        return "*".join(str(f) for f in _flatten_mul(self))


@dataclass(frozen=True)
class Pow(Expr):
    base: Expr
    exponent: Expr

    def diff(self, variable):
        if isinstance(self.exponent, Constant):
            n = self.exponent.value
            return simplify(
                Mul(
                    Constant(n),
                    Mul(
                        Pow(self.base, Constant(n - 1)),
                        self.base.diff(variable),
                    ),
                )
            )

        u, v = self.base, self.exponent
        return simplify(
            Mul(
                Pow(u, v),
                Add(
                    Mul(v.diff(variable), Log(u)),
                    Mul(v, Mul(u.diff(variable), Pow(u, Constant(-1)))),
                ),
            )
        )

    def simplify(self):
        base, exponent = simplify(self.base), simplify(self.exponent)
        if isinstance(exponent, Constant):
            if exponent.value == 0:
                return Constant(1)
            if exponent.value == 1:
                return base
        if isinstance(base, Constant) and isinstance(exponent, Constant):
            return Constant(base.value ** exponent.value)
        return Pow(base, exponent)

    def __str__(self):
        return f"{self.base}^{self.exponent}"


@dataclass(frozen=True)
class Sin(Expr):
    arg: Expr

    def diff(self, variable):
        return simplify(Mul(Cos(self.arg), self.arg.diff(variable)))

    def simplify(self):
        return Sin(simplify(self.arg))

    def __str__(self):
        return f"sin({self.arg})"


@dataclass(frozen=True)
class Cos(Expr):
    arg: Expr

    def diff(self, variable):
        return simplify(Mul(Constant(-1), Mul(Sin(self.arg), self.arg.diff(variable))))

    def simplify(self):
        return Cos(simplify(self.arg))

    def __str__(self):
        return f"cos({self.arg})"


@dataclass(frozen=True)
class Exp(Expr):
    arg: Expr

    def diff(self, variable):
        return simplify(Mul(Exp(self.arg), self.arg.diff(variable)))

    def simplify(self):
        return Exp(simplify(self.arg))

    def __str__(self):
        return f"exp({self.arg})"


@dataclass(frozen=True)
class Log(Expr):
    arg: Expr

    def diff(self, variable):
        return simplify(Mul(self.arg.diff(variable), Pow(self.arg, Constant(-1))))

    def simplify(self):
        return Log(simplify(self.arg))

    def __str__(self):
        return f"log({self.arg})"


def simplify(expr):
    current = expr
    for _ in range(32):
        nxt = current.simplify()
        if nxt == current:
            return nxt
        current = nxt
    return current


def differentiate(expr, variable):
    return simplify(as_expr(expr).diff(variable))
