# Physics Lab v1.1.9

PhysicsIR-first symbolic mechanics with typed dimensions, constrained Euler–Lagrange mechanics, and an explicit ComputationalIR execution contract.

## v1.1.9 — Constraint-aware ComputationalIR

This release makes the computational representation explicit about the structure a future solver/runtime will consume.

### Added

- `EquationKind`: `DIFFERENTIAL` vs `ALGEBRAIC`.
- `StateVariable` with physical dimension and derivative order.
- `EquationOfMotion` metadata for derivative order and residual dimension.
- `AlgebraicEquation` for first-class constraints in ComputationalIR.
- ComputationalIR now carries:
  - differential equations
  - algebraic constraint equations
  - state variables
- Typed transformations preserve coordinate, velocity, acceleration, and constraint dimensions.
- Legacy dimensionless models remain compatible.

### Pipeline

```text
PhysicsIR
   ↓
Typed SymbolicIR
   ↓
Dimensional + structural validation
   ↓
Euler–Lagrange / constrained mechanics
   ↓
ComputationalIR
   ├── Differential equations
   ├── Algebraic constraints
   └── State variables
   ↓
Future Solver / Runtime
```

## Example

A one-dimensional model with `x` as a length coordinate produces computational state variables:

- `x` — derivative order 0, dimension L
- `x_dot` — derivative order 1, dimension L/T

Its Euler–Lagrange equation is classified as a second-order differential equation, while a constraint such as `x = 0` is represented separately as an algebraic equation.

## Verification

**44 tests passed.**

No numerical solver is introduced yet. v1.1.9 establishes the IR contract needed before implementing solver/runtime execution.
