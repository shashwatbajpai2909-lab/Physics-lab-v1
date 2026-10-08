from __future__ import annotations
from dataclasses import dataclass
from .dimensions import Dimension
from .physics_ir import LagrangianModel


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def validate_model(model: LagrangianModel) -> tuple[ValidationIssue, ...]:
    issues = []

    if not model.name:
        issues.append(ValidationIssue("EMPTY_NAME", "Model name cannot be empty"))

    names = [q.name for q in model.coordinates]
    if len(names) != len(set(names)):
        issues.append(ValidationIssue("DUPLICATE_COORDINATE", "Coordinate names must be unique"))

    parameter_names = [p.name for p in model.parameters]
    if len(parameter_names) != len(set(parameter_names)):
        issues.append(ValidationIssue("DUPLICATE_PARAMETER", "Parameter names must be unique"))

    overlap = set(names) & set(parameter_names)
    if overlap:
        issues.append(ValidationIssue(
            "PARAMETER_COORDINATE_COLLISION",
            "Parameter and coordinate names must be distinct: " + ", ".join(sorted(overlap)),
        ))

    constraint_names = [c.name for c in model.constraints]
    if len(constraint_names) != len(set(constraint_names)):
        issues.append(ValidationIssue("DUPLICATE_CONSTRAINT", "Constraint names must be unique"))

    reserved = set(names) | set(parameter_names)
    for constraint in model.constraints:
        if not constraint.name:
            issues.append(ValidationIssue("EMPTY_CONSTRAINT_NAME", "Constraint names cannot be empty"))
        multiplier_name = f"lambda_{constraint.name}"
        if multiplier_name in reserved:
            issues.append(ValidationIssue(
                "CONSTRAINT_MULTIPLIER_COLLISION",
                f"Constraint multiplier name collides with a model symbol: {multiplier_name}",
            ))

    return tuple(issues)


def assert_valid(model: LagrangianModel) -> None:
    issues = validate_model(model)
    if issues:
        raise ValidationError("; ".join(i.message for i in issues))
