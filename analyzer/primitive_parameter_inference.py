from __future__ import annotations

import ast
from dataclasses import dataclass


_PRIMITIVE_TYPES = frozenset({"int", "float", "str", "bool"})
_NUMERIC_TYPES = frozenset({"int", "float"})
_NUMERIC_EVIDENCE = "numeric"


@dataclass(frozen=True, slots=True)
class PrimitiveParameterInference:
    inferred_types: tuple[tuple[str, str], ...]
    rejection_reasons: tuple[tuple[str, str], ...]

    def type_for(self, parameter_name: str) -> str | None:
        return dict(self.inferred_types).get(parameter_name)

    def rejection_for(self, parameter_name: str) -> str | None:
        return dict(self.rejection_reasons).get(parameter_name)


class _PrimitiveEvidenceCollector(ast.NodeVisitor):
    def __init__(self, parameter_names: set[str]) -> None:
        self._parameter_names = parameter_names
        self.evidence = {name: set() for name in parameter_names}
        self.unsafe_parameters: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = (node.left, *node.comparators)
        for operator, left, right in zip(
            node.ops,
            operands[:-1],
            operands[1:],
            strict=True,
        ):
            self._comparison_evidence(operator, left, right)
            self._comparison_evidence(operator, right, left)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            self._numeric_operation_evidence(node.left, node.right)
            self._numeric_operation_evidence(node.right, node.left)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            self._numeric_operation_evidence(node.target, node.value)
            self._numeric_operation_evidence(node.value, node.target)
            if _is_direct_self_attribute(node.target):
                self._add_name_evidence(node.value, _NUMERIC_EVIDENCE)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            self._mark_unsafe(node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            self._mark_unsafe(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._mark_unsafe(node.func)
        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            for descendant in ast.walk(argument):
                if isinstance(descendant, ast.Name):
                    self._mark_unsafe(descendant)
        self.generic_visit(node)

    def _comparison_evidence(
        self,
        operator: ast.cmpop,
        candidate: ast.expr,
        other: ast.expr,
    ) -> None:
        if not isinstance(candidate, ast.Name):
            return
        literal_type = primitive_literal_type(other)
        if literal_type is None:
            return
        if isinstance(operator, (ast.Is, ast.IsNot)):
            if literal_type == "bool":
                self._add_name_evidence(candidate, "bool")
            return
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            self._add_name_evidence(
                candidate,
                _NUMERIC_EVIDENCE
                if literal_type in _NUMERIC_TYPES
                else literal_type,
            )
            return
        if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            if literal_type in _NUMERIC_TYPES:
                self._add_name_evidence(candidate, _NUMERIC_EVIDENCE)

    def _numeric_operation_evidence(
        self,
        candidate: ast.expr,
        other: ast.expr,
    ) -> None:
        if not isinstance(candidate, ast.Name):
            return
        literal_type = primitive_literal_type(other)
        if literal_type in _NUMERIC_TYPES:
            self._add_name_evidence(candidate, _NUMERIC_EVIDENCE)
        elif isinstance(other, ast.Name) and other.id in self._parameter_names:
            self._add_name_evidence(candidate, _NUMERIC_EVIDENCE)

    def _add_name_evidence(self, node: ast.expr, evidence_type: str) -> None:
        if isinstance(node, ast.Name) and node.id in self._parameter_names:
            self.evidence[node.id].add(evidence_type)

    def _mark_unsafe(self, node: ast.Name) -> None:
        if node.id in self._parameter_names:
            self.unsafe_parameters.add(node.id)


def infer_primitive_parameter_types(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_names: set[str],
) -> PrimitiveParameterInference:
    evidence = {name: set() for name in parameter_names}
    for name, default in _defaults_by_name(node).items():
        if name not in evidence:
            continue
        default_type = primitive_literal_type(default)
        if default_type is not None:
            evidence[name].add(default_type)

    collector = _PrimitiveEvidenceCollector(parameter_names)
    for statement in node.body:
        collector.visit(statement)
    for name, values in collector.evidence.items():
        evidence[name].update(values)

    inferred: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    for name in sorted(parameter_names):
        if name in collector.unsafe_parameters:
            rejected.append((name, "is not a supported primitive."))
            continue
        type_name, reason = _resolve_evidence(evidence[name])
        if type_name is not None:
            inferred.append((name, type_name))
        elif reason is not None:
            rejected.append((name, reason))
    return PrimitiveParameterInference(tuple(inferred), tuple(rejected))


def primitive_annotation_type(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name) and annotation.id in _PRIMITIVE_TYPES:
        return annotation.id
    return None


def primitive_literal_type(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Constant) and type(value.value) in (int, float, str, bool):
        return type(value.value).__name__
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, (ast.UAdd, ast.USub))
        and isinstance(value.operand, ast.Constant)
        and type(value.operand.value) in (int, float)
    ):
        return type(value.operand.value).__name__
    return None


def _resolve_evidence(values: set[str]) -> tuple[str | None, str | None]:
    exact_types = values - {_NUMERIC_EVIDENCE}
    if len(exact_types) > 1:
        return None, "has conflicting primitive evidence."
    if exact_types:
        exact_type = next(iter(exact_types))
        if _NUMERIC_EVIDENCE in values and exact_type not in _NUMERIC_TYPES:
            return None, "has conflicting primitive evidence."
        return exact_type, None
    if _NUMERIC_EVIDENCE in values:
        return "int", None
    return None, None


def _defaults_by_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.expr]:
    positional = (*node.args.posonlyargs, *node.args.args)
    defaults: dict[str, ast.expr] = {
        argument.arg: default
        for argument, default in zip(
            positional[-len(node.args.defaults) :] if node.args.defaults else (),
            node.args.defaults,
            strict=True,
        )
    }
    defaults.update(
        {
            argument.arg: default
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
                strict=True,
            )
            if default is not None
        }
    )
    return defaults


def _is_direct_self_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )
