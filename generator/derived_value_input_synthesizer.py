from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

from cfg.path_analyzer import ExecutionPath


@dataclass(frozen=True, slots=True, order=True)
class SymbolicInputReference:
    """Affine bir ifadenin gerçek dış girdi kaynağı."""

    parameter_name: str
    element_index: int | None = None


@dataclass(frozen=True, slots=True)
class AffineExpression:
    """Allowlist AST işlemleriyle kurulmuş değişmez affine ifade."""

    coefficients: tuple[tuple[SymbolicInputReference, float], ...] = ()
    constant: float = 0.0

    @classmethod
    def reference(cls, reference: SymbolicInputReference) -> AffineExpression:
        return cls(coefficients=((reference, 1.0),))

    @classmethod
    def literal(cls, value: int | float) -> AffineExpression:
        return cls(constant=float(value))

    def add(self, other: AffineExpression) -> AffineExpression:
        values = dict(self.coefficients)
        for reference, coefficient in other.coefficients:
            values[reference] = values.get(reference, 0.0) + coefficient
        return AffineExpression(
            coefficients=tuple(
                sorted(
                    (reference, coefficient)
                    for reference, coefficient in values.items()
                    if not math.isclose(coefficient, 0.0)
                )
            ),
            constant=self.constant + other.constant,
        )

    def scale(self, factor: float) -> AffineExpression:
        return AffineExpression(
            coefficients=tuple(
                (reference, coefficient * factor)
                for reference, coefficient in self.coefficients
            ),
            constant=self.constant * factor,
        )

    def subtract(self, other: AffineExpression) -> AffineExpression:
        return self.add(other.scale(-1.0))

    @property
    def is_constant(self) -> bool:
        return not self.coefficients


@dataclass(slots=True)
class _LoopActivation:
    node_id: int
    target_name: str
    iterable_name: str
    iteration_index: int = -1


class DerivedValueSynthesisError(ValueError):
    """Bir derived koşul güvenli affine input'a geri yayılamadığında oluşur."""


class UnsupportedDerivedValueSynthesisError(DerivedValueSynthesisError):
    """Derived provenance desteklenen güvenli affine modelin dışındadır."""


class DerivedValueInputSynthesizer:
    """Path-local affine provenance'i gerçek parametrelere geri yayar."""

    _NEGATED_OPERATOR = {
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "==": "!=",
        "!=": "==",
    }

    def apply(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """Desteklenen derived condition'ları yerinde dış girdilere uygular."""
        parameter_set = set(parameter_names)
        expressions: dict[str, AffineExpression] = {}
        unsupported_names: set[str] = set()
        active_loops: list[_LoopActivation] = []

        for step in path.steps:
            if step.node_type == "for":
                self._update_loop_context(
                    step_node_id=step.node_id,
                    step_label=step.node_label,
                    outgoing_edge=step.outgoing_edge_label,
                    parameter_names=parameter_set,
                    expressions=expressions,
                    active_loops=active_loops,
                )
                continue

            if step.node_type in {"Assign", "AnnAssign", "AugAssign"}:
                self._apply_statement(
                    statement_text=step.node_label,
                    parameter_names=parameter_set,
                    expressions=expressions,
                    unsupported_names=unsupported_names,
                )
                continue

            if (
                step.node_type in {"if", "while"}
                and step.outgoing_edge_label in {"True", "False"}
            ):
                self._apply_condition(
                    condition=step.node_label,
                    desired_result=(step.outgoing_edge_label == "True"),
                    require_single_parameter_affine=(
                        step.node_type == "while"
                    ),
                    parameter_names=parameter_set,
                    expressions=expressions,
                    unsupported_names=unsupported_names,
                    direct_values=direct_values,
                )

    def _update_loop_context(
        self,
        *,
        step_node_id: int,
        step_label: str,
        outgoing_edge: str | None,
        parameter_names: set[str],
        expressions: dict[str, AffineExpression],
        active_loops: list[_LoopActivation],
    ) -> None:
        binding = self._parse_loop_binding(step_label, parameter_names)
        matching_index = next(
            (
                index
                for index in range(len(active_loops) - 1, -1, -1)
                if active_loops[index].node_id == step_node_id
            ),
            None,
        )

        if outgoing_edge == "Iterate" and binding is not None:
            if matching_index is None:
                target_name, iterable_name = binding
                activation = _LoopActivation(
                    node_id=step_node_id,
                    target_name=target_name,
                    iterable_name=iterable_name,
                )
                active_loops.append(activation)
            else:
                del active_loops[matching_index + 1:]
                activation = active_loops[matching_index]

            activation.iteration_index += 1
            expressions[activation.target_name] = AffineExpression.reference(
                SymbolicInputReference(
                    parameter_name=activation.iterable_name,
                    element_index=activation.iteration_index,
                )
            )
            return

        if outgoing_edge == "Complete" and matching_index is not None:
            removed = active_loops[matching_index:]
            del active_loops[matching_index:]
            for activation in removed:
                expressions.pop(activation.target_name, None)

    def _apply_statement(
        self,
        *,
        statement_text: str,
        parameter_names: set[str],
        expressions: dict[str, AffineExpression],
        unsupported_names: set[str],
    ) -> None:
        try:
            statement = ast.parse(statement_text).body[0]
        except (SyntaxError, IndexError):
            return

        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                return
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name) or statement.value is None:
                return
            target_name = statement.target.id
            value = statement.value
        elif isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                return
            target_name = statement.target.id
            current = expressions.get(target_name)
            additional = self._parse_expression(
                statement.value,
                parameter_names=parameter_names,
                expressions=expressions,
                resolving={target_name},
            )
            if current is None and target_name in parameter_names:
                return
            if (
                current is None
                or additional is None
                or not isinstance(statement.op, (ast.Add, ast.Sub))
            ):
                expressions.pop(target_name, None)
                unsupported_names.add(target_name)
                return
            expressions[target_name] = (
                current.add(additional)
                if isinstance(statement.op, ast.Add)
                else current.subtract(additional)
            )
            unsupported_names.discard(target_name)
            return
        else:
            return

        expression = self._parse_expression(
            value,
            parameter_names=parameter_names,
            expressions=expressions,
            resolving={target_name},
        )
        if expression is None:
            expressions.pop(target_name, None)
            unsupported_names.add(target_name)
        else:
            expressions[target_name] = expression
            unsupported_names.discard(target_name)

    def _parse_expression(
        self,
        node: ast.expr,
        *,
        parameter_names: set[str],
        expressions: dict[str, AffineExpression],
        resolving: set[str],
    ) -> AffineExpression | None:
        numeric = self._numeric_literal(node)
        if numeric is not None:
            return AffineExpression.literal(numeric)

        if isinstance(node, ast.Name):
            if node.id in resolving:
                return None
            if node.id in expressions:
                return expressions[node.id]
            if node.id in parameter_names:
                return AffineExpression.reference(
                    SymbolicInputReference(parameter_name=node.id)
                )
            return None

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._parse_expression(
                node.operand,
                parameter_names=parameter_names,
                expressions=expressions,
                resolving=resolving,
            )
            return None if operand is None else operand.scale(-1.0)

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id not in parameter_names:
                return None
            try:
                index_value = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                return None
            if (
                isinstance(index_value, bool)
                or not isinstance(index_value, int)
                or index_value < 0
            ):
                return None
            return AffineExpression.reference(
                SymbolicInputReference(node.value.id, index_value)
            )

        if not isinstance(node, ast.BinOp):
            return None

        left = self._parse_expression(
            node.left,
            parameter_names=parameter_names,
            expressions=expressions,
            resolving=resolving,
        )
        right = self._parse_expression(
            node.right,
            parameter_names=parameter_names,
            expressions=expressions,
            resolving=resolving,
        )
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left.add(right)
        if isinstance(node.op, ast.Sub):
            return left.subtract(right)
        if isinstance(node.op, ast.Mult):
            if left.is_constant:
                return right.scale(left.constant)
            if right.is_constant:
                return left.scale(right.constant)
            return None
        if isinstance(node.op, ast.Div):
            if not right.is_constant or right.constant <= 0:
                return None
            return left.scale(1.0 / right.constant)
        return None

    def _apply_condition(
        self,
        *,
        condition: str,
        desired_result: bool,
        require_single_parameter_affine: bool,
        parameter_names: set[str],
        expressions: dict[str, AffineExpression],
        unsupported_names: set[str],
        direct_values: dict[str, Any],
    ) -> None:
        try:
            expression = ast.parse(condition, mode="eval").body
        except SyntaxError:
            return

        comparisons = self._required_comparisons(expression, desired_result)
        for comparison, comparison_result in comparisons:
            referenced_names = {
                node.id
                for node in ast.walk(comparison)
                if isinstance(node, ast.Name)
            }
            ambiguous_names = referenced_names & unsupported_names
            if ambiguous_names:
                error_type = (
                    UnsupportedDerivedValueSynthesisError
                    if require_single_parameter_affine
                    else DerivedValueSynthesisError
                )
                raise error_type(
                    "Derived koşul desteklenmeyen provenance içeriyor: "
                    + ", ".join(sorted(ambiguous_names))
                )
            if require_single_parameter_affine:
                self._validate_single_parameter_while_bindings(
                    referenced_names=referenced_names,
                    expressions=expressions,
                )
            self._satisfy_comparison(
                comparison=comparison,
                desired_result=comparison_result,
                parameter_names=parameter_names,
                expressions=expressions,
                direct_values=direct_values,
            )

    @staticmethod
    def _validate_single_parameter_while_bindings(
        *,
        referenced_names: set[str],
        expressions: dict[str, AffineExpression],
    ) -> None:
        """Derived while local'lerini tek dış parametreli affine biçimle sınırlar."""
        for name in referenced_names:
            expression = expressions.get(name)
            if expression is None or expression.is_constant:
                continue

            if len(expression.coefficients) != 1:
                raise UnsupportedDerivedValueSynthesisError(
                    "Derived while başlangıcı tam olarak bir dış "
                    f"parametreye bağlı olmalıdır: {name}"
                )

            reference, coefficient = expression.coefficients[0]
            if (
                reference.element_index is not None
                or not (
                    math.isclose(coefficient, 1.0)
                    or math.isclose(coefficient, -1.0)
                )
            ):
                raise UnsupportedDerivedValueSynthesisError(
                    "Derived while başlangıcı yalnız ±1 katsayılı "
                    f"doğrudan parametre bağını destekler: {name}"
                )

    def _required_comparisons(
        self,
        expression: ast.expr,
        desired_result: bool,
    ) -> tuple[tuple[ast.Compare, bool], ...]:
        if isinstance(expression, ast.Compare):
            return ((expression, desired_result),)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return self._required_comparisons(expression.operand, not desired_result)
        if isinstance(expression, ast.BoolOp):
            all_required = (
                isinstance(expression.op, ast.And) and desired_result
            ) or (
                isinstance(expression.op, ast.Or) and not desired_result
            )
            if all_required:
                return tuple(
                    comparison
                    for child in expression.values
                    for comparison in self._required_comparisons(child, desired_result)
                )
        return ()

    def _satisfy_comparison(
        self,
        *,
        comparison: ast.Compare,
        desired_result: bool,
        parameter_names: set[str],
        expressions: dict[str, AffineExpression],
        direct_values: dict[str, Any],
    ) -> None:
        if len(comparison.ops) != 1 or len(comparison.comparators) != 1:
            return
        if not any(
            isinstance(node, ast.Name) and node.id in expressions
            for node in ast.walk(comparison)
        ):
            return
        operator = self._operator_text(comparison.ops[0])
        if operator is None:
            return
        if not desired_result:
            operator = self._NEGATED_OPERATOR[operator]

        left = self._parse_expression(
            comparison.left,
            parameter_names=parameter_names,
            expressions=expressions,
            resolving=set(),
        )
        right = self._parse_expression(
            comparison.comparators[0],
            parameter_names=parameter_names,
            expressions=expressions,
            resolving=set(),
        )
        if left is None or right is None:
            return
        difference = left.subtract(right)
        if not difference.coefficients:
            return
        current = self._evaluate(difference, direct_values)
        if current is None or self._compare(current, operator, 0.0):
            return

        target = self._target_value(operator)
        for reference, coefficient in difference.coefficients:
            old_value = self._reference_value(reference, direct_values)
            if old_value is None or math.isclose(coefficient, 0.0):
                continue
            raw_value = old_value + ((target - current) / coefficient)
            candidates = self._numeric_candidates(raw_value, old_value)
            for candidate in candidates:
                self._set_reference_value(reference, candidate, direct_values)
                updated = self._evaluate(difference, direct_values)
                if updated is not None and self._compare(updated, operator, 0.0):
                    return
            self._set_reference_value(reference, old_value, direct_values)

        raise DerivedValueSynthesisError(
            f"Derived koşul dış girdilere geri yayılamadı: {ast.unparse(comparison)}"
        )

    @staticmethod
    def _parse_loop_binding(
        label: str,
        parameter_names: set[str],
    ) -> tuple[str, str] | None:
        try:
            expression = ast.parse(label, mode="eval").body
        except SyntaxError:
            return None
        if (
            not isinstance(expression, ast.Compare)
            or len(expression.ops) != 1
            or not isinstance(expression.ops[0], ast.In)
            or not isinstance(expression.left, ast.Name)
            or len(expression.comparators) != 1
            or not isinstance(expression.comparators[0], ast.Name)
            or expression.comparators[0].id not in parameter_names
        ):
            return None
        return expression.left.id, expression.comparators[0].id

    @staticmethod
    def _numeric_literal(node: ast.expr) -> float | None:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _operator_text(operator: ast.cmpop) -> str | None:
        return {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }.get(type(operator))

    @staticmethod
    def _target_value(operator: str) -> float:
        if operator in {">", "!="}:
            return 1.0
        if operator == "<":
            return -1.0
        return 0.0

    def _evaluate(
        self,
        expression: AffineExpression,
        direct_values: dict[str, Any],
    ) -> float | None:
        result = expression.constant
        for reference, coefficient in expression.coefficients:
            value = self._reference_value(reference, direct_values)
            if value is None:
                return None
            result += coefficient * value
        return result

    @staticmethod
    def _reference_value(
        reference: SymbolicInputReference,
        direct_values: dict[str, Any],
    ) -> float | None:
        if reference.element_index is None:
            value = direct_values.get(reference.parameter_name, 0)
        else:
            collection = direct_values.get(reference.parameter_name)
            if not isinstance(collection, (list, tuple)):
                return None
            if reference.element_index >= len(collection):
                return None
            value = collection[reference.element_index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _set_reference_value(
        reference: SymbolicInputReference,
        value: int | float,
        direct_values: dict[str, Any],
    ) -> None:
        if reference.element_index is None:
            direct_values[reference.parameter_name] = value
            return
        collection = direct_values.get(reference.parameter_name)
        if not isinstance(collection, (list, tuple)):
            return
        values = list(collection)
        values[reference.element_index] = value
        direct_values[reference.parameter_name] = (
            tuple(values) if isinstance(collection, tuple) else values
        )

    @staticmethod
    def _numeric_candidates(raw_value: float, old_value: float) -> tuple[int | float, ...]:
        if float(old_value).is_integer():
            values = {math.floor(raw_value), math.ceil(raw_value)}
            values.update({value - 1 for value in tuple(values)})
            values.update({value + 1 for value in tuple(values)})
            return tuple(sorted(values, key=lambda value: (abs(value - old_value), value)))
        return (raw_value, raw_value - 1.0, raw_value + 1.0)

    @staticmethod
    def _compare(left: float, operator: str, right: float) -> bool:
        return {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": math.isclose(left, right),
            "!=": not math.isclose(left, right),
        }[operator]
