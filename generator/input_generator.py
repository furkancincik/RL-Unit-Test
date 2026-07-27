from __future__ import annotations

import ast
from dataclasses import dataclass

from cfg.control_flow_graph import CFGNode, ControlFlowGraph
from cfg.path_analyzer import ExecutionPath


@dataclass(frozen=True, slots=True)
class GeneratedInput:
    """Bir fonksiyon parametresi için üretilen test girdisini temsil eder."""

    parameter_name: str
    value: object
    condition: str
    branch: str


@dataclass(frozen=True, slots=True)
class TestInputResult:
    """Bir yürütme yolu için üretilen test girdilerini temsil eder."""

    values: tuple[GeneratedInput, ...]


class TestInputGenerator:
    """CFG koşullarına göre test girdileri üretir."""

    def generate(
        self,
        graph: ControlFlowGraph,
        path: ExecutionPath,
    ) -> TestInputResult:
        """
        Yürütme yolundaki basit koşulları analiz eder.

        Desteklenen koşullar:
        - Değişken ile sayısal sabit karşılaştırması
        - Değişken ile Boolean sabit karşılaştırması
        - Sabitin karşılaştırmanın sağında veya solunda olması
        - Doğrudan Boolean değişken kullanımı
        - not ile terslenen Boolean değişken kullanımı
        """
        generated_inputs: list[GeneratedInput] = []

        for index, edge_label in enumerate(path.edge_labels):
            node_id = path.node_ids[index]
            node = self._find_node(graph, node_id)

            if node.node_type not in {"if", "while"}:
                continue

            if edge_label not in {"True", "False"}:
                continue

            generated_input = self._generate_from_condition(
                condition=node.label,
                branch=edge_label,
            )

            if generated_input is not None:
                generated_inputs.append(generated_input)

        return TestInputResult(values=tuple(generated_inputs))

    @staticmethod
    def _find_node(
        graph: ControlFlowGraph,
        node_id: int,
    ) -> CFGNode:
        """Kimliğine göre CFG düğümünü bulur."""
        for node in graph.nodes:
            if node.node_id == node_id:
                return node

        raise ValueError(
            f"CFG içinde {node_id} kimlikli düğüm bulunamadı."
        )

    def _generate_from_condition(
        self,
        condition: str,
        branch: str,
    ) -> GeneratedInput | None:
        """Basit bir koşuldan test girdisi üretir."""

        try:
            expression = ast.parse(
                condition,
                mode="eval",
            ).body
        except SyntaxError:
            return None

        direct_boolean_input = self._generate_direct_boolean_input(
            expression=expression,
            condition=condition,
            branch=branch,
        )

        if direct_boolean_input is not None:
            return direct_boolean_input

        if not isinstance(expression, ast.Compare):
            return None

        if len(expression.ops) != 1:
            return None

        if len(expression.comparators) != 1:
            return None

        operator = expression.ops[0]
        comparator = expression.comparators[0]

        parameter_name: str
        constant_value: int | float | bool | None

        # Örnek:
        # x > 10
        # x >= -5
        # is_active == True
        if isinstance(expression.left, ast.Name):
            parameter_name = expression.left.id
            constant_value = self._extract_constant_value(comparator)

        # Örnek:
        # 10 < x
        # -5 < x
        # True == is_active
        elif isinstance(comparator, ast.Name):
            parameter_name = comparator.id
            constant_value = self._extract_constant_value(
                expression.left
            )

            reversed_operator = self._reverse_operator(operator)

            if reversed_operator is None:
                return None

            operator = reversed_operator

        else:
            return None

        if constant_value is None:
            return None

        if isinstance(constant_value, bool):
            generated_value = self._create_boolean_value(
                operator=operator,
                constant_value=constant_value,
                branch=branch,
            )
        else:
            generated_value = self._create_numeric_value(
                operator=operator,
                constant_value=constant_value,
                branch=branch,
            )

        if generated_value is None:
            return None

        return GeneratedInput(
            parameter_name=parameter_name,
            value=generated_value,
            condition=condition,
            branch=branch,
        )

    @staticmethod
    def _generate_direct_boolean_input(
        expression: ast.expr,
        condition: str,
        branch: str,
    ) -> GeneratedInput | None:
        """Doğrudan kullanılan Boolean koşulundan test girdisi üretir."""

        is_true_branch = branch == "True"

        # Örnek: if is_active:
        if isinstance(expression, ast.Name):
            return GeneratedInput(
                parameter_name=expression.id,
                value=is_true_branch,
                condition=condition,
                branch=branch,
            )

        # Örnek: if not is_active:
        if (
            isinstance(expression, ast.UnaryOp)
            and isinstance(expression.op, ast.Not)
            and isinstance(expression.operand, ast.Name)
        ):
            return GeneratedInput(
                parameter_name=expression.operand.id,
                value=not is_true_branch,
                condition=condition,
                branch=branch,
            )

        return None

    @staticmethod
    def _extract_constant_value(
        node: ast.expr,
    ) -> int | float | bool | None:
        """AST düğümünden sayısal veya Boolean sabit değeri çıkarır."""

        if isinstance(node, ast.Constant):
            value = node.value

            if isinstance(value, bool):
                return value

            if isinstance(value, (int, float)):
                return value

            return None

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
        ):
            value = node.operand.value

            if isinstance(value, bool):
                return None

            if isinstance(value, (int, float)):
                return -value

        return None

    @staticmethod
    def _reverse_operator(
        operator: ast.cmpop,
    ) -> ast.cmpop | None:
        """Sabit soldaysa karşılaştırma operatörünü tersine çevirir."""

        if isinstance(operator, ast.Gt):
            return ast.Lt()

        if isinstance(operator, ast.GtE):
            return ast.LtE()

        if isinstance(operator, ast.Lt):
            return ast.Gt()

        if isinstance(operator, ast.LtE):
            return ast.GtE()

        if isinstance(operator, ast.Eq):
            return ast.Eq()

        if isinstance(operator, ast.NotEq):
            return ast.NotEq()

        return None

    @staticmethod
    def _create_boolean_value(
        operator: ast.cmpop,
        constant_value: bool,
        branch: str,
    ) -> bool | None:
        """Boolean karşılaştırmasına göre test değeri üretir."""

        is_true_branch = branch == "True"

        if isinstance(operator, ast.Eq):
            return (
                constant_value
                if is_true_branch
                else not constant_value
            )

        if isinstance(operator, ast.NotEq):
            return (
                not constant_value
                if is_true_branch
                else constant_value
            )

        return None

    @staticmethod
    def _create_numeric_value(
        operator: ast.cmpop,
        constant_value: int | float,
        branch: str,
    ) -> int | float | None:
        """Sayısal karşılaştırmaya göre test değeri üretir."""

        is_true_branch = branch == "True"

        if isinstance(operator, ast.Gt):
            return (
                constant_value + 1
                if is_true_branch
                else constant_value
            )

        if isinstance(operator, ast.GtE):
            return (
                constant_value
                if is_true_branch
                else constant_value - 1
            )

        if isinstance(operator, ast.Lt):
            return (
                constant_value - 1
                if is_true_branch
                else constant_value
            )

        if isinstance(operator, ast.LtE):
            return (
                constant_value
                if is_true_branch
                else constant_value + 1
            )

        if isinstance(operator, ast.Eq):
            return (
                constant_value
                if is_true_branch
                else constant_value + 1
            )

        if isinstance(operator, ast.NotEq):
            return (
                constant_value + 1
                if is_true_branch
                else constant_value
            )

        return None