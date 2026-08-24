from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from analyzer.simple_instance_method import find_analysis_target


class DataFlowOperationType(str, Enum):
    ASSIGNMENT = "ASSIGNMENT"
    AUGMENTED_ASSIGNMENT = "AUGMENTED_ASSIGNMENT"
    DERIVED_ASSIGNMENT = "DERIVED_ASSIGNMENT"


@dataclass(frozen=True, slots=True)
class VariableUpdate:
    variable_name: str
    operation_type: DataFlowOperationType
    expression: str
    dependencies: tuple[str, ...]
    line_number: int | None
    inside_loop: bool


@dataclass(frozen=True, slots=True)
class VariableDataFlow:
    variable_name: str
    updates: tuple[VariableUpdate, ...]

    @property
    def is_updated(self) -> bool:
        return len(self.updates) > 1

    @property
    def is_updated_inside_loop(self) -> bool:
        return any(
            update.inside_loop
            for update in self.updates
        )

    @property
    def dependencies(self) -> tuple[str, ...]:
        result: list[str] = []

        for update in self.updates:
            for dependency in update.dependencies:
                if dependency not in result:
                    result.append(dependency)

        return tuple(result)


@dataclass(frozen=True, slots=True)
class InferredNumericRange:
    """
    Statik analizle güvenli biçimde çıkarılabilen sayısal sınırlar.
    None değeri ilgili yönde güvenli bir sınır çıkarılamadığını gösterir.
    """

    variable_name: str
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True, slots=True)
class DataFlowAnalysisResult:
    variables: tuple[VariableDataFlow, ...]
    inferred_numeric_ranges: tuple[
        InferredNumericRange,
        ...
    ] = ()

    def get_variable(
        self,
        variable_name: str,
    ) -> VariableDataFlow | None:
        for variable in self.variables:
            if variable.variable_name == variable_name:
                return variable

        return None

    def get_numeric_range(
        self,
        variable_name: str,
    ) -> InferredNumericRange | None:
        for numeric_range in self.inferred_numeric_ranges:
            if numeric_range.variable_name == variable_name:
                return numeric_range

        return None


class DataFlowAnalyzer:
    """
    Python kaynak kodundaki değişken tanım ve güncellemelerini
    AST üzerinden analiz eder.

    Bu sınıf tam symbolic execution yapmaz.
    Amaç; değişkenlerin nereden geldiğini, hangi ifadelerle
    güncellendiğini ve loop içerisinde değişip değişmediğini
    genel biçimde belirlemektir.
    """

    def analyze_file(
        self,
        source_file: Path,
        function_name: str,
    ) -> DataFlowAnalysisResult:
        if not isinstance(source_file, Path):
            raise TypeError(
                "source_file Path olmalıdır."
            )

        if not source_file.exists():
            raise FileNotFoundError(
                source_file
            )

        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        if not function_name.strip():
            raise ValueError(
                "function_name boş olamaz."
            )

        source = source_file.read_text(
            encoding="utf-8"
        )

        return self.analyze_source(
            source=source,
            function_name=function_name,
        )

    def analyze_source(
        self,
        *,
        source: str,
        function_name: str,
    ) -> DataFlowAnalysisResult:
        if not isinstance(source, str):
            raise TypeError(
                "source string olmalıdır."
            )

        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        if not function_name.strip():
            raise ValueError(
                "function_name boş olamaz."
            )

        tree = ast.parse(source)

        function_node = self._find_function(
            tree=tree,
            function_name=function_name,
        )

        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ] = {}

        self._walk_statements(
            statements=function_node.body,
            updates_by_variable=updates_by_variable,
            inside_loop=False,
        )

        variables = tuple(
            VariableDataFlow(
                variable_name=variable_name,
                updates=tuple(updates),
            )
            for variable_name, updates
            in sorted(
                updates_by_variable.items()
            )
        )

        inferred_numeric_ranges = tuple(
            numeric_range
            for variable in variables
            if (
                numeric_range
                := self._infer_numeric_range(
                    variable
                )
            )
            is not None
        )

        return DataFlowAnalysisResult(
            variables=variables,
            inferred_numeric_ranges=(
                inferred_numeric_ranges
            ),
        )

    @staticmethod
    def _find_function(
        *,
        tree: ast.Module,
        function_name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        return find_analysis_target(tree, function_name)

    def _walk_statements(
        self,
        *,
        statements: list[ast.stmt],
        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ],
        inside_loop: bool,
    ) -> None:
        for statement in statements:

            if isinstance(
                statement,
                ast.Assign,
            ):
                self._handle_assign(
                    statement=statement,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                ast.AnnAssign,
            ):
                self._handle_ann_assign(
                    statement=statement,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                ast.AugAssign,
            ):
                self._handle_aug_assign(
                    statement=statement,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                ),
            ):
                self._walk_statements(
                    statements=statement.body,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=True,
                )

                self._walk_statements(
                    statements=statement.orelse,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                ast.If,
            ):
                self._walk_statements(
                    statements=statement.body,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

                self._walk_statements(
                    statements=statement.orelse,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                ast.Try,
            ):
                self._walk_statements(
                    statements=statement.body,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

                for handler in statement.handlers:
                    self._walk_statements(
                        statements=handler.body,
                        updates_by_variable=(
                            updates_by_variable
                        ),
                        inside_loop=inside_loop,
                    )

                self._walk_statements(
                    statements=statement.orelse,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

                self._walk_statements(
                    statements=statement.finalbody,
                    updates_by_variable=(
                        updates_by_variable
                    ),
                    inside_loop=inside_loop,
                )

    def _handle_assign(
        self,
        *,
        statement: ast.Assign,
        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ],
        inside_loop: bool,
    ) -> None:
        for target in statement.targets:
            if not isinstance(
                target,
                ast.Name,
            ):
                continue

            self._register_update(
                variable_name=target.id,
                value=statement.value,
                operation_type=(
                    self._classify_assignment(
                        statement.value
                    )
                ),
                line_number=getattr(
                    statement,
                    "lineno",
                    None,
                ),
                inside_loop=inside_loop,
                updates_by_variable=(
                    updates_by_variable
                ),
            )

    def _handle_ann_assign(
        self,
        *,
        statement: ast.AnnAssign,
        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ],
        inside_loop: bool,
    ) -> None:
        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return

        if statement.value is None:
            return

        self._register_update(
            variable_name=statement.target.id,
            value=statement.value,
            operation_type=(
                self._classify_assignment(
                    statement.value
                )
            ),
            line_number=getattr(
                statement,
                "lineno",
                None,
            ),
            inside_loop=inside_loop,
            updates_by_variable=(
                updates_by_variable
            ),
        )

    def _handle_aug_assign(
        self,
        *,
        statement: ast.AugAssign,
        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ],
        inside_loop: bool,
    ) -> None:
        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return

        expression = (
            f"{statement.target.id} "
            f"{self._operator_symbol(statement.op)}= "
            f"{ast.unparse(statement.value)}"
        )

        dependencies = tuple(
            dependency
            for dependency in self._extract_dependencies(
                statement.value
            )
            if dependency != statement.target.id
        )

        update = VariableUpdate(
            variable_name=statement.target.id,
            operation_type=(
                DataFlowOperationType.AUGMENTED_ASSIGNMENT
            ),
            expression=expression,
            dependencies=dependencies,
            line_number=getattr(
                statement,
                "lineno",
                None,
            ),
            inside_loop=inside_loop,
        )

        updates_by_variable.setdefault(
            statement.target.id,
            [],
        ).append(update)

    def _register_update(
        self,
        *,
        variable_name: str,
        value: ast.expr,
        operation_type: DataFlowOperationType,
        line_number: int | None,
        inside_loop: bool,
        updates_by_variable: dict[
            str,
            list[VariableUpdate],
        ],
    ) -> None:
        update = VariableUpdate(
            variable_name=variable_name,
            operation_type=operation_type,
            expression=ast.unparse(value),
            dependencies=(
                self._extract_dependencies(
                    value
                )
            ),
            line_number=line_number,
            inside_loop=inside_loop,
        )

        updates_by_variable.setdefault(
            variable_name,
            [],
        ).append(update)

    @staticmethod
    def _classify_assignment(
        value: ast.expr,
    ) -> DataFlowOperationType:
        if isinstance(
            value,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.Call,
                ast.Subscript,
            ),
        ):
            return (
                DataFlowOperationType.DERIVED_ASSIGNMENT
            )

        return DataFlowOperationType.ASSIGNMENT

    @staticmethod
    def _extract_dependencies(
        expression: ast.AST,
    ) -> tuple[str, ...]:
        dependencies: list[str] = []

        for node in ast.walk(
            expression
        ):
            if isinstance(
                node,
                ast.Name,
            ):
                if node.id not in dependencies:
                    dependencies.append(
                        node.id
                    )

        return tuple(
            dependencies
        )

    def _infer_numeric_range(
        self,
        variable: VariableDataFlow,
    ) -> InferredNumericRange | None:
        """
        Yalnızca kesin ve yönü belli pattern'lerden güvenli sınır çıkarır.

        Örnek:
            count = 0
            count += 1   -> count >= 0

            remaining = 10
            remaining -= 1 -> remaining <= 10

        Döngü sayısını tahmin etmez ve kesin son değer üretmez.
        """
        if not variable.updates:
            return None

        first_update = variable.updates[0]

        initial_value = self._parse_numeric_literal(
            first_update.expression
        )

        if initial_value is None:
            return None

        lower_bound: float | None = initial_value
        upper_bound: float | None = initial_value

        for update in variable.updates[1:]:
            if (
                update.operation_type
                != DataFlowOperationType.AUGMENTED_ASSIGNMENT
            ):
                # Sonraki normal/derived atama eski sınırı geçersiz
                # kılabileceği için güvenli çıkarım yapmıyoruz.
                return None

            delta = self._parse_augmented_numeric_delta(
                variable_name=variable.variable_name,
                expression=update.expression,
            )

            if delta is None:
                return None

            operator, value = delta

            if operator == "+":
                if value < 0:
                    return None

                # Artış varsa başlangıç değeri güvenli alt sınırdır.
                upper_bound = None

            elif operator == "-":
                if value < 0:
                    return None

                # Azalış varsa başlangıç değeri güvenli üst sınırdır.
                lower_bound = None

            else:
                return None

        return InferredNumericRange(
            variable_name=variable.variable_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @staticmethod
    def _parse_numeric_literal(
        expression: str,
    ) -> float | None:
        try:
            node = ast.parse(
                expression,
                mode="eval",
            ).body
        except SyntaxError:
            return None

        value: int | float | None = None

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            value = node.value

        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            raw_value = node.operand.value
            value = (
                -raw_value
                if isinstance(node.op, ast.USub)
                else raw_value
            )

        if value is None:
            return None

        return float(value)

    def _parse_augmented_numeric_delta(
        self,
        *,
        variable_name: str,
        expression: str,
    ) -> tuple[str, float] | None:
        try:
            module = ast.parse(expression)
        except SyntaxError:
            return None

        if (
            len(module.body) != 1
            or not isinstance(module.body[0], ast.AugAssign)
        ):
            return None

        statement = module.body[0]

        if (
            not isinstance(statement.target, ast.Name)
            or statement.target.id != variable_name
        ):
            return None

        value = self._numeric_value_from_ast(
            statement.value
        )

        if value is None:
            return None

        if isinstance(statement.op, ast.Add):
            return ("+", value)

        if isinstance(statement.op, ast.Sub):
            return ("-", value)

        return None

    @staticmethod
    def _numeric_value_from_ast(
        node: ast.AST,
    ) -> float | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            value = float(node.operand.value)

            if isinstance(node.op, ast.USub):
                return -value

            return value

        return None

    @staticmethod
    def _operator_symbol(
        operator: ast.operator,
    ) -> str:
        mapping: dict[type[ast.operator], str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }

        return mapping.get(
            type(operator),
            "?",
        )
