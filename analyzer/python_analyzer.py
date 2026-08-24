from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from analyzer.primitive_parameter_inference import infer_primitive_parameter_types
from analyzer.safe_custom_object import analyze_safe_custom_object_target
from analyzer.simple_instance_method import analyze_simple_instance_method


@dataclass
class FunctionInfo:
    """Bir Python fonksiyonuna ait analiz sonuçlarını temsil eder."""

    name: str
    parameters: list[str]
    line_number: int
    is_async: bool
    return_count: int
    branch_count: int
    cyclomatic_complexity: int
    risk_level: str
    has_docstring: bool
    has_return_type_hint: bool
    typed_parameter_count: int
    end_line_number: int = 0
    parameter_types: dict[str, str] = field(
        default_factory=dict
    )
    return_annotation: str | None = None
    qualified_name: str = ""
    is_nested: bool = False
    is_method: bool = False
    is_supported: bool = True
    unsupported_reason: str | None = None
    class_name: str | None = None
    constructor_parameters: list[str] = field(default_factory=list)
    constructor_parameter_types: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Bir Python dosyasına ait genel analiz sonuçlarını temsil eder."""

    file_name: str
    function_count: int
    class_count: int
    if_count: int
    for_count: int
    while_count: int
    try_count: int
    return_count: int
    functions: list[FunctionInfo]


class PythonAnalyzer:
    """Python kaynak dosyalarını AST kullanarak analiz eder."""

    def analyze_file(
        self,
        file_path: str | Path,
    ) -> AnalysisResult:
        path = Path(file_path)
        self._validate_file(path)

        source_code = path.read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source_code)
        functions = self._extract_functions(tree)

        return AnalysisResult(
            file_name=path.name,
            function_count=len(functions),
            class_count=self._count_nodes(
                tree,
                ast.ClassDef,
            ),
            if_count=self._count_nodes(
                tree,
                ast.If,
            ),
            for_count=self._count_nodes(
                tree,
                (
                    ast.For,
                    ast.AsyncFor,
                ),
            ),
            while_count=self._count_nodes(
                tree,
                ast.While,
            ),
            try_count=self._count_nodes(
                tree,
                ast.Try,
            ),
            return_count=self._count_nodes(
                tree,
                ast.Return,
            ),
            functions=functions,
        )

    @staticmethod
    def _validate_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Belirtilen yol bir dosya değil: {path}"
            )

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Yalnızca Python dosyaları analiz edilebilir."
            )

    def _extract_functions(
        self,
        tree: ast.AST,
    ) -> list[FunctionInfo]:
        functions: list[FunctionInfo] = []
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        function_nodes = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )

        for node in function_nodes:
            parameters = self._extract_parameters(node)
            parameter_types = self._extract_parameter_types(
                node
            )
            return_count = self._count_nodes(
                node,
                ast.Return,
            )
            branch_count = self._calculate_branch_count(
                node
            )
            cyclomatic_complexity = 1 + branch_count
            ancestry = self._function_ancestry(node, parents)
            is_nested = any(
                isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                for parent in ancestry
            )
            is_method = any(
                isinstance(parent, ast.ClassDef) for parent in ancestry
            )
            method_spec = None
            method_reason = None
            direct_parent = parents.get(node)
            if is_method and isinstance(direct_parent, ast.ClassDef):
                if not isinstance(parents.get(direct_parent), ast.Module):
                    method_reason = "Only top-level classes are supported."
                else:
                    method_spec, method_reason = analyze_simple_instance_method(
                        direct_parent,
                        node,
                    )
            unsupported_reason = self._unsupported_reason(
                node=node,
                direct_parent=direct_parent,
                is_nested=is_nested,
                is_method=is_method,
                method_reason=method_reason,
            )
            custom_object_reason = None
            if (
                isinstance(node, ast.FunctionDef)
                and (
                    isinstance(direct_parent, ast.Module)
                    or isinstance(direct_parent, ast.ClassDef)
                    and isinstance(parents.get(direct_parent), ast.Module)
                )
            ):
                _, custom_object_reason = analyze_safe_custom_object_target(
                    tree,
                    self._qualified_name(node, ancestry),
                )
            if unsupported_reason is None:
                unsupported_reason = custom_object_reason
            if method_spec is not None:
                parameters = [
                    parameter.name
                    for parameter in method_spec.method_parameters
                ]
                parameter_types = {
                    parameter.name: parameter.type_name
                    for parameter in method_spec.method_parameters
                }
            elif self._is_normal_instance_method(node, is_method=is_method):
                parameters = parameters[1:]
                parameter_types.pop("self", None)

            functions.append(
                FunctionInfo(
                    name=node.name,
                    parameters=parameters,
                    line_number=node.lineno,
                    is_async=isinstance(
                        node,
                        ast.AsyncFunctionDef,
                    ),
                    return_count=return_count,
                    branch_count=branch_count,
                    cyclomatic_complexity=(
                        cyclomatic_complexity
                    ),
                    risk_level=self._determine_risk_level(
                        cyclomatic_complexity
                    ),
                    has_docstring=(
                        ast.get_docstring(node) is not None
                    ),
                    has_return_type_hint=(
                        node.returns is not None
                    ),
                    typed_parameter_count=(
                        self._count_typed_parameters(node)
                    ),
                    end_line_number=(
                        node.end_lineno
                        if node.end_lineno is not None
                        else node.lineno
                    ),
                    parameter_types=parameter_types,
                    return_annotation=(
                        ast.unparse(node.returns)
                        if node.returns is not None
                        else None
                    ),
                    qualified_name=self._qualified_name(node, ancestry),
                    is_nested=is_nested,
                    is_method=is_method,
                    is_supported=unsupported_reason is None,
                    unsupported_reason=unsupported_reason,
                    class_name=(
                        method_spec.class_name
                        if method_spec is not None
                        else direct_parent.name
                        if isinstance(direct_parent, ast.ClassDef)
                        else None
                    ),
                    constructor_parameters=(
                        [
                            parameter.name
                            for parameter in method_spec.constructor_parameters
                        ]
                        if method_spec is not None
                        else []
                    ),
                    constructor_parameter_types=(
                        {
                            parameter.name: parameter.type_name
                            for parameter in method_spec.constructor_parameters
                        }
                        if method_spec is not None
                        else {}
                    ),
                )
            )

        return functions

    @staticmethod
    def _function_ancestry(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parents: dict[ast.AST, ast.AST],
    ) -> tuple[ast.AST, ...]:
        ancestry: list[ast.AST] = []
        current = parents.get(node)
        while current is not None:
            ancestry.append(current)
            current = parents.get(current)
        return tuple(ancestry)

    @staticmethod
    def _qualified_name(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        ancestry: tuple[ast.AST, ...],
    ) -> str:
        scope_names = [
            parent.name
            for parent in reversed(ancestry)
            if isinstance(
                parent,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ]
        return ".".join((*scope_names, node.name))

    @staticmethod
    def _unsupported_reason(
        *,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        direct_parent: ast.AST | None,
        is_nested: bool,
        is_method: bool,
        method_reason: str | None,
    ) -> str | None:
        if isinstance(node, ast.AsyncFunctionDef):
            return "Async functions are not supported by the production pipeline."
        if is_nested:
            return "Nested functions are not supported by the production pipeline."
        if is_method:
            return method_reason
        if not isinstance(direct_parent, ast.Module):
            return "Conditionally defined functions are not supported."
        if node.name.startswith("__") and node.name.endswith("__"):
            return "Dunder functions are excluded from automatic production runs."
        if node.args.posonlyargs:
            return "Positional-only parameters are not supported."
        if node.args.vararg is not None or node.args.kwarg is not None:
            return "Variadic function signatures are not supported."
        return None

    @staticmethod
    def _extract_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        return [
            argument.arg
            for argument in (*node.args.args, *node.args.kwonlyargs)
        ]

    @staticmethod
    def _extract_parameter_types(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, str]:
        """
        Fonksiyon parametrelerinin type hint ifadelerini metin
        biçiminde döndürür.

        Örnekler:
            ``int``, ``str``, ``list[int]``, ``dict[str, int]``.
        """
        parameter_types: dict[str, str] = {}

        arguments = (*node.args.args, *node.args.kwonlyargs)
        for argument in arguments:
            if argument.annotation is None:
                continue

            parameter_types[argument.arg] = ast.unparse(
                argument.annotation
            )

        inference = infer_primitive_parameter_types(
            node,
            {
                argument.arg
                for argument in arguments
                if argument.annotation is None
            },
        )
        parameter_types.update(inference.inferred_types)

        return parameter_types

    @staticmethod
    def _is_normal_instance_method(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_method: bool,
    ) -> bool:
        if not is_method or not node.args.args:
            return False
        if node.args.args[0].arg != "self":
            return False
        decorator_names = {
            decorator.id
            if isinstance(decorator, ast.Name)
            else decorator.attr
            if isinstance(decorator, ast.Attribute)
            else None
            for decorator in node.decorator_list
        }
        return not decorator_names.intersection({"staticmethod", "classmethod"})

    @staticmethod
    def _count_typed_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        return sum(
            argument.annotation is not None
            for argument in (*node.args.args, *node.args.kwonlyargs)
        )

    @staticmethod
    def _calculate_branch_count(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        branch_types = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.Match,
        )

        return sum(
            isinstance(child, branch_types)
            for child in ast.walk(node)
        )

    @staticmethod
    def _count_nodes(
        tree: ast.AST,
        node_types: (
            type[ast.AST]
            | tuple[type[ast.AST], ...]
        ),
    ) -> int:
        return sum(
            isinstance(node, node_types)
            for node in ast.walk(tree)
        )

    @staticmethod
    def _determine_risk_level(
        cyclomatic_complexity: int,
    ) -> str:
        if cyclomatic_complexity <= 5:
            return "Low"

        if cyclomatic_complexity <= 10:
            return "Medium"

        return "High"
