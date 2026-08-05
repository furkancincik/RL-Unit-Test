from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


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

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

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
                )
            )

        return functions

    @staticmethod
    def _extract_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        return [
            argument.arg
            for argument in node.args.args
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

        for argument in node.args.args:
            if argument.annotation is None:
                continue

            parameter_types[argument.arg] = ast.unparse(
                argument.annotation
            )

        return parameter_types

    @staticmethod
    def _count_typed_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        return sum(
            argument.annotation is not None
            for argument in node.args.args
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