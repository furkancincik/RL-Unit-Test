from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionInfo:
    """Bir Python fonksiyonuna ait analiz sonuÃ§larÄ±nÄ± temsil eder."""

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


@dataclass
class AnalysisResult:
    """Bir Python dosyasÄ±na ait genel analiz sonuÃ§larÄ±nÄ± temsil eder."""

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
    """Python kaynak dosyalarÄ±nÄ± AST kullanarak analiz eder."""

    def analyze_file(self, file_path: str | Path) -> AnalysisResult:
        """
        Verilen Python dosyasÄ±nÄ± analiz eder.

        Args:
            file_path: Analiz edilecek Python dosyasÄ±nÄ±n yolu.

        Returns:
            Dosyaya ait analiz sonuÃ§larÄ±.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            ValueError: Dosya Python dosyasÄ± deÄŸilse.
            SyntaxError: Dosyada geÃ§ersiz Python sÃ¶zdizimi varsa.
        """
        path = Path(file_path)

        self._validate_file(path)

        source_code = path.read_text(encoding="utf-8")
        tree = ast.parse(source_code)

        functions = self._extract_functions(tree)

        return AnalysisResult(
            file_name=path.name,
            function_count=len(functions),
            class_count=self._count_nodes(tree, ast.ClassDef),
            if_count=self._count_nodes(tree, ast.If),
            for_count=self._count_nodes(
                tree,
                (ast.For, ast.AsyncFor),
            ),
            while_count=self._count_nodes(tree, ast.While),
            try_count=self._count_nodes(tree, ast.Try),
            return_count=self._count_nodes(tree, ast.Return),
            functions=functions,
        )

    @staticmethod
    def _validate_file(path: Path) -> None:
        """
        DosyanÄ±n varlÄ±ÄŸÄ±nÄ± ve uzantÄ±sÄ±nÄ± kontrol eder.

        Args:
            path: Kontrol edilecek dosya yolu.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            ValueError: Dosya Python dosyasÄ± deÄŸilse.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadÄ±: {path}")

        if not path.is_file():
            raise ValueError(f"Belirtilen yol bir dosya deÄŸil: {path}")

        if path.suffix.lower() != ".py":
            raise ValueError(
                "YalnÄ±zca Python dosyalarÄ± analiz edilebilir."
            )

    def _extract_functions(
        self,
        tree: ast.AST,
    ) -> list[FunctionInfo]:
        """
        AST iÃ§erisindeki fonksiyonlarÄ± ve Ã¶zelliklerini Ã§Ä±karÄ±r.

        Args:
            tree: Analiz edilecek AST aÄŸacÄ±.

        Returns:
            Fonksiyon analiz bilgilerinin listesi.
        """
        functions: list[FunctionInfo] = []

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            parameters = self._extract_parameters(node)

            return_count = self._count_nodes(
                node,
                ast.Return,
            )

            branch_count = self._calculate_branch_count(node)

            cyclomatic_complexity = 1 + branch_count

            risk_level = self._determine_risk_level(
                cyclomatic_complexity
            )

            has_docstring = ast.get_docstring(node) is not None
            has_return_type_hint = node.returns is not None

            typed_parameter_count = self._count_typed_parameters(
                node
            )

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
                    cyclomatic_complexity=cyclomatic_complexity,
                    risk_level=risk_level,
                    has_docstring=has_docstring,
                    has_return_type_hint=has_return_type_hint,
                    typed_parameter_count=typed_parameter_count,
                )
            )

        return functions

    @staticmethod
    def _extract_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """
        Fonksiyonun normal parametre isimlerini Ã§Ä±karÄ±r.

        Args:
            node: Fonksiyon AST dÃ¼ÄŸÃ¼mÃ¼.

        Returns:
            Parametre isimlerinin listesi.
        """
        return [
            argument.arg
            for argument in node.args.args
        ]

    @staticmethod
    def _count_typed_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """
        Type hint bulunan parametre sayÄ±sÄ±nÄ± hesaplar.

        Args:
            node: Fonksiyon AST dÃ¼ÄŸÃ¼mÃ¼.

        Returns:
            Type hint verilmiÅŸ parametre sayÄ±sÄ±.
        """
        return sum(
            argument.annotation is not None
            for argument in node.args.args
        )

    @staticmethod
    def _calculate_branch_count(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """
        Fonksiyon iÃ§erisindeki karar noktalarÄ±nÄ± hesaplar.

        Args:
            node: Fonksiyon AST dÃ¼ÄŸÃ¼mÃ¼.

        Returns:
            Karar noktasÄ± sayÄ±sÄ±.
        """
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
        node_types: type[ast.AST] | tuple[type[ast.AST], ...],
    ) -> int:
        """
        AST iÃ§erisinde belirtilen dÃ¼ÄŸÃ¼m tÃ¼rlerini sayar.

        Args:
            tree: Analiz edilecek AST aÄŸacÄ±.
            node_types: SayÄ±lacak AST dÃ¼ÄŸÃ¼m tÃ¼rleri.

        Returns:
            Bulunan dÃ¼ÄŸÃ¼m sayÄ±sÄ±.
        """
        return sum(
            isinstance(node, node_types)
            for node in ast.walk(tree)
        )

    @staticmethod
    def _determine_risk_level(
        cyclomatic_complexity: int,
    ) -> str:
        """
        Cyclomatic Complexity deÄŸerine gÃ¶re risk seviyesini belirler.

        Args:
            cyclomatic_complexity: Fonksiyonun karmaÅŸÄ±klÄ±k deÄŸeri.

        Returns:
            Low, Medium veya High risk seviyesi.
        """
        if cyclomatic_complexity <= 5:
            return "Low"

        if cyclomatic_complexity <= 10:
            return "Medium"

        return "High"

