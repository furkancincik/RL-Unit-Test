from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CFGNode:
    """Kontrol akış grafiğindeki bir düğümü temsil eder."""

    node_id: int
    label: str
    node_type: str
    line_number: int | None = None


@dataclass
class CFGEdge:
    """İki CFG düğümü arasındaki yönlü bağlantıyı temsil eder."""

    source_id: int
    target_id: int
    label: str | None = None


@dataclass
class ControlFlowGraph:
    """Bir fonksiyona ait kontrol akış grafiğini temsil eder."""

    function_name: str
    nodes: list[CFGNode] = field(default_factory=list)
    edges: list[CFGEdge] = field(default_factory=list)


class ControlFlowGraphBuilder:
    """Python fonksiyonlarından kontrol akış grafiği üretir."""

    def __init__(self) -> None:
        self._node_counter = 0
        self._graph: ControlFlowGraph | None = None

    def build_from_file(
        self,
        file_path: str | Path,
    ) -> list[ControlFlowGraph]:
        """
        Python dosyasındaki fonksiyonlar için CFG üretir.

        Args:
            file_path: Analiz edilecek Python dosyasının yolu.

        Returns:
            Fonksiyonlara ait kontrol akış graflarının listesi.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            ValueError: Dosya Python dosyası değilse.
            SyntaxError: Python sözdizimi geçersizse.
        """
        path = Path(file_path)

        self._validate_file(path)

        source_code = path.read_text(encoding="utf-8")
        tree = ast.parse(source_code)

        graphs: list[ControlFlowGraph] = []

        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                graphs.append(self._build_function_graph(node))

        return graphs

    def _build_function_graph(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ControlFlowGraph:
        """Tek bir fonksiyon için CFG oluşturur."""
        self._node_counter = 0
        self._graph = ControlFlowGraph(
            function_name=function_node.name,
        )

        start_node = self._create_node(
            label="START",
            node_type="start",
            line_number=function_node.lineno,
        )

        end_node = self._create_node(
            label="END",
            node_type="end",
        )

        remaining_paths = self._build_block(
            statements=function_node.body,
            incoming_paths=[(start_node.node_id, None)],
            end_node_id=end_node.node_id,
        )

        for source_id, edge_label in remaining_paths:
            self._add_edge(
                source_id=source_id,
                target_id=end_node.node_id,
                label=edge_label,
            )

        return self._graph

    def _build_block(
        self,
        statements: list[ast.stmt],
        incoming_paths: list[tuple[int, str | None]],
        end_node_id: int,
    ) -> list[tuple[int, str | None]]:
        """
        Bir kod bloğundaki ifadeleri sırayla CFG'ye ekler.

        Returns:
            Bloğun devam edebilen çıkış yolları.
        """
        current_paths = incoming_paths

        for statement in statements:
            if not current_paths:
                break

            if isinstance(statement, ast.If):
                current_paths = self._build_if_statement(
                    statement=statement,
                    incoming_paths=current_paths,
                    end_node_id=end_node_id,
                )
                continue

            if isinstance(statement, ast.While):
                current_paths = self._build_while_statement(
                    statement=statement,
                    incoming_paths=current_paths,
                    end_node_id=end_node_id,
                )
                continue

            if isinstance(statement, (ast.For, ast.AsyncFor)):
                current_paths = self._build_for_statement(
                    statement=statement,
                    incoming_paths=current_paths,
                    end_node_id=end_node_id,
                )
                continue

            if isinstance(statement, ast.Try):
                current_paths = self._build_try_statement(
                    statement=statement,
                    incoming_paths=current_paths,
                    end_node_id=end_node_id,
                )
                continue

            if isinstance(statement, ast.Return):
                return_node = self._create_node(
                    label=self._statement_to_text(statement),
                    node_type="return",
                    line_number=statement.lineno,
                )

                self._connect_paths(
                    incoming_paths=current_paths,
                    target_id=return_node.node_id,
                )

                self._add_edge(
                    source_id=return_node.node_id,
                    target_id=end_node_id,
                )

                current_paths = []
                continue

            statement_node = self._create_node(
                label=self._statement_to_text(statement),
                node_type=type(statement).__name__,
                line_number=getattr(statement, "lineno", None),
            )

            self._connect_paths(
                incoming_paths=current_paths,
                target_id=statement_node.node_id,
            )

            current_paths = [(statement_node.node_id, None)]

        return current_paths

    def _build_if_statement(
        self,
        statement: ast.If,
        incoming_paths: list[tuple[int, str | None]],
        end_node_id: int,
    ) -> list[tuple[int, str | None]]:
        """Bir if ifadesinin True ve False yollarını oluşturur."""
        condition_node = self._create_node(
            label=ast.unparse(statement.test),
            node_type="if",
            line_number=statement.lineno,
        )

        self._connect_paths(
            incoming_paths=incoming_paths,
            target_id=condition_node.node_id,
        )

        true_paths = self._build_block(
            statements=statement.body,
            incoming_paths=[(condition_node.node_id, "True")],
            end_node_id=end_node_id,
        )

        if statement.orelse:
            false_paths = self._build_block(
                statements=statement.orelse,
                incoming_paths=[(condition_node.node_id, "False")],
                end_node_id=end_node_id,
            )
        else:
            false_paths = [
                (condition_node.node_id, "False")
            ]

        return true_paths + false_paths

    def _build_while_statement(
        self,
        statement: ast.While,
        incoming_paths: list[tuple[int, str | None]],
        end_node_id: int,
    ) -> list[tuple[int, str | None]]:
        """While döngüsünün kontrol akışını oluşturur."""
        condition_node = self._create_node(
            label=ast.unparse(statement.test),
            node_type="while",
            line_number=statement.lineno,
        )

        self._connect_paths(
            incoming_paths=incoming_paths,
            target_id=condition_node.node_id,
        )

        body_paths = self._build_block(
            statements=statement.body,
            incoming_paths=[
                (condition_node.node_id, "True")
            ],
            end_node_id=end_node_id,
        )

        for source_id, _ in body_paths:
            self._add_edge(
                source_id=source_id,
                target_id=condition_node.node_id,
                label="Loop",
            )

        if statement.orelse:
            return self._build_block(
                statements=statement.orelse,
                incoming_paths=[
                    (condition_node.node_id, "False")
                ],
                end_node_id=end_node_id,
            )

        return [
            (condition_node.node_id, "False")
        ]

    def _build_for_statement(
        self,
        statement: ast.For | ast.AsyncFor,
        incoming_paths: list[tuple[int, str | None]],
        end_node_id: int,
    ) -> list[tuple[int, str | None]]:
        """For döngüsünün kontrol akışını oluşturur."""
        loop_label = (
            f"{ast.unparse(statement.target)} in "
            f"{ast.unparse(statement.iter)}"
        )

        loop_node = self._create_node(
            label=loop_label,
            node_type="for",
            line_number=statement.lineno,
        )

        self._connect_paths(
            incoming_paths=incoming_paths,
            target_id=loop_node.node_id,
        )

        body_paths = self._build_block(
            statements=statement.body,
            incoming_paths=[
                (loop_node.node_id, "Iterate")
            ],
            end_node_id=end_node_id,
        )

        for source_id, _ in body_paths:
            self._add_edge(
                source_id=source_id,
                target_id=loop_node.node_id,
                label="Next",
            )

        if statement.orelse:
            return self._build_block(
                statements=statement.orelse,
                incoming_paths=[
                    (loop_node.node_id, "Complete")
                ],
                end_node_id=end_node_id,
            )

        return [
            (loop_node.node_id, "Complete")
        ]

    def _build_try_statement(
        self,
        statement: ast.Try,
        incoming_paths: list[tuple[int, str | None]],
        end_node_id: int,
    ) -> list[tuple[int, str | None]]:
        """Try/except yapısının kontrol akışını oluşturur."""
        try_node = self._create_node(
            label="try",
            node_type="try",
            line_number=statement.lineno,
        )

        self._connect_paths(
            incoming_paths=incoming_paths,
            target_id=try_node.node_id,
        )

        normal_paths = self._build_block(
            statements=statement.body,
            incoming_paths=[
                (try_node.node_id, "Success")
            ],
            end_node_id=end_node_id,
        )

        exception_paths: list[tuple[int, str | None]] = []

        for handler in statement.handlers:
            if handler.type is None:
                handler_label = "except"
            else:
                handler_label = (
                    f"except {ast.unparse(handler.type)}"
                )

            handler_node = self._create_node(
                label=handler_label,
                node_type="except",
                line_number=handler.lineno,
            )

            self._add_edge(
                source_id=try_node.node_id,
                target_id=handler_node.node_id,
                label="Exception",
            )

            handler_paths = self._build_block(
                statements=handler.body,
                incoming_paths=[
                    (handler_node.node_id, None)
                ],
                end_node_id=end_node_id,
            )

            exception_paths.extend(handler_paths)

        if statement.orelse:
            normal_paths = self._build_block(
                statements=statement.orelse,
                incoming_paths=normal_paths,
                end_node_id=end_node_id,
            )

        combined_paths = normal_paths + exception_paths

        if statement.finalbody:
            combined_paths = self._build_block(
                statements=statement.finalbody,
                incoming_paths=combined_paths,
                end_node_id=end_node_id,
            )

        return combined_paths

    def _create_node(
        self,
        label: str,
        node_type: str,
        line_number: int | None = None,
    ) -> CFGNode:
        """Yeni bir CFG düğümü oluşturur."""
        if self._graph is None:
            raise RuntimeError("CFG henüz başlatılmadı.")

        self._node_counter += 1

        node = CFGNode(
            node_id=self._node_counter,
            label=label,
            node_type=node_type,
            line_number=line_number,
        )

        self._graph.nodes.append(node)

        return node

    def _add_edge(
        self,
        source_id: int,
        target_id: int,
        label: str | None = None,
    ) -> None:
        """CFG'ye yönlü bir kenar ekler."""
        if self._graph is None:
            raise RuntimeError("CFG henüz başlatılmadı.")

        self._graph.edges.append(
            CFGEdge(
                source_id=source_id,
                target_id=target_id,
                label=label,
            )
        )

    def _connect_paths(
        self,
        incoming_paths: list[tuple[int, str | None]],
        target_id: int,
    ) -> None:
        """Önceki yolları verilen hedef düğüme bağlar."""
        for source_id, edge_label in incoming_paths:
            self._add_edge(
                source_id=source_id,
                target_id=target_id,
                label=edge_label,
            )

    @staticmethod
    def _statement_to_text(statement: ast.stmt) -> str:
        """AST ifadesini okunabilir Python metnine dönüştürür."""
        return ast.unparse(statement)

    @staticmethod
    def _validate_file(path: Path) -> None:
        """Analiz edilecek dosyanın geçerliliğini kontrol eder."""
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