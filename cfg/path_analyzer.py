from __future__ import annotations

from dataclasses import dataclass, field

from cfg.control_flow_graph import CFGNode, ControlFlowGraph


@dataclass(frozen=True, slots=True)
class PathStep:
    """
    Bir yürütme yolu üzerindeki tek bir CFG adımını temsil eder.

    Attributes:
        node_id:
            CFG düğümünün benzersiz kimliği.

        node_label:
            Düğümün temsil ettiği kaynak kod ifadesi.
            Örnek: ``score >= 85`` veya ``return "Orta"``.

        node_type:
            Düğümün türü.
            Örnek: ``if``, ``return``, ``start`` veya ``end``.

        line_number:
            Düğümün kaynak dosyadaki satır numarası.

        outgoing_edge_label:
            Bu düğümden sonraki düğüme geçerken kullanılan
            kenarın etiketi.
            Örnek: ``True``, ``False``, ``Loop`` veya ``None``.
    """

    node_id: int
    node_label: str
    node_type: str
    line_number: int | None
    outgoing_edge_label: str | None = None


@dataclass(slots=True)
class ExecutionPath:
    """
    CFG üzerinde bulunan bir yürütme yolunu temsil eder.

    Mevcut ``node_ids`` ve ``edge_labels`` alanları geriye dönük
    uyumluluk amacıyla korunmuştur. Yeni alanlar sayesinde yol
    üzerindeki koşul, return ve diğer CFG düğümlerinin içerikleri
    de sonraki katmanlara aktarılabilir.

    Attributes:
        node_ids:
            Yol üzerinde ziyaret edilen CFG düğümlerinin kimlikleri.

        edge_labels:
            Düğümler arasındaki geçişlerin etiketleri.

        node_labels:
            Yol üzerindeki CFG düğümlerinin okunabilir ifadeleri.

        node_types:
            Yol üzerindeki CFG düğümlerinin türleri.

        line_numbers:
            Düğümlerin kaynak dosyadaki satır numaraları.
    """

    node_ids: list[int]
    edge_labels: list[str | None]
    node_labels: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)
    line_numbers: list[int | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Yürütme yolu verilerinin kendi içinde tutarlı olmasını doğrular.

        Eski testlerde yalnızca ``node_ids`` ve ``edge_labels``
        kullanılmış olabileceğinden, yeni metadata alanlarının boş
        bırakılmasına izin verilir.
        """
        if not self.node_ids:
            raise ValueError(
                "ExecutionPath en az bir CFG düğümü içermelidir."
            )

        expected_edge_count = len(self.node_ids) - 1

        if len(self.edge_labels) != expected_edge_count:
            raise ValueError(
                "edge_labels sayısı, node_ids sayısından bir eksik "
                "olmalıdır."
            )

        self._validate_optional_metadata(
            field_name="node_labels",
            values=self.node_labels,
        )
        self._validate_optional_metadata(
            field_name="node_types",
            values=self.node_types,
        )
        self._validate_optional_metadata(
            field_name="line_numbers",
            values=self.line_numbers,
        )

    @property
    def steps(self) -> tuple[PathStep, ...]:
        """
        Yürütme yolunu ilişkili ``PathStep`` nesneleri olarak döndürür.

        Metadata bulunmayan eski ExecutionPath nesnelerinde boş tuple
        döndürülür.
        """
        if not self.has_node_metadata:
            return ()

        steps: list[PathStep] = []

        for index, node_id in enumerate(self.node_ids):
            outgoing_edge_label = (
                self.edge_labels[index]
                if index < len(self.edge_labels)
                else None
            )

            steps.append(
                PathStep(
                    node_id=node_id,
                    node_label=self.node_labels[index],
                    node_type=self.node_types[index],
                    line_number=self.line_numbers[index],
                    outgoing_edge_label=outgoing_edge_label,
                )
            )

        return tuple(steps)

    @property
    def has_node_metadata(self) -> bool:
        """Yolun CFG düğüm metadata bilgilerini içerip içermediğini belirtir."""
        return (
            len(self.node_labels) == len(self.node_ids)
            and len(self.node_types) == len(self.node_ids)
            and len(self.line_numbers) == len(self.node_ids)
        )

    @property
    def condition_steps(self) -> tuple[PathStep, ...]:
        """
        Yol üzerindeki karar düğümlerini döndürür.

        ``if`` ve ``while`` düğümleri koşul adımı olarak değerlendirilir.
        """
        return tuple(
            step
            for step in self.steps
            if step.node_type in {"if", "while"}
        )

    @property
    def loop_steps(self) -> tuple[PathStep, ...]:
        """
        Yol üzerinde bulunan ``while`` ve ``for`` döngü düğümlerini
        ziyaret sırasıyla döndürür.
        """
        return tuple(
            step
            for step in self.steps
            if step.node_type in {"while", "for"}
        )

    @property
    def contains_loop(self) -> bool:
        """Yürütme yolu en az bir döngü düğümü içeriyorsa True döndürür."""
        return bool(self.loop_steps)

    @property
    def loop_iteration_count(self) -> int:
        """
        Yol üzerinde tamamlanan sınırlı döngü tekrar sayısını döndürür.

        ``while`` döngülerinde ``Loop``, ``for`` döngülerinde ``Next``
        etiketli geri dönüş kenarları bir tamamlanmış iterasyonu temsil
        eder.
        """
        return sum(
            edge_label in {"Loop", "Next"}
            for edge_label in self.edge_labels
        )

    @property
    def is_zero_iteration_loop_path(self) -> bool:
        """
        Döngü düğümü içerip gövdesini hiç çalıştırmayan yolu belirtir.
        """
        return (
            self.contains_loop
            and not any(
                step.outgoing_edge_label in {"True", "Iterate"}
                for step in self.loop_steps
            )
        )

    @property
    def return_step(self) -> PathStep | None:
        """Yol üzerindeki son return düğümünü döndürür."""
        for step in reversed(self.steps):
            if step.node_type == "return":
                return step

        return None

    def _validate_optional_metadata(
        self,
        field_name: str,
        values: list[object],
    ) -> None:
        """
        İsteğe bağlı metadata alanının uzunluğunu doğrular.

        Alan boş olabilir; fakat doluysa düğüm sayısıyla aynı uzunlukta
        olmak zorundadır.
        """
        if values and len(values) != len(self.node_ids):
            raise ValueError(
                f"{field_name} boş olmalı veya node_ids ile aynı "
                "sayıda öğe içermelidir."
            )


class CFGPathAnalyzer:
    """Control Flow Graph üzerindeki yürütme yollarını analiz eder."""

    def find_paths(
        self,
        graph: ControlFlowGraph,
        max_visits_per_node: int = 2,
    ) -> list[ExecutionPath]:
        """
        START düğümünden END düğümüne giden yürütme yollarını çıkarır.

        Döngülerin sonsuz sayıda yol üretmesini önlemek için aynı
        düğümün ziyaret edilme sayısı sınırlandırılır. Varsayılan
        ``max_visits_per_node=2`` değeri döngüler için sıfır ve bir
        iterasyonlu yolların çıkarılmasını sağlar. Daha yüksek değerler
        kontrollü biçimde ek iterasyon yolları üretir.

        Args:
            graph:
                Yürütme yolları çıkarılacak Control Flow Graph.

            max_visits_per_node:
                Bir CFG düğümünün tek bir yol içerisinde en fazla kaç
                kez ziyaret edilebileceği.

        Returns:
            CFG düğüm bilgilerini de içeren yürütme yolları.

        Raises:
            TypeError:
                graph geçerli bir ControlFlowGraph değilse veya ziyaret
                sınırı integer değilse.

            ValueError:
                Ziyaret sınırı geçersizse ya da START/END düğümü yoksa.
        """
        self._validate_graph(graph)
        self._validate_max_visits(max_visits_per_node)

        start_node_id = self._find_node_id(
            graph=graph,
            node_type="start",
        )
        end_node_id = self._find_node_id(
            graph=graph,
            node_type="end",
        )

        adjacency = self._build_adjacency(graph)
        node_lookup = self._build_node_lookup(graph)

        paths: list[ExecutionPath] = []

        self._depth_first_search(
            current_id=start_node_id,
            end_id=end_node_id,
            adjacency=adjacency,
            node_lookup=node_lookup,
            current_nodes=[start_node_id],
            current_labels=[],
            visit_counts={start_node_id: 1},
            paths=paths,
            max_visits_per_node=max_visits_per_node,
        )

        return paths

    def _depth_first_search(
        self,
        current_id: int,
        end_id: int,
        adjacency: dict[int, list[tuple[int, str | None]]],
        node_lookup: dict[int, CFGNode],
        current_nodes: list[int],
        current_labels: list[str | None],
        visit_counts: dict[int, int],
        paths: list[ExecutionPath],
        max_visits_per_node: int,
    ) -> None:
        """
        CFG üzerinde sınırlı derinlik öncelikli arama gerçekleştirir.
        """
        if current_id == end_id:
            paths.append(
                self._create_execution_path(
                    node_ids=current_nodes,
                    edge_labels=current_labels,
                    node_lookup=node_lookup,
                )
            )
            return

        for target_id, edge_label in adjacency.get(current_id, []):
            visit_count = visit_counts.get(target_id, 0)

            if visit_count >= max_visits_per_node:
                continue

            visit_counts[target_id] = visit_count + 1
            current_nodes.append(target_id)
            current_labels.append(edge_label)

            self._depth_first_search(
                current_id=target_id,
                end_id=end_id,
                adjacency=adjacency,
                node_lookup=node_lookup,
                current_nodes=current_nodes,
                current_labels=current_labels,
                visit_counts=visit_counts,
                paths=paths,
                max_visits_per_node=max_visits_per_node,
            )

            current_labels.pop()
            current_nodes.pop()

            if visit_count == 0:
                del visit_counts[target_id]
            else:
                visit_counts[target_id] = visit_count

    @staticmethod
    def _create_execution_path(
        node_ids: list[int],
        edge_labels: list[str | None],
        node_lookup: dict[int, CFGNode],
    ) -> ExecutionPath:
        """
        CFG düğümlerinden metadata içeren ExecutionPath oluşturur.
        """
        path_nodes: list[CFGNode] = []

        for node_id in node_ids:
            try:
                path_nodes.append(node_lookup[node_id])
            except KeyError as error:
                raise RuntimeError(
                    f"CFG düğümü bulunamadı: {node_id}"
                ) from error

        return ExecutionPath(
            node_ids=node_ids.copy(),
            edge_labels=edge_labels.copy(),
            node_labels=[
                node.label
                for node in path_nodes
            ],
            node_types=[
                node.node_type
                for node in path_nodes
            ],
            line_numbers=[
                node.line_number
                for node in path_nodes
            ],
        )

    @staticmethod
    def _build_adjacency(
        graph: ControlFlowGraph,
    ) -> dict[int, list[tuple[int, str | None]]]:
        """CFG kenarlarından komşuluk listesi oluşturur."""
        adjacency: dict[int, list[tuple[int, str | None]]] = {}

        for edge in graph.edges:
            adjacency.setdefault(
                edge.source_id,
                [],
            ).append(
                (
                    edge.target_id,
                    edge.label,
                )
            )

        return adjacency

    @staticmethod
    def _build_node_lookup(
        graph: ControlFlowGraph,
    ) -> dict[int, CFGNode]:
        """CFG düğümleri için kimlik tabanlı erişim sözlüğü oluşturur."""
        node_lookup: dict[int, CFGNode] = {}

        for node in graph.nodes:
            if node.node_id in node_lookup:
                raise ValueError(
                    f"Tekrarlanan CFG düğüm kimliği: {node.node_id}"
                )

            node_lookup[node.node_id] = node

        return node_lookup

    @staticmethod
    def _find_node_id(
        graph: ControlFlowGraph,
        node_type: str,
    ) -> int:
        """Belirtilen türdeki ilk CFG düğümünün kimliğini döndürür."""
        for node in graph.nodes:
            if node.node_type == node_type:
                return node.node_id

        raise ValueError(
            f"CFG içinde '{node_type}' türünde düğüm bulunamadı."
        )

    @staticmethod
    def _validate_graph(
        graph: ControlFlowGraph,
    ) -> None:
        if not isinstance(graph, ControlFlowGraph):
            raise TypeError(
                "graph bir ControlFlowGraph örneği olmalıdır."
            )

    @staticmethod
    def _validate_max_visits(
        max_visits_per_node: int,
    ) -> None:
        if isinstance(max_visits_per_node, bool):
            raise TypeError(
                "max_visits_per_node bir integer olmalıdır."
            )

        if not isinstance(max_visits_per_node, int):
            raise TypeError(
                "max_visits_per_node bir integer olmalıdır."
            )

        if max_visits_per_node < 1:
            raise ValueError(
                "max_visits_per_node en az 1 olmalıdır."
            )
