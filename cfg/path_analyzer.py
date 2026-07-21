from __future__ import annotations

from dataclasses import dataclass

from cfg.control_flow_graph import ControlFlowGraph


@dataclass
class ExecutionPath:
    """CFG üzerinde bulunan bir yürütme yolunu temsil eder."""

    node_ids: list[int]
    edge_labels: list[str | None]


class CFGPathAnalyzer:
    """Control Flow Graph üzerindeki yürütme yollarını analiz eder."""

    def find_paths(
        self,
        graph: ControlFlowGraph,
        max_visits_per_node: int = 2,
    ) -> list[ExecutionPath]:
        """
        START düğümünden END düğümüne giden yolları çıkarır.

        Döngüler sonsuz yol üretmesin diye aynı düğümün ziyaret
        sayısı sınırlandırılır.
        """
        start_node = self._find_node_id(graph, "start")
        end_node = self._find_node_id(graph, "end")

        adjacency: dict[int, list[tuple[int, str | None]]] = {}

        for edge in graph.edges:
            adjacency.setdefault(edge.source_id, []).append(
                (edge.target_id, edge.label)
            )

        paths: list[ExecutionPath] = []

        self._depth_first_search(
            current_id=start_node,
            end_id=end_node,
            adjacency=adjacency,
            current_nodes=[start_node],
            current_labels=[],
            visit_counts={start_node: 1},
            paths=paths,
            max_visits_per_node=max_visits_per_node,
        )

        return paths

    def _depth_first_search(
        self,
        current_id: int,
        end_id: int,
        adjacency: dict[int, list[tuple[int, str | None]]],
        current_nodes: list[int],
        current_labels: list[str | None],
        visit_counts: dict[int, int],
        paths: list[ExecutionPath],
        max_visits_per_node: int,
    ) -> None:
        """CFG üzerinde sınırlı derinlik öncelikli arama yapar."""
        if current_id == end_id:
            paths.append(
                ExecutionPath(
                    node_ids=current_nodes.copy(),
                    edge_labels=current_labels.copy(),
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
    def _find_node_id(
        graph: ControlFlowGraph,
        node_type: str,
    ) -> int:
        """Belirtilen türdeki ilk düğümün kimliğini bulur."""
        for node in graph.nodes:
            if node.node_type == node_type:
                return node.node_id

        raise ValueError(
            f"CFG içinde '{node_type}' türünde düğüm bulunamadı."
        )