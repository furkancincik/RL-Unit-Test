from __future__ import annotations

import ast
from pathlib import Path

import pytest

from analyzer.safe_custom_object import analyze_safe_custom_object_target
from analyzer.simple_instance_method import method_spec_for_target
from cfg.path_analyzer import ExecutionPath
from generator.path_input_generator import GeneratedTestInput
from generator.safe_method_setup_plan import SafeSetupPlanRejection
from services.real_rl_training_service import RealRLTrainingService


def _source(
    *,
    item_class: str = "Unit",
    owner_class: str = "Ledger",
) -> str:
    return f'''
class {item_class}:
    def __init__(self, code: str, amount: int, stock: int):
        self.code = code
        self.amount = amount
        self.stock = stock

    def adjust(self, delta: int):
        if self.stock + delta < 0:
            raise ValueError("invalid")
        self.stock += delta

class {owner_class}:
    def __init__(self):
        self.entries = {{}}

    def attach(self, unit, count: int):
        if count <= 0:
            raise ValueError("invalid")
        if unit.stock < count:
            return False
        if unit.code in self.entries:
            self.entries[unit.code]["count"] += count
        else:
            self.entries[unit.code] = {{"unit": unit, "count": count}}
        unit.adjust(-count)
        return True

    def total(self, threshold: int = 10):
        aggregate = 0
        for record in self.entries.values():
            unit = record["unit"]
            count = record["count"]
            aggregate += unit.amount * count
        if aggregate >= threshold:
            aggregate -= 1
        return aggregate

    def select(self):
        if not self.entries:
            return None
        chosen = None
        for record in self.entries.values():
            unit = record["unit"]
            if chosen is None:
                chosen = unit
            elif unit.amount > chosen.amount:
                chosen = unit
        return chosen.code

    def records(self):
        result = []
        for code, record in self.entries.items():
            result.append({{
                "code": code,
                "count": record["count"],
                "unit_amount": record["unit"].amount,
            }})
        return result
'''


def _context(target: str, *, source: str | None = None):
    from generator.safe_method_setup_plan import (
        analyze_safe_method_collection_context,
    )

    tree = ast.parse(source or _source())
    spec = method_spec_for_target(tree, target)
    assert spec is not None
    context, reason = analyze_safe_method_collection_context(
        tree,
        module_identity="fixtures.correlated_collection",
        target_name=target,
        method_spec=spec,
    )
    assert reason is None
    assert context is not None
    return context


def _path(*labels: tuple[str, str | None, str]) -> ExecutionPath:
    return ExecutionPath(
        node_ids=list(range(1, len(labels) + 1)),
        edge_labels=[edge for _, edge, _ in labels[:-1]],
        node_labels=[label for label, _, _ in labels],
        node_types=[kind for _, _, kind in labels],
        line_numbers=list(range(1, len(labels) + 1)),
    )


def test_scalar_fold_context_preserves_correlated_value_and_count_fields() -> None:
    context = _context("Ledger.total")

    assert context.pattern_kind == "SCALAR_FOLD"
    assert context.item_projection_paths == (
        ("unit", "amount"),
        ("count",),
    )


def test_selected_item_context_preserves_argmax_identity_projection() -> None:
    context = _context("Ledger.select")

    assert context.pattern_kind == "ARG_EXTREME"
    assert context.selection_operator == ">"
    assert context.selected_identity_projection == ("unit", "code")
    assert context.selected_value_projection == ("unit", "amount")


def test_record_projection_context_keeps_fields_on_one_logical_item() -> None:
    context = _context("Ledger.records")

    assert context.pattern_kind == "RECORD_PROJECTION"
    assert context.record_projection_paths == (
        ("code", ("$key",)),
        ("count", ("count",)),
        ("unit_amount", ("unit", "amount")),
    )


def test_same_item_multiple_constraints_materialize_one_atomic_witness() -> None:
    context = _context("Ledger.total")
    path = _path(
        ("START", None, "start"),
        ("__self_entries = {}", None, "Assign"),
        ("aggregate = 0", None, "Assign"),
        ("record in ()", "Iterate", "for"),
        ("unit = record['unit']", None, "Assign"),
        ("count = record['count']", None, "Assign"),
        ("aggregate += unit.amount * count", "Next", "AugAssign"),
        ("record in ()", "Complete", "for"),
        ("aggregate >= threshold", "True", "if"),
        ("aggregate -= 1", None, "AugAssign"),
        ("return aggregate", None, "return"),
        ("END", None, "end"),
    )
    rewritten = context.rewrite_path(path)
    names, _ = context.analysis_metadata(("threshold",), {"threshold": "int"})
    generated = GeneratedTestInput(
        keyword_arguments=tuple((name, 11 if name == "threshold" else 12) for name in names),
        expected_result=11,
    )

    bound = context.bind_generated_input(generated, path=rewritten)

    assert bound.setup_plan is not None
    assert len(bound.setup_plan.setup_calls) == 1
    assert len(bound.setup_plan.object_slots) == 2


def test_two_item_witness_is_bounded_and_deterministic() -> None:
    context = _context("Ledger.select")
    witness = context.correlated_witness_for(
        item_values=(0, 1),
        collection_nonempty=True,
    )
    repeated = context.correlated_witness_for(
        item_values=(0, 1),
        collection_nonempty=True,
    )

    assert witness == repeated
    assert len(witness.items) == 2
    assert witness.canonical_payload == repeated.canonical_payload


def test_conflicting_correlated_item_constraints_are_rejected() -> None:
    context = _context("Ledger.select")

    with pytest.raises(SafeSetupPlanRejection, match="correlated"):
        context.correlated_witness_for(
            item_values=(1, 0),
            collection_nonempty=False,
        )


def test_unsupported_collection_mutation_is_not_promoted_to_witness_support() -> None:
    source = _source().replace(
        "    def records(self):\n",
        "    def unsafe(self):\n"
        "        self.entries.clear()\n"
        "        return []\n\n"
        "    def records(self):\n",
    )
    tree = ast.parse(source)

    spec, reason = analyze_safe_custom_object_target(tree, "Ledger.unsafe")

    assert spec is None
    assert reason is None
    with pytest.raises(ValueError, match="mutation|method calls"):
        method_spec_for_target(tree, "Ledger.unsafe")


@pytest.mark.parametrize(
    ("item_class", "owner_class"),
    (("Parcel", "Warehouse"), ("Token", "Registry"), ("Entry", "Catalog")),
)
def test_correlated_context_is_identifier_independent(
    item_class: str,
    owner_class: str,
) -> None:
    context = _context(
        f"{owner_class}.total",
        source=_source(item_class=item_class, owner_class=owner_class),
    )

    assert context.pattern_kind == "SCALAR_FOLD"
    assert context.maximum_item_count == 2


@pytest.mark.parametrize(
    ("target_name", "minimum_scenarios", "minimum_branch_coverage"),
    (
        ("Ledger.total", 2, 75.0),
        ("Ledger.select", 4, 100.0),
        ("Ledger.records", 3, 100.0),
    ),
)
def test_correlated_witness_runs_through_concrete_coverage_and_rl(
    tmp_path: Path,
    target_name: str,
    minimum_scenarios: int,
    minimum_branch_coverage: float,
) -> None:
    source_file = tmp_path / "correlated_runtime_slice.py"
    source_file.write_text(_source(), encoding="utf-8")

    result = RealRLTrainingService().run(
        source_file=source_file,
        module_path="correlated_runtime_slice",
        function_name=target_name,
        output_directory=tmp_path / target_name.replace(".", "_"),
        import_root=tmp_path,
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        random_seed=42,
        run_greedy_baseline=True,
        timeout_seconds=30.0,
    )

    assert result.scenario_count >= minimum_scenarios
    assert result.diagnostic is not None
    assert result.diagnostic.funnel.concrete_validation_accepted_count >= 1
    assert result.function_coverage is not None
    assert result.function_coverage.line_coverage_percent == 100.0
    assert (
        result.function_coverage.branch_coverage_percent
        >= minimum_branch_coverage
    )
    assert result.q_table_state_count > 0
    assert result.minimization_result is not None
    assert result.minimization_result.coverage_preserved is True
    assert any(scenario.setup_plan is not None for scenario in result.scenarios)
