from __future__ import annotations

import json

import pytest

from models.project_analysis_result import (
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)


def test_qualified_target_selector_accepts_function_and_instance_method() -> None:
    function = QualifiedTargetSelector("package.module", "top_level")
    method = QualifiedTargetSelector("package.module", "Owner.execute")

    assert function.qualified_name == "top_level"
    assert method.qualified_name == "Owner.execute"


@pytest.mark.parametrize(
    "value",
    (
        "",
        ".method",
        "Owner.",
        "Owner..method",
        "Owner.method.extra",
        "Owner/method",
        "Owner\\method",
        "Owner.method()",
        "Owner[method]",
        "class",
        "Owner.class",
        " Owner.method",
        "Owner.method ",
        "Owner.\nmethod",
    ),
)
def test_qualified_target_selector_rejects_malformed_names(value: str) -> None:
    with pytest.raises(ValueError):
        QualifiedTargetSelector("package.module", value)


@pytest.mark.parametrize(
    "module_identity",
    ("", ".module", "package.", "package/module", " package.module"),
)
def test_selector_rejects_unsafe_module_identity(module_identity: str) -> None:
    with pytest.raises(ValueError):
        QualifiedTargetSelector(module_identity, "target")


def test_explicit_selection_deduplicates_in_first_seen_order() -> None:
    first = QualifiedTargetSelector("first.module", "Owner.run")
    second = QualifiedTargetSelector("second.module", "Owner.run")

    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (first, first, second),
    )

    assert selection.selectors == (first, second)
    assert selection.for_module("first.module") == ("Owner.run",)
    assert selection.for_module("second.module") == ("Owner.run",)


def test_default_selection_has_no_explicit_selectors() -> None:
    selection = TargetSelection()

    assert selection.mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    assert selection.selectors == ()


def test_selector_public_payload_contains_only_safe_identities() -> None:
    selection = TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        (QualifiedTargetSelector("package.module", "Owner.run"),),
    )

    serialized = json.dumps(selection.to_dict())

    assert "package.module" in serialized
    assert "Owner.run" in serialized
    for forbidden in (
        "source_file",
        "keyword_arguments",
        "constructor_arguments",
        "expected_result",
        "actual_result",
        "traceback",
        "environment",
    ):
        assert forbidden not in serialized
