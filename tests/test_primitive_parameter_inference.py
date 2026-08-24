from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer


def _analyze_source(tmp_path: Path, source: str, filename: str = "target.py"):
    source_file = tmp_path / filename
    source_file.write_text(source, encoding="utf-8")
    return PythonAnalyzer().analyze_file(source_file).functions


def test_infers_each_primitive_default_without_name_heuristics(tmp_path: Path) -> None:
    functions = _analyze_source(
        tmp_path,
        "def transform(alpha=7, beta=2.5, gamma='seed', delta=False, epsilon=None):\n"
        "    return alpha, beta, gamma, delta, epsilon\n",
    )

    assert functions[0].parameter_types == {
        "alpha": "int",
        "beta": "float",
        "gamma": "str",
        "delta": "bool",
    }


@pytest.mark.parametrize(
    ("expression", "expected_type"),
    [
        ("token == 'member'", "str"),
        ("token < 17", "int"),
        ("token + 3", "int"),
        ("token * 1.25", "int"),
        ("token is True", "bool"),
    ],
)
def test_infers_primitive_type_from_safe_ast_evidence(
    tmp_path: Path,
    expression: str,
    expected_type: str,
) -> None:
    functions = _analyze_source(
        tmp_path,
        f"def inspect(token):\n    return {expression}\n",
    )

    assert functions[0].parameter_types == {"token": expected_type}


def test_consistent_multiple_evidence_is_accepted(tmp_path: Path) -> None:
    functions = _analyze_source(
        tmp_path,
        "def measure(quantity):\n"
        "    if quantity <= 31:\n"
        "        return quantity + 6\n"
        "    return quantity * 2\n",
    )

    assert functions[0].parameter_types == {"quantity": "int"}


def test_exact_numeric_default_is_consistent_with_generic_numeric_usage(
    tmp_path: Path,
) -> None:
    functions = _analyze_source(
        tmp_path,
        "def scale(ratio=0.5):\n"
        "    if ratio > 1:\n"
        "        return ratio * 2\n"
        "    return ratio\n",
    )

    assert functions[0].parameter_types == {"ratio": "float"}


def test_augmented_arithmetic_infers_each_untyped_operand_as_numeric(
    tmp_path: Path,
) -> None:
    functions = _analyze_source(
        tmp_path,
        "def adjust(aggregate, decrement):\n"
        "    aggregate -= decrement\n"
        "    return aggregate\n",
    )

    assert functions[0].parameter_types == {
        "aggregate": "int",
        "decrement": "int",
    }


def test_conflicting_evidence_keeps_instance_method_controlled_unsupported(
    tmp_path: Path,
) -> None:
    functions = _analyze_source(
        tmp_path,
        "class Container:\n"
        "    def resolve(self, entry):\n"
        "        if entry == 'text':\n"
        "            return 1\n"
        "        if entry < 4:\n"
        "            return 2\n"
        "        return 3\n",
    )
    method = functions[0]

    assert method.is_supported is False
    assert "conflicting primitive evidence" in (method.unsupported_reason or "").lower()


def test_none_default_alone_keeps_instance_method_unsupported(tmp_path: Path) -> None:
    functions = _analyze_source(
        tmp_path,
        "class Holder:\n"
        "    def choose(self, candidate=None):\n"
        "        return candidate\n",
    )

    assert functions[0].is_supported is False
    assert "not a supported primitive" in (functions[0].unsupported_reason or "").lower()


@pytest.mark.parametrize(
    "body",
    [
        "return payload.attribute",
        "return payload()",
        "return convert(payload)",
        "return convert(payload + 1)",
    ],
)
def test_custom_attribute_or_call_usage_remains_unsupported(
    tmp_path: Path,
    body: str,
) -> None:
    functions = _analyze_source(
        tmp_path,
        "class Reader:\n"
        "    def read(self, payload):\n"
        f"        {body}\n",
    )

    assert functions[0].is_supported is False
    assert "not a supported primitive" in (functions[0].unsupported_reason or "").lower()


def test_explicit_annotation_has_priority_over_usage_evidence(tmp_path: Path) -> None:
    functions = _analyze_source(
        tmp_path,
        "def preserve(subject: str):\n"
        "    if subject < 9:\n"
        "        return 'low'\n"
        "    return subject\n",
    )

    assert functions[0].parameter_types == {"subject": "str"}
    assert functions[0].is_supported is True


def test_top_level_and_instance_method_receive_inferred_types(tmp_path: Path) -> None:
    functions = _analyze_source(
        tmp_path,
        "def authorize(identity):\n"
        "    return identity == 'operator'\n\n"
        "class Ledger:\n"
        "    def __init__(self, reserve=12):\n"
        "        self.reserve = reserve\n\n"
        "    def debit(self, change):\n"
        "        if change <= 3:\n"
        "            self.reserve -= change\n"
        "        return self.reserve\n",
    )
    top_level = next(item for item in functions if item.qualified_name == "authorize")
    method = next(item for item in functions if item.qualified_name == "Ledger.debit")

    assert top_level.parameter_types == {"identity": "str"}
    assert method.is_supported is True
    assert method.parameter_types == {"change": "int"}
    assert method.constructor_parameter_types == {"reserve": "int"}


@pytest.mark.parametrize(
    ("filename", "class_name", "method_name", "parameter_name", "literal"),
    [
        ("first_shape.py", "Vault", "lower", "units", 2),
        ("renamed_module.py", "Register", "reduce", "portion", 19),
        ("third_source.py", "Reservoir", "consume", "volume", 43),
    ],
)
def test_equivalent_shapes_are_name_file_and_literal_independent(
    tmp_path: Path,
    filename: str,
    class_name: str,
    method_name: str,
    parameter_name: str,
    literal: int,
) -> None:
    functions = _analyze_source(
        tmp_path,
        f"class {class_name}:\n"
        "    def __init__(self, initial=5):\n"
        "        self.current = initial\n\n"
        f"    def {method_name}(self, {parameter_name}):\n"
        f"        if {parameter_name} <= {literal}:\n"
        f"            self.current -= {parameter_name}\n"
        "        return self.current\n",
        filename,
    )
    method = next(item for item in functions if item.name == method_name)

    assert method.is_supported is True
    assert method.parameter_types == {parameter_name: "int"}
    assert method.constructor_parameter_types == {"initial": "int"}


@pytest.mark.parametrize(
    ("class_name", "method_name", "parameter_name"),
    [
        ("Archive", "inspect", "record"),
        ("Catalog", "review", "document"),
        ("Repository", "check", "object_value"),
    ],
)
def test_equivalent_unsupported_shapes_share_the_controlled_category(
    tmp_path: Path,
    class_name: str,
    method_name: str,
    parameter_name: str,
) -> None:
    functions = _analyze_source(
        tmp_path,
        f"class {class_name}:\n"
        f"    def {method_name}(self, {parameter_name}):\n"
        f"        return {parameter_name}.dynamic\n",
        f"{class_name.lower()}_source.py",
    )

    assert functions[0].is_supported is False
    assert "not a supported primitive" in (functions[0].unsupported_reason or "").lower()
