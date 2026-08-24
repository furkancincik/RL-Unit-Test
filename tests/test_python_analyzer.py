from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder


def test_analyze_file_returns_correct_counts() -> None:
    analyzer = PythonAnalyzer()

    sample_file = Path("datasets/sample_code.py")
    result = analyzer.analyze_file(sample_file)

    assert result.file_name == "sample_code.py"
    assert result.function_count == 1
    assert result.class_count == 0
    assert result.if_count == 2
    assert result.for_count == 0
    assert result.while_count == 0
    assert result.try_count == 0
    assert result.return_count == 3

    assert len(result.functions) == 1

    function = result.functions[0]

    assert function.name == "calculate_score"
    assert function.parameters == ["score"]
    assert function.line_number == 1
    assert function.is_async is False
    assert function.return_count == 3
    assert function.branch_count == 2
    assert function.cyclomatic_complexity == 3
    assert function.risk_level == "Low"


def test_function_discovery_preserves_source_order_and_scope_metadata(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "targets.py"
    source_file.write_text(
        """
def first(value: int) -> int:
    return value

async def asynchronous() -> None:
    return None

def outer() -> int:
    def nested() -> int:
        return 1
    return nested()

class Handler:
    def method(self) -> int:
        return 1

def last(*, enabled: bool) -> bool:
    return enabled
""",
        encoding="utf-8",
    )

    functions = PythonAnalyzer().analyze_file(source_file).functions

    assert [function.qualified_name for function in functions] == [
        "first",
        "asynchronous",
        "outer",
        "outer.nested",
        "Handler.method",
        "last",
    ]
    assert functions[0].return_annotation == "int"
    assert functions[0].is_supported is True
    assert functions[1].is_async is True
    assert functions[1].is_supported is False
    assert functions[3].is_nested is True
    assert functions[3].is_supported is False
    assert functions[4].is_method is True
    assert functions[4].is_supported is True
    assert functions[5].parameters == ["enabled"]
    assert functions[5].is_supported is True


def test_discovers_safe_instance_method_with_separate_constructor_metadata(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "counter.py"
    source_file.write_text(
        "class Counter:\n"
        "    def __init__(self, value: int = 0):\n"
        "        self.value = value\n\n"
        "    def classify(self, delta: int) -> str:\n"
        "        if self.value + delta >= 10:\n"
        "            return 'high'\n"
        "        return 'low'\n",
        encoding="utf-8",
    )

    functions = PythonAnalyzer().analyze_file(source_file).functions
    target = next(item for item in functions if item.name == "classify")

    assert target.qualified_name == "Counter.classify"
    assert target.is_method is True
    assert target.is_supported is True
    assert target.parameters == ["delta"]
    assert "self" not in target.parameters
    assert target.class_name == "Counter"
    assert target.constructor_parameters == ["value"]
    assert target.constructor_parameter_types == {"value": "int"}


@pytest.mark.parametrize(
    ("source", "reason_fragment"),
    [
        (
            "class Child(Base):\n"
            "    def run(self) -> int:\n"
            "        return 1\n",
            "inheritance",
        ),
        (
            "class Target:\n"
            "    @property\n"
            "    def value(self) -> int:\n"
            "        return 1\n",
            "decorated",
        ),
        (
            "class Target:\n"
            "    @classmethod\n"
            "    def run(cls) -> int:\n"
            "        return 1\n",
            "decorated",
        ),
        (
            "class Target:\n"
            "    @staticmethod\n"
            "    def run() -> int:\n"
            "        return 1\n",
            "decorated",
        ),
        (
            "class Target(metaclass=type):\n"
            "    def run(self) -> int:\n"
            "        return 1\n",
            "metaclass",
        ),
        (
            "if True:\n"
            "    class Target:\n"
            "        def run(self) -> int:\n"
            "            return 1\n",
            "top-level",
        ),
        (
            "class Target:\n"
            "    def values(self):\n"
            "        yield 1\n",
            "generator",
        ),
        (
            "class Target:\n"
            "    def __init__(self, dependency: object):\n"
            "        self.dependency = dependency\n\n"
            "    def run(self) -> int:\n"
            "        return 1\n",
            "constructor",
        ),
    ],
)
def test_unsupported_instance_method_shapes_remain_explicit(
    tmp_path: Path,
    source: str,
    reason_fragment: str,
) -> None:
    source_file = tmp_path / "unsupported_method.py"
    source_file.write_text(source, encoding="utf-8")

    methods = [
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.is_method and item.name != "__init__"
    ]

    assert len(methods) == 1
    assert methods[0].is_supported is False
    assert reason_fragment in (methods[0].unsupported_reason or "").lower()


def test_parameterless_class_supports_method_without_self_state(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "plain_method.py"
    source_file.write_text(
        "class Formatter:\n"
        "    def label(self, enabled: bool) -> str:\n"
        "        if enabled:\n"
        "            return 'yes'\n"
        "        return 'no'\n",
        encoding="utf-8",
    )

    target = PythonAnalyzer().analyze_file(source_file).functions[0]

    assert target.is_supported is True
    assert target.parameters == ["enabled"]
    assert target.constructor_parameters == []


def test_unsupported_instance_method_metadata_excludes_implicit_self(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "unsupported_receiver.py"
    source_file.write_text(
        "class Target:\n"
        "    def __init__(self, dependency: object):\n"
        "        self.dependency = dependency\n\n"
        "    def run(self, value: int) -> int:\n"
        "        return value\n",
        encoding="utf-8",
    )

    functions = PythonAnalyzer().analyze_file(source_file).functions
    constructor = next(item for item in functions if item.name == "__init__")
    method = next(item for item in functions if item.name == "run")

    assert constructor.is_supported is False
    assert constructor.parameters == ["dependency"]
    assert constructor.unsupported_reason == (
        "Constructor methods are not standalone analysis targets."
    )
    assert method.is_supported is False
    assert method.parameters == ["value"]
    assert "constructor" in (method.unsupported_reason or "").lower()


def test_self_named_top_level_parameter_and_staticmethod_parameters_are_preserved(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "receiver_boundaries.py"
    source_file.write_text(
        "def top_level(self: int) -> int:\n"
        "    return self\n\n"
        "class Target:\n"
        "    @staticmethod\n"
        "    def run(self: int, value: int) -> int:\n"
        "        return self + value\n",
        encoding="utf-8",
    )

    functions = PythonAnalyzer().analyze_file(source_file).functions
    top_level = next(item for item in functions if item.name == "top_level")
    static_method = next(item for item in functions if item.name == "run")

    assert top_level.parameters == ["self"]
    assert top_level.is_method is False
    assert static_method.parameters == ["self", "value"]
    assert static_method.is_supported is False
    assert static_method.unsupported_reason == (
        "Decorated instance methods are unsupported."
    )


def test_cfg_normalizes_self_state_comparison_and_mutation(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "stateful_method.py"
    source_file.write_text(
        "class Accumulator:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "    def adjust(self, delta: int) -> str:\n"
        "        self.value += delta\n"
        "        if self.value > delta:\n"
        "            return 'above'\n"
        "        return 'other'\n",
        encoding="utf-8",
    )

    graph = next(
        item
        for item in ControlFlowGraphBuilder().build_from_file(source_file)
        if item.function_name == "Accumulator.adjust"
    )
    labels = [node.label for node in graph.nodes]

    assert "__self_value += delta" in labels
    assert "__self_value > delta" in labels
    assert all("self." not in label for label in labels)


@pytest.mark.parametrize(
    ("initializer", "method_body"),
    [
        ("{}", "        return not self.records\n"),
        ("[]", "        return bool(self.entries)\n"),
        (
            "{}",
            "        if key in self.records:\n"
            "            return 'present'\n"
            "        return 'missing'\n",
        ),
        (
            "[]",
            "        if key not in self.entries:\n"
            "            return 'missing'\n"
            "        return 'present'\n",
        ),
        (
            "[]",
            "        count = 0\n"
            "        for entry in self.entries:\n"
            "            count += 1\n"
            "        return count\n",
        ),
        (
            "{}",
            "        count = 0\n"
            "        for key, value in self.records.items():\n"
            "            count += 1\n"
            "        for value in self.records.values():\n"
            "            count += 1\n"
            "        return count\n",
        ),
    ],
)
def test_safe_empty_collection_state_is_analyzable(
    tmp_path: Path,
    initializer: str,
    method_body: str,
) -> None:
    attribute = "records" if initializer == "{}" else "entries"
    source_file = tmp_path / "empty_state.py"
    source_file.write_text(
        "class Store:\n"
        "    def __init__(self):\n"
        f"        self.{attribute} = {initializer}\n\n"
        "    def inspect(self, key: str) -> object:\n"
        f"{method_body}",
        encoding="utf-8",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == "inspect"
    )

    assert target.is_supported is True
    assert target.unsupported_reason is None


@pytest.mark.parametrize(
    ("constructor_statement", "method_body", "reason_fragment"),
    [
        ("self.data = {'key': 1}", "return len(self.data)", "empty"),
        ("self.data = [1]", "return len(self.data)", "empty"),
        ("self.data = [[]]", "return len(self.data)", "empty"),
        (
            "self.data = {object(): 1}",
            "return len(self.data)",
            "empty",
        ),
        ("self.data = []", "self.data.append(value)\n        return 0", "mutation"),
        ("self.data = {}", "self.data[value] = 1\n        return 0", "mutation"),
        (
            "self.data = []",
            "alias = self.data\n        alias.append(value)\n        return 0",
            "alias",
        ),
    ],
)
def test_unsafe_collection_state_remains_controlled_unsupported(
    tmp_path: Path,
    constructor_statement: str,
    method_body: str,
    reason_fragment: str,
) -> None:
    source_file = tmp_path / "unsupported_collection.py"
    source_file.write_text(
        "class Container:\n"
        "    def __init__(self):\n"
        f"        {constructor_statement}\n\n"
        "    def inspect(self, value: int) -> int:\n"
        f"        {method_body}\n",
        encoding="utf-8",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == "inspect"
    )

    assert target.is_supported is False
    assert reason_fragment in (target.unsupported_reason or "").lower()


@pytest.mark.parametrize(
    ("module_name", "class_name", "method_name", "attribute_name", "parameter_name"),
    [
        ("first_shape", "Archive", "locate", "index", "needle"),
        ("second_shape", "Registry", "lookup", "mapping", "token"),
        ("third_shape", "Ledger", "contains", "entries", "label"),
    ],
)
def test_empty_collection_support_is_independent_of_source_identifiers(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    method_name: str,
    attribute_name: str,
    parameter_name: str,
) -> None:
    source_file = tmp_path / f"{module_name}.py"
    source_file.write_text(
        f"class {class_name}:\n"
        "    def __init__(self):\n"
        f"        self.{attribute_name} = {{}}\n\n"
        f"    def {method_name}(self, {parameter_name}: str) -> bool:\n"
        f"        return {parameter_name} in self.{attribute_name}\n",
        encoding="utf-8",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == method_name
    )

    assert target.qualified_name == f"{class_name}.{method_name}"
    assert target.is_supported is True
    assert target.parameters == [parameter_name]


def test_analyze_file_raises_error_when_file_does_not_exist() -> None:
    analyzer = PythonAnalyzer()

    with pytest.raises(FileNotFoundError):
        analyzer.analyze_file("datasets/olmayan_dosya.py")


def test_analyze_file_rejects_non_python_files(
    tmp_path: Path,
) -> None:
    analyzer = PythonAnalyzer()

    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "Ã¶rnek iÃ§erik",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        analyzer.analyze_file(text_file)


def test_analyze_file_raises_error_for_invalid_python_syntax(
    tmp_path: Path,
) -> None:
    analyzer = PythonAnalyzer()

    invalid_file = tmp_path / "invalid_code.py"
    invalid_file.write_text(
        "def broken_function(\n",
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError):
        analyzer.analyze_file(invalid_file)


@pytest.mark.parametrize(
    ("complexity", "expected_risk"),
    [
        (1, "Low"),
        (5, "Low"),
        (6, "Medium"),
        (10, "Medium"),
        (11, "High"),
        (20, "High"),
    ],
)
def test_determine_risk_level(
    complexity: int,
    expected_risk: str,
) -> None:
    result = PythonAnalyzer._determine_risk_level(complexity)

    assert result == expected_risk

