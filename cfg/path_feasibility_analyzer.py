from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from cfg.path_analyzer import ExecutionPath


ConstraintAtom: TypeAlias = int | float | str | bool
ConstraintValue: TypeAlias = ConstraintAtom | tuple[ConstraintAtom, ...]


class FeasibilityStatus(str, Enum):
    """
    Bir execution path'in feasibility durumunu temsil eder.
    """

    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PathConstraint:
    """
    Tek bir path koşulunu temsil eder.

    Desteklenen örnekler:
        score >= 50
        attendance < 40
        customer_type == "VIP"
        coupon not in ("NONE", "")
        items truthy / falsy
    """

    variable_name: str
    operator: str
    value: ConstraintValue

    SUPPORTED_OPERATORS = {
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        "in",
        "not in",
        "truthy",
        "falsy",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.variable_name, str):
            raise TypeError(
                "variable_name string olmalıdır."
            )

        if not self.variable_name.strip():
            raise ValueError(
                "variable_name boş olamaz."
            )

        if not isinstance(self.operator, str):
            raise TypeError(
                "operator string olmalıdır."
            )

        if self.operator not in self.SUPPORTED_OPERATORS:
            raise ValueError(
                "Desteklenmeyen constraint operatorü: "
                f"{self.operator}"
            )

        self._validate_value()

    def _validate_value(self) -> None:
        if self.operator in {
            "<",
            "<=",
            ">",
            ">=",
        }:
            if (
                isinstance(self.value, bool)
                or not isinstance(
                    self.value,
                    (int, float),
                )
            ):
                raise TypeError(
                    "sayısal karşılaştırma constraint value "
                    "değeri sayısal olmalıdır."
                )

            if not math.isfinite(
                float(self.value)
            ):
                raise ValueError(
                    "constraint value sonlu olmalıdır."
                )

            return

        if self.operator in {
            "in",
            "not in",
        }:
            if (
                not isinstance(self.value, tuple)
                or not self.value
            ):
                raise TypeError(
                    "membership constraint value "
                    "boş olmayan tuple olmalıdır."
                )

            if any(
                not self._is_supported_atom(item)
                for item in self.value
            ):
                raise TypeError(
                    "membership constraint yalnızca "
                    "basit literal değerler içermelidir."
                )

            return

        if self.operator in {
            "truthy",
            "falsy",
        }:
            if self.value is not True:
                raise ValueError(
                    "truthy/falsy constraint value True olmalıdır."
                )

            return

        if not self._is_supported_atom(
            self.value
        ):
            raise TypeError(
                "constraint value desteklenen bir "
                "literal olmalıdır."
            )

        if (
            isinstance(self.value, float)
            and not math.isfinite(
                self.value
            )
        ):
            raise ValueError(
                "constraint value sonlu olmalıdır."
            )

    @staticmethod
    def _is_supported_atom(
        value: object,
    ) -> bool:
        return isinstance(
            value,
            (int, float, str, bool),
        )


@dataclass(frozen=True, slots=True)
class PathConstraintExtractionResult:
    """
    Bir ExecutionPath üzerinden çıkarılan constraint'leri ve
    mevcut analyzer'ın yorumlayamadığı koşulları temsil eder.
    """

    constraints: tuple[PathConstraint, ...]
    unsupported_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PathFeasibilityResult:
    """
    Bir execution path'in feasibility analiz sonucudur.

    UNKNOWN durumu, path üzerinde henüz desteklenmeyen bir koşul
    bulunduğunda ve path'in infeasible olduğu kesin olarak
    kanıtlanamadığında kullanılır.
    """

    status: FeasibilityStatus
    constraints: tuple[PathConstraint, ...]
    conflicts: tuple[str, ...]
    unsupported_conditions: tuple[str, ...] = ()

    @property
    def is_feasible(self) -> bool:
        return (
            self.status
            == FeasibilityStatus.FEASIBLE
        )

    @property
    def is_infeasible(self) -> bool:
        return (
            self.status
            == FeasibilityStatus.INFEASIBLE
        )

    @property
    def is_unknown(self) -> bool:
        return (
            self.status
            == FeasibilityStatus.UNKNOWN
        )


@dataclass(slots=True)
class _VariableDomain:
    """
    Tek bir değişken için path boyunca oluşan constraint
    durumunu tutar.

    Sayısal sınırlar ile categorical/string eşitlikleri aynı
    değişken domain'i içerisinde güvenli biçimde saklanır.
    """

    lower_bound: float | None = None
    lower_inclusive: bool = True

    upper_bound: float | None = None
    upper_inclusive: bool = True

    exact_value: ConstraintAtom | None = None
    has_exact_value: bool = False
    conflicting_exact_values: bool = False

    excluded_values: set[ConstraintAtom] = field(
        default_factory=set
    )

    allowed_values: set[ConstraintAtom] | None = None

    requires_truthy: bool = False
    requires_falsy: bool = False


class PathFeasibilityAnalyzer:
    """
    Execution path üzerindeki desteklenen Python koşullarını analiz eder.

    Desteklenen başlıca yapılar:
    - Numeric karşılaştırmalar
    - String/bool equality ve inequality
    - Basit truthiness / ``not variable``
    - ``and`` ifadesinin True kolu
    - ``or`` ifadesinin False kolu
    - ``in`` / ``not in`` membership kontrolleri

    Güvenlik ilkesi:
    Karmaşık bir koşul kesin biçimde yorumlanamıyorsa path yanlışlıkla
    INFEASIBLE olarak işaretlenmez. Kesin bir çelişki de yoksa UNKNOWN
    sonucu döndürülür.
    """

    _AST_OPERATOR_TO_TEXT = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.In: "in",
        ast.NotIn: "not in",
    }

    _NEGATED_OPERATOR = {
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "==": "!=",
        "!=": "==",
        "in": "not in",
        "not in": "in",
    }

    _REVERSED_OPERATOR = {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }

    def analyze_constraints(
        self,
        constraints: tuple[PathConstraint, ...],
    ) -> PathFeasibilityResult:
        """
        Doğrudan verilen constraint koleksiyonunu analiz eder.
        """
        self._validate_constraints(
            constraints
        )

        conflicts = self._find_conflicts(
            constraints
        )

        if conflicts:
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.INFEASIBLE
                ),
                constraints=constraints,
                conflicts=conflicts,
            )

        return PathFeasibilityResult(
            status=FeasibilityStatus.FEASIBLE,
            constraints=constraints,
            conflicts=(),
        )

    def extract_constraints(
        self,
        path: ExecutionPath,
    ) -> PathConstraintExtractionResult:
        """
        ExecutionPath içerisindeki if/while koşullarından constraint üretir.

        Desteklenmeyen veya kesin yorumlanamayan koşullar
        ``unsupported_conditions`` içerisinde korunur.
        """
        self._validate_path(
            path
        )

        if not path.has_node_metadata:
            return PathConstraintExtractionResult(
                constraints=(),
                unsupported_conditions=(
                    "ExecutionPath node metadata içermiyor.",
                ),
            )

        constraints: list[PathConstraint] = []
        unsupported_conditions: list[str] = []

        for step in path.condition_steps:
            edge_label = (
                step.outgoing_edge_label
            )

            if edge_label not in {
                "True",
                "False",
            }:
                unsupported_conditions.append(
                    self._format_unsupported_condition(
                        condition=step.node_label,
                        reason=(
                            "True/False edge bilgisi bulunamadı"
                        ),
                    )
                )
                continue

            (
                extracted,
                unsupported,
            ) = self._parse_condition_constraints(
                condition=step.node_label,
                condition_is_true=(
                    edge_label == "True"
                ),
            )

            constraints.extend(
                extracted
            )

            unsupported_conditions.extend(
                self._format_unsupported_condition(
                    condition=condition,
                    reason=reason,
                )
                for condition, reason
                in unsupported
            )

        return PathConstraintExtractionResult(
            constraints=tuple(
                constraints
            ),
            unsupported_conditions=tuple(
                unsupported_conditions
            ),
        )

    def analyze_path(
        self,
        path: ExecutionPath,
    ) -> PathFeasibilityResult:
        """
        ExecutionPath üzerindeki koşulları otomatik çıkarır ve path'in
        FEASIBLE, INFEASIBLE veya UNKNOWN olduğunu belirler.

        Desteklenmeyen koşullar bulunsa bile desteklenen constraint'ler
        içinde kesin bir çelişki varsa INFEASIBLE sonucu korunur.
        """
        extraction = self.extract_constraints(
            path
        )

        constraint_result = (
            self.analyze_constraints(
                extraction.constraints
            )
        )

        if constraint_result.is_infeasible:
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.INFEASIBLE
                ),
                constraints=(
                    extraction.constraints
                ),
                conflicts=(
                    constraint_result.conflicts
                ),
                unsupported_conditions=(
                    extraction.unsupported_conditions
                ),
            )

        if extraction.unsupported_conditions:
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.UNKNOWN
                ),
                constraints=(
                    extraction.constraints
                ),
                conflicts=(),
                unsupported_conditions=(
                    extraction.unsupported_conditions
                ),
            )

        return PathFeasibilityResult(
            status=(
                FeasibilityStatus.FEASIBLE
            ),
            constraints=(
                extraction.constraints
            ),
            conflicts=(),
            unsupported_conditions=(),
        )

    def analyze_paths(
        self,
        paths: tuple[ExecutionPath, ...],
    ) -> tuple[PathFeasibilityResult, ...]:
        """
        Birden fazla ExecutionPath nesnesini giriş sırasını koruyarak
        analiz eder.
        """
        self._validate_paths(
            paths
        )

        return tuple(
            self.analyze_path(
                path
            )
            for path in paths
        )

    def _find_conflicts(
        self,
        constraints: tuple[
            PathConstraint,
            ...
        ],
    ) -> tuple[str, ...]:
        domains: dict[
            str,
            _VariableDomain,
        ] = {}

        conflicts: list[str] = []

        for constraint in constraints:
            domain = domains.setdefault(
                constraint.variable_name,
                _VariableDomain(),
            )

            self._apply_constraint(
                domain=domain,
                constraint=constraint,
            )

            conflict = self._detect_conflict(
                variable_name=(
                    constraint.variable_name
                ),
                domain=domain,
            )

            if (
                conflict is not None
                and conflict not in conflicts
            ):
                conflicts.append(
                    conflict
                )

        return tuple(
            conflicts
        )

    def _apply_constraint(
        self,
        *,
        domain: _VariableDomain,
        constraint: PathConstraint,
    ) -> None:
        operator = constraint.operator
        value = constraint.value

        if operator == "truthy":
            domain.requires_truthy = True
            return

        if operator == "falsy":
            domain.requires_falsy = True
            return

        if operator == "in":
            assert isinstance(value, tuple)

            new_values = set(value)

            if domain.allowed_values is None:
                domain.allowed_values = new_values
            else:
                domain.allowed_values.intersection_update(
                    new_values
                )

            return

        if operator == "not in":
            assert isinstance(value, tuple)

            domain.excluded_values.update(
                value
            )
            return

        if operator == "==":
            assert not isinstance(value, tuple)

            if (
                domain.has_exact_value
                and domain.exact_value != value
            ):
                domain.conflicting_exact_values = True
            else:
                domain.exact_value = value
                domain.has_exact_value = True

            return

        if operator == "!=":
            assert not isinstance(value, tuple)

            domain.excluded_values.add(
                value
            )
            return

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            return

        numeric_value = float(
            value
        )

        match operator:
            case ">":
                self._update_lower_bound(
                    domain=domain,
                    value=numeric_value,
                    inclusive=False,
                )

            case ">=":
                self._update_lower_bound(
                    domain=domain,
                    value=numeric_value,
                    inclusive=True,
                )

            case "<":
                self._update_upper_bound(
                    domain=domain,
                    value=numeric_value,
                    inclusive=False,
                )

            case "<=":
                self._update_upper_bound(
                    domain=domain,
                    value=numeric_value,
                    inclusive=True,
                )

    @staticmethod
    def _update_lower_bound(
        *,
        domain: _VariableDomain,
        value: float,
        inclusive: bool,
    ) -> None:
        if domain.lower_bound is None:
            domain.lower_bound = value
            domain.lower_inclusive = inclusive
            return

        if value > domain.lower_bound:
            domain.lower_bound = value
            domain.lower_inclusive = inclusive
            return

        if value == domain.lower_bound:
            domain.lower_inclusive = (
                domain.lower_inclusive
                and inclusive
            )

    @staticmethod
    def _update_upper_bound(
        *,
        domain: _VariableDomain,
        value: float,
        inclusive: bool,
    ) -> None:
        if domain.upper_bound is None:
            domain.upper_bound = value
            domain.upper_inclusive = inclusive
            return

        if value < domain.upper_bound:
            domain.upper_bound = value
            domain.upper_inclusive = inclusive
            return

        if value == domain.upper_bound:
            domain.upper_inclusive = (
                domain.upper_inclusive
                and inclusive
            )

    @staticmethod
    def _detect_conflict(
        *,
        variable_name: str,
        domain: _VariableDomain,
    ) -> str | None:
        if domain.conflicting_exact_values:
            return (
                f"{variable_name}: birden fazla farklı "
                "exact değer aynı anda gerekli."
            )

        if (
            domain.requires_truthy
            and domain.requires_falsy
        ):
            return (
                f"{variable_name}: aynı anda truthy ve "
                "falsy olması gerekiyor."
            )

        if (
            domain.allowed_values is not None
            and not (
                domain.allowed_values
                - domain.excluded_values
            )
        ):
            return (
                f"{variable_name}: membership constraint'leri "
                "geçerli değer bırakmıyor."
            )

        if domain.has_exact_value:
            exact = domain.exact_value

            if exact in domain.excluded_values:
                return (
                    f"{variable_name}: "
                    f"{exact!r} hem gerekli hem yasak."
                )

            if (
                domain.allowed_values is not None
                and exact not in domain.allowed_values
            ):
                return (
                    f"{variable_name}: "
                    f"{exact!r} izin verilen değerler "
                    "kümesiyle çelişiyor."
                )

            if domain.requires_truthy and not bool(exact):
                return (
                    f"{variable_name}: "
                    f"{exact!r} truthy olma şartıyla çelişiyor."
                )

            if domain.requires_falsy and bool(exact):
                return (
                    f"{variable_name}: "
                    f"{exact!r} falsy olma şartıyla çelişiyor."
                )

            if (
                isinstance(exact, (int, float))
                and not isinstance(exact, bool)
            ):
                numeric_exact = float(
                    exact
                )

                if (
                    domain.lower_bound
                    is not None
                ):
                    if (
                        numeric_exact
                        < domain.lower_bound
                    ):
                        return (
                            f"{variable_name}: "
                            f"{numeric_exact} alt sınır "
                            f"{domain.lower_bound} ile çelişiyor."
                        )

                    if (
                        numeric_exact
                        == domain.lower_bound
                        and not domain.lower_inclusive
                    ):
                        return (
                            f"{variable_name}: "
                            f"{numeric_exact} strict alt "
                            "sınırla çelişiyor."
                        )

                if (
                    domain.upper_bound
                    is not None
                ):
                    if (
                        numeric_exact
                        > domain.upper_bound
                    ):
                        return (
                            f"{variable_name}: "
                            f"{numeric_exact} üst sınır "
                            f"{domain.upper_bound} ile çelişiyor."
                        )

                    if (
                        numeric_exact
                        == domain.upper_bound
                        and not domain.upper_inclusive
                    ):
                        return (
                            f"{variable_name}: "
                            f"{numeric_exact} strict üst "
                            "sınırla çelişiyor."
                        )

        if (
            domain.lower_bound is not None
            and domain.upper_bound is not None
        ):
            if (
                domain.lower_bound
                > domain.upper_bound
            ):
                return (
                    f"{variable_name}: "
                    "alt sınır üst sınırdan büyük."
                )

            if (
                domain.lower_bound
                == domain.upper_bound
                and (
                    not domain.lower_inclusive
                    or not domain.upper_inclusive
                )
            ):
                return (
                    f"{variable_name}: "
                    "eşit sınırlar strict olduğu için "
                    "geçerli değer kalmıyor."
                )

            if (
                domain.lower_bound
                == domain.upper_bound
                and domain.lower_inclusive
                and domain.upper_inclusive
                and domain.lower_bound
                in domain.excluded_values
            ):
                return (
                    f"{variable_name}: tek geçerli değer "
                    f"{domain.lower_bound} aynı zamanda yasak."
                )

        return None

    def _parse_condition_constraints(
        self,
        *,
        condition: str,
        condition_is_true: bool,
    ) -> tuple[
        tuple[PathConstraint, ...],
        tuple[tuple[str, str], ...],
    ]:
        try:
            expression = ast.parse(
                condition,
                mode="eval",
            ).body
        except SyntaxError:
            return (
                (),
                (
                    (
                        condition,
                        "Python ifadesi parse edilemedi",
                    ),
                ),
            )

        return self._extract_from_expression(
            expression=expression,
            condition_is_true=condition_is_true,
            original_condition=condition,
        )

    def _extract_from_expression(
        self,
        *,
        expression: ast.expr,
        condition_is_true: bool,
        original_condition: str,
    ) -> tuple[
        tuple[PathConstraint, ...],
        tuple[tuple[str, str], ...],
    ]:
        if isinstance(
            expression,
            ast.BoolOp,
        ):
            return self._extract_from_boolean_operation(
                expression=expression,
                condition_is_true=condition_is_true,
                original_condition=original_condition,
            )

        if isinstance(
            expression,
            ast.UnaryOp,
        ) and isinstance(
            expression.op,
            ast.Not,
        ):
            return self._extract_from_expression(
                expression=expression.operand,
                condition_is_true=(
                    not condition_is_true
                ),
                original_condition=original_condition,
            )

        if isinstance(
            expression,
            ast.Name,
        ):
            return (
                (
                    PathConstraint(
                        variable_name=expression.id,
                        operator=(
                            "truthy"
                            if condition_is_true
                            else "falsy"
                        ),
                        value=True,
                    ),
                ),
                (),
            )

        if isinstance(
            expression,
            ast.Compare,
        ):
            constraint = (
                self._parse_compare_expression(
                    expression=expression,
                    condition_is_true=condition_is_true,
                )
            )

            if constraint is not None:
                return (
                    (
                        constraint,
                    ),
                    (),
                )

        return (
            (),
            (
                (
                    original_condition,
                    "desteklenen constraint yapısına "
                    "dönüştürülemedi",
                ),
            ),
        )

    def _extract_from_boolean_operation(
        self,
        *,
        expression: ast.BoolOp,
        condition_is_true: bool,
        original_condition: str,
    ) -> tuple[
        tuple[PathConstraint, ...],
        tuple[tuple[str, str], ...],
    ]:
        safe_to_expand = (
            isinstance(
                expression.op,
                ast.And,
            )
            and condition_is_true
        ) or (
            isinstance(
                expression.op,
                ast.Or,
            )
            and not condition_is_true
        )

        if not safe_to_expand:
            reason = (
                "AND False kolu alternatif olasılıklar içeriyor"
                if isinstance(
                    expression.op,
                    ast.And,
                )
                else (
                    "OR True kolu alternatif olasılıklar içeriyor"
                )
            )

            return (
                (),
                (
                    (
                        original_condition,
                        reason,
                    ),
                ),
            )

        constraints: list[PathConstraint] = []
        unsupported: list[tuple[str, str]] = []

        for value in expression.values:
            (
                nested_constraints,
                nested_unsupported,
            ) = self._extract_from_expression(
                expression=value,
                condition_is_true=condition_is_true,
                original_condition=ast.unparse(
                    value
                ),
            )

            constraints.extend(
                nested_constraints
            )
            unsupported.extend(
                nested_unsupported
            )

        return (
            tuple(
                constraints
            ),
            tuple(
                unsupported
            ),
        )

    def _parse_compare_expression(
        self,
        *,
        expression: ast.Compare,
        condition_is_true: bool,
    ) -> PathConstraint | None:
        if (
            len(expression.ops) != 1
            or len(expression.comparators) != 1
        ):
            return None

        operator = self._operator_to_text(
            expression.ops[0]
        )

        if operator is None:
            return None

        left = expression.left
        right = expression.comparators[0]

        if operator in {
            "in",
            "not in",
        }:
            if not isinstance(
                left,
                ast.Name,
            ):
                return None

            membership_values = (
                self._extract_membership_values(
                    right
                )
            )

            if membership_values is None:
                return None

            if not condition_is_true:
                operator = (
                    self._NEGATED_OPERATOR[
                        operator
                    ]
                )

            return PathConstraint(
                variable_name=left.id,
                operator=operator,
                value=membership_values,
            )

        variable_name: str | None = None
        literal_value: ConstraintAtom | None = None

        if isinstance(
            left,
            ast.Name,
        ):
            variable_name = left.id
            literal_value = (
                self._extract_literal(
                    right
                )
            )

        elif isinstance(
            right,
            ast.Name,
        ):
            variable_name = right.id
            literal_value = (
                self._extract_literal(
                    left
                )
            )

            if (
                literal_value is not None
            ):
                operator = (
                    self._REVERSED_OPERATOR[
                        operator
                    ]
                )

        if (
            variable_name is None
            or literal_value is None
        ):
            return None

        if (
            operator in {
                "<",
                "<=",
                ">",
                ">=",
            }
            and (
                isinstance(
                    literal_value,
                    bool,
                )
                or not isinstance(
                    literal_value,
                    (int, float),
                )
            )
        ):
            return None

        if not condition_is_true:
            operator = (
                self._NEGATED_OPERATOR[
                    operator
                ]
            )

        return PathConstraint(
            variable_name=variable_name,
            operator=operator,
            value=literal_value,
        )

    @classmethod
    def _operator_to_text(
        cls,
        operator: ast.cmpop,
    ) -> str | None:
        for (
            operator_type,
            operator_text,
        ) in cls._AST_OPERATOR_TO_TEXT.items():
            if isinstance(
                operator,
                operator_type,
            ):
                return operator_text

        return None

    @classmethod
    def _extract_literal(
        cls,
        node: ast.expr,
    ) -> ConstraintAtom | None:
        if isinstance(
            node,
            ast.Constant,
        ):
            value = node.value

            if isinstance(
                value,
                (int, float, str, bool),
            ):
                if (
                    isinstance(
                        value,
                        float,
                    )
                    and not math.isfinite(
                        value
                    )
                ):
                    return None

                return value

            return None

        if (
            isinstance(
                node,
                ast.UnaryOp,
            )
            and isinstance(
                node.op,
                (ast.USub, ast.UAdd),
            )
            and isinstance(
                node.operand,
                ast.Constant,
            )
            and not isinstance(
                node.operand.value,
                bool,
            )
            and isinstance(
                node.operand.value,
                (int, float),
            )
        ):
            value = node.operand.value

            if isinstance(
                node.op,
                ast.USub,
            ):
                value = -value

            numeric_value = float(
                value
            )

            if not math.isfinite(
                numeric_value
            ):
                return None

            return value

        return None

    @classmethod
    def _extract_membership_values(
        cls,
        node: ast.expr,
    ) -> tuple[ConstraintAtom, ...] | None:
        if not isinstance(
            node,
            (
                ast.Tuple,
                ast.List,
                ast.Set,
            ),
        ):
            return None

        values: list[ConstraintAtom] = []

        for element in node.elts:
            value = cls._extract_literal(
                element
            )

            if value is None:
                return None

            values.append(
                value
            )

        if not values:
            return None

        return tuple(
            values
        )

    @staticmethod
    def _format_unsupported_condition(
        *,
        condition: str,
        reason: str,
    ) -> str:
        return (
            f"{condition}: {reason}."
        )

    @staticmethod
    def _validate_constraints(
        constraints: tuple[
            PathConstraint,
            ...
        ],
    ) -> None:
        if not isinstance(
            constraints,
            tuple,
        ):
            raise TypeError(
                "constraints bir PathConstraint tuple'ı olmalıdır."
            )

        if any(
            not isinstance(
                constraint,
                PathConstraint,
            )
            for constraint in constraints
        ):
            raise TypeError(
                "constraints yalnızca PathConstraint "
                "nesneleri içermelidir."
            )

    @staticmethod
    def _validate_path(
        path: ExecutionPath,
    ) -> None:
        if not isinstance(
            path,
            ExecutionPath,
        ):
            raise TypeError(
                "path bir ExecutionPath örneği olmalıdır."
            )

    @classmethod
    def _validate_paths(
        cls,
        paths: tuple[
            ExecutionPath,
            ...
        ],
    ) -> None:
        if not isinstance(
            paths,
            tuple,
        ):
            raise TypeError(
                "paths bir ExecutionPath tuple'ı olmalıdır."
            )

        if any(
            not isinstance(
                path,
                ExecutionPath,
            )
            for path in paths
        ):
            raise TypeError(
                "paths yalnızca ExecutionPath "
                "nesneleri içermelidir."
            )
