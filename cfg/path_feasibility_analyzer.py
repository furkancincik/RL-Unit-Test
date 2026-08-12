from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from cfg.data_flow_analyzer import DataFlowAnalysisResult
from cfg.path_analyzer import ExecutionPath
from cfg.path_state_analyzer import PathSymbolicState


ConstraintAtom: TypeAlias = int | float | str | bool
ConstraintValue: TypeAlias = ConstraintAtom | tuple[ConstraintAtom, ...]
ConstraintClause: TypeAlias = tuple["PathConstraint", ...]
RelationalOperator: TypeAlias = str


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
class RelationalConstraint:
    """
    İki değişken arasındaki ilişkiyi temsil eder.

    Örnekler:
        stock < valid_item_count
        current_balance >= required_balance
        x == y
    """

    left_variable: str
    operator: RelationalOperator
    right_variable: str

    SUPPORTED_OPERATORS = {
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
    }

    def __post_init__(self) -> None:
        for name, value in (
            ("left_variable", self.left_variable),
            ("right_variable", self.right_variable),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} string olmalıdır."
                )

            if not value.strip():
                raise ValueError(
                    f"{name} boş olamaz."
                )

        if not isinstance(self.operator, str):
            raise TypeError(
                "operator string olmalıdır."
            )

        if self.operator not in self.SUPPORTED_OPERATORS:
            raise ValueError(
                "Desteklenmeyen relational operator: "
                f"{self.operator}"
            )


@dataclass(frozen=True, slots=True)
class ConstraintAlternativeGroup:
    """
    Aynı path koşulunun alternatif constraint kümelerini temsil eder.

    Örnek:
        A or B koşulunun True kolu

        Alternative 1:
            A == True

        Alternative 2:
            B == True

    Path'in bu grup bakımından mümkün olması için alternatiflerden
    en az birinin mevcut zorunlu constraint'lerle birlikte mümkün
    olması yeterlidir.
    """

    alternatives: tuple[ConstraintClause, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.alternatives,
            tuple,
        ):
            raise TypeError(
                "alternatives bir tuple olmalıdır."
            )

        if not self.alternatives:
            raise ValueError(
                "alternatives boş olamaz."
            )

        for alternative in self.alternatives:
            if not isinstance(
                alternative,
                tuple,
            ):
                raise TypeError(
                    "her alternative bir PathConstraint "
                    "tuple'ı olmalıdır."
                )

            if any(
                not isinstance(
                    constraint,
                    PathConstraint,
                )
                for constraint in alternative
            ):
                raise TypeError(
                    "alternatives yalnızca PathConstraint "
                    "nesneleri içermelidir."
                )


@dataclass(frozen=True, slots=True)
class PathConstraintExtractionResult:
    """
    Bir ExecutionPath üzerinden çıkarılan constraint bilgisini temsil eder.

    constraints:
        Bütün alternatiflerde ortak olan ve kesin olarak uygulanması
        gereken zorunlu constraint'ler.

    alternative_groups:
        AND False / OR True gibi birden fazla mantıksal olasılık
        içeren koşulların alternatif constraint kümeleri.

    unsupported_conditions:
        Mevcut analyzer'ın güvenli biçimde yorumlayamadığı koşullar.
    """

    constraints: tuple[PathConstraint, ...]
    unsupported_conditions: tuple[str, ...]
    alternative_groups: tuple[
        ConstraintAlternativeGroup,
        ...
    ] = ()
    relational_constraints: tuple[
        RelationalConstraint,
        ...
    ] = ()


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
    alternative_groups: tuple[
        ConstraintAlternativeGroup,
        ...
    ] = ()
    relational_constraints: tuple[
        RelationalConstraint,
        ...
    ] = ()

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

    v4.6 ile desteklenen başlıca yapılar:
    - Numeric karşılaştırmalar
    - String/bool equality ve inequality
    - Basit truthiness / ``not variable``
    - ``in`` / ``not in`` membership kontrolleri
    - AND True
    - OR False
    - AND False için alternatif constraint kümeleri
    - OR True için alternatif constraint kümeleri
    - İç içe boolean ifadelerin güvenli DNF-benzeri ayrıştırılması
    - Variable-to-variable relational constraint temsil ve basit çıkarım desteği
    - DataFlowAnalysisResult üzerinden inferred numeric range entegrasyonu
    - PathSymbolicState üzerinden path-sensitive numeric state entegrasyonu
    - Relational constraint'ler için bound tabanlı PROVEN_TRUE / PROVEN_FALSE / UNRESOLVED değerlendirmesi
    - UNRESOLVED relational constraint kümeleri için güvenli concrete-witness satisfiability araması

    Güvenlik ilkesi:
    Bir koşul kesin biçimde yorumlanamıyorsa path yanlışlıkla
    INFEASIBLE olarak işaretlenmez. Desteklenen bilgiler içinde
    kesin bir çelişki kanıtlanabiliyorsa INFEASIBLE korunur;
    aksi durumda UNKNOWN sonucu döndürülür.
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

    _MAX_ALTERNATIVE_COMBINATIONS = 4096
    _MAX_RELATIONAL_WITNESS_COMBINATIONS = 4096

    def analyze_constraints(
        self,
        constraints: tuple[PathConstraint, ...],
    ) -> PathFeasibilityResult:
        """
        Doğrudan verilen zorunlu constraint koleksiyonunu analiz eder.
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

        Basit ve deterministik koşullar ``constraints`` içerisine,
        alternatif mantıksal olasılıklar ``alternative_groups`` içerisine
        alınır.
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
                alternative_groups=(),
                relational_constraints=(),
            )

        mandatory_constraints: list[
            PathConstraint
        ] = []

        alternative_groups: list[
            ConstraintAlternativeGroup
        ] = []

        relational_constraints: list[
            RelationalConstraint
        ] = []

        unsupported_conditions: list[str] = []

        # Path içinde local olarak atanıp daha sonra güncellenen bir değişkeni
        # kullanan while koşulları statik constraint değildir. Aynı değişkenin
        # farklı iterasyonlardaki değerlerini tek domain'e sıkıştırmak,
        # örn. ``counter > 0`` True ... False akışını sahte çelişkiye dönüştürür.
        local_mutable_while_variables = (
            self._mutable_local_while_variable_names(
                path
            )
        )

        for step in path.condition_steps:
            edge_label = (
                step.outgoing_edge_label
            )

            if (
                step.node_type == "while"
                and self._condition_references_any_name(
                    condition=step.node_label,
                    variable_names=(
                        local_mutable_while_variables
                    ),
                )
            ):
                continue

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

            relational_constraint = (
                self._parse_relational_condition(
                    condition=step.node_label,
                    condition_is_true=(
                        edge_label == "True"
                    ),
                )
            )

            if relational_constraint is not None:
                relational_constraints.append(
                    relational_constraint
                )
                continue

            (
                clauses,
                unsupported,
            ) = self._parse_condition_clauses(
                condition=step.node_label,
                condition_is_true=(
                    edge_label == "True"
                ),
            )

            unsupported_conditions.extend(
                self._format_unsupported_condition(
                    condition=condition,
                    reason=reason,
                )
                for condition, reason
                in unsupported
            )

            if not clauses:
                continue

            (
                common_constraints,
                alternative_group,
            ) = self._split_common_constraints(
                clauses
            )

            mandatory_constraints.extend(
                common_constraints
            )

            if alternative_group is not None:
                alternative_groups.append(
                    alternative_group
                )

        return PathConstraintExtractionResult(
            constraints=tuple(
                self._deduplicate_constraints(
                    mandatory_constraints
                )
            ),
            unsupported_conditions=tuple(
                self._deduplicate_strings(
                    unsupported_conditions
                )
            ),
            alternative_groups=tuple(
                alternative_groups
            ),
            relational_constraints=tuple(
                self._deduplicate_relational_constraints(
                    relational_constraints
                )
            ),
        )

    def analyze_path(
        self,
        path: ExecutionPath,
        data_flow_result: DataFlowAnalysisResult | None = None,
        path_state: PathSymbolicState | None = None,
    ) -> PathFeasibilityResult:
        """
        ExecutionPath'i zorunlu, alternatif ve relational constraint'lerle
        analiz eder.

        ``data_flow_result`` verilirse DataFlowAnalyzer tarafından güvenli
        biçimde çıkarılmış sayısal alt/üst sınırlar relational reasoning
        sırasında ek domain bilgisi olarak kullanılır.
        """
        extraction = self.extract_constraints(
            path
        )

        (
            local_while_status,
            local_while_conflicts,
            local_while_unsupported,
        ) = self._analyze_local_while_paths(
            path
        )

        if (
            local_while_status
            == FeasibilityStatus.INFEASIBLE
        ):
            return PathFeasibilityResult(
                status=FeasibilityStatus.INFEASIBLE,
                constraints=extraction.constraints,
                conflicts=local_while_conflicts,
                unsupported_conditions=tuple(
                    self._deduplicate_strings(
                        [
                            *extraction.unsupported_conditions,
                            *local_while_unsupported,
                        ]
                    )
                ),
                alternative_groups=(
                    extraction.alternative_groups
                ),
                relational_constraints=(
                    extraction.relational_constraints
                ),
            )

        mandatory_result = (
            self.analyze_constraints(
                extraction.constraints
            )
        )

        if mandatory_result.is_infeasible:
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.INFEASIBLE
                ),
                constraints=(
                    extraction.constraints
                ),
                conflicts=(
                    mandatory_result.conflicts
                ),
                unsupported_conditions=(
                    extraction.unsupported_conditions
                ),
                alternative_groups=(
                    extraction.alternative_groups
                ),
                relational_constraints=(
                    extraction.relational_constraints
                ),
            )

        (
            alternatives_status,
            alternative_conflicts,
            alternative_limit_reached,
        ) = self._analyze_alternative_groups(
            mandatory_constraints=(
                extraction.constraints
            ),
            alternative_groups=(
                extraction.alternative_groups
            ),
        )

        (
            relational_status,
            relational_conflicts,
        ) = self._analyze_relational_constraints(
            literal_constraints=(
                extraction.constraints
            ),
            relational_constraints=(
                extraction.relational_constraints
            ),
            data_flow_result=data_flow_result,
            path_state=path_state,
        )

        if (
            relational_status
            == FeasibilityStatus.INFEASIBLE
        ):
            return PathFeasibilityResult(
                status=FeasibilityStatus.INFEASIBLE,
                constraints=extraction.constraints,
                conflicts=relational_conflicts,
                unsupported_conditions=(
                    extraction.unsupported_conditions
                ),
                alternative_groups=(
                    extraction.alternative_groups
                ),
                relational_constraints=(
                    extraction.relational_constraints
                ),
            )

        unsupported_conditions = [
            *extraction.unsupported_conditions,
            *local_while_unsupported,
        ]

        if alternative_limit_reached:
            unsupported_conditions.append(
                "Alternative constraint kombinasyon sınırı aşıldı."
            )

        unsupported_tuple = tuple(
            self._deduplicate_strings(
                unsupported_conditions
            )
        )

        if (
            alternatives_status
            == FeasibilityStatus.INFEASIBLE
        ):
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.INFEASIBLE
                ),
                constraints=(
                    extraction.constraints
                ),
                conflicts=(
                    alternative_conflicts
                ),
                unsupported_conditions=(
                    unsupported_tuple
                ),
                alternative_groups=(
                    extraction.alternative_groups
                ),
                relational_constraints=(
                    extraction.relational_constraints
                ),
            )

        if (
            unsupported_tuple
            or alternatives_status
            == FeasibilityStatus.UNKNOWN
            or relational_status
            == FeasibilityStatus.UNKNOWN
        ):
            return PathFeasibilityResult(
                status=(
                    FeasibilityStatus.UNKNOWN
                ),
                constraints=(
                    extraction.constraints
                ),
                conflicts=(),
                unsupported_conditions=(
                    unsupported_tuple
                ),
                alternative_groups=(
                    extraction.alternative_groups
                ),
                relational_constraints=(
                    extraction.relational_constraints
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
            alternative_groups=(
                extraction.alternative_groups
            ),
            relational_constraints=(
                extraction.relational_constraints
            ),
        )

    def _analyze_local_while_paths(
        self,
        path: ExecutionPath,
    ) -> tuple[
        FeasibilityStatus,
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """
        Path üzerinde local olarak atanıp döngü içinde güncellenen basit
        while sayaçlarını path-sensitive biçimde simüle eder.

        Desteklenen güvenli çekirdek:
        - ``name = numeric_literal``
        - tek isim kullanan basit numeric while comparison
        - ``name += numeric_literal``
        - ``name -= numeric_literal``

        Bir while koşulunun beklenen True/False CFG kenarı, simüle edilen
        concrete local state ile kesin olarak çelişirse path INFEASIBLE'dır.

        Başlangıç değeri veya update güvenli biçimde çözülemiyorsa yanlış bir
        INFEASIBLE kararı verilmez; durum UNKNOWN olarak üst analize taşınır.
        """
        self._validate_path(
            path
        )

        local_variables = (
            self._mutable_local_while_variable_names(
                path
            )
        )

        if not local_variables:
            return (
                FeasibilityStatus.FEASIBLE,
                (),
                (),
            )

        environment: dict[str, float] = {}
        unknown_variables: set[str] = set()
        unsupported: list[str] = []

        for step in path.steps:
            if step.node_type == "Assign":
                assignment = self._parse_local_numeric_assignment(
                    step.node_label
                )

                if assignment is not None:
                    variable_name, value = assignment

                    if variable_name in local_variables:
                        environment[variable_name] = value
                        unknown_variables.discard(
                            variable_name
                        )
                    continue

                assigned_name = (
                    self._extract_simple_assignment_target(
                        step.node_label
                    )
                )

                if (
                    assigned_name is not None
                    and assigned_name in local_variables
                ):
                    environment.pop(
                        assigned_name,
                        None,
                    )
                    unknown_variables.add(
                        assigned_name
                    )
                    unsupported.append(
                        "Local while başlangıç değeri kesin "
                        "olarak çözümlenemedi: "
                        f"{assigned_name}."
                    )

                continue

            if step.node_type == "while":
                referenced_names = (
                    self._condition_name_set(
                        step.node_label
                    )
                )

                tracked_names = (
                    referenced_names
                    & local_variables
                )

                if not tracked_names:
                    continue

                if step.outgoing_edge_label not in {
                    "True",
                    "False",
                }:
                    unsupported.append(
                        "Local while koşulunda True/False "
                        "edge bilgisi bulunamadı: "
                        f"{step.node_label!r}."
                    )
                    continue

                if (
                    tracked_names & unknown_variables
                    or any(
                        name not in environment
                        for name in tracked_names
                    )
                ):
                    unsupported.append(
                        "Local while koşulu için concrete "
                        "başlangıç state'i çözümlenemedi: "
                        f"{step.node_label!r}."
                    )
                    continue

                condition_value = (
                    self._evaluate_local_while_condition(
                        condition=step.node_label,
                        environment=environment,
                    )
                )

                if condition_value is None:
                    unsupported.append(
                        "Local while koşulu güvenli biçimde "
                        "değerlendirilemedi: "
                        f"{step.node_label!r}."
                    )
                    continue

                expected_value = (
                    step.outgoing_edge_label
                    == "True"
                )

                if condition_value != expected_value:
                    state_text = ", ".join(
                        f"{name}={environment[name]!r}"
                        for name in sorted(
                            tracked_names
                        )
                        if name in environment
                    )

                    return (
                        FeasibilityStatus.INFEASIBLE,
                        (
                            "Local while iterasyon akışı "
                            "CFG path'iyle çelişiyor: "
                            f"{step.node_label!r}; "
                            f"state=({state_text}); "
                            "beklenen edge="
                            f"{step.outgoing_edge_label}, "
                            "hesaplanan="
                            f"{condition_value}.",
                        ),
                        tuple(
                            self._deduplicate_strings(
                                unsupported
                            )
                        ),
                    )

                continue

            if step.node_type == "AugAssign":
                update = self._parse_local_numeric_augassign(
                    step.node_label
                )

                if update is None:
                    target_name = (
                        self._extract_augassign_target(
                            step.node_label
                        )
                    )

                    if (
                        target_name is not None
                        and target_name
                        in local_variables
                    ):
                        environment.pop(
                            target_name,
                            None,
                        )
                        unknown_variables.add(
                            target_name
                        )
                        unsupported.append(
                            "Local while update'i desteklenen "
                            "+= / -= numeric literal biçiminde "
                            "değil: "
                            f"{step.node_label!r}."
                        )

                    continue

                variable_name, delta = update

                if variable_name not in local_variables:
                    continue

                if variable_name not in environment:
                    unknown_variables.add(
                        variable_name
                    )
                    unsupported.append(
                        "Local while update'i için önceki "
                        "concrete state bilinmiyor: "
                        f"{variable_name}."
                    )
                    continue

                environment[variable_name] += delta

        unsupported_tuple = tuple(
            self._deduplicate_strings(
                unsupported
            )
        )

        if unsupported_tuple:
            return (
                FeasibilityStatus.UNKNOWN,
                (),
                unsupported_tuple,
            )

        return (
            FeasibilityStatus.FEASIBLE,
            (),
            (),
        )

    def _mutable_local_while_variable_names(
        self,
        path: ExecutionPath,
    ) -> set[str]:
        """
        Bir while koşulunda kullanılan, ilk while ziyaretinden önce local
        assignment alan ve path üzerinde AugAssign ile güncellenen isimleri
        döndürür.

        İsimler veya literal değerler dataset'e özel olarak hard-code edilmez.
        """
        if not path.has_node_metadata:
            return set()

        while_names: set[str] = set()
        first_while_index_by_name: dict[str, int] = {}
        assigned_indices_by_name: dict[
            str,
            list[int],
        ] = {}
        augmented_names: set[str] = set()

        for index, step in enumerate(
            path.steps
        ):
            if step.node_type == "while":
                for name in self._condition_name_set(
                    step.node_label
                ):
                    while_names.add(
                        name
                    )
                    first_while_index_by_name.setdefault(
                        name,
                        index,
                    )

            elif step.node_type == "Assign":
                target = (
                    self._extract_simple_assignment_target(
                        step.node_label
                    )
                )

                if target is not None:
                    assigned_indices_by_name.setdefault(
                        target,
                        [],
                    ).append(
                        index
                    )

            elif step.node_type == "AugAssign":
                target = self._extract_augassign_target(
                    step.node_label
                )

                if target is not None:
                    augmented_names.add(
                        target
                    )

        result: set[str] = set()

        for name in (
            while_names & augmented_names
        ):
            first_while_index = (
                first_while_index_by_name.get(
                    name
                )
            )

            if first_while_index is None:
                continue

            if any(
                assignment_index
                < first_while_index
                for assignment_index
                in assigned_indices_by_name.get(
                    name,
                    [],
                )
            ):
                result.add(
                    name
                )

        return result

    @staticmethod
    def _condition_references_any_name(
        *,
        condition: str,
        variable_names: set[str],
    ) -> bool:
        if not variable_names:
            return False

        return bool(
            PathFeasibilityAnalyzer._condition_name_set(
                condition
            )
            & variable_names
        )

    @staticmethod
    def _condition_name_set(
        condition: str,
    ) -> set[str]:
        try:
            expression = ast.parse(
                condition,
                mode="eval",
            )
        except SyntaxError:
            return set()

        return {
            node.id
            for node in ast.walk(
                expression
            )
            if isinstance(
                node,
                ast.Name,
            )
        }

    @staticmethod
    def _extract_simple_assignment_target(
        statement_text: str,
    ) -> str | None:
        try:
            module = ast.parse(
                statement_text
            )
        except SyntaxError:
            return None

        if (
            len(module.body) != 1
            or not isinstance(
                module.body[0],
                ast.Assign,
            )
        ):
            return None

        statement = module.body[0]

        if (
            len(statement.targets) != 1
            or not isinstance(
                statement.targets[0],
                ast.Name,
            )
        ):
            return None

        return statement.targets[0].id

    @classmethod
    def _parse_local_numeric_assignment(
        cls,
        statement_text: str,
    ) -> tuple[str, float] | None:
        target_name = (
            cls._extract_simple_assignment_target(
                statement_text
            )
        )

        if target_name is None:
            return None

        try:
            statement = ast.parse(
                statement_text
            ).body[0]
        except (
            SyntaxError,
            IndexError,
        ):
            return None

        assert isinstance(
            statement,
            ast.Assign,
        )

        value = cls._numeric_ast_literal(
            statement.value
        )

        if value is None:
            return None

        return (
            target_name,
            value,
        )

    @staticmethod
    def _extract_augassign_target(
        statement_text: str,
    ) -> str | None:
        try:
            module = ast.parse(
                statement_text
            )
        except SyntaxError:
            return None

        if (
            len(module.body) != 1
            or not isinstance(
                module.body[0],
                ast.AugAssign,
            )
        ):
            return None

        statement = module.body[0]

        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return None

        return statement.target.id

    @classmethod
    def _parse_local_numeric_augassign(
        cls,
        statement_text: str,
    ) -> tuple[str, float] | None:
        try:
            module = ast.parse(
                statement_text
            )
        except SyntaxError:
            return None

        if (
            len(module.body) != 1
            or not isinstance(
                module.body[0],
                ast.AugAssign,
            )
        ):
            return None

        statement = module.body[0]

        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return None

        amount = cls._numeric_ast_literal(
            statement.value
        )

        if amount is None:
            return None

        if isinstance(
            statement.op,
            ast.Add,
        ):
            delta = amount
        elif isinstance(
            statement.op,
            ast.Sub,
        ):
            delta = -amount
        else:
            return None

        return (
            statement.target.id,
            delta,
        )

    @staticmethod
    def _numeric_ast_literal(
        expression: ast.expr,
    ) -> float | None:
        if (
            isinstance(
                expression,
                ast.Constant,
            )
            and isinstance(
                expression.value,
                (int, float),
            )
            and not isinstance(
                expression.value,
                bool,
            )
        ):
            value = float(
                expression.value
            )

            if math.isfinite(
                value
            ):
                return value

            return None

        if (
            isinstance(
                expression,
                ast.UnaryOp,
            )
            and isinstance(
                expression.op,
                (ast.UAdd, ast.USub),
            )
        ):
            operand = (
                PathFeasibilityAnalyzer
                ._numeric_ast_literal(
                    expression.operand
                )
            )

            if operand is None:
                return None

            if isinstance(
                expression.op,
                ast.USub,
            ):
                return -operand

            return operand

        return None

    @classmethod
    def _evaluate_local_while_condition(
        cls,
        *,
        condition: str,
        environment: dict[str, float],
    ) -> bool | None:
        """
        Yalnızca yan etkisiz, tek numeric comparison içeren while koşulunu
        concrete local environment altında değerlendirir.
        """
        try:
            expression = ast.parse(
                condition,
                mode="eval",
            ).body
        except SyntaxError:
            return None

        if (
            not isinstance(
                expression,
                ast.Compare,
            )
            or len(expression.ops) != 1
            or len(expression.comparators) != 1
        ):
            return None

        left = cls._local_numeric_operand_value(
            expression.left,
            environment=environment,
        )
        right = cls._local_numeric_operand_value(
            expression.comparators[0],
            environment=environment,
        )

        if (
            left is None
            or right is None
        ):
            return None

        operator = expression.ops[0]

        if isinstance(operator, ast.Lt):
            return left < right

        if isinstance(operator, ast.LtE):
            return left <= right

        if isinstance(operator, ast.Gt):
            return left > right

        if isinstance(operator, ast.GtE):
            return left >= right

        if isinstance(operator, ast.Eq):
            return left == right

        if isinstance(operator, ast.NotEq):
            return left != right

        return None

    @classmethod
    def _local_numeric_operand_value(
        cls,
        expression: ast.expr,
        *,
        environment: dict[str, float],
    ) -> float | None:
        if isinstance(
            expression,
            ast.Name,
        ):
            return environment.get(
                expression.id
            )

        return cls._numeric_ast_literal(
            expression
        )

    def analyze_paths(
        self,
        paths: tuple[ExecutionPath, ...],
        data_flow_result: DataFlowAnalysisResult | None = None,
        path_states: tuple[PathSymbolicState, ...] | None = None,
    ) -> tuple[PathFeasibilityResult, ...]:
        """
        Birden fazla ExecutionPath nesnesini giriş sırasını koruyarak
        analiz eder.

        Aynı fonksiyona ait ``data_flow_result`` bütün path'lere ortak
        statik data-flow bilgisi olarak uygulanabilir.
        """
        self._validate_paths(
            paths
        )

        if path_states is not None:
            if not isinstance(path_states, tuple):
                raise TypeError(
                    "path_states bir PathSymbolicState tuple'ı olmalıdır."
                )

            if len(path_states) != len(paths):
                raise ValueError(
                    "path_states ve paths aynı uzunlukta olmalıdır."
                )

            if any(
                not isinstance(state, PathSymbolicState)
                for state in path_states
            ):
                raise TypeError(
                    "path_states yalnızca PathSymbolicState "
                    "nesneleri içermelidir."
                )

        return tuple(
            self.analyze_path(
                path,
                data_flow_result=data_flow_result,
                path_state=(
                    path_states[index]
                    if path_states is not None
                    else None
                ),
            )
            for index, path in enumerate(paths)
        )

    def _analyze_alternative_groups(
        self,
        *,
        mandatory_constraints: tuple[
            PathConstraint,
            ...
        ],
        alternative_groups: tuple[
            ConstraintAlternativeGroup,
            ...
        ],
    ) -> tuple[
        FeasibilityStatus,
        tuple[str, ...],
        bool,
    ]:
        """
        Alternative group'ları birlikte değerlendirir.

        Her grup için en az bir alternatif seçilmelidir. Gruplar arası
        etkileşimleri kaçırmamak için yalnız tek tek alternatiflere değil,
        mümkün kombinasyonlara bakılır. Her adımda infeasible kombinasyonlar
        elenerek arama alanı küçültülür.
        """
        if not alternative_groups:
            return (
                FeasibilityStatus.FEASIBLE,
                (),
                False,
            )

        active_combinations: list[
            tuple[PathConstraint, ...]
        ] = [
            mandatory_constraints
        ]

        last_conflicts: list[str] = []

        for group in alternative_groups:
            next_combinations: list[
                tuple[PathConstraint, ...]
            ] = []

            for base_constraints in (
                active_combinations
            ):
                for alternative in (
                    group.alternatives
                ):
                    combined = tuple(
                        self._deduplicate_constraints(
                            [
                                *base_constraints,
                                *alternative,
                            ]
                        )
                    )

                    conflicts = self._find_conflicts(
                        combined
                    )

                    if conflicts:
                        last_conflicts.extend(
                            conflicts
                        )
                        continue

                    if (
                        combined
                        not in next_combinations
                    ):
                        next_combinations.append(
                            combined
                        )

                    if (
                        len(next_combinations)
                        > self._MAX_ALTERNATIVE_COMBINATIONS
                    ):
                        return (
                            FeasibilityStatus.UNKNOWN,
                            (),
                            True,
                        )

            if not next_combinations:
                conflicts = tuple(
                    self._deduplicate_strings(
                        last_conflicts
                    )
                )

                if not conflicts:
                    conflicts = (
                        "Alternatif constraint kümelerinin "
                        "hiçbiri uygulanabilir değil.",
                    )

                return (
                    FeasibilityStatus.INFEASIBLE,
                    conflicts,
                    False,
                )

            active_combinations = (
                next_combinations
            )

        return (
            FeasibilityStatus.FEASIBLE,
            (),
            False,
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
            assert isinstance(
                value,
                tuple,
            )

            new_values = set(
                value
            )

            if domain.allowed_values is None:
                domain.allowed_values = (
                    new_values
                )
            else:
                domain.allowed_values.intersection_update(
                    new_values
                )

            return

        if operator == "not in":
            assert isinstance(
                value,
                tuple,
            )

            domain.excluded_values.update(
                value
            )
            return

        if operator == "==":
            assert not isinstance(
                value,
                tuple,
            )

            if (
                domain.has_exact_value
                and domain.exact_value != value
            ):
                domain.conflicting_exact_values = (
                    True
                )
            else:
                domain.exact_value = value
                domain.has_exact_value = True

            return

        if operator == "!=":
            assert not isinstance(
                value,
                tuple,
            )

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
                and exact
                not in domain.allowed_values
            ):
                return (
                    f"{variable_name}: "
                    f"{exact!r} izin verilen değerler "
                    "kümesiyle çelişiyor."
                )

            if (
                domain.requires_truthy
                and not bool(exact)
            ):
                return (
                    f"{variable_name}: "
                    f"{exact!r} truthy olma şartıyla çelişiyor."
                )

            if (
                domain.requires_falsy
                and bool(exact)
            ):
                return (
                    f"{variable_name}: "
                    f"{exact!r} falsy olma şartıyla çelişiyor."
                )

            if (
                isinstance(
                    exact,
                    (int, float),
                )
                and not isinstance(
                    exact,
                    bool,
                )
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

    def _parse_relational_condition(
        self,
        *,
        condition: str,
        condition_is_true: bool,
    ) -> RelationalConstraint | None:
        """
        Basit variable-to-variable comparison ifadelerini relational
        constraint olarak ayrıştırır.
        """
        try:
            expression = ast.parse(
                condition,
                mode="eval",
            ).body
        except SyntaxError:
            return None

        if not isinstance(
            expression,
            ast.Compare,
        ):
            return None

        if (
            len(expression.ops) != 1
            or len(expression.comparators) != 1
        ):
            return None

        left = expression.left
        right = expression.comparators[0]

        if (
            not isinstance(left, ast.Name)
            or not isinstance(right, ast.Name)
        ):
            return None

        operator = self._operator_to_text(
            expression.ops[0]
        )

        if operator not in {
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
        }:
            return None

        if not condition_is_true:
            operator = self._NEGATED_OPERATOR[
                operator
            ]

        return RelationalConstraint(
            left_variable=left.id,
            operator=operator,
            right_variable=right.id,
        )

    def _analyze_relational_constraints(
        self,
        *,
        literal_constraints: tuple[
            PathConstraint,
            ...
        ],
        relational_constraints: tuple[
            RelationalConstraint,
            ...
        ],
        data_flow_result: DataFlowAnalysisResult | None = None,
        path_state: PathSymbolicState | None = None,
    ) -> tuple[
        FeasibilityStatus,
        tuple[str, ...],
    ]:
        """
        Relational constraint'leri literal, data-flow ve path-state
        domain bilgileriyle birlikte güvenli biçimde değerlendirir.

        v4.6 iki aşamalı reasoning uygular:

        1. Universal truth proving
           - PROVEN_FALSE -> INFEASIBLE
           - PROVEN_TRUE  -> relation kesin doğru
           - UNRESOLVED   -> ikinci aşamaya geçilir

        2. Concrete witness satisfiability
           UNRESOLVED relation'lar için, bütün numeric domain ve
           relational constraint'leri aynı anda sağlayan somut bir değer
           ataması aranır.

           Bir witness bulunursa path'in relational bölümü FEASIBLE kabul
           edilir. Witness bulunamaması INFEASIBLE kanıtı sayılmaz; arama
           bilinçli olarak incomplete olduğu için UNKNOWN korunur.

        Bu yaklaşım ``x >= y`` ilişkisinin bütün domain üzerinde doğru
        olmasını şart koşmaz. Path feasibility için en az bir ortak çözümün
        varlığı yeterlidir.
        """
        if not relational_constraints:
            return (
                FeasibilityStatus.FEASIBLE,
                (),
            )

        domains = self._build_domains(
            literal_constraints
        )

        if data_flow_result is not None:
            self._apply_data_flow_ranges(
                domains=domains,
                data_flow_result=data_flow_result,
            )

        if path_state is not None:
            self._apply_path_symbolic_state(
                domains=domains,
                path_state=path_state,
            )

        conflicts: list[str] = []
        unresolved_relations: list[
            RelationalConstraint
        ] = []

        for relation in relational_constraints:
            truth_value = self._evaluate_relational_truth(
                relation=relation,
                domains=domains,
            )

            if truth_value is False:
                conflicts.append(
                    self._format_relational_false_conflict(
                        relation
                    )
                )
                continue

            if truth_value is None:
                unresolved_relations.append(
                    relation
                )

        if conflicts:
            return (
                FeasibilityStatus.INFEASIBLE,
                tuple(
                    self._deduplicate_strings(
                        conflicts
                    )
                ),
            )

        if not unresolved_relations:
            return (
                FeasibilityStatus.FEASIBLE,
                (),
            )

        witness = self._find_relational_witness(
            domains=domains,
            relational_constraints=(
                relational_constraints
            ),
        )

        if witness is not None:
            return (
                FeasibilityStatus.FEASIBLE,
                (),
            )

        return (
            FeasibilityStatus.UNKNOWN,
            (),
        )

    def _build_domains(
        self,
        constraints: tuple[
            PathConstraint,
            ...
        ],
    ) -> dict[str, _VariableDomain]:
        domains: dict[
            str,
            _VariableDomain,
        ] = {}

        for constraint in constraints:
            domain = domains.setdefault(
                constraint.variable_name,
                _VariableDomain(),
            )
            self._apply_constraint(
                domain=domain,
                constraint=constraint,
            )

        return domains

    def _apply_data_flow_ranges(
        self,
        *,
        domains: dict[str, _VariableDomain],
        data_flow_result: DataFlowAnalysisResult,
    ) -> None:
        """
        DataFlowAnalyzer'ın güvenli inferred numeric range sonuçlarını
        mevcut feasibility domain'leriyle birleştirir.

        Data-flow bilgisi path constraint'lerinin yerine geçmez; yalnızca
        ek statik sınır bilgisi sağlar.
        """
        if not isinstance(
            data_flow_result,
            DataFlowAnalysisResult,
        ):
            raise TypeError(
                "data_flow_result bir DataFlowAnalysisResult "
                "örneği olmalıdır."
            )

        for numeric_range in (
            data_flow_result.inferred_numeric_ranges
        ):
            domain = domains.setdefault(
                numeric_range.variable_name,
                _VariableDomain(),
            )

            if numeric_range.lower_bound is not None:
                self._update_lower_bound(
                    domain=domain,
                    value=float(
                        numeric_range.lower_bound
                    ),
                    inclusive=True,
                )

            if numeric_range.upper_bound is not None:
                self._update_upper_bound(
                    domain=domain,
                    value=float(
                        numeric_range.upper_bound
                    ),
                    inclusive=True,
                )

    def _apply_path_symbolic_state(
        self,
        *,
        domains: dict[str, _VariableDomain],
        path_state: PathSymbolicState,
    ) -> None:
        """
        PathStateAnalyzer tarafından tek bir execution path için
        çıkarılan symbolic numeric state bilgisini domain'lerle birleştirir.

        Path-local bilgi global data-flow bilgisinden daha özeldir.
        Ancak mevcut literal path constraint'lerini silmez; domain'e
        ek bilgi olarak uygulanır. Böylece çelişkiler güvenli biçimde
        relational reasoning tarafından kanıtlanabilir.
        """
        if not isinstance(path_state, PathSymbolicState):
            raise TypeError(
                "path_state bir PathSymbolicState örneği olmalıdır."
            )

        for variable_state in path_state.variables:
            domain = domains.setdefault(
                variable_state.variable_name,
                _VariableDomain(),
            )

            if variable_state.exact_value is not None:
                exact_value = float(
                    variable_state.exact_value
                )

                self._update_lower_bound(
                    domain=domain,
                    value=exact_value,
                    inclusive=True,
                )
                self._update_upper_bound(
                    domain=domain,
                    value=exact_value,
                    inclusive=True,
                )

            if variable_state.lower_bound is not None:
                self._update_lower_bound(
                    domain=domain,
                    value=float(
                        variable_state.lower_bound
                    ),
                    inclusive=True,
                )

            if variable_state.upper_bound is not None:
                self._update_upper_bound(
                    domain=domain,
                    value=float(
                        variable_state.upper_bound
                    ),
                    inclusive=True,
                )

    @staticmethod
    def _domain_exact_numeric(
        domain: _VariableDomain | None,
    ) -> float | None:
        """
        Domain kesin olarak tek bir sayısal değere indirgenmişse
        bu değeri döndürür.

        Kesin değer iki kaynaktan gelebilir:
        1. Doğrudan ``variable == literal`` constraint'i.
        2. Inclusive ve birbirine eşit alt/üst sınırlar.

        İkinci durum özellikle DataFlowAnalyzer tarafından üretilen
        ``[x, x]`` inferred numeric range bilgisinin relational
        reasoning içinde exact value olarak kullanılmasını sağlar.
        """
        if domain is None:
            return None

        if domain.has_exact_value:
            exact = domain.exact_value

            if (
                isinstance(exact, bool)
                or not isinstance(
                    exact,
                    (int, float),
                )
            ):
                return None

            return float(exact)

        if (
            domain.lower_bound is not None
            and domain.upper_bound is not None
            and domain.lower_bound
            == domain.upper_bound
            and domain.lower_inclusive
            and domain.upper_inclusive
        ):
            return float(
                domain.lower_bound
            )

        return None

    def _evaluate_relational_truth(
        self,
        *,
        relation: RelationalConstraint,
        domains: dict[
            str,
            _VariableDomain,
        ],
    ) -> bool | None:
        """
        Relation'ın mevcut domain'ler altında kesin doğruluk durumunu döndürür.

        True:
            Relation bütün mümkün değerlerde doğrudur.

        False:
            Relation bütün mümkün değerlerde yanlıştır / çelişkilidir.

        None:
            Mevcut domain bilgisi kesin karar vermeye yetmez.
        """
        left = domains.get(
            relation.left_variable
        )
        right = domains.get(
            relation.right_variable
        )

        if left is None or right is None:
            return None

        left_exact = self._domain_exact_numeric(
            left
        )
        right_exact = self._domain_exact_numeric(
            right
        )

        if (
            left_exact is not None
            and right_exact is not None
        ):
            return self._compare_numbers(
                left_exact,
                relation.operator,
                right_exact,
            )

        operator = relation.operator

        if operator == "<":
            if self._domains_prove_less_than(
                left=left,
                right=right,
                strict=True,
            ):
                return True

            if self._domains_prove_less_than_impossible(
                left=left,
                right=right,
                strict=True,
            ):
                return False

            return None

        if operator == "<=":
            if self._domains_prove_less_than(
                left=left,
                right=right,
                strict=False,
            ):
                return True

            if self._domains_prove_less_than_impossible(
                left=left,
                right=right,
                strict=False,
            ):
                return False

            return None

        if operator == ">":
            return self._evaluate_relational_truth(
                relation=RelationalConstraint(
                    left_variable=relation.right_variable,
                    operator="<",
                    right_variable=relation.left_variable,
                ),
                domains=domains,
            )

        if operator == ">=":
            return self._evaluate_relational_truth(
                relation=RelationalConstraint(
                    left_variable=relation.right_variable,
                    operator="<=",
                    right_variable=relation.left_variable,
                ),
                domains=domains,
            )

        if operator == "==":
            if (
                left_exact is not None
                and right_exact is not None
            ):
                return left_exact == right_exact

            if self._domains_are_disjoint(
                left=left,
                right=right,
            ):
                return False

            return None

        if operator == "!=":
            if (
                left_exact is not None
                and right_exact is not None
            ):
                return left_exact != right_exact

            if self._domains_are_disjoint(
                left=left,
                right=right,
            ):
                return True

            return None

        return None

    @classmethod
    def _domain_effective_lower(
        cls,
        domain: _VariableDomain,
    ) -> tuple[float, bool] | None:
        """
        Domain'in kullanılabilir sayısal alt ucunu döndürür.

        Doğrudan ``x == 0`` gibi exact constraint'ler _VariableDomain
        üzerinde lower_bound/upper_bound alanlarını doldurmayabilir.
        Relational truth proving sırasında exact numeric değer, güvenli
        biçimde hem alt hem üst sınır olarak kullanılabilir.
        """
        exact = cls._domain_exact_numeric(
            domain
        )

        if exact is not None:
            return (
                exact,
                True,
            )

        if domain.lower_bound is None:
            return None

        return (
            domain.lower_bound,
            domain.lower_inclusive,
        )

    @classmethod
    def _domain_effective_upper(
        cls,
        domain: _VariableDomain,
    ) -> tuple[float, bool] | None:
        """
        Domain'in kullanılabilir sayısal üst ucunu döndürür.

        Exact numeric değer varsa bu değer inclusive üst sınır olarak
        değerlendirilir.
        """
        exact = cls._domain_exact_numeric(
            domain
        )

        if exact is not None:
            return (
                exact,
                True,
            )

        if domain.upper_bound is None:
            return None

        return (
            domain.upper_bound,
            domain.upper_inclusive,
        )

    @classmethod
    def _domains_prove_less_than(
        cls,
        *,
        left: _VariableDomain,
        right: _VariableDomain,
        strict: bool,
    ) -> bool:
        """
        Domain sınırlarından ``left < right`` veya ``left <= right``
        ilişkisinin bütün olası değerlerde doğru olduğunu kanıtlar.

        Exact numeric değerler de [x, x] inclusive domain gibi ele alınır.
        """
        left_upper = cls._domain_effective_upper(
            left
        )
        right_lower = cls._domain_effective_lower(
            right
        )

        if (
            left_upper is None
            or right_lower is None
        ):
            return False

        (
            left_upper_value,
            left_upper_inclusive,
        ) = left_upper

        (
            right_lower_value,
            right_lower_inclusive,
        ) = right_lower

        if (
            left_upper_value
            < right_lower_value
        ):
            return True

        if (
            left_upper_value
            > right_lower_value
        ):
            return False

        if not strict:
            return True

        return (
            not left_upper_inclusive
            or not right_lower_inclusive
        )

    @classmethod
    def _domains_prove_less_than_impossible(
        cls,
        *,
        left: _VariableDomain,
        right: _VariableDomain,
        strict: bool,
    ) -> bool:
        """
        Domain sınırlarından ``left < right`` veya ``left <= right``
        ilişkisinin hiçbir olası değerde sağlanamayacağını kanıtlar.

        Exact numeric değerler de effective lower/upper bound olarak
        değerlendirilir.
        """
        left_lower = cls._domain_effective_lower(
            left
        )
        right_upper = cls._domain_effective_upper(
            right
        )

        if (
            left_lower is None
            or right_upper is None
        ):
            return False

        (
            left_lower_value,
            left_lower_inclusive,
        ) = left_lower

        (
            right_upper_value,
            right_upper_inclusive,
        ) = right_upper

        if (
            left_lower_value
            > right_upper_value
        ):
            return True

        if (
            left_lower_value
            < right_upper_value
        ):
            return False

        if strict:
            return True

        return (
            not left_lower_inclusive
            or not right_upper_inclusive
        )

    @classmethod
    def _domains_are_disjoint(
        cls,
        *,
        left: _VariableDomain,
        right: _VariableDomain,
    ) -> bool:
        """
        İki numeric domain'in ortak değer içeremediğini güvenli biçimde
        kanıtlar. Exact numeric değerler effective bound olarak kullanılır.
        """
        left_upper = cls._domain_effective_upper(
            left
        )
        left_lower = cls._domain_effective_lower(
            left
        )
        right_upper = cls._domain_effective_upper(
            right
        )
        right_lower = cls._domain_effective_lower(
            right
        )

        if (
            left_upper is not None
            and right_lower is not None
        ):
            (
                left_upper_value,
                left_upper_inclusive,
            ) = left_upper

            (
                right_lower_value,
                right_lower_inclusive,
            ) = right_lower

            if (
                left_upper_value
                < right_lower_value
            ):
                return True

            if (
                left_upper_value
                == right_lower_value
                and (
                    not left_upper_inclusive
                    or not right_lower_inclusive
                )
            ):
                return True

        if (
            right_upper is not None
            and left_lower is not None
        ):
            (
                right_upper_value,
                right_upper_inclusive,
            ) = right_upper

            (
                left_lower_value,
                left_lower_inclusive,
            ) = left_lower

            if (
                right_upper_value
                < left_lower_value
            ):
                return True

            if (
                right_upper_value
                == left_lower_value
                and (
                    not right_upper_inclusive
                    or not left_lower_inclusive
                )
            ):
                return True

        return False

    def _find_relational_witness(
        self,
        *,
        domains: dict[
            str,
            _VariableDomain,
        ],
        relational_constraints: tuple[
            RelationalConstraint,
            ...
        ],
    ) -> dict[str, float] | None:
        """
        Bütün relational constraint'leri aynı anda sağlayan somut numeric
        bir atama arar.

        Bu bir tam SMT solver değildir. Aday değerler domain sınırlarından,
        exact değerlerden, excluded değerlerin komşularından ve ilişkideki
        diğer değişkenlerin sınırlarından türetilir.

        Güvenlik özelliği:
        - Witness bulunduysa gerçekten bütün domain/relation kontrollerinden
          geçirilmiştir; bu nedenle FEASIBLE kararı güvenlidir.
        - Witness bulunamazsa INFEASIBLE denmez, UNKNOWN korunur.
        """
        variable_names = tuple(
            sorted(
                {
                    variable_name
                    for relation in relational_constraints
                    for variable_name in (
                        relation.left_variable,
                        relation.right_variable,
                    )
                }
            )
        )

        if not variable_names:
            return {}

        candidate_map: dict[
            str,
            tuple[float, ...],
        ] = {}

        for variable_name in variable_names:
            domain = domains.get(
                variable_name
            )

            if domain is None:
                return None

            candidates = self._numeric_domain_candidates(
                variable_name=variable_name,
                domain=domain,
                domains=domains,
                relational_constraints=(
                    relational_constraints
                ),
            )

            if not candidates:
                return None

            candidate_map[
                variable_name
            ] = candidates

        combination_count = 1

        for variable_name in variable_names:
            combination_count *= len(
                candidate_map[
                    variable_name
                ]
            )

            if (
                combination_count
                > self._MAX_RELATIONAL_WITNESS_COMBINATIONS
            ):
                return None

        candidate_lists = [
            candidate_map[
                variable_name
            ]
            for variable_name in variable_names
        ]

        for values in itertools.product(
            *candidate_lists
        ):
            assignment = dict(
                zip(
                    variable_names,
                    values,
                    strict=True,
                )
            )

            if all(
                self._numeric_value_satisfies_domain(
                    value=assignment[
                        variable_name
                    ],
                    domain=domains[
                        variable_name
                    ],
                )
                for variable_name
                in variable_names
            ) and all(
                self._assignment_satisfies_relation(
                    assignment=assignment,
                    relation=relation,
                )
                for relation
                in relational_constraints
            ):
                return assignment

        return None

    def _numeric_domain_candidates(
        self,
        *,
        variable_name: str,
        domain: _VariableDomain,
        domains: dict[
            str,
            _VariableDomain,
        ],
        relational_constraints: tuple[
            RelationalConstraint,
            ...
        ],
    ) -> tuple[float, ...]:
        """
        Witness araması için küçük ama kullanışlı bir numeric aday kümesi
        üretir. Her aday daha sonra domain doğrulamasından geçirilir.
        """
        exact = self._domain_exact_numeric(
            domain
        )

        if exact is not None:
            if self._numeric_value_satisfies_domain(
                value=exact,
                domain=domain,
            ):
                return (
                    exact,
                )

            return ()

        raw_candidates: set[float] = {
            -1.0,
            0.0,
            1.0,
        }

        lower = self._domain_effective_lower(
            domain
        )
        upper = self._domain_effective_upper(
            domain
        )

        if lower is not None:
            lower_value, lower_inclusive = lower

            if lower_inclusive:
                raw_candidates.add(
                    lower_value
                )

            raw_candidates.add(
                lower_value + 1.0
            )
            raw_candidates.add(
                lower_value + 2.0
            )

        if upper is not None:
            upper_value, upper_inclusive = upper

            if upper_inclusive:
                raw_candidates.add(
                    upper_value
                )

            raw_candidates.add(
                upper_value - 1.0
            )
            raw_candidates.add(
                upper_value - 2.0
            )

        for excluded in domain.excluded_values:
            if (
                isinstance(excluded, bool)
                or not isinstance(
                    excluded,
                    (int, float),
                )
            ):
                continue

            excluded_value = float(
                excluded
            )

            raw_candidates.add(
                excluded_value - 1.0
            )
            raw_candidates.add(
                excluded_value + 1.0
            )

        # Relation'ın diğer tarafındaki sınırlar da iyi witness
        # adaylarıdır. Örn. x < y ve her ikisi >= 1 ise x=1, y=2.
        for relation in relational_constraints:
            if (
                relation.left_variable
                == variable_name
            ):
                other_name = (
                    relation.right_variable
                )
            elif (
                relation.right_variable
                == variable_name
            ):
                other_name = (
                    relation.left_variable
                )
            else:
                continue

            other_domain = domains.get(
                other_name
            )

            if other_domain is None:
                continue

            other_lower = (
                self._domain_effective_lower(
                    other_domain
                )
            )
            other_upper = (
                self._domain_effective_upper(
                    other_domain
                )
            )

            if other_lower is not None:
                other_lower_value = (
                    other_lower[0]
                )

                raw_candidates.update(
                    {
                        other_lower_value - 1.0,
                        other_lower_value,
                        other_lower_value + 1.0,
                        other_lower_value + 2.0,
                    }
                )

            if other_upper is not None:
                other_upper_value = (
                    other_upper[0]
                )

                raw_candidates.update(
                    {
                        other_upper_value - 2.0,
                        other_upper_value - 1.0,
                        other_upper_value,
                        other_upper_value + 1.0,
                    }
                )

        valid_candidates = sorted(
            candidate
            for candidate
            in raw_candidates
            if (
                math.isfinite(
                    candidate
                )
                and self._numeric_value_satisfies_domain(
                    value=candidate,
                    domain=domain,
                )
            )
        )

        return tuple(
            valid_candidates
        )

    @staticmethod
    def _numeric_value_satisfies_domain(
        *,
        value: float,
        domain: _VariableDomain,
    ) -> bool:
        if not math.isfinite(
            value
        ):
            return False

        if domain.conflicting_exact_values:
            return False

        if domain.has_exact_value:
            exact = domain.exact_value

            if (
                isinstance(exact, bool)
                or not isinstance(
                    exact,
                    (int, float),
                )
            ):
                return False

            if value != float(
                exact
            ):
                return False

        if domain.lower_bound is not None:
            if value < domain.lower_bound:
                return False

            if (
                value == domain.lower_bound
                and not domain.lower_inclusive
            ):
                return False

        if domain.upper_bound is not None:
            if value > domain.upper_bound:
                return False

            if (
                value == domain.upper_bound
                and not domain.upper_inclusive
            ):
                return False

        for excluded in domain.excluded_values:
            if (
                isinstance(excluded, bool)
                or not isinstance(
                    excluded,
                    (int, float),
                )
            ):
                continue

            if value == float(
                excluded
            ):
                return False

        if domain.allowed_values is not None:
            numeric_allowed = {
                float(item)
                for item in domain.allowed_values
                if (
                    not isinstance(
                        item,
                        bool,
                    )
                    and isinstance(
                        item,
                        (int, float),
                    )
                )
            }

            if (
                numeric_allowed
                and value
                not in numeric_allowed
            ):
                return False

        if (
            domain.requires_truthy
            and value == 0.0
        ):
            return False

        if (
            domain.requires_falsy
            and value != 0.0
        ):
            return False

        return True

    def _assignment_satisfies_relation(
        self,
        *,
        assignment: dict[
            str,
            float,
        ],
        relation: RelationalConstraint,
    ) -> bool:
        left = assignment.get(
            relation.left_variable
        )
        right = assignment.get(
            relation.right_variable
        )

        if (
            left is None
            or right is None
        ):
            return False

        return self._compare_numbers(
            left,
            relation.operator,
            right,
        )

    @staticmethod
    def _format_relational_false_conflict(
        relation: RelationalConstraint,
    ) -> str:
        return (
            f"{relation.left_variable} "
            f"{relation.operator} "
            f"{relation.right_variable}: "
            "mevcut domain sınırları ilişkiyi kesin olarak dışlıyor."
        )

    def _detect_relational_conflict(
        self,
        *,
        relation: RelationalConstraint,
        domains: dict[
            str,
            _VariableDomain,
        ],
    ) -> str | None:
        left = domains.get(
            relation.left_variable
        )
        right = domains.get(
            relation.right_variable
        )

        left_exact = self._domain_exact_numeric(
            left
        )
        right_exact = self._domain_exact_numeric(
            right
        )

        if (
            left_exact is not None
            and right_exact is not None
        ):
            satisfied = self._compare_numbers(
                left_exact,
                relation.operator,
                right_exact,
            )

            if not satisfied:
                return (
                    f"{relation.left_variable} "
                    f"{relation.operator} "
                    f"{relation.right_variable}: "
                    "exact değerler ilişkiyle çelişiyor."
                )

            return None

        if (
            left is None
            or right is None
        ):
            return None

        if relation.operator in {
            "<",
            "<=",
        }:
            if (
                left.lower_bound is not None
                and right.upper_bound is not None
            ):
                if (
                    left.lower_bound
                    > right.upper_bound
                ):
                    return (
                        f"{relation.left_variable} "
                        f"{relation.operator} "
                        f"{relation.right_variable}: "
                        "sol alt sınır sağ üst sınırı aşıyor."
                    )

                if (
                    left.lower_bound
                    == right.upper_bound
                    and relation.operator == "<"
                    and (
                        left.lower_inclusive
                        and right.upper_inclusive
                    )
                ):
                    return (
                        f"{relation.left_variable} < "
                        f"{relation.right_variable}: "
                        "strict ilişki için geçerli değer kalmıyor."
                    )

        if relation.operator in {
            ">",
            ">=",
        }:
            reversed_relation = RelationalConstraint(
                left_variable=relation.right_variable,
                operator=(
                    "<"
                    if relation.operator == ">"
                    else "<="
                ),
                right_variable=relation.left_variable,
            )

            return self._detect_relational_conflict(
                relation=reversed_relation,
                domains=domains,
            )

        if relation.operator == "==":
            if (
                left_exact is not None
                and right_exact is not None
                and left_exact != right_exact
            ):
                return (
                    f"{relation.left_variable} == "
                    f"{relation.right_variable}: "
                    "exact değerler eşit değil."
                )

        return None

    @staticmethod
    def _compare_numbers(
        left: float,
        operator: str,
        right: float,
    ) -> bool:
        match operator:
            case "<":
                return left < right
            case "<=":
                return left <= right
            case ">":
                return left > right
            case ">=":
                return left >= right
            case "==":
                return left == right
            case "!=":
                return left != right

        return False

    def _parse_condition_clauses(
        self,
        *,
        condition: str,
        condition_is_true: bool,
    ) -> tuple[
        tuple[ConstraintClause, ...],
        tuple[tuple[str, str], ...],
    ]:
        """
        Bir condition'ı DNF-benzeri alternatif constraint clause'larına
        dönüştürür.

        Her clause kendi içinde AND anlamına gelir.
        Clause'lar birbirine OR ile bağlı alternatiflerdir.
        """
        try:
            expression = ast.parse(
                condition,
                mode="eval",
            ).body
        except SyntaxError:
            return (
                (
                    (),
                ),
                (
                    (
                        condition,
                        "Python ifadesi parse edilemedi",
                    ),
                ),
            )

        clauses, unsupported = (
            self._expression_to_clauses(
                expression=expression,
                condition_is_true=(
                    condition_is_true
                ),
                original_condition=condition,
            )
        )

        return (
            self._normalize_clauses(
                clauses
            ),
            unsupported,
        )

    def _expression_to_clauses(
        self,
        *,
        expression: ast.expr,
        condition_is_true: bool,
        original_condition: str,
    ) -> tuple[
        tuple[ConstraintClause, ...],
        tuple[tuple[str, str], ...],
    ]:
        if isinstance(
            expression,
            ast.BoolOp,
        ):
            return self._boolean_expression_to_clauses(
                expression=expression,
                condition_is_true=condition_is_true,
                original_condition=original_condition,
            )

        if (
            isinstance(
                expression,
                ast.UnaryOp,
            )
            and isinstance(
                expression.op,
                ast.Not,
            )
        ):
            return self._expression_to_clauses(
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
            constraint = PathConstraint(
                variable_name=expression.id,
                operator=(
                    "truthy"
                    if condition_is_true
                    else "falsy"
                ),
                value=True,
            )

            return (
                (
                    (
                        constraint,
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
                        (
                            constraint,
                        ),
                    ),
                    (),
                )

        return (
            (
                (),
            ),
            (
                (
                    original_condition,
                    "desteklenen constraint yapısına "
                    "dönüştürülemedi",
                ),
            ),
        )

    def _boolean_expression_to_clauses(
        self,
        *,
        expression: ast.BoolOp,
        condition_is_true: bool,
        original_condition: str,
    ) -> tuple[
        tuple[ConstraintClause, ...],
        tuple[tuple[str, str], ...],
    ]:
        is_and = isinstance(
            expression.op,
            ast.And,
        )

        is_or = isinstance(
            expression.op,
            ast.Or,
        )

        if not (
            is_and
            or is_or
        ):
            return (
                (
                    (),
                ),
                (
                    (
                        original_condition,
                        "desteklenmeyen boolean operator",
                    ),
                ),
            )

        # AND True  -> bütün operandlar True olmalı.
        # OR False  -> bütün operandlar False olmalı.
        combine_with_and = (
            is_and
            and condition_is_true
        ) or (
            is_or
            and not condition_is_true
        )

        # AND False -> operandlardan en az biri False.
        # OR True   -> operandlardan en az biri True.
        combine_with_or = not combine_with_and

        collected_clause_sets: list[
            tuple[ConstraintClause, ...]
        ] = []

        unsupported: list[
            tuple[str, str]
        ] = []

        for value in expression.values:
            child_clauses, child_unsupported = (
                self._expression_to_clauses(
                    expression=value,
                    condition_is_true=(
                        condition_is_true
                        if combine_with_and
                        else condition_is_true
                    ),
                    original_condition=ast.unparse(
                        value
                    ),
                )
            )

            if combine_with_or:
                child_clauses, child_unsupported = (
                    self._expression_to_clauses(
                        expression=value,
                        condition_is_true=(
                            False
                            if is_and
                            else True
                        ),
                        original_condition=ast.unparse(
                            value
                        ),
                    )
                )

            collected_clause_sets.append(
                child_clauses
            )

            unsupported.extend(
                child_unsupported
            )

        if combine_with_and:
            clauses = (
                (
                    (),
                )
            )

            for child_clauses in (
                collected_clause_sets
            ):
                clauses = (
                    self._and_clause_sets(
                        clauses,
                        child_clauses,
                    )
                )

            return (
                clauses,
                tuple(
                    unsupported
                ),
            )

        combined_alternatives: list[
            ConstraintClause
        ] = []

        for child_clauses in (
            collected_clause_sets
        ):
            combined_alternatives.extend(
                child_clauses
            )

        return (
            tuple(
                combined_alternatives
            ),
            tuple(
                unsupported
            ),
        )

    @staticmethod
    def _and_clause_sets(
        left_clauses: tuple[
            ConstraintClause,
            ...
        ],
        right_clauses: tuple[
            ConstraintClause,
            ...
        ],
    ) -> tuple[
        ConstraintClause,
        ...
    ]:
        combined: list[
            ConstraintClause
        ] = []

        for left in left_clauses:
            for right in right_clauses:
                combined.append(
                    tuple(
                        [
                            *left,
                            *right,
                        ]
                    )
                )

        return tuple(
            combined
        )

    def _split_common_constraints(
        self,
        clauses: tuple[
            ConstraintClause,
            ...
        ],
    ) -> tuple[
        tuple[PathConstraint, ...],
        ConstraintAlternativeGroup | None,
    ]:
        normalized = self._normalize_clauses(
            clauses
        )

        if not normalized:
            return (
                (),
                None,
            )

        if len(normalized) == 1:
            return (
                normalized[0],
                None,
            )

        common = list(
            normalized[0]
        )

        for clause in normalized[1:]:
            common = [
                constraint
                for constraint in common
                if constraint in clause
            ]

        common = (
            self._deduplicate_constraints(
                common
            )
        )

        alternatives: list[
            ConstraintClause
        ] = []

        for clause in normalized:
            remainder = tuple(
                constraint
                for constraint in clause
                if constraint
                not in common
            )

            if remainder not in alternatives:
                alternatives.append(
                    remainder
                )

        if len(alternatives) == 1:
            return (
                tuple(
                    [
                        *common,
                        *alternatives[0],
                    ]
                ),
                None,
            )

        return (
            tuple(
                common
            ),
            ConstraintAlternativeGroup(
                alternatives=tuple(
                    alternatives
                )
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
                literal_value
                is not None
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

        values: list[
            ConstraintAtom
        ] = []

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

    def _normalize_clauses(
        self,
        clauses: tuple[
            ConstraintClause,
            ...
        ],
    ) -> tuple[
        ConstraintClause,
        ...
    ]:
        normalized: list[
            ConstraintClause
        ] = []

        for clause in clauses:
            deduplicated = tuple(
                self._deduplicate_constraints(
                    clause
                )
            )

            if deduplicated not in normalized:
                normalized.append(
                    deduplicated
                )

        return tuple(
            normalized
        )

    @staticmethod
    def _deduplicate_constraints(
        constraints: list[PathConstraint]
        | tuple[PathConstraint, ...],
    ) -> list[PathConstraint]:
        result: list[
            PathConstraint
        ] = []

        for constraint in constraints:
            if constraint not in result:
                result.append(
                    constraint
                )

        return result

    @staticmethod
    def _deduplicate_relational_constraints(
        constraints: list[
            RelationalConstraint
        ]
        | tuple[
            RelationalConstraint,
            ...
        ],
    ) -> list[RelationalConstraint]:
        result: list[
            RelationalConstraint
        ] = []

        for constraint in constraints:
            if constraint not in result:
                result.append(
                    constraint
                )

        return result

    @staticmethod
    def _deduplicate_strings(
        values: list[str]
        | tuple[str, ...],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            if value not in result:
                result.append(
                    value
                )

        return result

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
