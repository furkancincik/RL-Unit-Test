
import ast
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from analyzer.safe_custom_object import (
    UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER,
)

from cfg.path_analyzer import ExecutionPath, PathStep
from generator.derived_value_input_synthesizer import (
    DerivedValueInputSynthesizer,
    DerivedValueSynthesisError,
    UnsupportedDerivedValueSynthesisError,
)

if TYPE_CHECKING:
    from generator.safe_method_setup_plan import SafeObjectSetupPlan


_SHADOWED_SAFE_CALL = object()
_MISSING_COLLECTION_VALUE = object()
_COLLECTION_WITNESS_SEARCH_LIMIT = 256

_RUNTIME_TYPE_ALLOWLIST: dict[str, type[Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
}

class UnreachablePathError(ValueError):
    """Bir yürütme yolundaki kısıtlar çelişkili olduğunda oluşur."""


class UnsupportedExpectedResultError(ValueError):
    """Beklenen sonuç güvenli replay allowlist'iyle çözülemediğinde oluşur."""

    category = "UNSUPPORTED_EXPECTED_RESULT"

    def __init__(
        self,
        *,
        return_expression: str,
        detail: str,
    ) -> None:
        self.return_expression = return_expression
        self.detail = detail
        super().__init__(
            "Dinamik return ifadesi güvenli biçimde "
            "hesaplanamadı: "
            f"{return_expression}. Ayrıntı: {detail}"
        )


class UnsupportedInputSynthesisError(ValueError):
    """Bir path girdisi güvenli synthesis allowlist'iyle çözülemediğinde oluşur."""

    category = "UNSUPPORTED_INPUT_SYNTHESIS"


@dataclass(frozen=True, slots=True)
class GeneratedTestInput:
    """
    Bir yürütme yolunu çalıştırmak için üretilen somut test
    girdilerini ve beklenen sonucu temsil eder.

    Attributes:
        keyword_arguments:
            Fonksiyona isimleriyle gönderilecek parametre değerleri.

        expected_result:
            Fonksiyon çağrısından beklenen dönüş değeri.

        expected_exception:
            Beklenen exception sınıfının adı.
            Exception beklenmiyorsa None olur.
    """

    keyword_arguments: tuple[tuple[str, Any], ...]
    expected_result: Any = None
    expected_exception: str | None = None
    setup_plan: SafeObjectSetupPlan | None = field(default=None, repr=False)

    @property
    def keyword_argument_dict(self) -> dict[str, Any]:
        """
        Parametreleri sözlük biçiminde döndürür.
        """
        return dict(self.keyword_arguments)


@dataclass(frozen=True, slots=True)
class _VariableConstraint:
    """
    Tek bir değişken için çıkarılan basit kısıtları temsil eder.
    """

    minimum: float | int | None = None
    minimum_inclusive: bool = True

    maximum: float | int | None = None
    maximum_inclusive: bool = True

    equal_value: Any = None
    has_equal_value: bool = False

    forbidden_values: tuple[Any, ...] = ()
    allowed_values: tuple[Any, ...] | None = None
    required_collection_members: tuple[Any, ...] = ()
    forbidden_collection_members: tuple[Any, ...] = ()
    required_runtime_type_groups: tuple[tuple[str, ...], ...] = ()
    forbidden_runtime_types: tuple[str, ...] = ()


@dataclass(slots=True)
class _ForLoopActivation:
    """Tek bir dinamik ``for`` aktivasyonunun yol üzerindeki durumu."""

    node_id: int
    activation_id: int
    parent_context: tuple[tuple[int, int], ...]
    target_name: str
    iterable_name: str
    iteration_index: int = -1
    iteration_constraints: dict[int, _VariableConstraint] = field(
        default_factory=dict
    )
    local_iteration_constraints: dict[
        int, dict[str, _VariableConstraint]
    ] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _StructuredInputReference:
    """Gerçek bir parametre içindeki güvenli list/dict değer yolu."""

    parameter_name: str
    access_path: tuple[int | str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DictionaryLookup:
    """Yerel bir değerin doğrulanmış ``dict.get`` provenance'ı."""

    mapping: _StructuredInputReference
    key: _StructuredInputReference | int | float | str | bool | None
    default: _StructuredInputReference | int | float | str | bool | None
    has_default: bool


class PathInputGenerator:
    """
    ExecutionPath üzerindeki koşullardan somut test girdileri üretir.

    Desteklenen temel koşullar:

    - x > değer
    - x >= değer
    - x < değer
    - x <= değer
    - x == değer
    - x != değer
    - x in (değerler)
    - x not in (değerler)
    - değer < x
    - değer <= x
    - bool parametre kontrolleri
    - and/or içeren bileşik koşullar
    - iç içe not ifadeleri
    - sınırlı while döngüsü girdileri
    - for döngüleri için iterable girdileri
    - path replay sırasında for hedeflerinin iterasyon değerleri
    - upstream katmanda çözülenmiş değişkenler arası ilişkiler
    - sabit ve güvenli dinamik return ifadeleri
    - güvenli f-string return ifadeleri

    Bu sınıf pytest kodu üretmez. Yalnızca yürütme yolunu
    çalıştırabilecek girdileri ve beklenen sonucu hesaplar.
    """

    def generate(
        self,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str] | None = None,
        candidate_values: dict[str, Any] | None = None,
    ) -> GeneratedTestInput:
        """
        Yürütme yolundan test girdisi ve beklenen sonuç üretir.

        Args:
            path:
                CFG metadata bilgilerini içeren yürütme yolu.

            parameter_names:
                Test edilen fonksiyonun parametre adları.

            parameter_types:
                Parametre type hint eşlemesi.

            candidate_values:
                Feasibility / InputCandidateGenerator katmanından gelen
                somut aday değerler. Yalnızca gerçek fonksiyon
                parametreleri başlangıç direct-value girdisi olarak
                uygulanır; yerel değişken adayları burada yok sayılır.

                Döngü ve exception gibi path'e özgü daha güçlü yapısal
                girdiler gerekirse bu başlangıç değerlerini güvenli biçimde
                geçersiz kılabilir.

        Returns:
            Somut test girdilerini ve beklenen sonucu içeren nesne.

        Raises:
            TypeError:
                Girdi türlerinden biri geçersizse.

            ValueError:
                Yol metadata içermiyorsa veya koşullar çözülemiyorsa.
        """
        self._validate_input(
            path=path,
            parameter_names=parameter_names,
            candidate_values=candidate_values,
        )

        normalized_parameter_types = (
            self._normalize_parameter_types(
                parameter_types=parameter_types,
                parameter_names=parameter_names,
            )
        )

        self._reject_unsupported_custom_object_method_path(path)

        self._validate_fixed_empty_collection_path(path)

        direct_values = self._initialize_candidate_values(
            candidate_values=candidate_values,
            parameter_names=parameter_names,
        )
        candidate_seed_values = dict(direct_values)

        handled_loop_node_ids = self._apply_loop_inputs(
            path=path,
            parameter_names=parameter_names,
            direct_values=direct_values,
        )

        self._apply_caught_exception_inputs(
            path=path,
            parameter_names=parameter_names,
            direct_values=direct_values,
        )

        constraints, loop_activations = (
            self._collect_iteration_scoped_constraints(
                path=path,
                parameter_names=parameter_names,
                handled_loop_node_ids=handled_loop_node_ids,
            )
        )

        loop_iterable_names = self._collect_for_loop_iterable_names(
            path=path,
            parameter_names=parameter_names,
        )

        restored_membership_loop_candidates = (
            self._restore_membership_loop_candidate_values(
                parameter_types=normalized_parameter_types,
                constraints=constraints,
                loop_activations=loop_activations,
                loop_iterable_names=loop_iterable_names,
                candidate_seed_values=candidate_seed_values,
                direct_values=direct_values,
            )
        )

        self._resize_for_loop_iterables(
            loop_activations=loop_activations,
            direct_values=direct_values,
        )

        self._apply_local_alias_constraints(
            path=path,
            parameter_names=parameter_names,
            parameter_types=normalized_parameter_types,
            constraints=constraints,
            direct_values=direct_values,
        )

        self._apply_direct_name_alias_constraints(
            path=path,
            parameter_names=parameter_names,
            constraints=constraints,
        )

        self._apply_loop_variable_constraints(
            parameter_types=normalized_parameter_types,
            loop_activations=loop_activations,
            direct_values=direct_values,
            restored_membership_loop_candidates=(
                restored_membership_loop_candidates
            ),
        )

        self._apply_dictionary_lookup_constraints(
            path=path,
            parameter_names=parameter_names,
            parameter_types=normalized_parameter_types,
            constraints=constraints,
            direct_values=direct_values,
            loop_activations=loop_activations,
        )

        self._apply_runtime_type_overrides(
            parameter_names=parameter_names,
            parameter_types=normalized_parameter_types,
            constraints=constraints,
            direct_values=direct_values,
        )

        try:
            DerivedValueInputSynthesizer().apply(
                path=path,
                parameter_names=parameter_names,
                direct_values=direct_values,
            )
        except UnsupportedDerivedValueSynthesisError as error:
            raise UnsupportedInputSynthesisError(str(error)) from error
        except DerivedValueSynthesisError as error:
            raise UnreachablePathError(str(error)) from error

        self._apply_collection_membership_constraints(
            parameter_names=parameter_names,
            parameter_types=normalized_parameter_types,
            constraints=constraints,
            direct_values=direct_values,
            loop_activations=loop_activations,
            loop_iterable_names=loop_iterable_names,
        )

        self._validate_collection_alias_constraints(
            path=path,
            parameter_names=parameter_names,
            constraints=constraints,
            direct_values=direct_values,
        )

        self._validate_direct_values_against_constraints(
            direct_values=direct_values,
            constraints=constraints,
        )

        keyword_argument_values: list[tuple[str, Any]] = []

        for parameter_name in parameter_names:
            constraint = constraints.get(parameter_name)
            parameter_type = normalized_parameter_types.get(parameter_name)
            value = (
                direct_values[parameter_name]
                if parameter_name in direct_values
                else self._create_parameter_value(
                    parameter_name=parameter_name,
                    constraint=constraint,
                    parameter_type=parameter_type,
                )
            )
            keyword_argument_values.append(
                (
                    parameter_name,
                    self._coerce_value_to_parameter_type(
                        value=value,
                        parameter_type=(
                            None
                            if self._has_runtime_type_constraint(constraint)
                            else parameter_type
                        ),
                    ),
                )
            )

        keyword_arguments = tuple(keyword_argument_values)

        expected_exception = self._extract_expected_exception(path)

        expected_result = (
            None
            if expected_exception is not None
            else self._extract_expected_result(
                path=path,
                keyword_arguments=keyword_arguments,
            )
        )

        return GeneratedTestInput(
            keyword_arguments=keyword_arguments,
            expected_result=expected_result,
            expected_exception=expected_exception,
        )

    @staticmethod
    def _validate_input(
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        candidate_values: dict[str, Any] | None,
    ) -> None:
        if not isinstance(path, ExecutionPath):
            raise TypeError(
                "path bir ExecutionPath örneği olmalıdır."
            )

        if not path.has_node_metadata:
            raise ValueError(
                "ExecutionPath düğüm metadata bilgilerini "
                "içermelidir."
            )

        if not isinstance(parameter_names, tuple):
            raise TypeError(
                "parameter_names bir tuple olmalıdır."
            )

        if any(
            not isinstance(parameter_name, str)
            or not parameter_name.strip()
            for parameter_name in parameter_names
        ):
            raise ValueError(
                "parameter_names yalnızca boş olmayan string "
                "değerler içermelidir."
            )

        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(
                "parameter_names tekrar eden değer içeremez."
            )

        if candidate_values is not None:
            if not isinstance(candidate_values, dict):
                raise TypeError(
                    "candidate_values bir dict veya None olmalıdır."
                )

            for variable_name in candidate_values:
                if (
                    not isinstance(variable_name, str)
                    or not variable_name.strip()
                ):
                    raise ValueError(
                        "candidate_values anahtarları boş olmayan "
                        "string değerler olmalıdır."
                    )

    @staticmethod
    def _initialize_candidate_values(
        *,
        candidate_values: dict[str, Any] | None,
        parameter_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Feasibility katmanından gelen adayları direct-value başlangıç
        girdilerine dönüştürür.

        InputCandidateGenerator yerel symbolic değişkenler için de değer
        üretebilir. PathInputGenerator yalnızca gerçek fonksiyon
        parametrelerini doğrudan çağrı girdisi yapabildiği için diğer
        değişkenler burada bilinçli olarak filtrelenir.
        """
        if candidate_values is None:
            return {}

        parameter_name_set = set(parameter_names)

        return {
            variable_name: value
            for variable_name, value
            in candidate_values.items()
            if variable_name in parameter_name_set
        }

    @staticmethod
    def _normalize_parameter_types(
        *,
        parameter_types: dict[str, str] | None,
        parameter_names: tuple[str, ...],
    ) -> dict[str, str]:
        """Type hint eşlemesini doğrular ve normalize eder."""
        if parameter_types is None:
            return {}

        if not isinstance(parameter_types, dict):
            raise TypeError(
                "parameter_types bir dict veya None olmalıdır."
            )

        normalized: dict[str, str] = {}

        for parameter_name, type_name in parameter_types.items():
            if parameter_name not in parameter_names:
                raise ValueError(
                    "parameter_types bilinmeyen parametre içeriyor: "
                    f"{parameter_name}"
                )

            if (
                not isinstance(type_name, str)
                or not type_name.strip()
            ):
                raise ValueError(
                    "parameter_types değerleri boş olmayan "
                    "string değerler olmalıdır."
                )

            normalized[parameter_name] = (
                type_name.strip().replace(" ", "")
            )

        return normalized

    def _apply_caught_exception_inputs(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """
        Yakalanan exception yolunu çalıştıracak doğrudan girdileri üretir.
        """
        steps = path.steps

        for index, step in enumerate(steps):
            if (
                step.outgoing_edge_label != "Exception"
                or index + 1 >= len(steps)
            ):
                continue

            except_step = steps[index + 1]

            if except_step.node_type != "except":
                continue

            exception_names = self._extract_handler_exception_names(
                except_step
            )

            if exception_names is None:
                continue

            if len(exception_names) == 1:
                self._apply_exception_source_input(
                    source_step=step,
                    exception_name=exception_names[0],
                    parameter_names=parameter_names,
                    direct_values=direct_values,
                )
                continue

            for exception_name in exception_names:
                if exception_name not in {
                    "ZeroDivisionError",
                    "IndexError",
                    "KeyError",
                }:
                    continue
                candidate_values = dict(direct_values)
                try:
                    self._apply_exception_source_input(
                        source_step=step,
                        exception_name=exception_name,
                        parameter_names=parameter_names,
                        direct_values=candidate_values,
                    )
                except (ValueError, UnreachablePathError):
                    continue
                direct_values.clear()
                direct_values.update(candidate_values)
                break
            else:
                names = ", ".join(exception_names)
                raise UnsupportedInputSynthesisError(
                    "Tuple except handler için güvenli input "
                    f"sentezlenemedi: {names}"
                )

    @staticmethod
    def _reject_unsupported_custom_object_method_path(
        path: ExecutionPath,
    ) -> None:
        for step in path.steps:
            try:
                statement = ast.parse(step.node_label)
            except SyntaxError:
                continue
            if any(
                isinstance(node, ast.Name)
                and node.id == UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER
                for node in ast.walk(statement)
            ):
                raise UnsupportedInputSynthesisError(
                    "Selected path requires unsupported custom object "
                    "method execution."
                )

    @staticmethod
    def _extract_handler_exception_names(
        except_step: PathStep,
    ) -> tuple[str, ...] | None:
        """Güvenli exception adlarını kaynak sırasıyla çıkarır."""
        normalized_label = except_step.node_label.strip()

        if normalized_label == "except":
            return None

        prefix = "except "

        if not normalized_label.startswith(prefix):
            raise UnsupportedInputSynthesisError(
                "Except düğümü geçerli bir exception etiketi "
                "içermiyor: "
                f"{except_step.node_label}"
            )

        expression_text = normalized_label[len(prefix):]

        try:
            expression = ast.parse(
                expression_text,
                mode="eval",
            ).body
        except SyntaxError as error:
            raise UnsupportedInputSynthesisError(
                "Except exception türü çözümlenemedi: "
                f"{except_step.node_label}"
            ) from error

        def extract_names(node: ast.expr) -> tuple[str, ...]:
            if isinstance(node, ast.Name):
                return (node.id,)
            if isinstance(node, ast.Attribute) and (
                PathInputGenerator._is_safe_exception_attribute(node)
            ):
                return (node.attr,)
            if isinstance(node, ast.Tuple):
                names: list[str] = []
                for element in node.elts:
                    names.extend(extract_names(element))
                if names:
                    return tuple(names)
            raise UnsupportedInputSynthesisError(
                "Desteklenmeyen except exception türü: "
                f"{except_step.node_label}"
            )

        return extract_names(expression)

    @staticmethod
    def _is_safe_exception_attribute(expression: ast.Attribute) -> bool:
        """Yalnız isim köklü attribute zincirlerini güvenli kabul eder."""
        value: ast.expr = expression.value
        while isinstance(value, ast.Attribute):
            value = value.value
        return isinstance(value, ast.Name)

    def _apply_exception_source_input(
        self,
        *,
        source_step: PathStep,
        exception_name: str | None,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """Exception türüne göre kaynak ifadeden test girdisi üretir."""
        if exception_name is None:
            return

        statement = self._parse_exception_source_statement(
            source_step
        )
        expression = self._extract_statement_expression(
            statement
        )

        if exception_name == "ZeroDivisionError":
            self._apply_zero_division_input(
                expression=expression,
                parameter_names=parameter_names,
                direct_values=direct_values,
            )
            return

        if exception_name == "IndexError":
            self._apply_index_error_input(
                expression=expression,
                parameter_names=parameter_names,
                direct_values=direct_values,
            )
            return

        if exception_name == "KeyError":
            self._apply_key_error_input(
                expression=expression,
                parameter_names=parameter_names,
                direct_values=direct_values,
            )
            return

    @staticmethod
    def _parse_exception_source_statement(
        source_step: PathStep,
    ) -> ast.stmt:
        """Exception kaynak düğümündeki Python ifadesini parse eder."""
        try:
            module = ast.parse(
                source_step.node_label
            )
        except SyntaxError as error:
            raise ValueError(
                "Exception kaynak ifadesi çözümlenemedi: "
                f"{source_step.node_label}"
            ) from error

        if len(module.body) != 1:
            raise ValueError(
                "Exception kaynak düğümü tek bir ifade "
                "içermelidir."
            )

        return module.body[0]

    @staticmethod
    def _extract_statement_expression(
        statement: ast.stmt,
    ) -> ast.expr:
        """Statement içindeki çalıştırılan temel ifadeyi döndürür."""
        if isinstance(statement, ast.Assign):
            return statement.value

        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                raise ValueError(
                    "Değersiz AnnAssign exception kaynağı "
                    "olarak kullanılamaz."
                )
            return statement.value

        if isinstance(statement, ast.Expr):
            return statement.value

        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise ValueError(
                    "Boş return exception kaynağı olarak "
                    "kullanılamaz."
                )
            return statement.value

        if isinstance(statement, ast.Raise):
            if statement.exc is None:
                raise ValueError(
                    "Boş raise exception kaynağı olarak "
                    "kullanılamaz."
                )
            return statement.exc

        raise ValueError(
            "Desteklenmeyen exception kaynak düğümü: "
            f"{type(statement).__name__}"
        )

    @staticmethod
    def _apply_zero_division_input(
        *,
        expression: ast.expr,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """Bölme ifadesindeki parametre böleni sıfır yapar."""
        division_node = next(
            (
                node
                for node in ast.walk(expression)
                if isinstance(node, ast.BinOp)
                and isinstance(
                    node.op,
                    (ast.Div, ast.FloorDiv, ast.Mod),
                )
            ),
            None,
        )

        if division_node is None:
            raise ValueError(
                "ZeroDivisionError için bölme ifadesi bulunamadı."
            )

        denominator = division_node.right

        if not isinstance(denominator, ast.Name):
            raise UnreachablePathError(
                "ZeroDivisionError yolu için bölen doğrudan "
                "kontrol edilebilir bir değişken değildir."
            )

        if denominator.id not in parameter_names:
            raise UnreachablePathError(
                "ZeroDivisionError yolu yerel bir böleni sıfır "
                "yapmayı gerektiriyor ve dış girdilerle doğrudan "
                "kontrol edilemiyor: "
                f"{denominator.id}"
            )

        direct_values[denominator.id] = 0

    @staticmethod
    def _apply_index_error_input(
        *,
        expression: ast.expr,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """Sabit indeksli erişim için yetersiz uzunlukta liste üretir."""
        subscript_node = next(
            (
                node
                for node in ast.walk(expression)
                if isinstance(node, ast.Subscript)
            ),
            None,
        )

        if subscript_node is None:
            raise ValueError(
                "IndexError için indeks erişimi bulunamadı."
            )

        if (
            not isinstance(subscript_node.value, ast.Name)
            or subscript_node.value.id not in parameter_names
        ):
            raise ValueError(
                "IndexError koleksiyonu doğrudan bir "
                "fonksiyon parametresi olmalıdır."
            )

        try:
            index_value = ast.literal_eval(
                subscript_node.slice
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                "IndexError için yalnızca sabit indeksler "
                "desteklenmektedir."
            ) from error

        if not isinstance(index_value, int) or isinstance(
            index_value,
            bool,
        ):
            raise ValueError(
                "IndexError indeksi bir tam sayı olmalıdır."
            )

        direct_values[subscript_node.value.id] = (
            [0 for _ in range(index_value)]
            if index_value >= 0
            else []
        )

    @staticmethod
    def _apply_key_error_input(
        *,
        expression: ast.expr,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """Sözlük erişimi için istenen anahtarı içermeyen sözlük üretir."""
        subscript_node = next(
            (
                node
                for node in ast.walk(expression)
                if isinstance(node, ast.Subscript)
            ),
            None,
        )

        if subscript_node is None:
            raise ValueError(
                "KeyError için anahtar erişimi bulunamadı."
            )

        if (
            not isinstance(subscript_node.value, ast.Name)
            or subscript_node.value.id not in parameter_names
        ):
            # Loop-element dictionary gibi güvenli nested provenance,
            # sıralı structured-input katmanında çözümlenir.
            return

        try:
            ast.literal_eval(subscript_node.slice)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "KeyError için yalnızca sabit anahtarlar "
                "desteklenmektedir."
            ) from error

        direct_values[subscript_node.value.id] = {}

    def _apply_loop_inputs(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> set[int]:
        """
        Döngü yolları için doğrudan başlangıç girdileri üretir.

        Returns:
            Genel koşul çözümlemesinde tekrar işlenmemesi gereken
            ``while`` düğüm kimlikleri.
        """
        handled_while_node_ids: set[int] = set()

        unique_loop_steps: dict[int, PathStep] = {}

        for loop_step in path.loop_steps:
            unique_loop_steps.setdefault(
                loop_step.node_id,
                loop_step,
            )

        for loop_step in unique_loop_steps.values():
            iteration_count = self._count_loop_iterations(
                path=path,
                loop_step=loop_step,
            )

            if loop_step.node_type == "for":
                self._apply_for_loop_input(
                    step=loop_step,
                    iteration_count=iteration_count,
                    parameter_names=parameter_names,
                    direct_values=direct_values,
                )
                continue

            if loop_step.node_type == "while":
                self._apply_while_loop_input(
                    path=path,
                    step=loop_step,
                    iteration_count=iteration_count,
                    parameter_names=parameter_names,
                    direct_values=direct_values,
                )
                handled_while_node_ids.add(
                    loop_step.node_id
                )

        return handled_while_node_ids

    @staticmethod
    def _count_loop_iterations(
        *,
        path: ExecutionPath,
        loop_step: PathStep,
    ) -> int:
        """
        Belirli bir döngü düğümünün yol üzerindeki iterasyon sayısını
        hesaplar.

        Gerçek CFG yollarında aynı döngünün tekrarları aynı ``node_id``
        değerini taşır. Bazı birim test yardımcıları ise her ziyaret için
        yeni kimlik ürettiğinden, kimlik eşleşmesi bulunamazsa düğüm türü
        ve etiketi üzerinden geriye uyumlu bir eşleşme uygulanır.
        """
        body_edge_labels = (
            {"Iterate"}
            if loop_step.node_type == "for"
            else {"True"}
        )

        iteration_count = sum(
            path_step.node_id == loop_step.node_id
            and path_step.outgoing_edge_label
            in body_edge_labels
            for path_step in path.steps
        )

        if iteration_count > 0:
            return iteration_count

        return sum(
            path_step.node_type == loop_step.node_type
            and path_step.node_label == loop_step.node_label
            and path_step.outgoing_edge_label
            in body_edge_labels
            for path_step in path.steps
        )

    @staticmethod
    def _apply_for_loop_input(
        *,
        step: PathStep,
        iteration_count: int,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """
        ``for item in values`` biçimindeki döngüler için iterable üretir.
        """
        try:
            _, iterable_text = step.node_label.split(
                " in ",
                maxsplit=1,
            )
            iterable_expression = ast.parse(
                iterable_text,
                mode="eval",
            ).body
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                "For döngüsü ifadesi çözümlenemedi: "
                f"{step.node_label}"
            ) from error

        if PathInputGenerator._is_fixed_empty_collection_expression(
            iterable_expression,
            {},
        ):
            if iteration_count != 0:
                raise UnreachablePathError(
                    "Execution path boş literal iterable üzerinde "
                    f"iterasyon gerektiriyor: {step.node_label}"
                )
            return

        if not isinstance(iterable_expression, ast.Name):
            raise ValueError(
                "For döngüsü iterable değeri doğrudan bir "
                "fonksiyon parametresi olmalıdır: "
                f"{step.node_label}"
            )

        iterable_name = iterable_expression.id

        if iterable_name not in parameter_names:
            return

        direct_values[iterable_name] = [
            0
            for _ in range(iteration_count)
        ]

    def _apply_while_loop_input(
        self,
        *,
        path: ExecutionPath,
        step: PathStep,
        iteration_count: int,
        parameter_names: tuple[str, ...],
        direct_values: dict[str, Any],
    ) -> None:
        """
        Basit sayısal while döngüsü için başlangıç durumunu çözümler.

        Döngü değişkeni bir fonksiyon parametresi ise uygun test girdisi
        üretilir. Yerel değişkense, döngüden önceki sabit Assign veya
        AnnAssign ifadesi izlenerek yolun uygulanabilirliği doğrulanır.

        Desteklenen güncellemeler:
            ``x += sabit`` ve ``x -= sabit``.
        """
        try:
            condition_expression = ast.parse(
                step.node_label,
                mode="eval",
            ).body
        except SyntaxError as error:
            raise ValueError(
                "While koşulu çözümlenemedi: "
                f"{step.node_label}"
            ) from error

        variable_names = {
            node.id
            for node in ast.walk(condition_expression)
            if isinstance(node, ast.Name)
        }

        if len(variable_names) != 1:
            raise ValueError(
                "While koşulu tam olarak bir sayısal değişken "
                "üzerinden tanımlanmalıdır: "
                f"{step.node_label}"
            )

        variable_name = next(iter(variable_names))

        if variable_name in parameter_names:
            update_delta = self._extract_loop_update_delta(
                path=path,
                variable_name=variable_name,
            )

            if iteration_count > 0 and update_delta is None:
                raise UnsupportedInputSynthesisError(
                    "While döngüsü için desteklenen bir değişken "
                    "güncellemesi bulunamadı."
                )

            candidate = self._find_while_initial_value(
                expression=condition_expression,
                variable_name=variable_name,
                iteration_count=iteration_count,
                update_delta=update_delta or 0,
            )

            direct_values[variable_name] = candidate
            return

        local_initial_expression = (
            self._extract_local_loop_initial_expression(
                path=path,
                loop_step=step,
                variable_name=variable_name,
            )
        )

        if local_initial_expression is None:
            raise UnsupportedInputSynthesisError(
                "While koşulundaki yerel değişken için döngüden "
                "önce bir başlangıç ataması bulunamadı: "
                f"{variable_name}"
            )

        try:
            local_initial_value = ast.literal_eval(
                local_initial_expression
            )
        except (ValueError, TypeError):
            # Affine ve desteklenmeyen derived ifadeler tek symbolic
            # kaynağın sahibi olan DerivedValueInputSynthesizer'da ayrılır.
            return

        if (
            isinstance(local_initial_value, bool)
            or not isinstance(local_initial_value, (int, float))
        ):
            raise UnsupportedInputSynthesisError(
                "While başlangıç ataması sayısal veya güvenli affine "
                f"bir ifade olmalıdır: {variable_name}"
            )

        update_delta = self._extract_loop_update_delta(
            path=path,
            variable_name=variable_name,
        )

        if iteration_count > 0 and update_delta is None:
            raise UnsupportedInputSynthesisError(
                "While döngüsü için desteklenen bir değişken "
                "güncellemesi bulunamadı."
            )

        if not self._matches_while_iteration_count(
            expression=condition_expression,
            variable_name=variable_name,
            initial_value=local_initial_value,
            iteration_count=iteration_count,
            update_delta=update_delta or 0,
        ):
            raise UnreachablePathError(
                "Yerel while değişkeninin başlangıç değeri, CFG "
                "yolundaki iterasyon sayısıyla uyuşmuyor: "
                f"{variable_name}={local_initial_value!r}"
            )

    @classmethod
    def _validate_fixed_empty_collection_path(
        cls,
        path: ExecutionPath,
    ) -> None:
        """Doğrudan boş literal state ile çelişen path kenarlarını reddeder."""
        fixed_collections: dict[str, list[Any] | dict[Any, Any]] = {}

        for step in path.steps:
            if step.node_type in {"Assign", "AnnAssign"}:
                assignment = cls._parse_single_assignment(step.node_label)
                if assignment is None:
                    continue
                target_name, value = assignment
                fixed_value = cls._empty_collection_literal(value)
                if fixed_value is None:
                    fixed_collections.pop(target_name, None)
                else:
                    fixed_collections[target_name] = fixed_value
                continue

            if (
                step.node_type in {"if", "while"}
                and step.outgoing_edge_label in {"True", "False"}
            ):
                try:
                    condition = ast.parse(step.node_label, mode="eval").body
                except SyntaxError:
                    continue
                actual = cls._fixed_collection_condition(
                    condition,
                    fixed_collections,
                )
                if actual is not None and actual is not (
                    step.outgoing_edge_label == "True"
                ):
                    raise UnreachablePathError(
                        "Execution path, doğrulanmış boş koleksiyon "
                        f"state'iyle çelişiyor: {step.node_label}"
                    )
                continue

            if step.node_type == "for" and step.outgoing_edge_label in {
                "Iterate",
                "Complete",
            }:
                iterable_is_empty = cls._fixed_empty_loop_iterable(
                    step.node_label,
                    fixed_collections,
                )
                if iterable_is_empty and step.outgoing_edge_label == "Iterate":
                    raise UnreachablePathError(
                        "Execution path doğrulanmış boş koleksiyon üzerinde "
                        f"iterasyon gerektiriyor: {step.node_label}"
                    )

    @staticmethod
    def _empty_collection_literal(
        expression: ast.expr,
    ) -> list[Any] | dict[Any, Any] | None:
        if isinstance(expression, ast.List) and not expression.elts:
            return []
        if (
            isinstance(expression, ast.Dict)
            and not expression.keys
            and not expression.values
        ):
            return {}
        return None

    @classmethod
    def _fixed_collection_condition(
        cls,
        expression: ast.expr,
        fixed_collections: dict[str, list[Any] | dict[Any, Any]],
    ) -> bool | None:
        if isinstance(expression, ast.Name) and expression.id in fixed_collections:
            return bool(fixed_collections[expression.id])
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            operand = cls._fixed_collection_condition(
                expression.operand,
                fixed_collections,
            )
            return None if operand is None else not operand
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "bool"
            and len(expression.args) == 1
            and not expression.keywords
        ):
            return cls._fixed_collection_condition(
                expression.args[0],
                fixed_collections,
            )
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], (ast.In, ast.NotIn))
            and len(expression.comparators) == 1
            and cls._is_fixed_empty_collection_expression(
                expression.comparators[0],
                fixed_collections,
            )
        ):
            return isinstance(expression.ops[0], ast.NotIn)
        return None

    @classmethod
    def _fixed_empty_loop_iterable(
        cls,
        label: str,
        fixed_collections: dict[str, list[Any] | dict[Any, Any]],
    ) -> bool:
        try:
            expression = ast.parse(label, mode="eval").body
        except SyntaxError:
            return False
        return bool(
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], ast.In)
            and len(expression.comparators) == 1
            and cls._is_fixed_empty_collection_expression(
                expression.comparators[0],
                fixed_collections,
            )
        )

    @staticmethod
    def _is_fixed_empty_collection_expression(
        expression: ast.expr,
        fixed_collections: dict[str, list[Any] | dict[Any, Any]],
    ) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id in fixed_collections
        return (
            isinstance(expression, (ast.List, ast.Tuple))
            and not expression.elts
            or isinstance(expression, ast.Dict)
            and not expression.keys
            and not expression.values
        )

    @staticmethod
    def _extract_local_loop_initial_expression(
        *,
        path: ExecutionPath,
        loop_step: PathStep,
        variable_name: str,
    ) -> ast.expr | None:
        """
        Döngüden önceki son yerel değişken atamasının ifadesini döndürür.
        """
        initial_expression: ast.expr | None = None

        for path_step in path.steps:
            if path_step.node_id == loop_step.node_id:
                break

            if path_step.node_type not in {
                "Assign",
                "AnnAssign",
            }:
                continue

            try:
                statement = ast.parse(
                    path_step.node_label
                ).body[0]
            except SyntaxError:
                continue

            target: ast.expr
            value_expression: ast.expr | None

            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    continue

                target = statement.targets[0]
                value_expression = statement.value
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
                value_expression = statement.value
            else:
                continue

            if (
                not isinstance(target, ast.Name)
                or target.id != variable_name
                or value_expression is None
            ):
                continue

            initial_expression = value_expression

        return initial_expression

    @staticmethod
    def _matches_while_iteration_count(
        *,
        expression: ast.expr,
        variable_name: str,
        initial_value: int | float,
        iteration_count: int,
        update_delta: int | float,
    ) -> bool:
        """
        Yerel başlangıç değerinin koşulu beklenen sayıda True ve
        ardından False yapıp yapmadığını doğrular.
        """
        compiled_expression = compile(
            ast.Expression(body=expression),
            filename="<while-condition>",
            mode="eval",
        )

        current_value: int | float = initial_value

        for _ in range(iteration_count):
            condition_result = bool(
                eval(
                    compiled_expression,
                    {"__builtins__": {}},
                    {variable_name: current_value},
                )
            )

            if not condition_result:
                return False

            current_value += update_delta

        return not bool(
            eval(
                compiled_expression,
                {"__builtins__": {}},
                {variable_name: current_value},
            )
        )

    @staticmethod
    def _extract_loop_update_delta(
        *,
        path: ExecutionPath,
        variable_name: str,
    ) -> int | float | None:
        """
        Yol üzerindeki ilk desteklenen ``AugAssign`` güncellemesini çıkarır.
        """
        for path_step in path.steps:
            if path_step.node_type != "AugAssign":
                continue

            try:
                statement = ast.parse(
                    path_step.node_label
                ).body[0]
            except SyntaxError:
                continue

            if not isinstance(statement, ast.AugAssign):
                continue

            if (
                not isinstance(statement.target, ast.Name)
                or statement.target.id != variable_name
            ):
                continue

            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                continue

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                continue

            if isinstance(statement.op, ast.Add):
                return value

            if isinstance(statement.op, ast.Sub):
                return -value

        return None

    @staticmethod
    def _find_while_initial_value(
        *,
        expression: ast.expr,
        variable_name: str,
        iteration_count: int,
        update_delta: int | float,
    ) -> int | float:
        """
        Koşulu belirtilen sayıda True, ardından False yapan değeri arar.
        """
        literal_values = [
            node.value
            for node in ast.walk(expression)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]

        center = (
            literal_values[0]
            if literal_values
            else 0
        )

        radius = max(
            20,
            int(abs(update_delta) * (iteration_count + 2)) + 5,
        )

        integer_candidates = range(
            int(center) - radius,
            int(center) + radius + 1,
        )

        ordered_candidates = sorted(
            integer_candidates,
            key=lambda value: (
                abs(value - center),
                value,
            ),
        )

        compiled_expression = compile(
            ast.Expression(body=expression),
            filename="<while-condition>",
            mode="eval",
        )

        for candidate in ordered_candidates:
            current_value: int | float = candidate
            valid = True

            for _ in range(iteration_count):
                if not bool(
                    eval(
                        compiled_expression,
                        {"__builtins__": {}},
                        {variable_name: current_value},
                    )
                ):
                    valid = False
                    break

                current_value += update_delta

            if not valid:
                continue

            if bool(
                eval(
                    compiled_expression,
                    {"__builtins__": {}},
                    {variable_name: current_value},
                )
            ):
                continue

            return candidate

        raise UnreachablePathError(
            "While döngüsü için istenen iterasyon sayısını "
            "sağlayan başlangıç değeri üretilemedi."
        )

    def _apply_condition_step(
        self,
        step: PathStep,
        constraints: dict[str, _VariableConstraint],
        shadowed_safe_calls: frozenset[str] = frozenset(),
    ) -> None:
        """
        Bir CFG koşul adımını değişken kısıtına dönüştürür.
        """
        if step.outgoing_edge_label not in {"True", "False"}:
            return

        try:
            expression = ast.parse(
                step.node_label,
                mode="eval",
            ).body
        except SyntaxError as error:
            raise ValueError(
                f"Koşul ifadesi çözümlenemedi: "
                f"{step.node_label}"
            ) from error

        self._apply_expression(
            expression=expression,
            desired_result=(
                step.outgoing_edge_label == "True"
            ),
            constraints=constraints,
            original_expression=step.node_label,
            shadowed_safe_calls=shadowed_safe_calls,
        )

    def _apply_expression(
        self,
        *,
        expression: ast.expr,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
        shadowed_safe_calls: frozenset[str],
    ) -> None:
        """
        Bir koşul AST ifadesini istenen Boolean sonuca göre uygular.
        """
        if self._is_fixed_empty_collection_expression(expression, {}):
            if desired_result:
                raise UnreachablePathError(
                    "Execution path doğrulanmış boş koleksiyon literalini "
                    f"truthy gerektiriyor: {original_expression}"
                )
            return

        if isinstance(expression, ast.Name):
            self._apply_boolean_constraint(
                variable_name=expression.id,
                desired_value=desired_result,
                constraints=constraints,
            )
            return

        if (
            isinstance(expression, ast.UnaryOp)
            and isinstance(expression.op, ast.Not)
        ):
            self._apply_expression(
                expression=expression.operand,
                desired_result=not desired_result,
                constraints=constraints,
                original_expression=original_expression,
                shadowed_safe_calls=shadowed_safe_calls,
            )
            return

        if isinstance(expression, ast.BoolOp):
            self._apply_boolean_operation(
                expression=expression,
                desired_result=desired_result,
                constraints=constraints,
                original_expression=original_expression,
                shadowed_safe_calls=shadowed_safe_calls,
            )
            return

        if isinstance(expression, ast.Call):
            self._apply_safe_predicate_constraint(
                expression=expression,
                desired_result=desired_result,
                constraints=constraints,
                original_expression=original_expression,
                shadowed_safe_calls=shadowed_safe_calls,
            )
            return

        if isinstance(expression, ast.Compare):
            if (
                len(expression.ops) != 1
                or len(expression.comparators) != 1
            ):
                raise ValueError(
                    "Zincirleme karşılaştırmalar henüz "
                    "desteklenmiyor: "
                    f"{original_expression}"
                )

            self._apply_comparison(
                left=expression.left,
                operator=expression.ops[0],
                right=expression.comparators[0],
                desired_result=desired_result,
                constraints=constraints,
                original_expression=original_expression,
            )
            return

        raise UnsupportedInputSynthesisError(
            "Desteklenmeyen koşul ifadesi: "
            f"{original_expression}"
        )

    def _apply_boolean_operation(
        self,
        *,
        expression: ast.BoolOp,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
        shadowed_safe_calls: frozenset[str],
    ) -> None:
        """
        ``and`` ve ``or`` ifadelerini kısıtlara dönüştürür.

        Bütün alt ifadelerin gerekli olduğu durumda hepsi uygulanır.
        Tek bir alt ifadenin yeterli olduğu durumda mevcut kısıtlarla
        çelişmeyen ilk deterministik alternatif seçilir.
        """
        all_operands_required = (
            isinstance(expression.op, ast.And)
            and desired_result
        ) or (
            isinstance(expression.op, ast.Or)
            and not desired_result
        )

        if all_operands_required:
            for child_expression in expression.values:
                self._apply_expression(
                    expression=child_expression,
                    desired_result=desired_result,
                    constraints=constraints,
                    original_expression=original_expression,
                    shadowed_safe_calls=shadowed_safe_calls,
                )
            return

        last_error: UnreachablePathError | None = None

        for child_expression in expression.values:
            candidate_constraints = dict(constraints)

            try:
                self._apply_expression(
                    expression=child_expression,
                    desired_result=desired_result,
                    constraints=candidate_constraints,
                    original_expression=original_expression,
                    shadowed_safe_calls=shadowed_safe_calls,
                )
            except UnreachablePathError as error:
                last_error = error
                continue

            constraints.clear()
            constraints.update(candidate_constraints)
            return

        raise UnreachablePathError(
            "Bileşik koşul için geçerli bir alternatif "
            f"bulunamadı: {original_expression}"
        ) from last_error

    def _apply_safe_predicate_constraint(
        self,
        *,
        expression: ast.Call,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
        shadowed_safe_calls: frozenset[str],
    ) -> None:
        """Allowlist içindeki ``isinstance`` çağrısını type constraint'e çevirir."""
        if (
            not isinstance(expression.func, ast.Name)
            or expression.func.id != "isinstance"
        ):
            raise UnsupportedInputSynthesisError(
                "Yalnızca doğrudan güvenli isinstance predicate çağrısı "
                f"desteklenir: {original_expression}"
            )

        if "isinstance" in shadowed_safe_calls:
            raise UnsupportedInputSynthesisError(
                "isinstance adı parametre, yerel binding veya import "
                f"tarafından gölgeleniyor: {original_expression}"
            )

        if expression.keywords or len(expression.args) != 2:
            raise UnsupportedInputSynthesisError(
                "isinstance tam olarak iki positional argüman ve sıfır "
                f"keyword içermelidir: {original_expression}"
            )

        if any(isinstance(argument, ast.Starred) for argument in expression.args):
            raise UnsupportedInputSynthesisError(
                f"isinstance starred argüman desteklemez: {original_expression}"
            )

        subject, type_expression = expression.args
        if not isinstance(subject, ast.Name):
            raise UnsupportedInputSynthesisError(
                "isinstance subject'i isim tabanlı parametre, loop hedefi "
                f"veya alias olmalıdır: {original_expression}"
            )

        type_names = self._parse_runtime_type_tokens(
            expression=type_expression,
            original_expression=original_expression,
        )
        current = constraints.get(subject.id, _VariableConstraint())

        if desired_result:
            updated = replace(
                current,
                required_runtime_type_groups=(
                    *current.required_runtime_type_groups,
                    type_names,
                ),
            )
        else:
            updated = replace(
                current,
                forbidden_runtime_types=tuple(
                    dict.fromkeys(
                        (*current.forbidden_runtime_types, *type_names)
                    )
                ),
            )

        if not self._runtime_type_constraint_has_candidate(updated):
            raise UnreachablePathError(
                "Runtime type kısıtları birbiriyle çelişiyor: "
                f"{original_expression}"
            )

        constraints[subject.id] = updated

    @staticmethod
    def _parse_runtime_type_tokens(
        *,
        expression: ast.expr,
        original_expression: str,
    ) -> tuple[str, ...]:
        candidates = (
            expression.elts
            if isinstance(expression, ast.Tuple)
            else (expression,)
        )

        if not candidates:
            raise UnsupportedInputSynthesisError(
                f"isinstance type tuple boş olamaz: {original_expression}"
            )

        type_names: list[str] = []
        for candidate in candidates:
            if (
                not isinstance(candidate, ast.Name)
                or candidate.id not in _RUNTIME_TYPE_ALLOWLIST
            ):
                raise UnsupportedInputSynthesisError(
                    "isinstance type argümanı yalnız güvenli built-in "
                    f"allowlist token'larını içerebilir: {original_expression}"
                )
            type_names.append(candidate.id)

        return tuple(dict.fromkeys(type_names))

    def _apply_comparison(
        self,
        left: ast.expr,
        operator: ast.cmpop,
        right: ast.expr,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
    ) -> None:
        """
        Basit karşılaştırma ifadesini değişken kısıtına dönüştürür.

        Her iki tarafı da isim olan ilişkiler literal kısıt değildir.
        Bu ilişkiler PathFeasibilityAnalyzer ve relational witness
        katmanında çözülür. Burada tekrar literal olarak yorumlanmaları,
        geçerli path'lerin aday üretimi sırasında hata vermesine neden olur.
        """
        if isinstance(operator, (ast.Is, ast.IsNot)):
            self._apply_none_identity_comparison(
                left=left,
                operator=operator,
                right=right,
                desired_result=desired_result,
                constraints=constraints,
                original_expression=original_expression,
            )
            return

        if (
            isinstance(operator, (ast.In, ast.NotIn))
            and not isinstance(right, ast.Name)
        ):
            unsupported_membership_detail = (
                "Koleksiyon üyelik kısıtının sağ operandı doğrudan "
                "güvenli bir koleksiyon referansı veya literal olmalıdır."
            )
            if not isinstance(left, ast.Name):
                raise UnsupportedInputSynthesisError(
                    unsupported_membership_detail
                )
            try:
                self._extract_literal(right)
            except ValueError as error:
                raise UnsupportedInputSynthesisError(
                    unsupported_membership_detail
                ) from error

        if (
            isinstance(left, ast.Name)
            and isinstance(right, ast.Name)
        ):
            return

        if isinstance(left, ast.Name):
            variable_name = left.id
            try:
                value = self._extract_literal(right)
            except ValueError as error:
                raise UnsupportedInputSynthesisError(
                    "Karşılaştırma operand provenance'ı güvenli literal "
                    "constraint'e indirgenemedi: "
                    f"expression={original_expression!r}, "
                    f"operator={type(operator).__name__}, "
                    f"left={type(left).__name__}, "
                    f"right={type(right).__name__}"
                ) from error
            normalized_operator = operator

        elif isinstance(right, ast.Name):
            if isinstance(operator, (ast.In, ast.NotIn)):
                try:
                    member = self._extract_literal(left)
                except ValueError as error:
                    raise UnsupportedInputSynthesisError(
                        "Koleksiyon üyelik kısıtının sol operandı "
                        "güvenli bir literal olmalıdır."
                    ) from error

                effective_operator = (
                    operator
                    if desired_result
                    else self._negate_operator(operator)
                )
                variable_name = right.id
                constraints[variable_name] = (
                    self._merge_collection_membership_constraint(
                        current=constraints.get(
                            variable_name,
                            _VariableConstraint(),
                        ),
                        operator=effective_operator,
                        member=member,
                    )
                )
                return

            variable_name = right.id
            try:
                value = self._extract_literal(left)
            except ValueError as error:
                raise UnsupportedInputSynthesisError(
                    "Karşılaştırma operand provenance'ı güvenli literal "
                    "constraint'e indirgenemedi: "
                    f"expression={original_expression!r}, "
                    f"operator={type(operator).__name__}, "
                    f"left={type(left).__name__}, "
                    f"right={type(right).__name__}"
                ) from error
            normalized_operator = self._reverse_operator(operator)

        elif isinstance(left, ast.BinOp) or isinstance(right, ast.BinOp):
            # Güvenli affine ifadeler aynı path üzerinde daha sonra tek
            # symbolic sahibi olan DerivedValueInputSynthesizer tarafından
            # dış parametrelere geri yayılır. Burada literal constraint gibi
            # ele alınmaları geçerli çok-parametreli affine yolları erkenden
            # reddeder.
            return

        else:
            raise ValueError(
                "Koşul içinde doğrudan bir parametre "
                "karşılaştırması bulunamadı: "
                f"{original_expression}"
            )

        effective_operator = (
            normalized_operator
            if desired_result
            else self._negate_operator(normalized_operator)
        )

        current_constraint = constraints.get(
            variable_name,
            _VariableConstraint(),
        )

        constraints[variable_name] = self._merge_constraint(
            current=current_constraint,
            operator=effective_operator,
            value=value,
        )

    def _merge_collection_membership_constraint(
        self,
        *,
        current: _VariableConstraint,
        operator: ast.cmpop,
        member: Any,
    ) -> _VariableConstraint:
        """Literal-sol koleksiyon üyeliğini güvenli kısıta dönüştürür."""
        if isinstance(operator, ast.In):
            if self._contains_equal_value(
                current.forbidden_collection_members,
                member,
            ):
                raise UnreachablePathError(
                    "Aynı koleksiyon üyelik kısıtları çelişiyor."
                )
            result = replace(
                current,
                required_collection_members=self._append_unique_value(
                    current.required_collection_members,
                    member,
                ),
            )
        elif isinstance(operator, ast.NotIn):
            if self._contains_equal_value(
                current.required_collection_members,
                member,
            ):
                raise UnreachablePathError(
                    "Aynı koleksiyon üyelik kısıtları çelişiyor."
                )
            result = replace(
                current,
                forbidden_collection_members=self._append_unique_value(
                    current.forbidden_collection_members,
                    member,
                ),
            )
        else:
            raise UnsupportedInputSynthesisError(
                "Koleksiyon üyelik kısıtı yalnız in/not in ile desteklenir."
            )

        self._validate_constraint_consistency(result)
        return result

    @staticmethod
    def _contains_equal_value(values: tuple[Any, ...], candidate: Any) -> bool:
        """Literal değerlerde hash gerektirmeden deterministik eşitlik arar."""
        return any(value == candidate for value in values)

    @classmethod
    def _append_unique_value(
        cls,
        values: tuple[Any, ...],
        candidate: Any,
    ) -> tuple[Any, ...]:
        if cls._contains_equal_value(values, candidate):
            return values
        return (*values, candidate)

    def _apply_none_identity_comparison(
        self,
        *,
        left: ast.expr,
        operator: ast.cmpop,
        right: ast.expr,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
    ) -> None:
        """Yalnız ``name is (not) None`` identity biçimlerini uygular."""
        left_is_none = isinstance(left, ast.Constant) and left.value is None
        right_is_none = isinstance(right, ast.Constant) and right.value is None

        if left_is_none and isinstance(right, ast.Name):
            variable_name = right.id
        elif right_is_none and isinstance(left, ast.Name):
            variable_name = left.id
        else:
            raise UnsupportedInputSynthesisError(
                "Identity comparison yalnızca bir değişken ile None "
                f"arasında desteklenmektedir: {original_expression}"
            )

        requires_none = isinstance(operator, ast.Is) is desired_result
        constraints[variable_name] = self._merge_constraint(
            current=constraints.get(variable_name, _VariableConstraint()),
            operator=ast.Eq() if requires_none else ast.NotEq(),
            value=None,
        )

    @staticmethod
    def _apply_boolean_constraint(
        variable_name: str,
        desired_value: bool,
        constraints: dict[str, _VariableConstraint],
    ) -> None:
        """
        Doğrudan bool parametre kontrolünü kısıta dönüştürür.
        """
        current = constraints.get(
            variable_name,
            _VariableConstraint(),
        )

        if (
            current.has_equal_value
            and current.equal_value is not desired_value
        ):
            raise UnreachablePathError(
                f"{variable_name} için çelişkili bool "
                "kısıtları bulundu."
            )

        constraints[variable_name] = replace(
            current,
            equal_value=desired_value,
            has_equal_value=True,
        )

    @staticmethod
    def _extract_literal(node: ast.expr) -> Any:
        """
        AST düğümünden güvenli sabit değer çıkarır.
        """
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Koşuldaki karşılaştırma değeri sabit "
                "bir değer olmalıdır."
            ) from error

    def _merge_constraint(
        self,
        current: _VariableConstraint,
        operator: ast.cmpop,
        value: Any,
    ) -> _VariableConstraint:
        """
        Yeni karşılaştırmayı mevcut değişken kısıtına ekler.
        """
        if isinstance(operator, ast.Eq):
            if (
                current.has_equal_value
                and current.equal_value != value
            ):
                raise UnreachablePathError(
                    "Aynı değişken için çelişkili eşitlik "
                    "kısıtları bulundu."
                )

            if value in current.forbidden_values:
                raise UnreachablePathError(
                    "Eşitlik ve eşitsizlik kısıtları çelişiyor."
                )

            result = replace(
                current,
                equal_value=value,
                has_equal_value=True,
            )

            self._validate_constraint_consistency(result)

            return result

        if isinstance(operator, ast.NotEq):
            if (
                current.has_equal_value
                and current.equal_value == value
            ):
                raise UnreachablePathError(
                    "Eşitlik ve eşitsizlik kısıtları çelişiyor."
                )

            result = replace(
                current,
                forbidden_values=tuple(
                    dict.fromkeys(
                        (
                            *current.forbidden_values,
                            value,
                        )
                    )
                ),
            )

            self._validate_constraint_consistency(result)

            return result

        if isinstance(operator, ast.In):
            allowed_values = self._normalize_membership_values(
                value
            )

            if current.allowed_values is None:
                merged_allowed_values = allowed_values
            else:
                merged_allowed_values = tuple(
                    candidate
                    for candidate in current.allowed_values
                    if candidate in allowed_values
                )

            merged_allowed_values = tuple(
                candidate
                for candidate in merged_allowed_values
                if candidate not in current.forbidden_values
            )

            if not merged_allowed_values:
                raise UnreachablePathError(
                    "Üyelik kısıtları ortak bir değer içermiyor."
                )

            result = replace(
                current,
                allowed_values=merged_allowed_values,
            )

            self._validate_constraint_consistency(result)
            return result

        if isinstance(operator, ast.NotIn):
            forbidden_members = self._normalize_membership_values(
                value
            )

            result = replace(
                current,
                forbidden_values=tuple(
                    dict.fromkeys(
                        (
                            *current.forbidden_values,
                            *forbidden_members,
                        )
                    )
                ),
            )

            self._validate_constraint_consistency(result)
            return result

        if not isinstance(value, (int, float)) or isinstance(
            value,
            bool,
        ):
            raise ValueError(
                "Sıralama karşılaştırmaları yalnızca sayısal "
                "sabitlerle desteklenmektedir."
            )

        if isinstance(operator, ast.Gt):
            return self._merge_minimum(
                current=current,
                value=value,
                inclusive=False,
            )

        if isinstance(operator, ast.GtE):
            return self._merge_minimum(
                current=current,
                value=value,
                inclusive=True,
            )

        if isinstance(operator, ast.Lt):
            return self._merge_maximum(
                current=current,
                value=value,
                inclusive=False,
            )

        if isinstance(operator, ast.LtE):
            return self._merge_maximum(
                current=current,
                value=value,
                inclusive=True,
            )

        raise ValueError(
            "Desteklenmeyen karşılaştırma operatörü."
        )

    @staticmethod
    def _normalize_membership_values(
        value: Any,
    ) -> tuple[Any, ...]:
        """
        ``in`` ve ``not in`` karşılaştırmalarındaki sabit koleksiyonu
        kararlı bir tuple biçimine dönüştürür.
        """
        if isinstance(value, (tuple, list, set, frozenset)):
            normalized = tuple(value)
        else:
            raise ValueError(
                "Üyelik karşılaştırmasının sağ tarafı sabit bir "
                "koleksiyon olmalıdır."
            )

        if not normalized:
            raise UnreachablePathError(
                "Boş koleksiyon için üyelik kısıtı sağlanamaz."
            )

        return normalized

    @staticmethod
    def _merge_minimum(
        current: _VariableConstraint,
        value: float | int,
        inclusive: bool,
    ) -> _VariableConstraint:
        minimum = current.minimum
        minimum_inclusive = current.minimum_inclusive

        if minimum is None or value > minimum:
            minimum = value
            minimum_inclusive = inclusive
        elif value == minimum:
            minimum_inclusive = (
                minimum_inclusive and inclusive
            )

        result = replace(
            current,
            minimum=minimum,
            minimum_inclusive=minimum_inclusive,
        )

        PathInputGenerator._validate_constraint_consistency(result)

        return result

    @staticmethod
    def _merge_maximum(
        current: _VariableConstraint,
        value: float | int,
        inclusive: bool,
    ) -> _VariableConstraint:
        maximum = current.maximum
        maximum_inclusive = current.maximum_inclusive

        if maximum is None or value < maximum:
            maximum = value
            maximum_inclusive = inclusive
        elif value == maximum:
            maximum_inclusive = (
                maximum_inclusive and inclusive
            )

        result = replace(
            current,
            maximum=maximum,
            maximum_inclusive=maximum_inclusive,
        )

        PathInputGenerator._validate_constraint_consistency(result)

        return result

    @staticmethod
    def _validate_constraint_consistency(
        constraint: _VariableConstraint,
    ) -> None:
        """
        Bir değişken için biriktirilen bütün kısıtların birlikte
        sağlanabilir olup olmadığını doğrular.
        """
        PathInputGenerator._validate_range(
            constraint
        )

        if any(
            PathInputGenerator._contains_equal_value(
                constraint.forbidden_collection_members,
                member,
            )
            for member in constraint.required_collection_members
        ):
            raise UnreachablePathError(
                "Aynı koleksiyon üyelik kısıtları çelişiyor."
            )

        if (
            constraint.has_equal_value
            and constraint.equal_value is False
            and constraint.required_collection_members
        ):
            raise UnreachablePathError(
                "Falsy koleksiyon kısıtı gerekli üyelikle çelişiyor."
            )

        if (
            constraint.has_equal_value
            and not isinstance(constraint.equal_value, bool)
            and (
                constraint.required_collection_members
                or constraint.forbidden_collection_members
            )
        ):
            for member in constraint.required_collection_members:
                if not PathInputGenerator._collection_contains_member(
                    constraint.equal_value,
                    member,
                ):
                    raise UnreachablePathError(
                        "Kesin koleksiyon değeri gerekli üyeliği sağlamıyor."
                    )
            for member in constraint.forbidden_collection_members:
                if PathInputGenerator._collection_contains_member(
                    constraint.equal_value,
                    member,
                ):
                    raise UnreachablePathError(
                        "Kesin koleksiyon değeri yasak üyelikle çelişiyor."
                    )

        if constraint.has_equal_value:
            equal_value = constraint.equal_value

            if equal_value in constraint.forbidden_values:
                raise UnreachablePathError(
                    "Eşitlik ve eşitsizlik kısıtları çelişiyor."
                )

            if (
                constraint.allowed_values is not None
                and equal_value not in constraint.allowed_values
            ):
                raise UnreachablePathError(
                    "Eşitlik değeri üyelik kısıtını sağlamıyor."
                )

            if isinstance(equal_value, bool):
                if (
                    constraint.minimum is not None
                    or constraint.maximum is not None
                ):
                    raise UnreachablePathError(
                        "Bool eşitlik kısıtı sayısal aralıkla "
                        "birlikte kullanılamaz."
                    )

                return

            if (
                constraint.minimum is not None
                or constraint.maximum is not None
            ) and (
                not isinstance(equal_value, (int, float))
                or isinstance(equal_value, bool)
            ):
                raise UnreachablePathError(
                    "Sayısal aralık ile sayısal olmayan eşitlik "
                    "kısıtı birlikte sağlanamaz."
                )

            if not PathInputGenerator._satisfies_minimum(
                value=equal_value,
                constraint=constraint,
            ):
                raise UnreachablePathError(
                    "Eşitlik değeri minimum kısıtını sağlamıyor."
                )

            if not PathInputGenerator._satisfies_maximum(
                value=equal_value,
                constraint=constraint,
            ):
                raise UnreachablePathError(
                    "Eşitlik değeri maksimum kısıtını sağlamıyor."
                )

        if constraint.allowed_values is not None:
            valid_allowed_values = tuple(
                value
                for value in constraint.allowed_values
                if (
                    value not in constraint.forbidden_values
                    and (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and PathInputGenerator._satisfies_minimum(
                            value=value,
                            constraint=constraint,
                        )
                        and PathInputGenerator._satisfies_maximum(
                            value=value,
                            constraint=constraint,
                        )
                        or (
                            constraint.minimum is None
                            and constraint.maximum is None
                        )
                    )
                )
            )

            if not valid_allowed_values:
                raise UnreachablePathError(
                    "Üyelik kısıtındaki hiçbir değer diğer "
                    "kısıtları sağlamıyor."
                )

        if (
            constraint.minimum is not None
            and constraint.maximum is not None
            and constraint.minimum == constraint.maximum
            and constraint.minimum_inclusive
            and constraint.maximum_inclusive
            and constraint.minimum in constraint.forbidden_values
        ):
            raise UnreachablePathError(
                "Tek mümkün değer eşitsizlik kısıtıyla yasaklandı."
            )

    @staticmethod
    def _validate_range(
        constraint: _VariableConstraint,
    ) -> None:
        if (
            constraint.minimum is None
            or constraint.maximum is None
        ):
            return

        if constraint.minimum > constraint.maximum:
            raise UnreachablePathError(
                "Minimum ve maksimum kısıtları çelişiyor."
            )

        if (
            constraint.minimum == constraint.maximum
            and (
                not constraint.minimum_inclusive
                or not constraint.maximum_inclusive
            )
        ):
            raise UnreachablePathError(
                "Belirlenen aralık geçerli bir değer içermiyor."
            )

    def _collect_iteration_scoped_constraints(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        handled_loop_node_ids: set[int],
    ) -> tuple[
        dict[str, _VariableConstraint],
        list[_ForLoopActivation],
    ]:
        """
        Yol koşullarını normal ve ``for``-iterasyon kısıtlarına ayırır.

        Aktif döngüler sıralı ``PathStep`` ziyaretlerinden çıkarılır.
        Aynı CFG düğümünün her ``Iterate`` ziyareti mevcut aktivasyonun
        sonraki elemanını, bir inner loop tamamlandıktan sonraki yeni
        ziyaret ise parent bağlamı altında yeni bir aktivasyonu temsil eder.
        """
        constraints: dict[str, _VariableConstraint] = {}
        active_loops: list[_ForLoopActivation] = []
        loop_activations: list[_ForLoopActivation] = []
        next_activation_id = 0
        shadowed_safe_calls = self._find_shadowed_condition_safe_calls(
            path=path,
            parameter_names=parameter_names,
        )

        for step in path.steps:
            if step.node_type == "for":
                binding = self._extract_for_loop_binding(
                    step=step,
                    parameter_names=parameter_names,
                )

                if binding is None:
                    continue

                matching_index = self._find_active_loop_index(
                    active_loops=active_loops,
                    node_id=step.node_id,
                )

                if step.outgoing_edge_label == "Iterate":
                    if matching_index is None:
                        next_activation_id += 1
                        loop_variable, iterable_name = binding
                        activation = _ForLoopActivation(
                            node_id=step.node_id,
                            activation_id=next_activation_id,
                            parent_context=tuple(
                                (
                                    active_loop.node_id,
                                    active_loop.activation_id,
                                )
                                for active_loop in active_loops
                            ),
                            target_name=loop_variable,
                            iterable_name=iterable_name,
                        )
                        active_loops.append(activation)
                        loop_activations.append(activation)
                    else:
                        del active_loops[matching_index + 1:]
                        activation = active_loops[matching_index]

                    activation.iteration_index += 1
                    activation.iteration_constraints.setdefault(
                        activation.iteration_index,
                        _VariableConstraint(),
                    )
                    activation.local_iteration_constraints.setdefault(
                        activation.iteration_index,
                        {},
                    )
                    continue

                if (
                    step.outgoing_edge_label == "Complete"
                    and matching_index is not None
                ):
                    del active_loops[matching_index:]

                continue

            if step.node_type in {"Assign", "AnnAssign"} and active_loops:
                assignment = self._parse_single_assignment(step.node_label)
                if assignment is not None:
                    target_name, _ = assignment
                    owner = active_loops[-1]
                    owner.local_iteration_constraints[
                        owner.iteration_index
                    ].setdefault(target_name, _VariableConstraint())
                continue

            if (
                step.node_type not in {"if", "while"}
                or step.node_id in handled_loop_node_ids
            ):
                continue

            self._apply_scoped_condition_step(
                step=step,
                constraints=constraints,
                active_loops=active_loops,
                shadowed_safe_calls=shadowed_safe_calls,
            )

        return constraints, loop_activations

    @staticmethod
    def _find_shadowed_condition_safe_calls(
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
    ) -> frozenset[str]:
        """Path metadata'sında güvenli predicate adını bağlayan isimleri bulur."""
        shadowed: set[str] = set(parameter_names) & {"isinstance"}

        for step in path.steps:
            if step.node_type in {"Import", "ImportFrom"}:
                try:
                    statement = ast.parse(step.node_label).body[0]
                except (SyntaxError, IndexError):
                    continue
                aliases = getattr(statement, "names", ())
                if any(
                    (alias.asname or alias.name.split(".", maxsplit=1)[0])
                    == "isinstance"
                    for alias in aliases
                ):
                    shadowed.add("isinstance")
                continue

            if step.node_type == "for":
                try:
                    target_text = step.node_label.split(" in ", maxsplit=1)[0]
                    target = ast.parse(target_text, mode="eval").body
                except (SyntaxError, ValueError):
                    continue
                if any(
                    isinstance(node, ast.Name) and node.id == "isinstance"
                    for node in ast.walk(target)
                ):
                    shadowed.add("isinstance")
                continue

            if step.node_type not in {"Assign", "AnnAssign"}:
                continue

            try:
                statement = ast.parse(step.node_label).body[0]
            except (SyntaxError, IndexError):
                continue

            targets: tuple[ast.expr, ...]
            if isinstance(statement, ast.Assign):
                targets = tuple(statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                targets = (statement.target,)
            else:
                continue

            if any(
                isinstance(node, ast.Name) and node.id == "isinstance"
                for target in targets
                for node in ast.walk(target)
            ):
                shadowed.add("isinstance")

        return frozenset(shadowed)

    @staticmethod
    def _find_active_loop_index(
        *,
        active_loops: list[_ForLoopActivation],
        node_id: int,
    ) -> int | None:
        """Aktif stack içindeki en içteki eşleşen CFG loop'unu bulur."""
        for index in range(len(active_loops) - 1, -1, -1):
            if active_loops[index].node_id == node_id:
                return index

        return None

    def _apply_scoped_condition_step(
        self,
        *,
        step: PathStep,
        constraints: dict[str, _VariableConstraint],
        active_loops: list[_ForLoopActivation],
        shadowed_safe_calls: frozenset[str],
    ) -> None:
        """Koşulu aktif loop target'ları için ilgili iterasyona yönlendirir."""
        routed_constraints = dict(constraints)

        for activation in active_loops:
            if activation.iteration_index < 0:
                continue

            routed_constraints[activation.target_name] = (
                activation.iteration_constraints[
                    activation.iteration_index
                ]
            )
            routed_constraints.update(
                activation.local_iteration_constraints.get(
                    activation.iteration_index,
                    {},
                )
            )

        self._apply_condition_step(
            step=step,
            constraints=routed_constraints,
            shadowed_safe_calls=shadowed_safe_calls,
        )

        for variable_name, constraint in routed_constraints.items():
            activation = self._find_active_loop_for_target(
                active_loops=active_loops,
                target_name=variable_name,
            )

            if activation is None:
                activation = self._find_active_loop_for_local(
                    active_loops=active_loops,
                    variable_name=variable_name,
                )
                if activation is None:
                    constraints[variable_name] = constraint
                    continue

                activation.local_iteration_constraints[
                    activation.iteration_index
                ][variable_name] = constraint
                continue

            activation.iteration_constraints[
                activation.iteration_index
            ] = constraint

    @staticmethod
    def _find_active_loop_for_target(
        *,
        active_loops: list[_ForLoopActivation],
        target_name: str,
    ) -> _ForLoopActivation | None:
        """Aynı adı gölgeleyen loop'larda en içteki binding'i döndürür."""
        for activation in reversed(active_loops):
            if activation.target_name == target_name:
                return activation

        return None

    @staticmethod
    def _find_active_loop_for_local(
        *,
        active_loops: list[_ForLoopActivation],
        variable_name: str,
    ) -> _ForLoopActivation | None:
        for activation in reversed(active_loops):
            if variable_name in activation.local_iteration_constraints.get(
                activation.iteration_index,
                {},
            ):
                return activation
        return None

    @staticmethod
    def _resize_for_loop_iterables(
        *,
        loop_activations: list[_ForLoopActivation],
        direct_values: dict[str, Any],
    ) -> None:
        """Iterable uzunluğunu aktivasyon başına en yüksek iterasyona ayarlar."""
        required_lengths: dict[str, int] = {}

        for activation in loop_activations:
            required_lengths[activation.iterable_name] = max(
                required_lengths.get(activation.iterable_name, 0),
                activation.iteration_index + 1,
            )

        for iterable_name, required_length in required_lengths.items():
            existing_value = direct_values.get(iterable_name)

            if isinstance(existing_value, tuple):
                direct_values[iterable_name] = tuple(
                    list(existing_value)[:required_length]
                )
            elif isinstance(existing_value, list):
                direct_values[iterable_name] = list(existing_value)[
                    :required_length
                ]

    def _restore_membership_loop_candidate_values(
        self,
        *,
        parameter_types: dict[str, str],
        constraints: dict[str, _VariableConstraint],
        loop_activations: list[_ForLoopActivation],
        loop_iterable_names: set[str],
        candidate_seed_values: dict[str, Any],
        direct_values: dict[str, Any],
    ) -> set[str]:
        """Üyelik seed'ini doğrulayıp bounded loop cardinality'siyle korur."""
        required_lengths: dict[str, int] = {}
        for activation in loop_activations:
            required_lengths[activation.iterable_name] = max(
                required_lengths.get(activation.iterable_name, 0),
                activation.iteration_index + 1,
            )

        restored: set[str] = set()
        for parameter_name in loop_iterable_names:
            required_length = required_lengths.get(parameter_name, 0)
            constraint = constraints.get(parameter_name)
            if (
                parameter_name not in candidate_seed_values
                or constraint is None
                or not (
                    constraint.required_collection_members
                    or constraint.forbidden_collection_members
                )
            ):
                continue

            kind, member_type, value_type = (
                self._collection_membership_schema(
                    parameter_types.get(parameter_name)
                )
            )
            candidate = self._copy_typed_collection(
                value=candidate_seed_values[parameter_name],
                kind=kind,
                member_type=member_type,
                value_type=value_type,
            )
            if len(candidate) != required_length:
                raise UnsupportedInputSynthesisError(
                    "Typed membership seed'i bounded loop iterasyon "
                    "sayısıyla aynı cardinality'ye sahip olmalıdır."
                )
            direct_values[parameter_name] = candidate
            restored.add(parameter_name)
        return restored

    @classmethod
    def _collect_for_loop_iterable_names(
        cls,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
    ) -> set[str]:
        iterable_names: set[str] = set()
        for step in path.loop_steps:
            if step.node_type != "for":
                continue
            binding = cls._extract_for_loop_binding(
                step=step,
                parameter_names=parameter_names,
            )
            if binding is not None:
                iterable_names.add(binding[1])
        return iterable_names

    def _apply_loop_variable_constraints(
        self,
        *,
        parameter_types: dict[str, str],
        loop_activations: list[_ForLoopActivation],
        direct_values: dict[str, Any],
        restored_membership_loop_candidates: set[str],
    ) -> None:
        """
        ``for item in items`` yapısındaki yerel döngü değişkeni
        üzerinde çıkarılan kısıtları iterable parametresinin
        elemanlarına geri yansıtır.
        """
        element_constraints: dict[
            tuple[str, int],
            _VariableConstraint,
        ] = {}

        for activation in loop_activations:
            for iteration_index, loop_constraint in (
                activation.iteration_constraints.items()
            ):
                element_key = (
                    activation.iterable_name,
                    iteration_index,
                )
                element_constraints[element_key] = (
                    self._combine_variable_constraints(
                        current=element_constraints.get(
                            element_key,
                            _VariableConstraint(),
                        ),
                        additional=loop_constraint,
                    )
                )

        for (
            iterable_name,
            iteration_index,
        ), loop_constraint in element_constraints.items():
            if (
                loop_constraint == _VariableConstraint()
                and iterable_name in restored_membership_loop_candidates
            ):
                continue

            existing_value = direct_values.get(iterable_name)

            if existing_value is None:
                continue

            if isinstance(existing_value, list):
                iterable_values = list(existing_value)
                restore_as_tuple = False
            elif isinstance(existing_value, tuple):
                iterable_values = list(existing_value)
                restore_as_tuple = True
            else:
                raise UnsupportedInputSynthesisError(
                    "Sırasız loop iterable eleman kısıtı güvenli bir "
                    "pozisyona geri yansıtılamıyor."
                )

            element_type = self._extract_element_type(
                parameter_types.get(iterable_name)
            )

            if iteration_index >= len(iterable_values):
                raise UnreachablePathError(
                    "For döngüsü iterasyon indeksi üretilen "
                    "koleksiyonda bulunmuyor: "
                    f"{iterable_name}[{iteration_index}]"
                )

            if (
                iterable_name in restored_membership_loop_candidates
                and self._value_satisfies_constraint(
                    value=iterable_values[iteration_index],
                    constraint=loop_constraint,
                )
            ):
                continue

            element_value = self._create_parameter_value(
                parameter_name=(
                    f"{iterable_name}[{iteration_index}]"
                ),
                constraint=loop_constraint,
                parameter_type=element_type,
            )

            iterable_values[iteration_index] = element_value

            direct_values[iterable_name] = (
                tuple(iterable_values)
                if restore_as_tuple
                else iterable_values
            )

    def _combine_variable_constraints(
        self,
        *,
        current: _VariableConstraint,
        additional: _VariableConstraint,
    ) -> _VariableConstraint:
        """Aynı somut koleksiyon elemanına ait kısıtları birleştirir."""
        result = current

        if additional.has_equal_value:
            result = self._merge_constraint(
                current=result,
                operator=ast.Eq(),
                value=additional.equal_value,
            )

        for forbidden_value in additional.forbidden_values:
            result = self._merge_constraint(
                current=result,
                operator=ast.NotEq(),
                value=forbidden_value,
            )

        if additional.allowed_values is not None:
            result = self._merge_constraint(
                current=result,
                operator=ast.In(),
                value=additional.allowed_values,
            )

        for member in additional.required_collection_members:
            result = self._merge_collection_membership_constraint(
                current=result,
                operator=ast.In(),
                member=member,
            )

        for member in additional.forbidden_collection_members:
            result = self._merge_collection_membership_constraint(
                current=result,
                operator=ast.NotIn(),
                member=member,
            )

        if additional.minimum is not None:
            result = self._merge_constraint(
                current=result,
                operator=(
                    ast.GtE()
                    if additional.minimum_inclusive
                    else ast.Gt()
                ),
                value=additional.minimum,
            )

        if additional.maximum is not None:
            result = self._merge_constraint(
                current=result,
                operator=(
                    ast.LtE()
                    if additional.maximum_inclusive
                    else ast.Lt()
                ),
                value=additional.maximum,
            )

        for required_group in additional.required_runtime_type_groups:
            result = replace(
                result,
                required_runtime_type_groups=(
                    *result.required_runtime_type_groups,
                    required_group,
                ),
            )

        if additional.forbidden_runtime_types:
            result = replace(
                result,
                forbidden_runtime_types=tuple(
                    dict.fromkeys(
                        (
                            *result.forbidden_runtime_types,
                            *additional.forbidden_runtime_types,
                        )
                    )
                ),
            )

        if (
            self._has_runtime_type_constraint(result)
            and not result.required_collection_members
            and not result.forbidden_collection_members
            and not self._runtime_type_constraint_has_candidate(result)
        ):
            raise UnreachablePathError(
                "Aynı değer için çıkarılan runtime type kısıtları çelişiyor."
            )

        return result

    @staticmethod
    def _extract_for_loop_binding(
        *,
        step: PathStep,
        parameter_names: tuple[str, ...],
    ) -> tuple[str, str] | None:
        """
        ``item in items`` biçimindeki CFG etiketinden döngü
        değişkeni ve iterable parametresini çıkarır.
        """
        try:
            target_text, iterable_text = step.node_label.split(
                " in ",
                maxsplit=1,
            )
            target_expression = ast.parse(
                target_text,
                mode="eval",
            ).body
            iterable_expression = ast.parse(
                iterable_text,
                mode="eval",
            ).body
        except (ValueError, SyntaxError):
            return None

        if (
            not isinstance(target_expression, ast.Name)
            or not isinstance(iterable_expression, ast.Name)
            or iterable_expression.id not in parameter_names
        ):
            return None

        return (
            target_expression.id,
            iterable_expression.id,
        )

    def _apply_local_alias_constraints(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
        constraints: dict[str, _VariableConstraint],
        direct_values: dict[str, Any],
    ) -> None:
        """
        Yerel değişkene atanmış parametre alt elemanı üzerindeki
        kısıtları asıl fonksiyon parametresine geri yansıtır.

        Desteklenen temel biçim:

            local_value = parameter[index]

        Örneğin ``first_item = items[0]`` sonrasında
        ``first_item < 0`` koşulu varsa ``items[0]`` için negatif
        bir değer üretilir. Böylece dışarıdan gelen kaynak kodda
        yerel takma adlar kullanılsa dahi hedef yürütme yoluna
        ulaşılabilir.
        """
        for step in path.steps:
            if step.node_type not in {
                "Assign",
                "AnnAssign",
            }:
                continue

            binding = self._extract_subscript_alias_binding(
                step=step,
                parameter_names=parameter_names,
            )

            if binding is None:
                continue

            (
                local_name,
                parameter_name,
                index_value,
            ) = binding

            local_constraint = constraints.get(local_name)

            if local_constraint is None:
                continue

            parameter_type = parameter_types.get(
                parameter_name
            )
            element_type = self._extract_element_type(
                parameter_type
            )

            element_value = self._create_parameter_value(
                parameter_name=local_name,
                constraint=local_constraint,
                parameter_type=element_type,
            )

            existing_value = direct_values.get(
                parameter_name
            )

            if existing_value is None:
                collection_values: list[Any] = []
                restore_as_tuple = self._is_tuple_type(
                    parameter_type
                )
            elif isinstance(existing_value, list):
                collection_values = list(existing_value)
                restore_as_tuple = False
            elif isinstance(existing_value, tuple):
                collection_values = list(existing_value)
                restore_as_tuple = True
            else:
                raise UnreachablePathError(
                    "Yerel alt eleman kısıtı, indekslenebilir "
                    "olmayan doğrudan parametre değeriyle "
                    "çelişiyor: "
                    f"{parameter_name}={existing_value!r}"
                )

            default_element = self._create_default_typed_value(
                element_type
            )

            while len(collection_values) <= index_value:
                collection_values.append(default_element)

            collection_values[index_value] = element_value

            direct_values[parameter_name] = (
                tuple(collection_values)
                if restore_as_tuple
                else collection_values
            )

    def _apply_direct_name_alias_constraints(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        constraints: dict[str, _VariableConstraint],
    ) -> None:
        """``local = parameter`` alias kısıtlarını gerçek parametreye taşır."""
        parameter_set = set(parameter_names)
        aliases: dict[str, str] = {}

        for step in path.steps:
            if step.node_type not in {"Assign", "AnnAssign"}:
                continue
            try:
                statement = ast.parse(step.node_label).body[0]
            except (SyntaxError, IndexError):
                continue

            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target = statement.targets[0]
                value = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                target = statement.target
                value = statement.value
            else:
                continue

            if not isinstance(value, ast.Name):
                aliases.pop(target.id, None)
                continue

            source_name = (
                value.id
                if value.id in parameter_set
                else aliases.get(value.id)
            )
            if source_name is None:
                aliases.pop(target.id, None)
            else:
                aliases[target.id] = source_name

        for local_name, parameter_name in aliases.items():
            local_constraint = constraints.get(local_name)
            if local_constraint is None:
                continue
            constraints[parameter_name] = self._combine_variable_constraints(
                current=constraints.get(parameter_name, _VariableConstraint()),
                additional=local_constraint,
            )

    def _apply_dictionary_lookup_constraints(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
        constraints: dict[str, _VariableConstraint],
        direct_values: dict[str, Any],
        loop_activations: list[_ForLoopActivation],
    ) -> None:
        """Güvenli subscript ve ``dict.get`` provenance'ını dış girdiye taşır."""
        references = {
            name: _StructuredInputReference(name)
            for name in parameter_names
        }
        lookups: dict[str, _DictionaryLookup] = {}
        loop_indices: dict[int, int] = {}
        active_structured_loops: list[_ForLoopActivation] = []
        used_activation_ids: set[int] = set()
        required_present: set[_StructuredInputReference] = set()
        required_absent: set[_StructuredInputReference] = set()

        for step in path.steps:
            if step.node_type == "for":
                if step.outgoing_edge_label == "Iterate":
                    binding = self._extract_for_loop_binding(
                        step=step,
                        parameter_names=parameter_names,
                    )
                    if binding is not None:
                        target, iterable = binding
                        index = loop_indices.get(step.node_id, -1) + 1
                        loop_indices[step.node_id] = index
                        activation = next(
                            (
                                candidate
                                for candidate in active_structured_loops
                                if candidate.node_id == step.node_id
                            ),
                            None,
                        )
                        if activation is None:
                            activation = next(
                                candidate
                                for candidate in loop_activations
                                if candidate.node_id == step.node_id
                                and candidate.activation_id not in used_activation_ids
                            )
                            used_activation_ids.add(activation.activation_id)
                            active_structured_loops.append(activation)
                        references[target] = _StructuredInputReference(
                            iterable, (index,)
                        )
                elif step.outgoing_edge_label == "Complete":
                    loop_indices.pop(step.node_id, None)
                    active_structured_loops = [
                        activation
                        for activation in active_structured_loops
                        if activation.node_id != step.node_id
                    ]
                continue

            if step.node_type not in {"Assign", "AnnAssign"}:
                continue
            assignment = self._parse_single_assignment(step.node_label)
            if assignment is None:
                continue
            target_name, expression = assignment

            if isinstance(expression, ast.Name):
                source = references.get(expression.id)
                if source is not None:
                    references[target_name] = source
                continue

            if isinstance(expression, ast.Subscript):
                reference = self._resolve_structured_subscript(
                    expression=expression,
                    references=references,
                    parameter_types=parameter_types,
                    direct_values=direct_values,
                    require_present=(step.outgoing_edge_label != "Exception"),
                    required_present=required_present,
                    required_absent=required_absent,
                )
                if reference is None and step.outgoing_edge_label == "Exception":
                    raise UnsupportedInputSynthesisError(
                        "KeyError dictionary provenance'ı gerçek input'a "
                        "güvenli biçimde bağlanamadı."
                    )
                if reference is not None:
                    references[target_name] = reference
                    constraint = self._structured_local_constraint(
                        variable_name=target_name,
                        constraints=constraints,
                        active_loops=active_structured_loops,
                        loop_indices=loop_indices,
                    )
                    if (
                        constraint is not None
                        and step.outgoing_edge_label != "Exception"
                    ):
                        self._materialize_structured_constraint(
                            reference=reference,
                            constraint=constraint,
                            parameter_types=parameter_types,
                            direct_values=direct_values,
                        )
                continue

            if isinstance(expression, ast.Call):
                constraint = self._structured_local_constraint(
                    variable_name=target_name,
                    constraints=constraints,
                    active_loops=active_structured_loops,
                    loop_indices=loop_indices,
                )
                if constraint is None:
                    continue
                lookup = self._parse_safe_dictionary_lookup(
                    expression=expression,
                    references=references,
                    parameter_types=parameter_types,
                    direct_values=direct_values,
                )
                lookups[target_name] = lookup
                if constraint is not None:
                    self._materialize_dictionary_lookup(
                        lookup=lookup,
                        constraint=constraint,
                        constraints=constraints,
                        parameter_types=parameter_types,
                        direct_values=direct_values,
                    )
                continue

            if target_name in constraints:
                references.pop(target_name, None)
                lookups.pop(target_name, None)

        for local_name, reference in references.items():
            if local_name in parameter_names:
                continue
            constraint = constraints.get(local_name)
            if constraint is not None:
                self._materialize_structured_constraint(
                    reference=reference,
                    constraint=constraint,
                    parameter_types=parameter_types,
                    direct_values=direct_values,
                )

        for local_name, lookup in lookups.items():
            constraint = constraints.get(local_name)
            if constraint is not None:
                self._materialize_dictionary_lookup(
                    lookup=lookup,
                    constraint=constraint,
                    constraints=constraints,
                    parameter_types=parameter_types,
                    direct_values=direct_values,
                )

    @staticmethod
    def _structured_local_constraint(
        *,
        variable_name: str,
        constraints: dict[str, _VariableConstraint],
        active_loops: list[_ForLoopActivation],
        loop_indices: dict[int, int],
    ) -> _VariableConstraint | None:
        for activation in reversed(active_loops):
            iteration_index = loop_indices.get(activation.node_id)
            if iteration_index is None:
                continue
            local_constraint = activation.local_iteration_constraints.get(
                iteration_index,
                {},
            ).get(variable_name)
            if local_constraint is not None:
                return local_constraint
        return constraints.get(variable_name)

    @staticmethod
    def _parse_single_assignment(
        statement_text: str,
    ) -> tuple[str, ast.expr] | None:
        try:
            statement = ast.parse(statement_text).body[0]
        except (SyntaxError, IndexError):
            return None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0].id, statement.value
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            return statement.target.id, statement.value
        return None

    def _resolve_structured_subscript(
        self,
        *,
        expression: ast.Subscript,
        references: dict[str, _StructuredInputReference],
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
        require_present: bool,
        required_present: set[_StructuredInputReference],
        required_absent: set[_StructuredInputReference],
    ) -> _StructuredInputReference | None:
        if not isinstance(expression.value, ast.Name):
            return None
        parent = references.get(expression.value.id)
        if parent is None:
            return None
        parent_type = self._structured_reference_type(
            reference=parent,
            parameter_types=parameter_types,
        )
        parent_value = direct_values.get(parent.parameter_name)
        if (
            not parent.access_path
            and not isinstance(parent_value, (list, tuple, dict))
            and not (
                self._is_dict_type(parent_type)
                or self._extract_element_type(parent_type) is not None
            )
        ):
            return None
        key = self._resolve_lookup_key(
            expression=expression.slice,
            references=references,
            parameter_types=parameter_types,
            direct_values=direct_values,
        )
        if not isinstance(key, (int, str)) or isinstance(key, bool):
            raise UnsupportedInputSynthesisError(
                "Structured dictionary/list anahtarı int veya str olmalıdır."
            )
        reference = _StructuredInputReference(
            parent.parameter_name, (*parent.access_path, key)
        )
        if require_present:
            if reference in required_absent:
                raise UnreachablePathError(
                    "Aynı dictionary key için present ve absent "
                    f"kısıtları çelişiyor: {self._format_structured_reference(reference)}"
                )
            required_present.add(reference)
            self._ensure_structured_reference(
                reference=reference,
                parameter_types=parameter_types,
                direct_values=direct_values,
            )
        else:
            if reference in required_present:
                raise UnreachablePathError(
                    "Aynı dictionary key için present ve absent "
                    f"kısıtları çelişiyor: {self._format_structured_reference(reference)}"
                )
            required_absent.add(reference)
            self._delete_structured_reference(
                reference=reference,
                direct_values=direct_values,
            )
        return reference

    def _parse_safe_dictionary_lookup(
        self,
        *,
        expression: ast.Call,
        references: dict[str, _StructuredInputReference],
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
    ) -> _DictionaryLookup:
        if not (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "get"
            and isinstance(expression.func.value, ast.Name)
        ):
            raise UnsupportedInputSynthesisError(
                "Yalnız doğrulanmış dictionary receiver üzerindeki dict.get "
                "lookup çağrıları desteklenmektedir."
            )
        if expression.keywords or len(expression.args) not in {1, 2} or any(
            isinstance(argument, ast.Starred) for argument in expression.args
        ):
            raise UnsupportedInputSynthesisError(
                "dict.get yalnız bir veya iki positional, starred olmayan "
                "argümanla desteklenmektedir."
            )
        mapping = references.get(expression.func.value.id)
        if mapping is None:
            raise UnsupportedInputSynthesisError(
                "dict.get receiver provenance'ı gerçek input'a bağlanamadı."
            )
        mapping_value = self._ensure_structured_reference(
            reference=mapping,
            parameter_types=parameter_types,
            direct_values=direct_values,
        )
        mapping_type = self._structured_reference_type(
            reference=mapping,
            parameter_types=parameter_types,
        )
        if type(mapping_value) is not dict and not self._is_dict_type(mapping_type):
            raise UnsupportedInputSynthesisError(
                "dict.get receiver gerçek dictionary parametresi veya alias olmalıdır."
            )
        if type(mapping_value) is not dict:
            self._set_structured_reference(mapping, {}, direct_values)

        key = self._resolve_lookup_key(
            expression=expression.args[0],
            references=references,
            parameter_types=parameter_types,
            direct_values=direct_values,
        )
        default: _StructuredInputReference | int | float | str | bool | None = None
        if len(expression.args) == 2:
            default = self._resolve_lookup_value(
                expression=expression.args[1],
                references=references,
            )
            if isinstance(default, _StructuredInputReference):
                default = self._ensure_structured_reference(
                    reference=default,
                    parameter_types=parameter_types,
                    direct_values=direct_values,
                )
        return _DictionaryLookup(
            mapping=mapping,
            key=key,
            default=default,
            has_default=(len(expression.args) == 2),
        )

    def _resolve_lookup_key(
        self,
        *,
        expression: ast.expr,
        references: dict[str, _StructuredInputReference],
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
    ) -> _StructuredInputReference | int | float | str | bool | None:
        value = self._resolve_lookup_value(
            expression=expression,
            references=references,
        )
        if isinstance(value, _StructuredInputReference):
            return self._ensure_structured_reference(
                reference=value,
                parameter_types=parameter_types,
                direct_values=direct_values,
            )
        return value

    @staticmethod
    def _resolve_lookup_value(
        *,
        expression: ast.expr,
        references: dict[str, _StructuredInputReference],
    ) -> _StructuredInputReference | int | float | str | bool | None:
        if isinstance(expression, ast.Name):
            reference = references.get(expression.id)
            if reference is not None:
                return reference
        try:
            value = ast.literal_eval(expression)
        except (ValueError, TypeError) as error:
            raise UnsupportedInputSynthesisError(
                "dict.get key/default ifadesi güvenli literal veya input "
                "provenance olmalıdır."
            ) from error
        if value is not None and not isinstance(value, (int, float, str, bool)):
            raise UnsupportedInputSynthesisError(
                "dict.get key/default yalnız primitive değerleri destekler."
            )
        return value

    def _materialize_structured_constraint(
        self,
        *,
        reference: _StructuredInputReference,
        constraint: _VariableConstraint,
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
    ) -> None:
        existing_value = self._ensure_structured_reference(
            reference=reference,
            parameter_types=parameter_types,
            direct_values=direct_values,
        )
        if self._value_satisfies_constraint(
            value=existing_value,
            constraint=constraint,
        ):
            return
        value = self._create_parameter_value(
            parameter_name=self._format_structured_reference(reference),
            constraint=constraint,
            parameter_type=self._structured_reference_type(
                reference=reference,
                parameter_types=parameter_types,
            ),
        )
        self._set_structured_reference(reference, value, direct_values)

    def _materialize_dictionary_lookup(
        self,
        *,
        lookup: _DictionaryLookup,
        constraint: _VariableConstraint,
        constraints: dict[str, _VariableConstraint],
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
    ) -> None:
        mapping = self._get_structured_reference(lookup.mapping, direct_values)
        if type(mapping) is not dict:
            raise UnsupportedInputSynthesisError(
                "Doğrulanmış dict.get receiver dictionary değerine dönüşmedi."
            )
        key = lookup.key
        default = lookup.default
        absent_value = default if lookup.has_default else None

        if self._value_satisfies_constraint(value=absent_value, constraint=constraint):
            mapping.pop(key, None)
            self._preserve_truthy_mapping(
                lookup=lookup,
                missing_key=key,
                mapping=mapping,
                constraints=constraints,
                parameter_types=parameter_types,
            )
            return

        value_type = self._dictionary_value_type(
            self._structured_reference_type(
                reference=lookup.mapping,
                parameter_types=parameter_types,
            )
        )
        mapping[key] = self._create_parameter_value(
            parameter_name=f"{self._format_structured_reference(lookup.mapping)}[{key!r}]",
            constraint=constraint,
            parameter_type=value_type,
        )

    def _preserve_truthy_mapping(
        self,
        *,
        lookup: _DictionaryLookup,
        missing_key: Any,
        mapping: dict[Any, Any],
        constraints: dict[str, _VariableConstraint],
        parameter_types: dict[str, str],
    ) -> None:
        if mapping:
            return
        root_constraint = constraints.get(lookup.mapping.parameter_name)
        if not (
            root_constraint is not None
            and root_constraint.has_equal_value
            and root_constraint.equal_value is True
        ):
            return
        key_type, value_type = self._dictionary_types(
            self._structured_reference_type(
                reference=lookup.mapping,
                parameter_types=parameter_types,
            )
        )
        sentinel: int | str = 0 if key_type == "int" else "__generated_key__"
        while sentinel == missing_key or sentinel in mapping:
            sentinel = f"{sentinel}_x" if isinstance(sentinel, str) else sentinel + 1
        mapping[sentinel] = self._create_default_typed_value(value_type)

    def _ensure_structured_reference(
        self,
        *,
        reference: _StructuredInputReference,
        parameter_types: dict[str, str],
        direct_values: dict[str, Any],
    ) -> Any:
        if reference.parameter_name not in direct_values:
            direct_values[reference.parameter_name] = self._create_default_typed_value(
                parameter_types.get(reference.parameter_name)
            )
        current = direct_values[reference.parameter_name]
        current_type = parameter_types.get(reference.parameter_name)
        for position, token in enumerate(reference.access_path):
            next_type = self._child_type(current_type, token)
            is_last = position == len(reference.access_path) - 1
            if isinstance(token, int) and isinstance(current, (list, tuple)):
                if isinstance(current, tuple):
                    if token >= len(current):
                        raise UnsupportedInputSynthesisError(
                            "Tuple structured provenance mevcut indeksin "
                            "dışına genişletilemez."
                        )
                    if is_last:
                        return current[token]
                    current = current[token]
                    current_type = next_type
                    continue
                while len(current) <= token:
                    current.append(self._create_default_typed_value(next_type))
                if is_last:
                    return current[token]
                current = current[token]
            elif isinstance(current, dict):
                if token not in current:
                    current[token] = self._create_default_typed_value(next_type)
                if is_last:
                    return current[token]
                current = current[token]
            else:
                raise UnsupportedInputSynthesisError(
                    "Structured input provenance list/dict değerine indirgenemedi: "
                    f"{self._format_structured_reference(reference)}"
                )
            current_type = next_type
        return current

    @staticmethod
    def _get_structured_reference(
        reference: _StructuredInputReference,
        direct_values: dict[str, Any],
    ) -> Any:
        current = direct_values[reference.parameter_name]
        for token in reference.access_path:
            current = current[token]
        return current

    @staticmethod
    def _set_structured_reference(
        reference: _StructuredInputReference,
        value: Any,
        direct_values: dict[str, Any],
    ) -> None:
        if not reference.access_path:
            direct_values[reference.parameter_name] = value
            return
        current = direct_values[reference.parameter_name]
        for token in reference.access_path[:-1]:
            current = current[token]
        current[reference.access_path[-1]] = value

    @staticmethod
    def _delete_structured_reference(
        *,
        reference: _StructuredInputReference,
        direct_values: dict[str, Any],
    ) -> None:
        if not reference.access_path or reference.parameter_name not in direct_values:
            return
        current = direct_values[reference.parameter_name]
        try:
            for token in reference.access_path[:-1]:
                current = current[token]
            if isinstance(current, dict):
                current.pop(reference.access_path[-1], None)
        except (KeyError, IndexError, TypeError):
            return

    def _structured_reference_type(
        self,
        *,
        reference: _StructuredInputReference,
        parameter_types: dict[str, str],
    ) -> str | None:
        current = parameter_types.get(reference.parameter_name)
        for token in reference.access_path:
            current = self._child_type(current, token)
        return current

    def _child_type(self, parent_type: str | None, token: int | str) -> str | None:
        if isinstance(token, int) and not self._is_dict_type(parent_type):
            return self._extract_element_type(parent_type)
        return self._dictionary_value_type(parent_type)

    @staticmethod
    def _is_dict_type(type_name: str | None) -> bool:
        normalized = (type_name or "").replace(" ", "")
        return normalized == "dict" or normalized.startswith(("dict[", "typing.Dict["))

    @classmethod
    def _dictionary_value_type(cls, type_name: str | None) -> str | None:
        return cls._dictionary_types(type_name)[1]

    @staticmethod
    def _dictionary_types(type_name: str | None) -> tuple[str | None, str | None]:
        normalized = (type_name or "").replace(" ", "")
        prefix = next(
            (prefix for prefix in ("dict[", "typing.Dict[") if normalized.startswith(prefix)),
            None,
        )
        if prefix is None or not normalized.endswith("]"):
            return None, None
        inner = normalized[len(prefix):-1]
        depth = 0
        for index, character in enumerate(inner):
            if character in "[({":
                depth += 1
            elif character in "])}":
                depth -= 1
            elif character == "," and depth == 0:
                return inner[:index], inner[index + 1:]
        return None, None

    @staticmethod
    def _format_structured_reference(reference: _StructuredInputReference) -> str:
        return reference.parameter_name + "".join(
            f"[{token!r}]" for token in reference.access_path
        )

    def _apply_runtime_type_overrides(
        self,
        *,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
        constraints: dict[str, _VariableConstraint],
        direct_values: dict[str, Any],
    ) -> None:
        """Explicit runtime predicate'i candidate/type-hint seed'inden üstün tutar."""
        for parameter_name in parameter_names:
            constraint = constraints.get(parameter_name)
            if not self._has_runtime_type_constraint(constraint):
                continue
            existing = direct_values.get(parameter_name)
            if (
                existing is not None
                and self._value_satisfies_constraint(
                    value=existing,
                    constraint=constraint,
                )
            ):
                continue
            direct_values[parameter_name] = self._create_parameter_value(
                parameter_name=parameter_name,
                constraint=constraint,
                parameter_type=parameter_types.get(parameter_name),
            )

    @staticmethod
    def _extract_subscript_alias_binding(
        *,
        step: PathStep,
        parameter_names: tuple[str, ...],
    ) -> tuple[str, str, int] | None:
        """
        ``local = parameter[index]`` atamasını çözümler.
        """
        try:
            statement = ast.parse(
                step.node_label
            ).body[0]
        except (SyntaxError, IndexError):
            return None

        target: ast.expr
        value: ast.expr | None

        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                return None

            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        else:
            return None

        if (
            not isinstance(target, ast.Name)
            or not isinstance(value, ast.Subscript)
            or not isinstance(value.value, ast.Name)
            or value.value.id not in parameter_names
        ):
            return None

        try:
            index_value = ast.literal_eval(value.slice)
        except (ValueError, TypeError):
            return None

        if (
            not isinstance(index_value, int)
            or isinstance(index_value, bool)
            or index_value < 0
        ):
            return None

        return (
            target.id,
            value.value.id,
            index_value,
        )

    @staticmethod
    def _extract_element_type(
        parameter_type: str | None,
    ) -> str | None:
        """
        ``list[int]`` veya ``tuple[str, ...]`` benzeri bir
        anotasyondan eleman tipini çıkarır.
        """
        if parameter_type is None:
            return None

        normalized = parameter_type.strip().replace(
            " ",
            "",
        )

        prefixes = (
            "list[",
            "typing.List[",
            "tuple[",
            "typing.Tuple[",
        )

        for prefix in prefixes:
            if not normalized.startswith(prefix):
                continue

            inner = normalized[
                len(prefix):-1
            ]

            return inner.split(",", maxsplit=1)[0]

        return None

    @staticmethod
    def _is_tuple_type(
        parameter_type: str | None,
    ) -> bool:
        if parameter_type is None:
            return False

        normalized = parameter_type.strip().replace(
            " ",
            "",
        )

        return (
            normalized == "tuple"
            or normalized.startswith("tuple[")
            or normalized.startswith("typing.Tuple[")
        )

    def _validate_collection_alias_constraints(
        self,
        *,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
        constraints: dict[str, _VariableConstraint],
        direct_values: dict[str, Any],
    ) -> None:
        """
        ``local = parameter[index]`` ilişkilerinin nihai koleksiyon
        değeriyle hâlâ uyumlu olduğunu doğrular.

        Yerel alias kısıtı uygulandıktan sonra for-döngüsü eleman
        üretimi aynı koleksiyon değerini değiştirebilir. Bu kontrol,
        aynı elemanın birbiriyle çelişen iki koşulu sağlamaya
        zorlandığı yolları ulaşılamaz olarak işaretler.
        """
        for step in path.steps:
            binding = self._extract_subscript_alias_binding(
                step=step,
                parameter_names=parameter_names,
            )

            if binding is None:
                continue

            (
                local_name,
                parameter_name,
                index_value,
            ) = binding

            local_constraint = constraints.get(local_name)

            if local_constraint is None:
                continue

            collection_value = direct_values.get(
                parameter_name
            )

            if not isinstance(
                collection_value,
                (list, tuple),
            ):
                raise UnreachablePathError(
                    "Yerel alt eleman kısıtı için indekslenebilir "
                    "bir koleksiyon üretilemedi: "
                    f"{parameter_name}={collection_value!r}"
                )

            if index_value >= len(collection_value):
                raise UnreachablePathError(
                    "Yerel alt eleman kısıtının gerektirdiği indeks "
                    "üretilen koleksiyonda bulunmuyor: "
                    f"{parameter_name}[{index_value}]"
                )

            element_value = collection_value[index_value]

            if not self._value_satisfies_constraint(
                value=element_value,
                constraint=local_constraint,
            ):
                raise UnreachablePathError(
                    "Aynı koleksiyon elemanı için çıkarılan alias "
                    "ve döngü kısıtları çelişiyor: "
                    f"{parameter_name}[{index_value}]="
                    f"{element_value!r}"
                )

    @classmethod
    def _value_satisfies_constraint(
        cls,
        *,
        value: Any,
        constraint: _VariableConstraint,
    ) -> bool:
        """Somut bir değerin tüm değişken kısıtlarını sağlayıp sağlamadığını döndürür."""
        if constraint.has_equal_value:
            expected_value = constraint.equal_value

            if isinstance(expected_value, bool):
                if bool(value) is not expected_value:
                    return False
            elif value != expected_value:
                return False

        if value in constraint.forbidden_values:
            return False

        if (
            constraint.allowed_values is not None
            and value not in constraint.allowed_values
        ):
            return False

        for member in constraint.required_collection_members:
            if not cls._collection_contains_member(value, member):
                return False

        for member in constraint.forbidden_collection_members:
            if cls._collection_contains_member(value, member):
                return False

        if (
            constraint.minimum is not None
            or constraint.maximum is not None
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                return False

            if not cls._satisfies_minimum(
                value=value,
                constraint=constraint,
            ):
                return False

            if not cls._satisfies_maximum(
                value=value,
                constraint=constraint,
            ):
                return False

        if not cls._value_satisfies_runtime_type_constraint(
            value=value,
            constraint=constraint,
        ):
            return False

        return True

    @staticmethod
    def _collection_contains_member(collection: Any, member: Any) -> bool:
        if not isinstance(
            collection,
            (dict, list, tuple, set, frozenset),
        ):
            return False
        if isinstance(collection, (dict, set, frozenset)):
            try:
                hash(member)
            except TypeError as error:
                raise UnsupportedInputSynthesisError(
                    "Hash tabanlı koleksiyon için üyelik literal'i "
                    "hashable olmalıdır."
                ) from error
        try:
            return member in collection
        except TypeError:
            return False

    def _apply_collection_membership_constraints(
        self,
        *,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
        constraints: dict[str, _VariableConstraint],
        direct_values: dict[str, Any],
        loop_activations: tuple[_ForLoopActivation, ...],
        loop_iterable_names: set[str],
    ) -> None:
        """Seed edilmiş koleksiyonları üyelik kısıtlarına copy-on-write uyarlar."""
        for parameter_name in parameter_names:
            constraint = constraints.get(parameter_name)
            if (
                parameter_name not in direct_values
                or constraint is None
                or not (
                    constraint.required_collection_members
                    or constraint.forbidden_collection_members
                )
            ):
                continue
            direct_values[parameter_name] = (
                self._materialize_collection_membership_value(
                    parameter_name=parameter_name,
                    parameter_type=parameter_types.get(parameter_name),
                    constraint=constraint,
                    seed=direct_values[parameter_name],
                    is_loop_iterable=(parameter_name in loop_iterable_names),
                    loop_iteration_constraints=(
                        self._loop_iteration_constraints_for_iterable(
                            iterable_name=parameter_name,
                            loop_activations=loop_activations,
                        )
                    ),
                )
            )

    def _loop_iteration_constraints_for_iterable(
        self,
        *,
        iterable_name: str,
        loop_activations: tuple[_ForLoopActivation, ...],
    ) -> dict[int, _VariableConstraint]:
        constraints: dict[int, _VariableConstraint] = {}
        for activation in loop_activations:
            if activation.iterable_name != iterable_name:
                continue
            for iteration_index, constraint in (
                activation.iteration_constraints.items()
            ):
                constraints[iteration_index] = (
                    self._combine_variable_constraints(
                        current=constraints.get(
                            iteration_index,
                            _VariableConstraint(),
                        ),
                        additional=constraint,
                    )
                )
        return constraints

    @staticmethod
    def _has_runtime_type_constraint(
        constraint: _VariableConstraint | None,
    ) -> bool:
        return bool(
            constraint is not None
            and (
                constraint.required_runtime_type_groups
                or constraint.forbidden_runtime_types
            )
        )

    @staticmethod
    def _value_satisfies_runtime_type_constraint(
        *,
        value: Any,
        constraint: _VariableConstraint,
    ) -> bool:
        for required_group in constraint.required_runtime_type_groups:
            if not isinstance(
                value,
                tuple(
                    _RUNTIME_TYPE_ALLOWLIST[type_name]
                    for type_name in required_group
                ),
            ):
                return False

        if constraint.forbidden_runtime_types and isinstance(
            value,
            tuple(
                _RUNTIME_TYPE_ALLOWLIST[type_name]
                for type_name in constraint.forbidden_runtime_types
            ),
        ):
            return False

        return True

    @classmethod
    def _runtime_type_constraint_has_candidate(
        cls,
        constraint: _VariableConstraint,
    ) -> bool:
        return any(
            cls._value_satisfies_runtime_type_constraint(
                value=value,
                constraint=constraint,
            )
            for value in cls._deterministic_runtime_type_values()
        )

    def _validate_direct_values_against_constraints(
        self,
        *,
        direct_values: dict[str, Any],
        constraints: dict[str, _VariableConstraint],
    ) -> None:
        """
        Döngü veya exception çözümlemesinden gelen doğrudan
        değerlerin, aynı yürütme yolundaki koşul kısıtlarıyla
        uyumlu olduğunu doğrular.
        """
        for variable_name, direct_value in direct_values.items():
            constraint = constraints.get(variable_name)

            if constraint is None:
                continue

            if constraint.has_equal_value:
                expected_value = constraint.equal_value

                if isinstance(expected_value, bool):
                    if bool(direct_value) is not expected_value:
                        raise UnreachablePathError(
                            "Doğrudan üretilen değer, Boolean yol "
                            "kısıtıyla çelişiyor: "
                            f"{variable_name}={direct_value!r}, "
                            f"beklenen doğruluk={expected_value}"
                        )
                elif direct_value != expected_value:
                    raise UnreachablePathError(
                        "Doğrudan üretilen değer eşitlik "
                        "kısıtıyla çelişiyor: "
                        f"{variable_name}={direct_value!r}, "
                        f"beklenen={expected_value!r}"
                    )

            if direct_value in constraint.forbidden_values:
                raise UnreachablePathError(
                    "Doğrudan üretilen değer eşitsizlik "
                    "kısıtıyla yasaklanıyor: "
                    f"{variable_name}={direct_value!r}"
                )

            if (
                constraint.allowed_values is not None
                and direct_value not in constraint.allowed_values
            ):
                raise UnreachablePathError(
                    "Doğrudan üretilen değer üyelik "
                    "kısıtını sağlamıyor: "
                    f"{variable_name}={direct_value!r}"
                )

            if (
                isinstance(direct_value, (int, float))
                and not isinstance(direct_value, bool)
            ):
                if not self._satisfies_minimum(
                    value=direct_value,
                    constraint=constraint,
                ):
                    raise UnreachablePathError(
                        "Doğrudan üretilen değer minimum "
                        "kısıtını sağlamıyor: "
                        f"{variable_name}={direct_value!r}"
                    )

                if not self._satisfies_maximum(
                    value=direct_value,
                    constraint=constraint,
                ):
                    raise UnreachablePathError(
                        "Doğrudan üretilen değer maksimum "
                        "kısıtını sağlamıyor: "
                        f"{variable_name}={direct_value!r}"
                    )

            if not self._value_satisfies_runtime_type_constraint(
                value=direct_value,
                constraint=constraint,
            ):
                raise UnreachablePathError(
                    "Doğrudan üretilen değer runtime type "
                    "kısıtını sağlamıyor: "
                    f"{variable_name}={direct_value!r}"
                )

            for member in constraint.required_collection_members:
                if not self._collection_contains_member(
                    direct_value,
                    member,
                ):
                    raise UnreachablePathError(
                        "Doğrudan üretilen koleksiyon gerekli üyelik "
                        "kısıtını sağlamıyor: "
                        f"{variable_name}"
                    )

            for member in constraint.forbidden_collection_members:
                if self._collection_contains_member(
                    direct_value,
                    member,
                ):
                    raise UnreachablePathError(
                        "Doğrudan üretilen koleksiyon yasak üyelik "
                        "kısıtıyla çelişiyor: "
                        f"{variable_name}"
                    )

    def _create_parameter_value(
        self,
        parameter_name: str,
        constraint: _VariableConstraint | None,
        parameter_type: str | None = None,
    ) -> Any:
        """
        Parametre kısıtından somut bir test değeri seçer.
        """
        if constraint is None:
            return self._create_default_typed_value(
                parameter_type
            )

        if (
            constraint.required_collection_members
            or constraint.forbidden_collection_members
        ):
            return self._create_collection_membership_value(
                parameter_name=parameter_name,
                parameter_type=parameter_type,
                constraint=constraint,
            )

        if self._has_runtime_type_constraint(constraint):
            return self._create_runtime_type_value(
                parameter_name=parameter_name,
                constraint=constraint,
            )

        if constraint.has_equal_value:
            self._validate_constraint_consistency(
                constraint
            )

            if isinstance(
                constraint.equal_value,
                bool,
            ):
                return self._create_typed_boolean_value(
                    desired_value=constraint.equal_value,
                    parameter_type=parameter_type,
                )

            return constraint.equal_value

        if constraint.allowed_values is not None:
            self._validate_constraint_consistency(
                constraint
            )

            for candidate in constraint.allowed_values:
                if candidate in constraint.forbidden_values:
                    continue

                if isinstance(candidate, bool):
                    if (
                        constraint.minimum is None
                        and constraint.maximum is None
                    ):
                        return candidate
                    continue

                if isinstance(candidate, (int, float)):
                    if (
                        self._satisfies_minimum(
                            value=candidate,
                            constraint=constraint,
                        )
                        and self._satisfies_maximum(
                            value=candidate,
                            constraint=constraint,
                        )
                    ):
                        return candidate
                    continue

                if (
                    constraint.minimum is None
                    and constraint.maximum is None
                ):
                    return candidate

            raise ValueError(
                f"{parameter_name} için üyelik kısıtına uygun "
                "değer üretilemedi."
            )

        if (
            constraint.minimum is None
            and constraint.maximum is None
            and constraint.forbidden_values
            and any(
                not isinstance(value, (int, float, bool))
                for value in constraint.forbidden_values
            )
        ):
            if None in constraint.forbidden_values:
                typed_default = self._create_default_typed_value(parameter_type)
                if self._value_satisfies_constraint(
                    value=typed_default,
                    constraint=constraint,
                ):
                    return typed_default
            candidate = "__generated_value__"

            while candidate in constraint.forbidden_values:
                candidate += "_x"

            return candidate

        value = self._select_numeric_value(constraint)

        while value in constraint.forbidden_values:
            value += 1

            if not self._satisfies_maximum(
                value=value,
                constraint=constraint,
            ):
                raise ValueError(
                    f"{parameter_name} için geçerli değer "
                    "üretilemedi."
                )

        return value

    def _create_collection_membership_value(
        self,
        *,
        parameter_name: str,
        parameter_type: str | None,
        constraint: _VariableConstraint,
    ) -> Any:
        """Typed koleksiyon için literal-sol üyelik witness'ı üretir."""
        return self._materialize_collection_membership_value(
            parameter_name=parameter_name,
            parameter_type=parameter_type,
            constraint=constraint,
        )

    def _materialize_collection_membership_value(
        self,
        *,
        parameter_name: str,
        parameter_type: str | None,
        constraint: _VariableConstraint,
        seed: Any = _MISSING_COLLECTION_VALUE,
        is_loop_iterable: bool = False,
        loop_iteration_constraints: dict[
            int, _VariableConstraint
        ] | None = None,
    ) -> Any:
        """Typed koleksiyon witness'ını caller değerini mutate etmeden üretir."""
        kind, member_type, value_type = self._collection_membership_schema(
            parameter_type
        )
        constrained_members = (
            *constraint.required_collection_members,
            *constraint.forbidden_collection_members,
        )
        if kind in {"dict", "set", "frozenset"}:
            for member in constrained_members:
                try:
                    hash(member)
                except TypeError as error:
                    raise UnsupportedInputSynthesisError(
                        "Hash tabanlı koleksiyon için üyelik literal'i "
                        "hashable olmalıdır."
                    ) from error

        for member in constrained_members:
            if not self._literal_matches_primitive_type(
                member,
                member_type,
            ):
                raise UnsupportedInputSynthesisError(
                    "Koleksiyon üyelik literal'i eleman türüyle uyumlu değil."
                )

        if (
            constraint.has_equal_value
            and not isinstance(constraint.equal_value, bool)
        ):
            candidate = self._copy_typed_collection(
                value=constraint.equal_value,
                kind=kind,
                member_type=member_type,
                value_type=value_type,
            )
            if not self._value_satisfies_constraint(
                value=candidate,
                constraint=constraint,
            ):
                raise UnreachablePathError(
                    "Kesin koleksiyon değeri üyelik kısıtlarını sağlamıyor."
                )
            if is_loop_iterable:
                original_candidate = (
                    self._empty_collection(kind)
                    if seed is _MISSING_COLLECTION_VALUE
                    else self._copy_typed_collection(
                        value=seed,
                        kind=kind,
                        member_type=member_type,
                        value_type=value_type,
                    )
                )
                self._validate_loop_collection_candidate(
                    original_candidate=original_candidate,
                    candidate=candidate,
                    iteration_constraints=(
                        loop_iteration_constraints or {}
                    ),
                )
            return candidate

        candidate = (
            self._empty_collection(kind)
            if seed is _MISSING_COLLECTION_VALUE
            else self._copy_typed_collection(
                value=seed,
                kind=kind,
                member_type=member_type,
                value_type=value_type,
            )
        )
        original_candidate = self._copy_typed_collection(
            value=candidate,
            kind=kind,
            member_type=member_type,
            value_type=value_type,
        )

        if self._value_satisfies_constraint(
            value=candidate,
            constraint=constraint,
        ):
            if is_loop_iterable:
                self._validate_loop_collection_candidate(
                    original_candidate=original_candidate,
                    candidate=candidate,
                    iteration_constraints=(
                        loop_iteration_constraints or {}
                    ),
                )
            return candidate

        candidate = self._remove_collection_members(
            collection=candidate,
            kind=kind,
            members=constraint.forbidden_collection_members,
        )
        for member in constraint.required_collection_members:
            candidate = self._add_collection_member(
                collection=candidate,
                kind=kind,
                member=member,
                value_type=value_type,
            )

        desired_truthiness = (
            constraint.equal_value
            if constraint.has_equal_value
            and isinstance(constraint.equal_value, bool)
            else None
        )
        if desired_truthiness is False:
            if constraint.required_collection_members:
                raise UnreachablePathError(
                    "Falsy koleksiyon kısıtı gerekli üyelikle çelişiyor."
                )
            candidate = self._empty_collection(kind)
        elif desired_truthiness is True and not candidate:
            sentinel = self._select_collection_membership_sentinel(
                member_type=member_type,
                forbidden_members=constraint.forbidden_collection_members,
            )
            candidate = self._add_collection_member(
                collection=candidate,
                kind=kind,
                member=sentinel,
                value_type=value_type,
            )

        if is_loop_iterable:
            self._validate_loop_collection_candidate(
                original_candidate=original_candidate,
                candidate=candidate,
                iteration_constraints=loop_iteration_constraints or {},
            )

        if not self._value_satisfies_constraint(
            value=candidate,
            constraint=constraint,
        ):
            raise UnreachablePathError(
                "Typed koleksiyon üyelik kısıtlarını birlikte sağlayan "
                f"input üretilemedi: {parameter_name}"
            )
        return candidate

    @classmethod
    def _validate_loop_collection_candidate(
        cls,
        *,
        original_candidate: Any,
        candidate: Any,
        iteration_constraints: dict[int, _VariableConstraint],
    ) -> None:
        if len(candidate) != len(original_candidate):
            raise UnsupportedInputSynthesisError(
                "Bounded loop iterable üyelik kısıtı mevcut iterasyon "
                "sayısını değiştirmeden sentezlenemedi."
            )

        constrained_iterations = {
            index: constraint
            for index, constraint in iteration_constraints.items()
            if constraint != _VariableConstraint()
        }
        if not constrained_iterations:
            return
        if not isinstance(candidate, (list, tuple)):
            raise UnsupportedInputSynthesisError(
                "Sırasız loop iterable üyelik mutation'ı iterasyon "
                "kısıtlarıyla güvenli biçimde doğrulanamıyor."
            )
        for index, constraint in constrained_iterations.items():
            if (
                index >= len(candidate)
                or not cls._value_satisfies_constraint(
                    value=candidate[index],
                    constraint=constraint,
                )
            ):
                raise UnsupportedInputSynthesisError(
                    "Loop membership mutation'ı daha önce sağlanan "
                    "iterasyon kısıtını bozuyor."
                )

    @staticmethod
    def _collection_membership_kind(
        parameter_type: str | None,
    ) -> str | None:
        normalized = (parameter_type or "").replace(" ", "")
        prefixes = {
            "dict": ("dict[", "typing.Dict["),
            "list": ("list[", "typing.List["),
            "tuple": ("tuple[", "typing.Tuple["),
            "set": ("set[", "typing.Set["),
            "frozenset": ("frozenset[", "typing.FrozenSet["),
        }
        for kind, typed_prefixes in prefixes.items():
            if normalized == kind or normalized.startswith(typed_prefixes):
                return kind
        return None

    @classmethod
    def _collection_membership_schema(
        cls,
        parameter_type: str | None,
    ) -> tuple[str, str, str | None]:
        kind = cls._collection_membership_kind(parameter_type)
        if kind is None:
            raise UnsupportedInputSynthesisError(
                "Literal-sol üyelik yalnız açıkça typed güvenli "
                "koleksiyon parametrelerinde sentezlenebilir."
            )
        arguments = cls._generic_type_arguments(parameter_type)
        if kind == "dict":
            if len(arguments) != 2:
                raise UnsupportedInputSynthesisError(
                    "Dict üyelik sentezi key/value türü gerektirir."
                )
            member_type, value_type = arguments
            if not cls._is_supported_primitive_type(member_type):
                raise UnsupportedInputSynthesisError(
                    "Dict üyelik sentezi primitive key türü gerektirir."
                )
            if not cls._is_supported_primitive_type(value_type):
                raise UnsupportedInputSynthesisError(
                    "Dict üyelik sentezi primitive value türü gerektirir."
                )
            return kind, member_type, value_type
        if kind == "tuple":
            if len(arguments) != 2 or arguments[1] != "...":
                raise UnsupportedInputSynthesisError(
                    "Üyelik sentezi yalnız homojen tuple[T, ...] için desteklenir."
                )
            member_type = arguments[0]
            if not cls._is_supported_primitive_type(member_type):
                raise UnsupportedInputSynthesisError(
                    "Tuple üyelik sentezi primitive eleman türü gerektirir."
                )
            return kind, member_type, None
        if len(arguments) != 1:
            raise UnsupportedInputSynthesisError(
                "Koleksiyon üyelik sentezi tek bir eleman türü gerektirir."
            )
        member_type = arguments[0]
        if not cls._is_supported_primitive_type(member_type):
            raise UnsupportedInputSynthesisError(
                "Koleksiyon üyelik sentezi primitive eleman türü gerektirir."
            )
        return kind, member_type, None

    @staticmethod
    def _generic_type_arguments(parameter_type: str | None) -> tuple[str, ...]:
        normalized = (parameter_type or "").replace(" ", "")
        if "[" not in normalized or not normalized.endswith("]"):
            return ()
        inner = normalized.split("[", maxsplit=1)[1][:-1]
        arguments: list[str] = []
        start = 0
        depth = 0
        for index, character in enumerate(inner):
            if character in "[({":
                depth += 1
            elif character in "])}":
                depth -= 1
            elif character == "," and depth == 0:
                arguments.append(inner[start:index])
                start = index + 1
        arguments.append(inner[start:])
        return tuple(arguments)

    @staticmethod
    def _is_supported_primitive_type(type_name: str | None) -> bool:
        return type_name in {
            "int", "builtins.int",
            "float", "builtins.float",
            "str", "builtins.str",
            "bool", "builtins.bool",
        }

    @classmethod
    def _literal_matches_primitive_type(
        cls,
        value: Any,
        type_name: str,
    ) -> bool:
        if not cls._is_supported_primitive_type(type_name):
            return False
        normalized = type_name.removeprefix("builtins.")
        if normalized == "bool":
            return type(value) is bool
        if normalized == "int":
            return type(value) is int
        if normalized == "float":
            return type(value) in {int, float}
        return type(value) is str

    @staticmethod
    def _empty_collection(kind: str) -> Any:
        if kind == "dict":
            return {}
        if kind == "list":
            return []
        if kind == "tuple":
            return ()
        if kind == "set":
            return set()
        return frozenset()

    @classmethod
    def _copy_typed_collection(
        cls,
        *,
        value: Any,
        kind: str,
        member_type: str,
        value_type: str | None,
    ) -> Any:
        expected_types = {
            "dict": dict,
            "list": list,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
        }
        expected_type = expected_types[kind]
        if type(value) is not expected_type:
            raise UnsupportedInputSynthesisError(
                "Seed edilen üyelik değeri typed koleksiyon türüyle uyumlu değil."
            )

        members = value.keys() if kind == "dict" else value
        if any(
            not cls._literal_matches_primitive_type(member, member_type)
            for member in members
        ):
            raise UnsupportedInputSynthesisError(
                "Seed edilen typed koleksiyon eleman şemasıyla uyumlu değil."
            )

        if kind == "dict" and (
            value_type is None
            or any(
                not cls._literal_matches_primitive_type(item, value_type)
                for item in value.values()
            )
        ):
            raise UnsupportedInputSynthesisError(
                "Seed edilen typed dict value şemasıyla uyumlu değil."
            )
        if kind == "dict":
            return dict(value)
        if kind == "list":
            return list(value)
        if kind == "tuple":
            return tuple(value)
        if kind == "set":
            return set(value)
        return frozenset(value)

    @classmethod
    def _remove_collection_members(
        cls,
        *,
        collection: Any,
        kind: str,
        members: tuple[Any, ...],
    ) -> Any:
        if kind == "dict":
            result = dict(collection)
            for member in members:
                result.pop(member, None)
            return result
        if kind in {"list", "tuple"}:
            retained = [
                value
                for value in collection
                if not cls._contains_equal_value(members, value)
            ]
            return retained if kind == "list" else tuple(retained)
        result = set(collection)
        for member in members:
            result.discard(member)
        return result if kind == "set" else frozenset(result)

    @classmethod
    def _add_collection_member(
        cls,
        *,
        collection: Any,
        kind: str,
        member: Any,
        value_type: str | None,
    ) -> Any:
        if cls._collection_contains_member(collection, member):
            return collection
        if kind == "dict":
            result = dict(collection)
            result[member] = cls._create_default_typed_value(value_type)
            return result
        if kind == "list":
            return [*collection, member]
        if kind == "tuple":
            return (*collection, member)
        result = set(collection)
        result.add(member)
        return result if kind == "set" else frozenset(result)

    @classmethod
    def _select_collection_membership_sentinel(
        cls,
        *,
        member_type: str,
        forbidden_members: tuple[Any, ...],
    ) -> Any:
        normalized = member_type.removeprefix("builtins.")
        if normalized == "bool":
            for candidate in (False, True):
                if not cls._contains_equal_value(forbidden_members, candidate):
                    return candidate
            raise UnreachablePathError(
                "Kapalı bool üye domain'inde izin verilen değer kalmadı."
            )

        if normalized not in {"str", "float", "int"}:
            raise UnsupportedInputSynthesisError(
                "Koleksiyon truthiness witness'ı primitive eleman türü gerektirir."
            )

        attempt_count = min(
            len(forbidden_members) + 1,
            _COLLECTION_WITNESS_SEARCH_LIMIT,
        )
        for index in range(attempt_count):
            if normalized == "str":
                candidate = (
                    "__generated_member__"
                    if index == 0
                    else f"__generated_member_{index + 1}__"
                )
            else:
                if index == 0:
                    integer_candidate = 0
                else:
                    magnitude = (index + 1) // 2
                    integer_candidate = (
                        magnitude if index % 2 else -magnitude
                    )
                candidate = (
                    float(integer_candidate)
                    if normalized == "float"
                    else integer_candidate
                )
            if not cls._contains_equal_value(forbidden_members, candidate):
                return candidate

        raise UnsupportedInputSynthesisError(
            "Geniş primitive domain için güvenli üyelik witness'ı "
            "bounded arama bütçesi içinde üretilemedi."
        )

    def _create_runtime_type_value(
        self,
        *,
        parameter_name: str,
        constraint: _VariableConstraint,
    ) -> Any:
        """Type ve mevcut value constraint'lerini sağlayan deterministik değer seçer."""
        candidates: list[Any] = []

        if constraint.has_equal_value:
            candidates.append(constraint.equal_value)
        if constraint.allowed_values is not None:
            candidates.extend(constraint.allowed_values)
        if constraint.minimum is not None or constraint.maximum is not None:
            try:
                numeric = self._select_numeric_value(constraint)
            except ValueError:
                numeric = None
            if numeric is not None:
                candidates.extend((numeric, float(numeric)))

        candidates.extend(self._deterministic_runtime_type_values())

        for candidate in candidates:
            if self._value_satisfies_constraint(
                value=candidate,
                constraint=constraint,
            ):
                return candidate

        raise UnreachablePathError(
            "Runtime type ve value kısıtlarına uygun input üretilemedi: "
            f"{parameter_name}"
        )

    @staticmethod
    def _deterministic_runtime_type_values() -> tuple[Any, ...]:
        return (
            0,
            1,
            -1,
            0.0,
            1.0,
            -1.0,
            "",
            "value",
            False,
            True,
            [],
            [0],
            (),
            (0,),
            set(),
            {0},
            {},
            {"key": 0},
        )

    @staticmethod
    def _coerce_value_to_parameter_type(
        *,
        value: Any,
        parameter_type: str | None,
    ) -> Any:
        """
        Üretilen değeri fonksiyon parametresinin type hint
        ifadesiyle uyumlu koleksiyon türüne dönüştürür.

        Kısıt çözümleme sırasında koleksiyonlar kolay değiştirilebilmesi
        için geçici olarak list biçiminde tutulabilir. Bu metot,
        ``GeneratedTestInput`` oluşturulmadan önce gerçek parametre
        türünü geri yükler.
        """
        if parameter_type is None:
            return value

        normalized = parameter_type.strip().replace(
            " ",
            "",
        )

        if (
            normalized == "tuple"
            or normalized.startswith("tuple[")
            or normalized.startswith("typing.Tuple[")
        ):
            if isinstance(value, tuple):
                return value

            if isinstance(
                value,
                (list, set, frozenset),
            ):
                return tuple(value)

            raise UnreachablePathError(
                "Tuple parametresi için iterable olmayan değer "
                "üretildi: "
                f"{value!r}"
            )

        if (
            normalized == "list"
            or normalized.startswith("list[")
            or normalized.startswith("typing.List[")
        ):
            if isinstance(value, list):
                return value

            if isinstance(
                value,
                (tuple, set, frozenset),
            ):
                return list(value)

            raise UnreachablePathError(
                "List parametresi için iterable olmayan değer "
                "üretildi: "
                f"{value!r}"
            )

        if (
            normalized == "set"
            or normalized.startswith("set[")
            or normalized.startswith("typing.Set[")
        ):
            if isinstance(value, set):
                return value

            if isinstance(
                value,
                (list, tuple, frozenset),
            ):
                return set(value)

            raise UnreachablePathError(
                "Set parametresi için iterable olmayan değer "
                "üretildi: "
                f"{value!r}"
            )

        if (
            normalized == "frozenset"
            or normalized.startswith("frozenset[")
            or normalized.startswith("typing.FrozenSet[")
        ):
            if isinstance(value, frozenset):
                return value

            if isinstance(
                value,
                (list, tuple, set),
            ):
                return frozenset(value)

            raise UnreachablePathError(
                "Frozenset parametresi için iterable olmayan "
                "değer üretildi: "
                f"{value!r}"
            )

        return value

    @staticmethod
    def _create_default_typed_value(
        parameter_type: str | None,
    ) -> Any:
        """Type hint bilgisine göre güvenli varsayılan değer üretir."""
        normalized = parameter_type or ""

        if normalized in {"str", "builtins.str"}:
            return ""

        if normalized in {"float", "builtins.float"}:
            return 0.0

        if normalized in {"bool", "builtins.bool"}:
            return False

        if (
            normalized == "list"
            or normalized.startswith("list[")
            or normalized.startswith("typing.List[")
        ):
            return []

        if (
            normalized == "tuple"
            or normalized.startswith("tuple[")
            or normalized.startswith("typing.Tuple[")
        ):
            return ()

        if (
            normalized == "dict"
            or normalized.startswith("dict[")
            or normalized.startswith("typing.Dict[")
        ):
            return {}

        if (
            normalized == "set"
            or normalized.startswith("set[")
            or normalized.startswith("typing.Set[")
        ):
            return set()

        return 0

    @classmethod
    def _create_typed_boolean_value(
        cls,
        *,
        desired_value: bool,
        parameter_type: str | None,
    ) -> Any:
        """
        Bir doğruluk koşulunu parametrenin gerçek tipine uygun
        somut değere dönüştürür.
        """
        normalized = parameter_type or ""

        if (
            normalized == "list"
            or normalized.startswith("list[")
            or normalized.startswith("typing.List[")
        ):
            return [0] if desired_value else []

        if (
            normalized == "tuple"
            or normalized.startswith("tuple[")
            or normalized.startswith("typing.Tuple[")
        ):
            return (0,) if desired_value else ()

        if (
            normalized == "dict"
            or normalized.startswith("dict[")
            or normalized.startswith("typing.Dict[")
        ):
            return {"key": 0} if desired_value else {}

        if (
            normalized == "set"
            or normalized.startswith("set[")
            or normalized.startswith("typing.Set[")
        ):
            return {0} if desired_value else set()

        if normalized in {"str", "builtins.str"}:
            return "value" if desired_value else ""

        if normalized in {"int", "builtins.int"}:
            return 1 if desired_value else 0

        if normalized in {"float", "builtins.float"}:
            return 1.0 if desired_value else 0.0

        return desired_value

    @staticmethod
    def _select_numeric_value(
        constraint: _VariableConstraint,
    ) -> int | float:
        minimum = constraint.minimum
        maximum = constraint.maximum

        if minimum is not None:
            value: int | float = minimum

            if not constraint.minimum_inclusive:
                value = (
                    minimum + 1
                    if isinstance(minimum, int)
                    else minimum + 0.1
                )

        elif maximum is not None:
            value = maximum

            if not constraint.maximum_inclusive:
                value = (
                    maximum - 1
                    if isinstance(maximum, int)
                    else maximum - 0.1
                )

        else:
            value = 0

        if not PathInputGenerator._satisfies_maximum(
            value=value,
            constraint=constraint,
        ):
            raise ValueError(
                "Belirlenen kısıtlara uygun sayısal değer "
                "üretilemedi."
            )

        return value

    @staticmethod
    def _satisfies_minimum(
        value: int | float,
        constraint: _VariableConstraint,
    ) -> bool:
        """Değerin minimum kısıtını sağlayıp sağlamadığını kontrol eder."""
        if constraint.minimum is None:
            return True

        if constraint.minimum_inclusive:
            return value >= constraint.minimum

        return value > constraint.minimum

    @staticmethod
    def _satisfies_maximum(
        value: int | float,
        constraint: _VariableConstraint,
    ) -> bool:
        if constraint.maximum is None:
            return True

        if constraint.maximum_inclusive:
            return value <= constraint.maximum

        return value < constraint.maximum

    @classmethod
    def _extract_expected_result(
        cls,
        *,
        path: ExecutionPath,
        keyword_arguments: tuple[tuple[str, Any], ...],
    ) -> Any:
        """
        Yürütme yolundaki return ifadesinden beklenen sonucu çıkarır.

        Sabit return değerlerinin yanında, üretilen parametreler ve yol
        üzerindeki basit atamalar kullanılarak güvenli biçimde
        hesaplanabilen dinamik ifadeleri de destekler.
        """
        return_step = path.return_step

        if return_step is None:
            return None

        try:
            statement = ast.parse(
                return_step.node_label,
            ).body[0]
        except SyntaxError as error:
            raise UnsupportedExpectedResultError(
                return_expression=return_step.node_label,
                detail="Return ifadesi çözümlenemedi.",
            ) from error

        if not isinstance(statement, ast.Return):
            raise ValueError(
                "Return düğümü geçerli bir return ifadesi "
                "içermiyor."
            )

        if statement.value is None:
            return None

        # Sabit dönüşlerde önceki atamaları çalıştırmak gereksizdir.
        # Bu erken dönüş; exception yolundaki hatalı ifadelerin ve
        # döngü gövdesindeki henüz çözülemeyen yerel atamaların
        # beklenen sonuç hesaplanırken yeniden çalıştırılmasını engeller.
        try:
            return ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            pass

        environment: dict[str, Any] = dict(
            keyword_arguments
        )

        try:
            cls._apply_path_assignments(
                path=path,
                return_node_id=return_step.node_id,
                environment=environment,
            )

            return cls._evaluate_safe_expression(
                expression=statement.value,
                environment=environment,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise UnsupportedExpectedResultError(
                return_expression=return_step.node_label,
                detail=str(error) or type(error).__name__,
            ) from error

    @classmethod
    def _apply_path_assignments(
        cls,
        *,
        path: ExecutionPath,
        return_node_id: int,
        environment: dict[str, Any],
    ) -> None:
        """
        Return düğümünden önceki basit atamaları değerlendirme ortamına
        uygular.

        Bir ``for`` düğümü ``Iterate`` kenarından geçildiğinde döngü
        hedefi iterable'ın sıradaki değerine bağlanır. Aynı döngü
        düğümünün tekrarlanan ziyaretleri aynı iterator'ı ilerletir.
        """
        loop_iterators: dict[int, Any] = {}

        for step in path.steps:
            if step.node_id == return_node_id:
                break

            if (
                step.node_type == "for"
                and step.outgoing_edge_label == "Iterate"
            ):
                cls._bind_for_iteration_target(
                    step=step,
                    environment=environment,
                    loop_iterators=loop_iterators,
                )
                continue

            if step.node_type in {"Import", "ImportFrom"}:
                cls._apply_import_shadowing(
                    statement_text=step.node_label,
                    environment=environment,
                )
                continue

            if step.node_type not in {
                "Assign",
                "AnnAssign",
                "AugAssign",
            }:
                continue

            try:
                statement = ast.parse(
                    step.node_label
                ).body[0]
            except SyntaxError as error:
                raise ValueError(
                    "Atama ifadesi çözümlenemedi: "
                    f"{step.node_label}"
                ) from error

            cls._apply_assignment_statement(
                statement=statement,
                environment=environment,
            )

    @staticmethod
    def _apply_import_shadowing(
        *,
        statement_text: str,
        environment: dict[str, Any],
    ) -> None:
        """Import ile bağlanan isimleri safe-call shadowing ortamına ekler."""
        try:
            statement = ast.parse(statement_text).body[0]
        except (SyntaxError, IndexError) as error:
            raise ValueError(
                f"Import ifadesi çözümlenemedi: {statement_text}"
            ) from error

        if isinstance(statement, ast.Import):
            bound_names = (
                alias.asname or alias.name.split(".", maxsplit=1)[0]
                for alias in statement.names
            )
        elif isinstance(statement, ast.ImportFrom):
            bound_names = (
                alias.asname or alias.name
                for alias in statement.names
                if alias.name != "*"
            )
        else:
            raise ValueError(
                f"Geçersiz import ifadesi: {statement_text}"
            )

        for bound_name in bound_names:
            environment[bound_name] = _SHADOWED_SAFE_CALL

    @classmethod
    def _bind_for_iteration_target(
        cls,
        *,
        step: PathStep,
        environment: dict[str, Any],
        loop_iterators: dict[int, Any],
    ) -> None:
        """
        Bir for path adımının hedefini sıradaki iterable değerine bağlar.

        CFG for etiketi ``target in iterable`` biçimindedir. Iterable
        ifadesi güvenli expression değerlendiricisiyle hesaplanır ve her
        döngü düğümü için ayrı bir iterator tutulur.
        """
        try:
            expression = ast.parse(
                step.node_label,
                mode="eval",
            ).body
        except SyntaxError as error:
            raise ValueError(
                "For döngüsü ifadesi çözümlenemedi: "
                f"{step.node_label}"
            ) from error

        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], ast.In)
            and len(expression.comparators) == 1
        ):
            raise ValueError(
                "For döngüsü etiketi 'target in iterable' "
                "biçiminde olmalıdır: "
                f"{step.node_label}"
            )

        iterator = loop_iterators.get(step.node_id)

        if iterator is None:
            iterable_value = cls._evaluate_safe_expression(
                expression=expression.comparators[0],
                environment=environment,
            )

            try:
                iterator = iter(iterable_value)
            except TypeError as error:
                raise ValueError(
                    "For döngüsü kaynağı iterable değil: "
                    f"{step.node_label}"
                ) from error

            loop_iterators[step.node_id] = iterator

        try:
            iteration_value = next(iterator)
        except StopIteration as error:
            raise ValueError(
                "Execution path, iterable uzunluğundan daha fazla "
                "for iterasyonu gerektiriyor: "
                f"{step.node_label}"
            ) from error

        cls._bind_assignment_target(
            target=expression.left,
            value=iteration_value,
            environment=environment,
        )

    @classmethod
    def _bind_assignment_target(
        cls,
        *,
        target: ast.expr,
        value: Any,
        environment: dict[str, Any],
    ) -> None:
        """For hedefindeki isim veya iç içe unpacking yapısını bağlar."""
        if isinstance(target, ast.Name):
            environment[target.id] = value
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            try:
                unpacked_values = tuple(value)
            except TypeError as error:
                raise ValueError(
                    "For döngüsü unpacking değeri iterable olmalıdır."
                ) from error

            if len(target.elts) != len(unpacked_values):
                raise ValueError(
                    "For döngüsü unpacking hedefi ile değer "
                    "uzunlukları uyuşmuyor."
                )

            for child_target, child_value in zip(
                target.elts,
                unpacked_values,
                strict=True,
            ):
                cls._bind_assignment_target(
                    target=child_target,
                    value=child_value,
                    environment=environment,
                )

            return

        raise ValueError(
            "Desteklenmeyen for döngüsü hedefi: "
            f"{type(target).__name__}"
        )

    @classmethod
    def _apply_assignment_statement(
        cls,
        *,
        statement: ast.stmt,
        environment: dict[str, Any],
    ) -> None:
        """Desteklenen atama ifadesini değerlendirme ortamına uygular."""
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                raise ValueError(
                    "Çoklu atama hedefleri desteklenmiyor."
                )

            target = statement.targets[0]

            if not isinstance(target, ast.Name):
                raise ValueError(
                    "Yalnızca isim tabanlı atamalar "
                    "desteklenmektedir."
                )

            environment[target.id] = cls._evaluate_safe_expression(
                expression=statement.value,
                environment=environment,
            )
            return

        if isinstance(statement, ast.AnnAssign):
            if (
                not isinstance(statement.target, ast.Name)
                or statement.value is None
            ):
                raise ValueError(
                    "Desteklenmeyen açıklamalı atama."
                )

            environment[statement.target.id] = (
                cls._evaluate_safe_expression(
                    expression=statement.value,
                    environment=environment,
                )
            )
            return

        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise ValueError(
                    "Yalnızca isim tabanlı artırımlı atamalar "
                    "desteklenmektedir."
                )

            variable_name = statement.target.id

            if variable_name not in environment:
                raise ValueError(
                    f"Atama değişkeni tanımlı değil: {variable_name}"
                )

            right_value = cls._evaluate_safe_expression(
                expression=statement.value,
                environment=environment,
            )

            environment[variable_name] = cls._apply_binary_operator(
                operator=statement.op,
                left_value=environment[variable_name],
                right_value=right_value,
            )
            return

        raise ValueError(
            "Desteklenmeyen atama ifadesi."
        )

    @classmethod
    def _evaluate_safe_expression(
        cls,
        *,
        expression: ast.expr,
        environment: dict[str, Any],
    ) -> Any:
        """
        Yan etkisiz ve izin verilen AST düğümlerini güvenli biçimde
        değerlendirir.
        """
        if isinstance(expression, ast.Constant):
            return expression.value

        if isinstance(expression, ast.Name):
            try:
                return environment[expression.id]
            except KeyError as error:
                raise ValueError(
                    f"Tanımsız değişken: {expression.id}"
                ) from error

        if isinstance(expression, ast.List):
            return [
                cls._evaluate_safe_expression(
                    expression=element,
                    environment=environment,
                )
                for element in expression.elts
            ]

        if isinstance(expression, ast.Tuple):
            return tuple(
                cls._evaluate_safe_expression(
                    expression=element,
                    environment=environment,
                )
                for element in expression.elts
            )

        if isinstance(expression, ast.Set):
            return {
                cls._evaluate_safe_expression(
                    expression=element,
                    environment=environment,
                )
                for element in expression.elts
            }

        if isinstance(expression, ast.Dict):
            return {
                cls._evaluate_safe_expression(
                    expression=key,
                    environment=environment,
                ): cls._evaluate_safe_expression(
                    expression=value,
                    environment=environment,
                )
                for key, value in zip(
                    expression.keys,
                    expression.values,
                    strict=True,
                )
                if key is not None
            }

        if isinstance(expression, ast.UnaryOp):
            operand = cls._evaluate_safe_expression(
                expression=expression.operand,
                environment=environment,
            )

            if isinstance(expression.op, ast.UAdd):
                return +operand

            if isinstance(expression.op, ast.USub):
                return -operand

            if isinstance(expression.op, ast.Not):
                return not operand

            raise ValueError(
                "Desteklenmeyen unary operatör."
            )

        if isinstance(expression, ast.BinOp):
            left_value = cls._evaluate_safe_expression(
                expression=expression.left,
                environment=environment,
            )
            right_value = cls._evaluate_safe_expression(
                expression=expression.right,
                environment=environment,
            )

            return cls._apply_binary_operator(
                operator=expression.op,
                left_value=left_value,
                right_value=right_value,
            )

        if isinstance(expression, ast.BoolOp):
            values = [
                cls._evaluate_safe_expression(
                    expression=value,
                    environment=environment,
                )
                for value in expression.values
            ]

            if isinstance(expression.op, ast.And):
                result: Any = values[0]

                for value in values[1:]:
                    if not result:
                        return result
                    result = value

                return result

            if isinstance(expression.op, ast.Or):
                result = values[0]

                for value in values[1:]:
                    if result:
                        return result
                    result = value

                return result

            raise ValueError(
                "Desteklenmeyen Boolean operatörü."
            )

        if isinstance(expression, ast.Compare):
            return cls._evaluate_comparison_chain(
                expression=expression,
                environment=environment,
            )

        if isinstance(expression, ast.IfExp):
            condition = cls._evaluate_safe_expression(
                expression=expression.test,
                environment=environment,
            )

            selected_expression = (
                expression.body
                if condition
                else expression.orelse
            )

            return cls._evaluate_safe_expression(
                expression=selected_expression,
                environment=environment,
            )

        if isinstance(expression, ast.Subscript):
            collection = cls._evaluate_safe_expression(
                expression=expression.value,
                environment=environment,
            )
            key = cls._evaluate_safe_expression(
                expression=expression.slice,
                environment=environment,
            )
            return collection[key]

        if isinstance(expression, ast.Call):
            return cls._evaluate_safe_call(
                expression=expression,
                environment=environment,
            )

        if isinstance(expression, ast.JoinedStr):
            return "".join(
                cls._evaluate_joined_string_part(
                    part=part,
                    environment=environment,
                )
                for part in expression.values
            )

        if isinstance(expression, ast.FormattedValue):
            return cls._evaluate_formatted_value(
                expression=expression,
                environment=environment,
            )

        raise ValueError(
            "Desteklenmeyen dinamik expression türü: "
            f"{type(expression).__name__}"
        )

    @classmethod
    def _evaluate_safe_call(
        cls,
        *,
        expression: ast.Call,
        environment: dict[str, Any],
    ) -> Any:
        """Allowlist'teki yan etkisiz built-in çağrıyı değerlendirir."""
        if isinstance(expression.func, ast.Attribute):
            if expression.func.attr != "get":
                raise ValueError("Desteklenmeyen güvenli çağrı hedefi.")
            if expression.keywords or len(expression.args) not in {1, 2} or any(
                isinstance(argument, ast.Starred) for argument in expression.args
            ):
                raise ValueError(
                    "Güvenli dict.get çağrısı bir veya iki positional "
                    "argüman almalıdır."
                )
            receiver = cls._evaluate_safe_expression(
                expression=expression.func.value,
                environment=environment,
            )
            if type(receiver) is not dict:
                raise ValueError(
                    "Güvenli dict.get receiver doğrulanmış dict olmalıdır."
                )
            key = cls._evaluate_safe_expression(
                expression=expression.args[0],
                environment=environment,
            )
            default = (
                cls._evaluate_safe_expression(
                    expression=expression.args[1],
                    environment=environment,
                )
                if len(expression.args) == 2
                else None
            )
            return receiver.get(key, default)

        if not (
            isinstance(expression.func, ast.Name)
            and expression.func.id == "round"
        ):
            raise ValueError(
                "Desteklenmeyen güvenli çağrı hedefi."
            )

        if "round" in environment:
            raise ValueError(
                "Shadow edilmiş round built-in olarak çağrılamaz."
            )

        if expression.keywords:
            raise ValueError(
                "Güvenli round çağrısında keyword desteklenmiyor."
            )

        if len(expression.args) not in {1, 2}:
            raise ValueError(
                "Güvenli round çağrısı bir veya iki argüman almalıdır."
            )

        if any(isinstance(argument, ast.Starred) for argument in expression.args):
            raise ValueError(
                "Güvenli round çağrısında starred argüman desteklenmiyor."
            )

        number = cls._evaluate_safe_expression(
            expression=expression.args[0],
            environment=environment,
        )

        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(
                "Güvenli round çağrısının number argümanı sayısal olmalıdır."
            )

        if len(expression.args) == 1:
            try:
                return round(number)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "Güvenli round çağrısı hesaplanamadı."
                ) from error

        ndigits = cls._evaluate_safe_expression(
            expression=expression.args[1],
            environment=environment,
        )

        if (
            ndigits is not None
            and (
                isinstance(ndigits, bool)
                or not isinstance(ndigits, int)
            )
        ):
            raise ValueError(
                "Güvenli round çağrısının ndigits argümanı "
                "int veya None olmalıdır."
            )

        try:
            return round(number, ndigits)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Güvenli round çağrısı hesaplanamadı."
            ) from error

    @classmethod
    def _evaluate_joined_string_part(
        cls,
        *,
        part: ast.expr,
        environment: dict[str, Any],
    ) -> str:
        """F-string içindeki sabit veya formatlanmış bir parçayı üretir."""
        if isinstance(part, ast.Constant):
            if not isinstance(part.value, str):
                raise ValueError(
                    "F-string sabit parçası string olmalıdır."
                )

            return part.value

        if isinstance(part, ast.FormattedValue):
            return cls._evaluate_formatted_value(
                expression=part,
                environment=environment,
            )

        raise ValueError(
            "Desteklenmeyen f-string parçası: "
            f"{type(part).__name__}"
        )

    @classmethod
    def _evaluate_formatted_value(
        cls,
        *,
        expression: ast.FormattedValue,
        environment: dict[str, Any],
    ) -> str:
        """Yan etkisiz bir f-string değerini Python format kurallarıyla üretir."""
        value = cls._evaluate_safe_expression(
            expression=expression.value,
            environment=environment,
        )

        if expression.conversion == ord("s"):
            value = str(value)
        elif expression.conversion == ord("r"):
            value = repr(value)
        elif expression.conversion == ord("a"):
            value = ascii(value)
        elif expression.conversion != -1:
            raise ValueError(
                "Desteklenmeyen f-string conversion değeri: "
                f"{expression.conversion}"
            )

        format_spec = ""

        if expression.format_spec is not None:
            evaluated_format_spec = cls._evaluate_safe_expression(
                expression=expression.format_spec,
                environment=environment,
            )

            if not isinstance(evaluated_format_spec, str):
                raise ValueError(
                    "F-string format spec değeri string olmalıdır."
                )

            format_spec = evaluated_format_spec

        try:
            return format(value, format_spec)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "F-string değeri belirtilen formatla üretilemedi."
            ) from error

    @staticmethod
    def _apply_binary_operator(
        *,
        operator: ast.operator,
        left_value: Any,
        right_value: Any,
    ) -> Any:
        """İzin verilen yan etkisiz ikili operatörü uygular."""
        if isinstance(operator, ast.Add):
            return left_value + right_value

        if isinstance(operator, ast.Sub):
            return left_value - right_value

        if isinstance(operator, ast.Mult):
            return left_value * right_value

        if isinstance(operator, ast.Div):
            return left_value / right_value

        if isinstance(operator, ast.FloorDiv):
            return left_value // right_value

        if isinstance(operator, ast.Mod):
            return left_value % right_value

        if isinstance(operator, ast.Pow):
            return left_value ** right_value

        raise ValueError(
            "Desteklenmeyen ikili operatör."
        )

    @classmethod
    def _evaluate_comparison_chain(
        cls,
        *,
        expression: ast.Compare,
        environment: dict[str, Any],
    ) -> bool:
        """Karşılaştırma zincirini soldan sağa değerlendirir."""
        left_value = cls._evaluate_safe_expression(
            expression=expression.left,
            environment=environment,
        )

        for operator, comparator in zip(
            expression.ops,
            expression.comparators,
            strict=True,
        ):
            right_value = cls._evaluate_safe_expression(
                expression=comparator,
                environment=environment,
            )

            if not cls._compare_values(
                operator=operator,
                left_value=left_value,
                right_value=right_value,
            ):
                return False

            left_value = right_value

        return True

    @staticmethod
    def _compare_values(
        *,
        operator: ast.cmpop,
        left_value: Any,
        right_value: Any,
    ) -> bool:
        """İzin verilen karşılaştırma operatörünü uygular."""
        if isinstance(operator, ast.Eq):
            return left_value == right_value

        if isinstance(operator, ast.NotEq):
            return left_value != right_value

        if isinstance(operator, ast.Lt):
            return left_value < right_value

        if isinstance(operator, ast.LtE):
            return left_value <= right_value

        if isinstance(operator, ast.Gt):
            return left_value > right_value

        if isinstance(operator, ast.GtE):
            return left_value >= right_value

        if isinstance(operator, ast.In):
            return left_value in right_value

        if isinstance(operator, ast.NotIn):
            return left_value not in right_value

        if isinstance(operator, ast.Is):
            return left_value is right_value

        if isinstance(operator, ast.IsNot):
            return left_value is not right_value

        raise ValueError(
            "Desteklenmeyen karşılaştırma operatörü."
        )

    @staticmethod
    def _extract_expected_exception(
        path: ExecutionPath,
    ) -> str | None:
        """
        Yol üzerinde doğrudan raise düğümü varsa exception adını çıkarır.
        """
        for step in path.steps:
            if step.node_type != "Raise":
                continue

            try:
                statement = ast.parse(
                    step.node_label,
                ).body[0]
            except SyntaxError as error:
                raise ValueError(
                    "Raise ifadesi çözümlenemedi: "
                    f"{step.node_label}"
                ) from error

            if not isinstance(statement, ast.Raise):
                continue

            exception_expression = statement.exc

            if isinstance(exception_expression, ast.Call):
                exception_expression = exception_expression.func

            if isinstance(exception_expression, ast.Name):
                return exception_expression.id

            if isinstance(exception_expression, ast.Attribute):
                return exception_expression.attr

            raise ValueError(
                "Exception sınıfı belirlenemedi: "
                f"{step.node_label}"
            )

        return None

    @staticmethod
    def _negate_operator(
        operator: ast.cmpop,
    ) -> ast.cmpop:
        """
        False dalı için karşılaştırma operatörünü tersine çevirir.
        """
        operator_map: dict[type[ast.cmpop], type[ast.cmpop]] = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In,
        }

        operator_type = type(operator)

        try:
            return operator_map[operator_type]()
        except KeyError as error:
            raise ValueError(
                "Desteklenmeyen karşılaştırma operatörü."
            ) from error

    @staticmethod
    def _reverse_operator(
        operator: ast.cmpop,
    ) -> ast.cmpop:
        """
        Sabit değerin solda olduğu karşılaştırmayı normalleştirir.

        Örnek:
            ``50 <= score`` ifadesi ``score >= 50`` biçimine çevrilir.
        """
        operator_map: dict[type[ast.cmpop], type[ast.cmpop]] = {
            ast.Eq: ast.Eq,
            ast.NotEq: ast.NotEq,
            ast.Gt: ast.Lt,
            ast.GtE: ast.LtE,
            ast.Lt: ast.Gt,
            ast.LtE: ast.GtE,
        }

        operator_type = type(operator)

        try:
            return operator_map[operator_type]()
        except KeyError as error:
            raise ValueError(
                "Desteklenmeyen karşılaştırma operatörü."
            ) from error
