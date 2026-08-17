
import ast
from dataclasses import dataclass, field, replace
from typing import Any

from cfg.path_analyzer import ExecutionPath, PathStep


class UnreachablePathError(ValueError):
    """Bir yürütme yolundaki kısıtlar çelişkili olduğunda oluşur."""


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

        direct_values = self._initialize_candidate_values(
            candidate_values=candidate_values,
            parameter_names=parameter_names,
        )

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

        self._apply_loop_variable_constraints(
            parameter_types=normalized_parameter_types,
            loop_activations=loop_activations,
            direct_values=direct_values,
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

        keyword_arguments = tuple(
            (
                parameter_name,
                self._coerce_value_to_parameter_type(
                    value=(
                        direct_values[parameter_name]
                        if parameter_name in direct_values
                        else self._create_parameter_value(
                            parameter_name=parameter_name,
                            constraint=constraints.get(
                                parameter_name
                            ),
                            parameter_type=(
                                normalized_parameter_types.get(
                                    parameter_name
                                )
                            ),
                        )
                    ),
                    parameter_type=(
                        normalized_parameter_types.get(
                            parameter_name
                        )
                    ),
                ),
            )
            for parameter_name in parameter_names
        )

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

            exception_name = self._extract_handler_exception_name(
                except_step
            )

            self._apply_exception_source_input(
                source_step=step,
                exception_name=exception_name,
                parameter_names=parameter_names,
                direct_values=direct_values,
            )

    @staticmethod
    def _extract_handler_exception_name(
        except_step: PathStep,
    ) -> str | None:
        """Except düğüm etiketinden exception sınıfının adını çıkarır."""
        normalized_label = except_step.node_label.strip()

        if normalized_label == "except":
            return None

        prefix = "except "

        if not normalized_label.startswith(prefix):
            raise ValueError(
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
            raise ValueError(
                "Except exception türü çözümlenemedi: "
                f"{except_step.node_label}"
            ) from error

        if isinstance(expression, ast.Name):
            return expression.id

        if isinstance(expression, ast.Attribute):
            return expression.attr

        raise ValueError(
            "Desteklenmeyen except exception türü: "
            f"{except_step.node_label}"
        )

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
            raise ValueError(
                "KeyError sözlüğü doğrudan bir fonksiyon "
                "parametresi olmalıdır."
            )

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

        update_delta = self._extract_loop_update_delta(
            path=path,
            variable_name=variable_name,
        )

        if iteration_count > 0 and update_delta is None:
            raise ValueError(
                "While döngüsü için desteklenen bir değişken "
                "güncellemesi bulunamadı."
            )

        if variable_name in parameter_names:
            candidate = self._find_while_initial_value(
                expression=condition_expression,
                variable_name=variable_name,
                iteration_count=iteration_count,
                update_delta=update_delta or 0,
            )

            direct_values[variable_name] = candidate
            return

        local_initial_value = (
            self._extract_local_loop_initial_value(
                path=path,
                loop_step=step,
                variable_name=variable_name,
            )
        )

        if local_initial_value is None:
            raise ValueError(
                "While koşulundaki yerel değişken için döngüden "
                "önce desteklenen bir başlangıç ataması bulunamadı: "
                f"{variable_name}"
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

    @staticmethod
    def _extract_local_loop_initial_value(
        *,
        path: ExecutionPath,
        loop_step: PathStep,
        variable_name: str,
    ) -> int | float | None:
        """
        Döngüden önceki son sabit yerel değişken atamasını döndürür.
        """
        initial_value: int | float | None = None

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

            try:
                value = ast.literal_eval(
                    value_expression
                )
            except (ValueError, TypeError):
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue

            initial_value = value

        return initial_value

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
        )

    def _apply_expression(
        self,
        *,
        expression: ast.expr,
        desired_result: bool,
        constraints: dict[str, _VariableConstraint],
        original_expression: str,
    ) -> None:
        """
        Bir koşul AST ifadesini istenen Boolean sonuca göre uygular.
        """
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
            )
            return

        if isinstance(expression, ast.BoolOp):
            self._apply_boolean_operation(
                expression=expression,
                desired_result=desired_result,
                constraints=constraints,
                original_expression=original_expression,
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

        raise ValueError(
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
        if (
            isinstance(left, ast.Name)
            and isinstance(right, ast.Name)
        ):
            return

        if isinstance(left, ast.Name):
            variable_name = left.id
            value = self._extract_literal(right)
            normalized_operator = operator

        elif isinstance(right, ast.Name):
            if isinstance(operator, (ast.In, ast.NotIn)):
                raise ValueError(
                    "Üyelik karşılaştırmasında değişken sol tarafta "
                    "olmalıdır: "
                    f"{original_expression}"
                )

            variable_name = right.id
            value = self._extract_literal(left)
            normalized_operator = self._reverse_operator(operator)

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
                    continue

                if (
                    step.outgoing_edge_label == "Complete"
                    and matching_index is not None
                ):
                    del active_loops[matching_index:]

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
            )

        return constraints, loop_activations

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

        self._apply_condition_step(
            step=step,
            constraints=routed_constraints,
        )

        for variable_name, constraint in routed_constraints.items():
            activation = self._find_active_loop_for_target(
                active_loops=active_loops,
                target_name=variable_name,
            )

            if activation is None:
                constraints[variable_name] = constraint
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

    def _apply_loop_variable_constraints(
        self,
        *,
        parameter_types: dict[str, str],
        loop_activations: list[_ForLoopActivation],
        direct_values: dict[str, Any],
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
                raise UnreachablePathError(
                    "For döngüsü iterable değeri indekslenebilir "
                    "bir koleksiyon olmalıdır: "
                    f"{iterable_name}={existing_value!r}"
                )

            element_type = self._extract_element_type(
                parameter_types.get(iterable_name)
            )

            element_value = self._create_parameter_value(
                parameter_name=(
                    f"{iterable_name}[{iteration_index}]"
                ),
                constraint=loop_constraint,
                parameter_type=element_type,
            )

            if iteration_index >= len(iterable_values):
                raise UnreachablePathError(
                    "For döngüsü iterasyon indeksi üretilen "
                    "koleksiyonda bulunmuyor: "
                    f"{iterable_name}[{iteration_index}]"
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

        return True

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
            raise ValueError(
                "Return ifadesi çözümlenemedi: "
                f"{return_step.node_label}"
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

        cls._apply_path_assignments(
            path=path,
            return_node_id=return_step.node_id,
            environment=environment,
        )

        try:
            return cls._evaluate_safe_expression(
                expression=statement.value,
                environment=environment,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(
                "Dinamik return ifadesi güvenli biçimde "
                "hesaplanamadı: "
                f"{return_step.node_label}"
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
