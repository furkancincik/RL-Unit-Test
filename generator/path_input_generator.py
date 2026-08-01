from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any

from cfg.path_analyzer import ExecutionPath, PathStep


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
    - değer < x
    - değer <= x
    - bool parametre kontrolleri
    - sabit return değerleri

    Bu sınıf pytest kodu üretmez. Yalnızca yürütme yolunu
    çalıştırabilecek girdileri ve beklenen sonucu hesaplar.
    """

    def generate(
        self,
        path: ExecutionPath,
        parameter_names: tuple[str, ...],
    ) -> GeneratedTestInput:
        """
        Yürütme yolundan test girdisi ve beklenen sonuç üretir.

        Args:
            path:
                CFG metadata bilgilerini içeren yürütme yolu.

            parameter_names:
                Test edilen fonksiyonun parametre adları.

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
        )

        constraints: dict[str, _VariableConstraint] = {}

        for condition_step in path.condition_steps:
            self._apply_condition_step(
                step=condition_step,
                constraints=constraints,
            )

        keyword_arguments = tuple(
            (
                parameter_name,
                self._create_parameter_value(
                    parameter_name=parameter_name,
                    constraint=constraints.get(parameter_name),
                ),
            )
            for parameter_name in parameter_names
        )

        expected_exception = self._extract_expected_exception(path)

        expected_result = (
            None
            if expected_exception is not None
            else self._extract_expected_result(path)
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

        desired_result = (
            step.outgoing_edge_label == "True"
        )

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
            and isinstance(expression.operand, ast.Name)
        ):
            self._apply_boolean_constraint(
                variable_name=expression.operand.id,
                desired_value=not desired_result,
                constraints=constraints,
            )
            return

        if not isinstance(expression, ast.Compare):
            raise ValueError(
                "Desteklenmeyen koşul ifadesi: "
                f"{step.node_label}"
            )

        if (
            len(expression.ops) != 1
            or len(expression.comparators) != 1
        ):
            raise ValueError(
                "Zincirleme karşılaştırmalar henüz "
                "desteklenmiyor: "
                f"{step.node_label}"
            )

        self._apply_comparison(
            left=expression.left,
            operator=expression.ops[0],
            right=expression.comparators[0],
            desired_result=desired_result,
            constraints=constraints,
            original_expression=step.node_label,
        )

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
        """
        if isinstance(left, ast.Name):
            variable_name = left.id
            value = self._extract_literal(right)
            normalized_operator = operator

        elif isinstance(right, ast.Name):
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
            raise ValueError(
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
                raise ValueError(
                    "Aynı değişken için çelişkili eşitlik "
                    "kısıtları bulundu."
                )

            if value in current.forbidden_values:
                raise ValueError(
                    "Eşitlik ve eşitsizlik kısıtları çelişiyor."
                )

            return replace(
                current,
                equal_value=value,
                has_equal_value=True,
            )

        if isinstance(operator, ast.NotEq):
            if (
                current.has_equal_value
                and current.equal_value == value
            ):
                raise ValueError(
                    "Eşitlik ve eşitsizlik kısıtları çelişiyor."
                )

            return replace(
                current,
                forbidden_values=(
                    *current.forbidden_values,
                    value,
                ),
            )

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

        PathInputGenerator._validate_range(result)

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

        PathInputGenerator._validate_range(result)

        return result

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
            raise ValueError(
                "Minimum ve maksimum kısıtları çelişiyor."
            )

        if (
            constraint.minimum == constraint.maximum
            and (
                not constraint.minimum_inclusive
                or not constraint.maximum_inclusive
            )
        ):
            raise ValueError(
                "Belirlenen aralık geçerli bir değer içermiyor."
            )

    def _create_parameter_value(
        self,
        parameter_name: str,
        constraint: _VariableConstraint | None,
    ) -> Any:
        """
        Parametre kısıtından somut bir test değeri seçer.
        """
        if constraint is None:
            return 0

        if constraint.has_equal_value:
            value = constraint.equal_value

            if value in constraint.forbidden_values:
                raise ValueError(
                    f"{parameter_name} için geçerli değer "
                    "üretilemedi."
                )

            return value

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
    def _satisfies_maximum(
        value: int | float,
        constraint: _VariableConstraint,
    ) -> bool:
        if constraint.maximum is None:
            return True

        if constraint.maximum_inclusive:
            return value <= constraint.maximum

        return value < constraint.maximum

    @staticmethod
    def _extract_expected_result(
        path: ExecutionPath,
    ) -> Any:
        """
        Yürütme yolundaki return ifadesinden beklenen sonucu çıkarır.
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

        try:
            return ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Dinamik return ifadeleri henüz "
                "desteklenmiyor: "
                f"{return_step.node_label}"
            ) from error

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