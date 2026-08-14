from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathConstraint,
    PathFeasibilityResult,
)


@dataclass(frozen=True, slots=True)
class InputCandidateValue:
    """
    Tek bir değişken için üretilen aday test değerini temsil eder.
    """

    variable_name: str
    value: Any
    source: str


@dataclass(frozen=True, slots=True)
class TestInputCandidate:
    """
    Bir FEASIBLE path için üretilen aday test girdilerini temsil eder.
    """

    values: tuple[InputCandidateValue, ...]

    @property
    def value_dict(self) -> dict[str, Any]:
        return {
            item.variable_name: item.value
            for item in self.values
        }

    @property
    def path_input_value_dict(
        self,
    ) -> dict[str, Any]:
        """
        PathInputGenerator'a doğrudan aktarılabilecek somut
        aday değerleri döndürür.

        ``truthy`` ve ``falsy`` constraint'lerinden üretilen
        ``True`` / ``False`` değerleri, gerçek parametre tipi
        bilinmeden oluşturulan sembolik doğruluk gereksinimleridir.
        Bu placeholder değerler doğrudan aktarılmaz.

        PathInputGenerator, path koşulunu ve parameter type
        bilgisini birlikte kullanarak list, tuple, string veya
        diğer parametre türleri için uygun somut değeri üretir.

        ``literal_exact`` ile üretilmiş gerçek Boolean değerler
        ve diğer somut candidate kaynakları korunur.
        """
        return {
            item.variable_name: item.value
            for item in self.values
            if item.source
            not in {
                "literal_truthy",
                "literal_falsy",
            }
        }


class InputCandidateGenerator:
    """
    PathFeasibilityAnalyzer çıktısını somut test input adaylarına dönüştürür.

    Bu sınıf path'i tekrar parse etmez.

    Kaynaklar:
    - literal constraint
    - relational witness
    """

    def generate(
        self,
        *,
        feasibility_result: PathFeasibilityResult,
        relational_witness: dict[str, float] | None = None,
    ) -> TestInputCandidate:
        """
        FEASIBLE feasibility sonucundan test input adayı üretir.
        """
        self._validate_feasibility_result(
            feasibility_result
        )

        if (
            feasibility_result.status
            != FeasibilityStatus.FEASIBLE
        ):
            raise ValueError(
                "Yalnızca FEASIBLE path'ler için "
                "input candidate üretilebilir."
            )

        candidate_values: dict[
            str,
            InputCandidateValue,
        ] = {}

        for constraint in (
            feasibility_result.constraints
        ):
            generated = (
                self._generate_from_constraint(
                    constraint
                )
            )

            if generated is None:
                continue

            existing = candidate_values.get(
                generated.variable_name
            )

            if existing is None:
                candidate_values[
                    generated.variable_name
                ] = generated
                continue

            candidate_values[
                generated.variable_name
            ] = self._merge_candidate_values(
                existing=existing,
                new=generated,
            )

        if relational_witness is not None:
            self._apply_relational_witness(
                candidate_values=candidate_values,
                relational_witness=relational_witness,
            )

        return TestInputCandidate(
            values=tuple(
                candidate_values.values()
            )
        )

    @staticmethod
    def _validate_feasibility_result(
        feasibility_result: PathFeasibilityResult,
    ) -> None:
        if not isinstance(
            feasibility_result,
            PathFeasibilityResult,
        ):
            raise TypeError(
                "feasibility_result bir "
                "PathFeasibilityResult olmalıdır."
            )

    def _generate_from_constraint(
        self,
        constraint: PathConstraint,
    ) -> InputCandidateValue | None:
        operator = constraint.operator
        value = constraint.value

        if operator == "==":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=value,
                source="literal_exact",
            )

        if operator == "!=":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=self._create_not_equal_value(
                    value
                ),
                source="literal_not_equal",
            )

        if operator == ">=":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=value,
                source="literal_lower_bound",
            )

        if operator == ">":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=self._next_numeric_value(
                    value
                ),
                source="literal_strict_lower_bound",
            )

        if operator == "<=":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=value,
                source="literal_upper_bound",
            )

        if operator == "<":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=self._previous_numeric_value(
                    value
                ),
                source="literal_strict_upper_bound",
            )

        if operator == "in":
            if (
                not isinstance(value, tuple)
                or not value
            ):
                return None

            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=value[0],
                source="literal_membership",
            )

        if operator == "not in":
            if not isinstance(value, tuple):
                return None

            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=self._create_not_in_value(
                    value
                ),
                source="literal_not_membership",
            )

        if operator == "truthy":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=True,
                source="literal_truthy",
            )

        if operator == "falsy":
            return InputCandidateValue(
                variable_name=(
                    constraint.variable_name
                ),
                value=False,
                source="literal_falsy",
            )

        return None

    def _merge_candidate_values(
        self,
        *,
        existing: InputCandidateValue,
        new: InputCandidateValue,
    ) -> InputCandidateValue:
        """
        Aynı değişken için birden fazla constraint olduğunda aday
        değerleri constraint yönünü dikkate alarak birleştirir.

        Kurallar:
        - exact değer her zaman önceliklidir.
        - birden fazla lower-bound için daha güçlü olan büyük değer seçilir.
        - birden fazla upper-bound için daha güçlü olan küçük değer seçilir.
        - lower + upper birlikteyse ortak aralıkta kalan aday tercih edilir.
        """
        if existing.source == "literal_exact":
            return existing

        if new.source == "literal_exact":
            return new

        if not (
            self._is_numeric(existing.value)
            and self._is_numeric(new.value)
        ):
            return existing

        existing_is_lower = self._is_lower_bound_source(
            existing.source
        )
        new_is_lower = self._is_lower_bound_source(
            new.source
        )

        existing_is_upper = self._is_upper_bound_source(
            existing.source
        )
        new_is_upper = self._is_upper_bound_source(
            new.source
        )

        if existing_is_lower and new_is_lower:
            merged_value = max(
                existing.value,
                new.value,
            )

            return InputCandidateValue(
                variable_name=existing.variable_name,
                value=merged_value,
                source="merged_lower_bound",
            )

        if existing_is_upper and new_is_upper:
            merged_value = min(
                existing.value,
                new.value,
            )

            return InputCandidateValue(
                variable_name=existing.variable_name,
                value=merged_value,
                source="merged_upper_bound",
            )

        if (
            (existing_is_lower and new_is_upper)
            or (existing_is_upper and new_is_lower)
        ):
            lower_candidate = (
                existing.value
                if existing_is_lower
                else new.value
            )

            upper_candidate = (
                existing.value
                if existing_is_upper
                else new.value
            )

            if lower_candidate > upper_candidate:
                # Feasibility katmanı normalde çelişkili aralığı
                # FEASIBLE olarak iletmemelidir. Yine de burada
                # geçersiz bir candidate üretmek yerine mevcut değeri
                # koruyoruz; PathInputGenerator son doğrulamayı yapar.
                return existing

            merged_value = max(
                existing.value,
                new.value,
            )

            return InputCandidateValue(
                variable_name=existing.variable_name,
                value=merged_value,
                source="merged_numeric_range",
            )

        return existing

    @staticmethod
    def _is_lower_bound_source(
        source: str,
    ) -> bool:
        """
        Candidate kaynağının alt sınır constraint'inden gelip
        gelmediğini döndürür.
        """
        return source in {
            "literal_lower_bound",
            "literal_strict_lower_bound",
            "merged_lower_bound",
            "merged_numeric_range",
        }

    @staticmethod
    def _is_upper_bound_source(
        source: str,
    ) -> bool:
        """
        Candidate kaynağının üst sınır constraint'inden gelip
        gelmediğini döndürür.
        """
        return source in {
            "literal_upper_bound",
            "literal_strict_upper_bound",
            "merged_upper_bound",
            "merged_numeric_range",
        }

    def _apply_relational_witness(
        self,
        *,
        candidate_values: dict[
            str,
            InputCandidateValue,
        ],
        relational_witness: dict[
            str,
            float,
        ],
    ) -> None:
        for (
            variable_name,
            value,
        ) in relational_witness.items():
            candidate_values[
                variable_name
            ] = InputCandidateValue(
                variable_name=variable_name,
                value=value,
                source="relational_witness",
            )

    @staticmethod
    def _create_not_equal_value(
        value: Any,
    ) -> Any:
        if isinstance(value, bool):
            return not value

        if isinstance(
            value,
            (int, float),
        ):
            return value + 1

        if isinstance(value, str):
            return (
                "__candidate__"
                if value != "__candidate__"
                else "__candidate_2__"
            )

        return None

    @staticmethod
    def _create_not_in_value(
        values: tuple[Any, ...],
    ) -> Any:
        numeric_values = {
            item
            for item in values
            if isinstance(
                item,
                (int, float),
            )
            and not isinstance(
                item,
                bool,
            )
        }

        if numeric_values:
            candidate = 0

            while candidate in numeric_values:
                candidate += 1

            return candidate

        string_values = {
            item
            for item in values
            if isinstance(item, str)
        }

        candidate = "__candidate__"

        while candidate in string_values:
            candidate += "_x"

        return candidate

    @staticmethod
    def _next_numeric_value(
        value: Any,
    ) -> int | float:
        if not InputCandidateGenerator._is_numeric(
            value
        ):
            raise TypeError(
                "Strict numeric alt sınır "
                "sayısal olmalıdır."
            )

        return value + 1

    @staticmethod
    def _previous_numeric_value(
        value: Any,
    ) -> int | float:
        if not InputCandidateGenerator._is_numeric(
            value
        ):
            raise TypeError(
                "Strict numeric üst sınır "
                "sayısal olmalıdır."
            )

        return value - 1

    @staticmethod
    def _is_numeric(
        value: Any,
    ) -> bool:
        return (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        )
