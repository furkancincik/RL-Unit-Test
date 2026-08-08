from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from generator.scenario_generator import Scenario
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)


CoverageMeasurement = Callable[
    [tuple[Scenario, ...]],
    CoverageResult | FunctionCoverageResult,
]


@dataclass(frozen=True, slots=True)
class GreedyScenarioSelectionStep:
    """
    Greedy minimizasyon sırasında gerçekleştirilen
    tek bir scenario seçim adımını temsil eder.

    Attributes:
        step_number:
            Seçimin kaçıncı greedy adımı olduğu.

        scenario:
            Bu adımda seçilen scenario.

        coverage_before:
            Scenario eklenmeden önceki coverage değeri.

        coverage_after:
            Scenario eklendikten sonraki coverage değeri.

        marginal_gain:
            Scenario'nun mevcut test paketine sağladığı
            ek coverage katkısı.
    """

    step_number: int
    scenario: Scenario
    coverage_before: float
    coverage_after: float
    marginal_gain: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_number, bool)
            or not isinstance(self.step_number, int)
        ):
            raise TypeError(
                "step_number bir tam sayı olmalıdır."
            )

        if self.step_number < 1:
            raise ValueError(
                "step_number 1 veya daha büyük olmalıdır."
            )

        if not isinstance(
            self.scenario,
            Scenario,
        ):
            raise TypeError(
                "scenario bir Scenario örneği olmalıdır."
            )

        self._validate_percentage(
            "coverage_before",
            self.coverage_before,
        )
        self._validate_percentage(
            "coverage_after",
            self.coverage_after,
        )

        self._validate_finite_number(
            "marginal_gain",
            self.marginal_gain,
        )

    @staticmethod
    def _validate_finite_number(
        name: str,
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                f"{name} sayısal olmalıdır."
            )

        if not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f"{name} sonlu olmalıdır."
            )

    @classmethod
    def _validate_percentage(
        cls,
        name: str,
        value: float,
    ) -> None:
        cls._validate_finite_number(
            name,
            value,
        )

        if not (
            0.0
            <= float(value)
            <= 100.0
        ):
            raise ValueError(
                f"{name} 0 ile 100 arasında olmalıdır."
            )


@dataclass(frozen=True, slots=True)
class GreedyScenarioMinimizationResult:
    """
    Greedy scenario minimizasyon işleminin sonucunu temsil eder.
    """

    selected_scenarios: tuple[Scenario, ...]
    steps: tuple[GreedyScenarioSelectionStep, ...]
    target_coverage_percentage: float
    final_coverage_percentage: float

    @property
    def selected_scenario_count(self) -> int:
        """
        Greedy algoritmanın seçtiği scenario sayısını döndürür.
        """
        return len(
            self.selected_scenarios
        )

    @property
    def target_reached(self) -> bool:
        """
        Hedef coverage seviyesine ulaşılıp ulaşılmadığını döndürür.
        """
        return (
            self.final_coverage_percentage
            >= self.target_coverage_percentage
        )


class GreedyScenarioMinimizer:
    """
    Mevcut scenario havuzundan yüksek coverage sağlayan
    küçük bir alt küme seçmek için greedy baseline uygular.

    Bu sınıf RL algoritmasının yerine geçmez.

    Amaç, RL test seçiminin karşılaştırılabileceği
    deterministik bir referans sonuç üretmektir.
    """

    __slots__ = (
        "_coverage_measurement",
    )

    def __init__(
        self,
        coverage_measurement: CoverageMeasurement,
    ) -> None:
        if not callable(
            coverage_measurement
        ):
            raise TypeError(
                "coverage_measurement callable olmalıdır."
            )

        self._coverage_measurement = (
            coverage_measurement
        )

    def minimize(
        self,
        scenarios: tuple[Scenario, ...],
        target_coverage_percentage: float,
    ) -> GreedyScenarioMinimizationResult:
        """
        Greedy marginal coverage yaklaşımı ile scenario seçer.

        Her turda, mevcut seçilmiş scenario paketine eklendiğinde
        en yüksek coverage değerini sağlayan scenario seçilir.

        Hedef coverage'a ulaşıldığında veya seçilecek scenario
        kalmadığında işlem sonlandırılır.
        """
        self._validate_scenarios(
            scenarios
        )
        self._validate_target_coverage(
            target_coverage_percentage
        )

        selected: list[Scenario] = []
        remaining = list(
            scenarios
        )

        steps: list[
            GreedyScenarioSelectionStep
        ] = []

        current_coverage = 0.0

        while (
            remaining
            and current_coverage
            < target_coverage_percentage
        ):
            best_scenario: Scenario | None = None
            best_coverage = current_coverage
            best_gain = float("-inf")

            for candidate in remaining:
                candidate_suite = tuple(
                    [
                        *selected,
                        candidate,
                    ]
                )

                coverage_result = (
                    self._coverage_measurement(
                        candidate_suite
                    )
                )

                candidate_coverage = (
                    self._extract_line_coverage(
                        coverage_result
                    )
                )

                marginal_gain = (
                    candidate_coverage
                    - current_coverage
                )

                if (
                    marginal_gain
                    > best_gain
                ):
                    best_scenario = candidate
                    best_coverage = (
                        candidate_coverage
                    )
                    best_gain = (
                        marginal_gain
                    )

            if best_scenario is None:
                break

            if best_gain <= 0.0:
                break

            coverage_before = (
                current_coverage
            )

            selected.append(
                best_scenario
            )
            remaining.remove(
                best_scenario
            )

            current_coverage = (
                best_coverage
            )

            steps.append(
                GreedyScenarioSelectionStep(
                    step_number=len(steps) + 1,
                    scenario=best_scenario,
                    coverage_before=coverage_before,
                    coverage_after=current_coverage,
                    marginal_gain=best_gain,
                )
            )

        return GreedyScenarioMinimizationResult(
            selected_scenarios=tuple(
                selected
            ),
            steps=tuple(
                steps
            ),
            target_coverage_percentage=float(
                target_coverage_percentage
            ),
            final_coverage_percentage=float(
                current_coverage
            ),
        )

    @staticmethod
    def _extract_line_coverage(
        coverage_result: (
            CoverageResult
            | FunctionCoverageResult
        ),
    ) -> float:
        """
        Coverage sonucundan karşılaştırmada kullanılacak
        line coverage değerini çıkarır.
        """
        if isinstance(
            coverage_result,
            FunctionCoverageResult,
        ):
            return float(
                coverage_result.line_coverage_percent
            )

        if isinstance(
            coverage_result,
            CoverageResult,
        ):
            return float(
                coverage_result.line_coverage_percent
            )

        raise TypeError(
            "coverage_measurement CoverageResult veya "
            "FunctionCoverageResult döndürmelidir."
        )

    @staticmethod
    def _validate_scenarios(
        scenarios: tuple[Scenario, ...],
    ) -> None:
        if not isinstance(
            scenarios,
            tuple,
        ):
            raise TypeError(
                "scenarios bir Scenario tuple'ı olmalıdır."
            )

        if not scenarios:
            raise ValueError(
                "scenarios boş olamaz."
            )

        if any(
            not isinstance(
                scenario,
                Scenario,
            )
            for scenario in scenarios
        ):
            raise TypeError(
                "scenarios yalnızca Scenario "
                "nesneleri içermelidir."
            )

        if (
            len(set(scenarios))
            != len(scenarios)
        ):
            raise ValueError(
                "scenarios tekrar eden eleman "
                "içeremez."
            )

    @staticmethod
    def _validate_target_coverage(
        target_coverage_percentage: float,
    ) -> None:
        if (
            isinstance(
                target_coverage_percentage,
                bool,
            )
            or not isinstance(
                target_coverage_percentage,
                (int, float),
            )
        ):
            raise TypeError(
                "target_coverage_percentage "
                "sayısal olmalıdır."
            )

        normalized_value = float(
            target_coverage_percentage
        )

        if not math.isfinite(
            normalized_value
        ):
            raise ValueError(
                "target_coverage_percentage "
                "sonlu olmalıdır."
            )

        if not (
            0.0
            < normalized_value
            <= 100.0
        ):
            raise ValueError(
                "target_coverage_percentage 0'dan "
                "büyük ve 100'den küçük veya "
                "eşit olmalıdır."
            )
