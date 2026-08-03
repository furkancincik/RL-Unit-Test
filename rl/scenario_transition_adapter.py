from __future__ import annotations

from collections.abc import Callable

from generator.scenario_generator import Scenario
from rl.action import Action
from rl.coverage_state import CoverageState
from rl.scenario_action_mapper import ScenarioActionMapper


ScenarioTransition = Callable[
    [CoverageState, Scenario],
    CoverageState,
]


class ScenarioTransitionAdapter:
    """
    RL Action nesnelerini gerçek Scenario nesnelerine dönüştürerek
    senaryo tabanlı durum geçişini çalıştırır.

    Bu sınıf, CoverageEnvironment ile test senaryosu katmanı arasında
    adaptör görevi görür.

    Akış:

        CoverageState + Action
                  ↓
        ScenarioActionMapper
                  ↓
        CoverageState + Scenario
                  ↓
        ScenarioTransition
                  ↓
        Yeni CoverageState
    """

    __slots__ = (
        "_mapper",
        "_transition_function",
    )

    def __init__(
        self,
        mapper: ScenarioActionMapper,
        transition_function: ScenarioTransition,
    ) -> None:
        """
        Adaptörün bağımlılıklarını hazırlar.

        Args:
            mapper:
                Action ve Scenario nesneleri arasındaki eşlemeyi
                yöneten bileşen.

            transition_function:
                Mevcut coverage durumu ve seçilen Scenario üzerinden
                yeni CoverageState oluşturan fonksiyon.
        """
        self._validate_mapper(mapper)
        self._validate_transition_function(
            transition_function
        )

        self._mapper = mapper
        self._transition_function = transition_function

    @property
    def mapper(self) -> ScenarioActionMapper:
        """Kullanılan ScenarioActionMapper nesnesini döndürür."""
        return self._mapper

    def __call__(
        self,
        state: CoverageState,
        action: Action,
    ) -> CoverageState:
        """
        Action nesnesini Scenario nesnesine dönüştürür ve senaryo
        tabanlı geçiş fonksiyonunu çalıştırır.

        Args:
            state:
                Aksiyon uygulanmadan önceki coverage durumu.

            action:
                RL ajanı tarafından seçilen aksiyon.

        Returns:
            Seçilen senaryonun uygulanması sonucunda oluşan yeni
            CoverageState.

        Raises:
            TypeError:
                state veya action geçersiz türdeyse ya da geçiş
                fonksiyonu CoverageState döndürmezse.

            ValueError:
                Action mapper içerisindeki bir senaryoya karşılık
                gelmiyorsa.
        """
        self._validate_state(state)
        self._validate_action(action)

        scenario = self._mapper.get_scenario(
            action=action,
        )

        next_state = self._transition_function(
            state,
            scenario,
        )

        if not isinstance(next_state, CoverageState):
            raise TypeError(
                "transition_function must return a "
                "CoverageState instance."
            )

        return next_state

    @staticmethod
    def _validate_mapper(
        mapper: ScenarioActionMapper,
    ) -> None:
        if not isinstance(
            mapper,
            ScenarioActionMapper,
        ):
            raise TypeError(
                "mapper must be a "
                "ScenarioActionMapper instance."
            )

    @staticmethod
    def _validate_transition_function(
        transition_function: ScenarioTransition,
    ) -> None:
        if not callable(transition_function):
            raise TypeError(
                "transition_function must be callable."
            )

    @staticmethod
    def _validate_state(
        state: CoverageState,
    ) -> None:
        if not isinstance(state, CoverageState):
            raise TypeError(
                "state must be a CoverageState instance."
            )

    @staticmethod
    def _validate_action(
        action: Action,
    ) -> None:
        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action instance."
            )