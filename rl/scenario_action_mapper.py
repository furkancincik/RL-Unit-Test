from __future__ import annotations

from collections.abc import Iterable

from generator.scenario_generator import Scenario
from rl.action import Action


class ScenarioActionMapper:
    """
    Scenario nesneleri ile RL Action nesneleri arasında
    çift yönlü eşleştirme sağlar.

    Her senaryo, iterable içindeki sırasına göre sıfır tabanlı
    bir Action.scenario_index değeriyle eşleştirilir.

    Scenario nesneleri liste, sözlük veya başka hashlenemeyen
    değerler içerebildiği için eşleştirme doğrudan Scenario
    nesnesinin hash değerine değil, benzersiz scenario_id alanına
    göre yapılır.
    """

    __slots__ = (
        "_scenarios",
        "_actions",
        "_scenario_id_to_action",
    )

    def __init__(
        self,
        scenarios: Iterable[Scenario],
    ) -> None:
        scenario_tuple = self._prepare_scenarios(
            scenarios=scenarios,
        )

        actions = tuple(
            Action(scenario_index=index)
            for index in range(len(scenario_tuple))
        )

        self._scenarios = scenario_tuple
        self._actions = actions
        self._scenario_id_to_action = {
            scenario.scenario_id: action
            for scenario, action in zip(
                scenario_tuple,
                actions,
                strict=True,
            )
        }

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        """Mapper içinde saklanan senaryoları döndürür."""
        return self._scenarios

    @property
    def actions(self) -> tuple[Action, ...]:
        """Senaryolara karşılık gelen aksiyonları döndürür."""
        return self._actions

    def get_scenario(
        self,
        action: Action,
    ) -> Scenario:
        """
        Verilen aksiyona karşılık gelen test senaryosunu döndürür.
        """
        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action instance."
            )

        if action.scenario_index >= len(self._scenarios):
            raise ValueError(
                "action does not correspond to a mapped scenario."
            )

        return self._scenarios[action.scenario_index]

    def get_action(
        self,
        scenario: Scenario,
    ) -> Action:
        """
        Verilen test senaryosuna karşılık gelen aksiyonu döndürür.

        Eşleştirme Scenario nesnesinin hash değerine göre değil,
        benzersiz scenario_id alanına göre yapılır.
        """
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "scenario must be a Scenario instance."
            )

        try:
            return self._scenario_id_to_action[
                scenario.scenario_id
            ]
        except KeyError as error:
            raise ValueError(
                "scenario is not registered in this mapper."
            ) from error

    def __len__(self) -> int:
        """Eşleştirilmiş senaryo sayısını döndürür."""
        return len(self._scenarios)

    def __contains__(
        self,
        scenario: object,
    ) -> bool:
        """Senaryonun mapper içinde bulunup bulunmadığını belirtir."""
        return (
            isinstance(scenario, Scenario)
            and scenario.scenario_id
            in self._scenario_id_to_action
        )

    @staticmethod
    def _prepare_scenarios(
        scenarios: Iterable[Scenario],
    ) -> tuple[Scenario, ...]:
        try:
            scenario_tuple = tuple(scenarios)
        except TypeError as error:
            raise TypeError(
                "scenarios must be an iterable of "
                "Scenario instances."
            ) from error

        if any(
            not isinstance(scenario, Scenario)
            for scenario in scenario_tuple
        ):
            raise TypeError(
                "scenarios must contain only "
                "Scenario instances."
            )

        scenario_ids = tuple(
            scenario.scenario_id
            for scenario in scenario_tuple
        )

        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError(
                "scenarios cannot contain duplicate "
                "scenario_id values."
            )

        return scenario_tuple