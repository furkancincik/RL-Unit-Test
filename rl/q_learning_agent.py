from __future__ import annotations

import math

from rl.action import Action
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_table import QTable
from rl.state_key import StateKey


class QLearningAgent:
    """
    Q-Learning algoritmasını kullanarak aksiyon seçer
    ve Q değerlerini günceller.
    """

    __slots__ = (
        "_q_table",
        "_policy",
        "_learning_rate",
        "_discount_factor",
    )

    def __init__(
        self,
        q_table: QTable,
        policy: EpsilonGreedyPolicy,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        self._validate_q_table(q_table)
        self._validate_policy(policy)
        self._validate_rate(
            name="learning_rate",
            value=learning_rate,
        )
        self._validate_rate(
            name="discount_factor",
            value=discount_factor,
        )

        self._q_table = q_table
        self._policy = policy
        self._learning_rate = float(learning_rate)
        self._discount_factor = float(discount_factor)

    @property
    def q_table(self) -> QTable:
        return self._q_table

    @property
    def policy(self) -> EpsilonGreedyPolicy:
        return self._policy

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def select_action(
        self,
        state_key: StateKey,
        actions: tuple[Action, ...],
    ) -> Action:
        """
        Policy kullanarak mevcut durum için aksiyon seçer.
        """
        return self._policy.select_action(
            state_key=state_key,
            actions=actions,
            q_table=self._q_table,
        )

    def update(
        self,
        state_key: StateKey,
        action: Action,
        reward: float,
        next_state_key: StateKey,
        next_actions: tuple[Action, ...],
        terminal: bool = False,
    ) -> float:
        """
        Q-Learning formülünü kullanarak state-action
        çiftinin Q değerini günceller.

        Terminal durumda gelecekteki Q değeri hesaba katılmaz.
        """
        self._validate_state_key(
            name="state_key",
            state_key=state_key,
        )
        self._validate_action(action)
        self._validate_reward(reward)
        self._validate_state_key(
            name="next_state_key",
            state_key=next_state_key,
        )
        self._validate_actions(next_actions)
        self._validate_terminal(terminal)

        old_q_value = self._q_table.get_value(
            state_key=state_key,
            action=action,
        )

        if terminal:
            next_max_q_value = 0.0
        else:
            next_max_q_value = self._q_table.max_value(
                state_key=next_state_key,
                actions=next_actions,
            )

        target_value = (
            float(reward)
            + self._discount_factor * next_max_q_value
        )

        new_q_value = (
            old_q_value
            + self._learning_rate
            * (
                target_value
                - old_q_value
            )
        )

        self._q_table.set_value(
            state_key=state_key,
            action=action,
            value=new_q_value,
        )

        return new_q_value

    @staticmethod
    def _validate_q_table(
        q_table: QTable,
    ) -> None:
        if not isinstance(q_table, QTable):
            raise TypeError(
                "q_table must be a QTable instance."
            )

    @staticmethod
    def _validate_policy(
        policy: EpsilonGreedyPolicy,
    ) -> None:
        if not isinstance(
            policy,
            EpsilonGreedyPolicy,
        ):
            raise TypeError(
                "policy must be an EpsilonGreedyPolicy instance."
            )

    @staticmethod
    def _validate_rate(
        name: str,
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be a finite value."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

    @staticmethod
    def _validate_state_key(
        name: str,
        state_key: StateKey,
    ) -> None:
        if not isinstance(state_key, StateKey):
            raise TypeError(
                f"{name} must be a StateKey instance."
            )

    @staticmethod
    def _validate_action(
        action: Action,
    ) -> None:
        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action instance."
            )

    @staticmethod
    def _validate_reward(
        reward: float,
    ) -> None:
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
        ):
            raise TypeError(
                "reward must be a numeric value."
            )

        if not math.isfinite(float(reward)):
            raise ValueError(
                "reward must be a finite value."
            )

    @staticmethod
    def _validate_actions(
        actions: tuple[Action, ...],
    ) -> None:
        if not isinstance(actions, tuple):
            raise TypeError(
                "next_actions must be a tuple of Action instances."
            )

        for action in actions:
            if not isinstance(action, Action):
                raise TypeError(
                    "next_actions must contain only Action instances."
                )

    @staticmethod
    def _validate_terminal(
        terminal: bool,
    ) -> None:
        if not isinstance(terminal, bool):
            raise TypeError(
                "terminal must be a bool value."
            )