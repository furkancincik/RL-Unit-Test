from __future__ import annotations

import math
import random

from rl.action import Action
from rl.q_table import QTable
from rl.state_key import StateKey


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy stratejisini kullanarak aksiyon seçer.

    Epsilon olasılığıyla rastgele bir aksiyon seçilir.
    Geri kalan olasılıkta QTable içerisindeki en iyi
    aksiyonlardan biri seçilir.

    Politika, episode'lar ilerledikçe keşif oranının
    kontrollü biçimde azaltılabilmesi için epsilon güncelleme
    ve decay desteği sağlar.
    """

    __slots__ = (
        "_epsilon",
        "_random",
    )

    def __init__(
        self,
        epsilon: float,
        random_generator: random.Random | None = None,
    ) -> None:
        self._validate_epsilon(epsilon)
        self._validate_random_generator(random_generator)

        self._epsilon = float(epsilon)
        self._random = (
            random_generator
            if random_generator is not None
            else random.Random()
        )

    @property
    def epsilon(self) -> float:
        """
        Politikanın güncel keşif oranını döndürür.
        """
        return self._epsilon

    def set_epsilon(
        self,
        epsilon: float,
    ) -> float:
        """
        Politikanın keşif oranını güvenli biçimde günceller.

        Returns:
            Güncellenmiş epsilon değeri.
        """
        self._validate_epsilon(
            epsilon
        )

        self._epsilon = float(
            epsilon
        )

        return self._epsilon

    def decay_epsilon(
        self,
        decay_rate: float,
        minimum_epsilon: float = 0.0,
    ) -> float:
        """
        Epsilon değerini decay_rate ile azaltır.

        Yeni epsilon değeri minimum_epsilon değerinin altına
        düşmez.

        Args:
            decay_rate:
                Güncel epsilon ile çarpılacak decay katsayısı.
                0 ile 1 arasında olmalıdır.

            minimum_epsilon:
                Keşif oranının düşebileceği en düşük değer.
                0 ile 1 arasında olmalıdır.

        Returns:
            Güncellenmiş epsilon değeri.
        """
        self._validate_epsilon(
            decay_rate
        )
        self._validate_epsilon(
            minimum_epsilon
        )

        if minimum_epsilon > self._epsilon:
            raise ValueError(
                "minimum_epsilon cannot be greater "
                "than current epsilon."
            )

        decayed_epsilon = (
            self._epsilon
            * float(decay_rate)
        )

        self._epsilon = max(
            float(minimum_epsilon),
            decayed_epsilon,
        )

        return self._epsilon

    def select_action(
        self,
        state_key: StateKey,
        actions: tuple[Action, ...],
        q_table: QTable,
    ) -> Action:
        """
        Epsilon-greedy stratejisine göre bir aksiyon seçer.

        Explore:
            Epsilon olasılığıyla bütün aksiyonlar arasından
            rastgele seçim yapılır.

        Exploit:
            En yüksek Q değerine sahip aksiyonlardan
            biri seçilir.
        """
        self._validate_state_key(state_key)
        self._validate_actions(actions)
        self._validate_q_table(q_table)

        if not actions:
            raise ValueError(
                "actions must contain at least one Action instance."
            )

        should_explore = (
            self._random.random()
            < self._epsilon
        )

        if should_explore:
            return self._random.choice(
                actions
            )

        best_actions = q_table.best_actions(
            state_key=state_key,
            actions=actions,
        )

        return self._random.choice(
            best_actions
        )

    @staticmethod
    def _validate_epsilon(
        epsilon: float,
    ) -> None:
        if (
            isinstance(epsilon, bool)
            or not isinstance(
                epsilon,
                (int, float),
            )
        ):
            raise TypeError(
                "epsilon must be a numeric value."
            )

        if not math.isfinite(
            float(epsilon)
        ):
            raise ValueError(
                "epsilon must be a finite value."
            )

        if not (
            0.0
            <= float(epsilon)
            <= 1.0
        ):
            raise ValueError(
                "epsilon must be between 0.0 and 1.0."
            )

    @staticmethod
    def _validate_random_generator(
        random_generator: random.Random | None,
    ) -> None:
        if (
            random_generator is not None
            and not isinstance(
                random_generator,
                random.Random,
            )
        ):
            raise TypeError(
                "random_generator must be a random.Random instance."
            )

    @staticmethod
    def _validate_state_key(
        state_key: StateKey,
    ) -> None:
        if not isinstance(
            state_key,
            StateKey,
        ):
            raise TypeError(
                "state_key must be a StateKey instance."
            )

    @staticmethod
    def _validate_actions(
        actions: tuple[Action, ...],
    ) -> None:
        if not isinstance(
            actions,
            tuple,
        ):
            raise TypeError(
                "actions must be a tuple of Action instances."
            )

        for action in actions:
            if not isinstance(
                action,
                Action,
            ):
                raise TypeError(
                    "actions must contain only Action instances."
                )

    @staticmethod
    def _validate_q_table(
        q_table: QTable,
    ) -> None:
        if not isinstance(
            q_table,
            QTable,
        ):
            raise TypeError(
                "q_table must be a QTable instance."
            )