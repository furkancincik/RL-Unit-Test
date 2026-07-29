from __future__ import annotations

import math

from rl.action import Action
from rl.state_key import StateKey


class QTable:
    """
    Q-Learning ajanının öğrendiği state-action değerlerini saklar.

    İç veri yapısı:

        StateKey
            └── Action -> Q değeri

    Bilinmeyen state-action çiftleri için varsayılan değer 0.0'dır.
    """

    def __init__(self) -> None:
        self._values: dict[StateKey, dict[Action, float]] = {}

    def get_value(
        self,
        state_key: StateKey,
        action: Action,
    ) -> float:
        """
        Belirtilen state-action çiftinin Q değerini döndürür.

        Çift daha önce kaydedilmemişse 0.0 döndürülür.
        """
        self._validate_state_key(state_key)
        self._validate_action(action)

        state_values = self._values.get(state_key)

        if state_values is None:
            return 0.0

        return state_values.get(action, 0.0)

    def set_value(
        self,
        state_key: StateKey,
        action: Action,
        value: float,
    ) -> None:
        """
        Belirtilen state-action çiftinin Q değerini kaydeder
        veya mevcut değeri günceller.
        """
        self._validate_state_key(state_key)
        self._validate_action(action)
        self._validate_value(value)

        state_values = self._values.setdefault(
            state_key,
            {},
        )

        state_values[action] = float(value)

    def contains_state(
        self,
        state_key: StateKey,
    ) -> bool:
        """
        State'in Q-Table içerisinde bulunup bulunmadığını döndürür.
        """
        self._validate_state_key(state_key)

        return state_key in self._values


    def max_value(
        self,
        state_key: StateKey,
        actions: tuple[Action, ...],
    ) -> float:
        """
        Verilen state için mevcut aksiyonlar arasındaki
        en yüksek Q değerini döndürür.

        Aksiyon koleksiyonu boşsa 0.0 döndürülür.
        """
        self._validate_state_key(state_key)
        self._validate_actions(actions)

        if not actions:
            return 0.0

        return max(
            self.get_value(state_key, action)
            for action in actions
        )

    def best_actions(
        self,
        state_key: StateKey,
        actions: tuple[Action, ...],
    ) -> tuple[Action, ...]:
        """
        Verilen state için en yüksek Q değerine sahip
        bütün aksiyonları döndürür.

        Birden fazla aksiyon aynı en yüksek Q değerine sahipse
        hepsi sonuçta yer alır.
        """
        self._validate_state_key(state_key)
        self._validate_actions(actions)

        if not actions:
            return ()

        highest_value = self.max_value(
            state_key=state_key,
            actions=actions,
        )

        return tuple(
            action
            for action in actions
            if self.get_value(state_key, action) == highest_value
        )




    def clear(self) -> None:
        """
        Q-Table içerisindeki bütün öğrenilmiş değerleri siler.
        """
        self._values.clear()

    def __len__(self) -> int:
        """
        Q-Table içerisindeki state sayısını döndürür.
        """
        return len(self._values)

    @staticmethod
    def _validate_state_key(
        state_key: StateKey,
    ) -> None:
        if not isinstance(state_key, StateKey):
            raise TypeError(
                "state_key must be a StateKey instance."
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
    def _validate_value(
        value: float,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(
                "value must be a finite number."
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                "value must be a finite number."
            )

        if not math.isfinite(value):
            raise ValueError(
                "value must be finite."
            )


    @staticmethod
    def _validate_actions(
        actions: tuple[Action, ...],
    ) -> None:
        if not isinstance(actions, tuple):
            raise TypeError(
                "actions must be a tuple of Action instances."
            )

        for action in actions:
            if not isinstance(action, Action):
                raise TypeError(
                    "actions must contain only Action instances."
                )