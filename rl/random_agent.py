import random
from collections.abc import Sequence

from rl.action import Action


class RandomAgent:
    """
    Kullanılabilir aksiyonlar arasından rastgele seçim yapan ajan.

    Bu ajan öğrenme gerçekleştirmez. RL ortamının ve aksiyon seçme
    mekanizmasının doğrulanması için temel karşılaştırma ajanı olarak
    kullanılır.
    """

    __slots__ = ("_random_generator",)

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            if isinstance(seed, bool):
                raise TypeError("seed must be an integer or None.")

            if not isinstance(seed, int):
                raise TypeError("seed must be an integer or None.")

        self._random_generator = random.Random(seed)

    def select_action(
        self,
        available_actions: Sequence[Action],
    ) -> Action:
        """
        Kullanılabilir aksiyonlardan rastgele birini seçer.

        Args:
            available_actions:
                Ajanın seçebileceği Action nesneleri.

        Returns:
            Rastgele seçilen Action nesnesi.

        Raises:
            TypeError:
                available_actions geçerli bir sequence değilse veya
                Action dışında değer içeriyorsa.

            ValueError:
                Kullanılabilir aksiyon bulunmuyorsa.
        """
        if isinstance(available_actions, (str, bytes)):
            raise TypeError(
                "available_actions must be a sequence of Action instances."
            )

        if not isinstance(available_actions, Sequence):
            raise TypeError(
                "available_actions must be a sequence of Action instances."
            )

        if any(
            not isinstance(action, Action)
            for action in available_actions
        ):
            raise TypeError(
                "available_actions must contain only Action instances."
            )

        if not available_actions:
            raise ValueError("available_actions cannot be empty.")

        return self._random_generator.choice(available_actions)