from collections.abc import Callable, Iterable

from rl.action import Action
from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.reward_calculator import RewardCalculator


StateTransition = Callable[[CoverageState, Action], CoverageState]


class CoverageEnvironment:
    """
    RL ajanının coverage optimizasyonu sırasında etkileşime girdiği ortamdır.

    Ortam;
    - başlangıç ve güncel coverage durumunu tutar,
    - kullanılabilir aksiyonları yönetir,
    - seçilen aksiyonu uygular,
    - ödül değerini hesaplar,
    - episode'un tamamlanıp tamamlanmadığını belirler.
    """

    __slots__ = (
        "_initial_state",
        "_current_state",
        "_initial_actions",
        "_remaining_actions",
        "_transition_function",
        "_reward_calculator",
    )

    def __init__(
        self,
        initial_state: CoverageState,
        actions: Iterable[Action],
        transition_function: StateTransition,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:
        self._validate_initial_state(initial_state)
        self._validate_transition_function(transition_function)
        self._validate_reward_calculator(reward_calculator)

        action_tuple = self._prepare_actions(actions)

        self._initial_state = initial_state
        self._current_state = initial_state
        self._initial_actions = action_tuple
        self._remaining_actions = list(action_tuple)
        self._transition_function = transition_function
        self._reward_calculator = (
            reward_calculator
            if reward_calculator is not None
            else RewardCalculator()
        )

    @property
    def current_state(self) -> CoverageState:
        """Güncel coverage durumunu döndürür."""
        return self._current_state

    @property
    def available_actions(self) -> tuple[Action, ...]:
        """Henüz uygulanmamış aksiyonları döndürür."""
        return tuple(self._remaining_actions)

    @property
    def is_done(self) -> bool:
        """
        Episode tamamlandıysa True döndürür.

        Episode şu durumlarda tamamlanır:
        - Yüzde 100 coverage elde edilmişse,
        - Kullanılabilir aksiyon kalmamışsa.
        """
        return (
            self._current_state.is_fully_covered
            or not self._remaining_actions
        )

    def reset(self) -> CoverageState:
        """
        Ortamı başlangıç durumuna döndürür.

        Kullanılan aksiyonlar yeniden kullanılabilir hâle gelir.
        """
        self._current_state = self._initial_state
        self._remaining_actions = list(self._initial_actions)

        return self._current_state

    def step(self, action: Action) -> EnvironmentStep:
        """
        Seçilen aksiyonu uygular ve oluşan sonucu döndürür.

        Raises:
            TypeError:
                Geçersiz action verilirse veya transition fonksiyonu
                CoverageState döndürmezse.

            RuntimeError:
                Episode tamamlandıktan sonra step çağrılırsa.

            ValueError:
                Aksiyon kullanılabilir değilse.
        """
        if not isinstance(action, Action):
            raise TypeError("action must be an Action instance.")

        if self.is_done:
            raise RuntimeError("episode is already completed.")

        if action not in self._remaining_actions:
            raise ValueError("action is not available.")

        previous_state = self._current_state

        next_state = self._transition_function(
            previous_state,
            action,
        )

        if not isinstance(next_state, CoverageState):
            raise TypeError(
                "transition_function must return a "
                "CoverageState instance."
            )

        reward = self._reward_calculator.calculate(
            current_state=previous_state,
            next_state=next_state,
        )

        self._current_state = next_state
        self._remaining_actions.remove(action)

        return EnvironmentStep(
            state=next_state,
            reward=reward,
            done=self.is_done,
        )

    @staticmethod
    def _validate_initial_state(
        initial_state: CoverageState,
    ) -> None:
        if not isinstance(initial_state, CoverageState):
            raise TypeError(
                "initial_state must be a CoverageState instance."
            )

    @staticmethod
    def _validate_transition_function(
        transition_function: StateTransition,
    ) -> None:
        if not callable(transition_function):
            raise TypeError(
                "transition_function must be callable."
            )

    @staticmethod
    def _validate_reward_calculator(
        reward_calculator: RewardCalculator | None,
    ) -> None:
        if (
            reward_calculator is not None
            and not isinstance(
                reward_calculator,
                RewardCalculator,
            )
        ):
            raise TypeError(
                "reward_calculator must be a RewardCalculator instance."
            )

    @staticmethod
    def _prepare_actions(
        actions: Iterable[Action],
    ) -> tuple[Action, ...]:
        try:
            action_tuple = tuple(actions)
        except TypeError as error:
            raise TypeError(
                "actions must be an iterable of Action instances."
            ) from error

        if any(
            not isinstance(action, Action)
            for action in action_tuple
        ):
            raise TypeError(
                "actions must contain only Action instances."
            )

        if len(set(action_tuple)) != len(action_tuple):
            raise ValueError(
                "actions cannot contain duplicates."
            )

        return action_tuple