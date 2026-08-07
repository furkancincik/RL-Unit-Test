from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from rl.action import Action
from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.reward_calculator import RewardCalculator


StateTransition = Callable[
    [CoverageState, Action],
    CoverageState,
]

EpisodeResetCallback = Callable[[], None]


class CoverageEnvironment:
    """
    RL ajanının coverage optimizasyonu sırasında etkileşime girdiği ortamdır.

    Ortam;
    - başlangıç ve güncel coverage durumunu tutar,
    - kullanılabilir aksiyonları yönetir,
    - seçilen aksiyonu uygular,
    - ödül değerini hesaplar,
    - episode'un tamamlanıp tamamlanmadığını belirler,
    - gerektiğinde episode'a özel dış durumu sıfırlar.

    target_coverage_percentage verilirse episode,
    yüzde 100 coverage beklenmeden tanımlanan hedef
    coverage seviyesine ulaşıldığında da tamamlanabilir.

    Parametre None ise eski davranış korunur.
    """

    __slots__ = (
        "_initial_state",
        "_current_state",
        "_initial_actions",
        "_remaining_actions",
        "_transition_function",
        "_reward_calculator",
        "_episode_reset_callback",
        "_target_coverage_percentage",
    )

    def __init__(
        self,
        initial_state: CoverageState,
        actions: Iterable[Action],
        transition_function: StateTransition,
        reward_calculator: RewardCalculator | None = None,
        episode_reset_callback: EpisodeResetCallback | None = None,
        target_coverage_percentage: float | None = None,
    ) -> None:
        """
        CoverageEnvironment bağımlılıklarını ve başlangıç durumunu hazırlar.

        Args:
            initial_state:
                Episode başlangıcındaki coverage durumu.

            actions:
                Episode içerisinde kullanılabilecek aksiyonlar.

            transition_function:
                Mevcut state ve seçilen action üzerinden yeni
                CoverageState oluşturan fonksiyon.

            reward_calculator:
                State değişimine göre reward hesaplayan bileşen.

            episode_reset_callback:
                Environment reset edildiğinde episode'a özel dış
                durumları temizlemek için çalıştırılacak fonksiyon.

            target_coverage_percentage:
                Episode için ulaşılması yeterli kabul edilen
                coverage yüzdesi.

                None verilirse episode yalnızca yüzde 100 coverage
                elde edildiğinde veya kullanılabilir aksiyon
                kalmadığında tamamlanır.
        """
        self._validate_initial_state(
            initial_state
        )
        self._validate_transition_function(
            transition_function
        )
        self._validate_reward_calculator(
            reward_calculator
        )
        self._validate_episode_reset_callback(
            episode_reset_callback
        )

        normalized_target_coverage = (
            self._normalize_target_coverage_percentage(
                target_coverage_percentage
            )
        )

        action_tuple = self._prepare_actions(
            actions
        )

        self._initial_state = initial_state
        self._current_state = initial_state
        self._initial_actions = action_tuple
        self._remaining_actions = list(
            action_tuple
        )
        self._transition_function = (
            transition_function
        )
        self._reward_calculator = (
            reward_calculator
            if reward_calculator is not None
            else RewardCalculator()
        )
        self._episode_reset_callback = (
            episode_reset_callback
        )
        self._target_coverage_percentage = (
            normalized_target_coverage
        )

    @property
    def current_state(
        self,
    ) -> CoverageState:
        """
        Güncel coverage durumunu döndürür.
        """
        return self._current_state

    @property
    def available_actions(
        self,
    ) -> tuple[Action, ...]:
        """
        Henüz uygulanmamış aksiyonları döndürür.
        """
        return tuple(
            self._remaining_actions
        )

    @property
    def target_coverage_percentage(
        self,
    ) -> float | None:
        """
        Episode için tanımlanmış coverage hedefini döndürür.

        Hedef tanımlanmamışsa None döner.
        """
        return self._target_coverage_percentage

    @property
    def has_reached_target_coverage(
        self,
    ) -> bool:
        """
        Tanımlanan hedef coverage seviyesine ulaşılıp
        ulaşılmadığını belirtir.

        Hedef coverage tanımlanmamışsa False döndürülür.
        """
        if (
            self._target_coverage_percentage
            is None
        ):
            return False

        return (
            self._current_state.coverage_percentage
            >= self._target_coverage_percentage
        )

    @property
    def is_done(
        self,
    ) -> bool:
        """
        Episode tamamlandıysa True döndürür.

        Episode şu durumlarda tamamlanır:
        - Yüzde 100 coverage elde edilmişse,
        - Tanımlanan hedef coverage seviyesine ulaşılmışsa,
        - Kullanılabilir aksiyon kalmamışsa.
        """
        return (
            self._current_state.is_fully_covered
            or self.has_reached_target_coverage
            or not self._remaining_actions
        )

    def reset(
        self,
    ) -> CoverageState:
        """
        Ortamı başlangıç durumuna döndürür.

        Kullanılan aksiyonlar yeniden kullanılabilir hâle gelir.

        Episode'a özel dış durum bulunuyorsa reset callback'i
        çalıştırılarak ilgili geçiş bileşeni temizlenir.

        target_coverage_percentage reset sırasında korunur.

        Returns:
            Ortamın başlangıç CoverageState nesnesi.
        """
        if (
            self._episode_reset_callback
            is not None
        ):
            self._episode_reset_callback()

        self._current_state = (
            self._initial_state
        )
        self._remaining_actions = list(
            self._initial_actions
        )

        return self._current_state

    def step(
        self,
        action: Action,
    ) -> EnvironmentStep:
        """
        Seçilen aksiyonu uygular ve oluşan sonucu döndürür.

        Args:
            action:
                RL ajanı tarafından seçilen aksiyon.

        Returns:
            Yeni state, reward ve episode tamamlanma
            bilgisini içeren EnvironmentStep nesnesi.

        Raises:
            TypeError:
                Geçersiz action verilirse veya transition
                fonksiyonu CoverageState döndürmezse.

            RuntimeError:
                Episode tamamlandıktan sonra step çağrılırsa.

            ValueError:
                Aksiyon kullanılabilir değilse.
        """
        if not isinstance(
            action,
            Action,
        ):
            raise TypeError(
                "action must be an Action instance."
            )

        if self.is_done:
            raise RuntimeError(
                "episode is already completed."
            )

        if (
            action
            not in self._remaining_actions
        ):
            raise ValueError(
                "action is not available."
            )

        previous_state = (
            self._current_state
        )

        next_state = (
            self._transition_function(
                previous_state,
                action,
            )
        )

        if not isinstance(
            next_state,
            CoverageState,
        ):
            raise TypeError(
                "transition_function must return a "
                "CoverageState instance."
            )

        reward = (
            self._reward_calculator.calculate(
                current_state=previous_state,
                next_state=next_state,
            )
        )

        self._current_state = (
            next_state
        )

        self._remaining_actions.remove(
            action
        )

        return EnvironmentStep(
            state=next_state,
            reward=reward,
            done=self.is_done,
        )

    @staticmethod
    def _validate_initial_state(
        initial_state: CoverageState,
    ) -> None:
        """
        Başlangıç coverage durumunu doğrular.
        """
        if not isinstance(
            initial_state,
            CoverageState,
        ):
            raise TypeError(
                "initial_state must be a "
                "CoverageState instance."
            )

    @staticmethod
    def _validate_transition_function(
        transition_function: StateTransition,
    ) -> None:
        """
        Transition fonksiyonunu doğrular.
        """
        if not callable(
            transition_function
        ):
            raise TypeError(
                "transition_function must be callable."
            )

    @staticmethod
    def _validate_reward_calculator(
        reward_calculator: RewardCalculator | None,
    ) -> None:
        """
        RewardCalculator bağımlılığını doğrular.
        """
        if (
            reward_calculator is not None
            and not isinstance(
                reward_calculator,
                RewardCalculator,
            )
        ):
            raise TypeError(
                "reward_calculator must be a "
                "RewardCalculator instance."
            )

    @staticmethod
    def _validate_episode_reset_callback(
        callback: EpisodeResetCallback | None,
    ) -> None:
        """
        Episode reset callback değerini doğrular.
        """
        if (
            callback is not None
            and not callable(
                callback
            )
        ):
            raise TypeError(
                "episode_reset_callback callable "
                "veya None olmalıdır."
            )

    @staticmethod
    def _normalize_target_coverage_percentage(
        value: float | None,
    ) -> float | None:
        """
        Opsiyonel episode coverage hedefini doğrular.

        Returns:
            Normalize edilmiş float coverage değeri
            veya hedef tanımlanmamışsa None.
        """
        if value is None:
            return None

        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                "target_coverage_percentage sayısal "
                "bir değer veya None olmalıdır."
            )

        normalized_value = float(
            value
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
                "target_coverage_percentage "
                "0'dan büyük ve 100'den "
                "küçük veya eşit olmalıdır."
            )

        return normalized_value

    @staticmethod
    def _prepare_actions(
        actions: Iterable[Action],
    ) -> tuple[Action, ...]:
        """
        Action koleksiyonunu doğrular ve tuple'a dönüştürür.
        """
        try:
            action_tuple = tuple(
                actions
            )
        except TypeError as error:
            raise TypeError(
                "actions must be an iterable "
                "of Action instances."
            ) from error

        if any(
            not isinstance(
                action,
                Action,
            )
            for action in action_tuple
        ):
            raise TypeError(
                "actions must contain only "
                "Action instances."
            )

        if (
            len(
                set(
                    action_tuple
                )
            )
            != len(
                action_tuple
            )
        ):
            raise ValueError(
                "actions cannot contain duplicates."
            )

        return action_tuple