from __future__ import annotations

from rl.coverage_environment import CoverageEnvironment
from rl.environment_step import EnvironmentStep
from rl.q_learning_agent import QLearningAgent
from rl.state_encoder import StateEncoder


class QLearningTrainer:
    """
    QLearningAgent ile CoverageEnvironment arasındaki
    eğitim sürecini yönetir.

    Her adımda:
    - ortamın mevcut durumunu encode eder,
    - ajan ile aksiyon seçer,
    - aksiyonu ortamda uygular,
    - elde edilen ödülle Q-Table'ı günceller.
    """

    __slots__ = (
        "_agent",
        "_state_encoder",
    )

    def __init__(
        self,
        agent: QLearningAgent,
        state_encoder: StateEncoder,
    ) -> None:
        self._validate_agent(agent)
        self._validate_state_encoder(state_encoder)

        self._agent = agent
        self._state_encoder = state_encoder

    @property
    def agent(self) -> QLearningAgent:
        """Eğitimde kullanılan Q-Learning ajanını döndürür."""
        return self._agent

    @property
    def state_encoder(self) -> StateEncoder:
        """Coverage durumlarını dönüştüren encoder'ı döndürür."""
        return self._state_encoder

    def train_step(
        self,
        environment: CoverageEnvironment,
    ) -> EnvironmentStep:
        """
        Ortam üzerinde tek bir Q-Learning adımı gerçekleştirir.

        Returns:
            Aksiyon sonucunda ortam tarafından oluşturulan
            EnvironmentStep nesnesi.

        Raises:
            TypeError:
                environment geçerli bir CoverageEnvironment değilse.

            RuntimeError:
                Episode daha önce tamamlanmışsa.
        """
        self._validate_environment(environment)

        if environment.is_done:
            raise RuntimeError(
                "cannot train on a completed episode."
            )

        current_state_key = self._state_encoder.encode(
            state=environment.current_state,
        )

        available_actions = environment.available_actions

        selected_action = self._agent.select_action(
            state_key=current_state_key,
            actions=available_actions,
        )

        environment_step = environment.step(
            action=selected_action,
        )

        next_state_key = self._state_encoder.encode(
            state=environment_step.state,
        )

        self._agent.update(
            state_key=current_state_key,
            action=selected_action,
            reward=environment_step.reward,
            next_state_key=next_state_key,
            next_actions=environment.available_actions,
            terminal=environment_step.done,
        )

        return environment_step

    def train_episode(
        self,
        environment: CoverageEnvironment,
        reset: bool = True,
    ) -> tuple[EnvironmentStep, ...]:
        """
        Episode tamamlanana kadar ortam üzerinde eğitim yapar.

        reset=True olduğunda eğitim başlamadan önce ortam
        başlangıç durumuna döndürülür.
        """
        self._validate_environment(environment)
        self._validate_reset(reset)

        if reset:
            environment.reset()

        steps: list[EnvironmentStep] = []

        while not environment.is_done:
            environment_step = self.train_step(
                environment=environment,
            )

            steps.append(environment_step)

        return tuple(steps)

    @staticmethod
    def _validate_agent(
        agent: QLearningAgent,
    ) -> None:
        if not isinstance(agent, QLearningAgent):
            raise TypeError(
                "agent must be a QLearningAgent instance."
            )

    @staticmethod
    def _validate_state_encoder(
        state_encoder: StateEncoder,
    ) -> None:
        if not isinstance(state_encoder, StateEncoder):
            raise TypeError(
                "state_encoder must be a StateEncoder instance."
            )

    @staticmethod
    def _validate_environment(
        environment: CoverageEnvironment,
    ) -> None:
        if not isinstance(
            environment,
            CoverageEnvironment,
        ):
            raise TypeError(
                "environment must be a "
                "CoverageEnvironment instance."
            )

    @staticmethod
    def _validate_reset(
        reset: bool,
    ) -> None:
        if not isinstance(reset, bool):
            raise TypeError(
                "reset must be a bool value."
            )