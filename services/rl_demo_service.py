from __future__ import annotations

import random
from dataclasses import dataclass

from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.state_encoder import StateEncoder
from rl.state_key import StateKey


@dataclass(frozen=True, slots=True)
class RLDemoStep:
    """
    Q-Learning demosundaki tek bir eğitim adımını temsil eder.

    Attributes:
        step_number:
            Episode içerisindeki adım numarası.

        previous_state:
            Aksiyon uygulanmadan önceki coverage durumu.

        previous_state_key:
            Önceki durumun encode edilmiş Q-Table anahtarı.

        selected_action:
            Ajan tarafından seçilen aksiyon.

        reward:
            Aksiyon sonrasında elde edilen ödül.

        next_state:
            Aksiyon sonrasında oluşan yeni coverage durumu.

        next_state_key:
            Yeni durumun encode edilmiş Q-Table anahtarı.

        old_q_value:
            Güncellemeden önceki Q değeri.

        new_q_value:
            Güncellemeden sonraki Q değeri.

        done:
            Episode'un tamamlanıp tamamlanmadığı.
    """

    step_number: int
    previous_state: CoverageState
    previous_state_key: StateKey
    selected_action: Action
    reward: float
    next_state: CoverageState
    next_state_key: StateKey
    old_q_value: float
    new_q_value: float
    done: bool


@dataclass(frozen=True, slots=True)
class RLDemoSummary:
    """
    Q-Learning eğitim demosunun toplu sonucunu temsil eder.
    """

    initial_state: CoverageState
    final_state: CoverageState
    steps: tuple[RLDemoStep, ...]

    @property
    def total_reward(self) -> float:
        """Episode boyunca elde edilen toplam ödülü döndürür."""
        return round(
            sum(step.reward for step in self.steps),
            2,
        )

    @property
    def step_count(self) -> int:
        """Gerçekleştirilen eğitim adımı sayısını döndürür."""
        return len(self.steps)

    @property
    def completed(self) -> bool:
        """Episode'un tamamlanıp tamamlanmadığını belirtir."""
        return self.final_state.is_fully_covered


class RLDemoService:
    """
    Mevcut Q-Learning altyapısını terminal üzerinde görünür hâle getirir.

    Bu servis henüz gerçek pytest ve coverage hattını kullanmaz.
    Coverage geçişleri kontrollü biçimde simüle edilir.

    Amaç aşağıdaki bileşenlerin birlikte çalışmasını göstermektir:

    CoverageState
        ↓
    StateEncoder
        ↓
    StateKey
        ↓
    EpsilonGreedyPolicy
        ↓
    QLearningAgent
        ↓
    CoverageEnvironment
        ↓
    RewardCalculator
        ↓
    QTable güncellemesi
    """

    def run(self) -> RLDemoSummary:
        """
        Tek bir kontrollü Q-Learning episode'u çalıştırır.

        Returns:
            Eğitim adımlarını ve final durumunu içeren özet.
        """
        initial_state = CoverageState(
            coverage_percentage=0.0,
            executed_tests=0,
            missing_lines=(1, 2, 3, 4, 5, 6),
            uncovered_branches=4,
        )

        actions = (
            Action(scenario_index=0),
            Action(scenario_index=1),
            Action(scenario_index=2),
        )

        q_table = QTable()

        policy = EpsilonGreedyPolicy(
            epsilon=1.0,
            random_generator=random.Random(42),
        )

        agent = QLearningAgent(
            q_table=q_table,
            policy=policy,
            learning_rate=0.5,
            discount_factor=0.9,
        )

        state_encoder = StateEncoder(
            coverage_bucket_size=10.0,
            missing_lines_bucket_size=2,
            uncovered_branches_bucket_size=1,
        )

        trainer = QLearningTrainer(
            agent=agent,
            state_encoder=state_encoder,
        )

        environment = CoverageEnvironment(
            initial_state=initial_state,
            actions=actions,
            transition_function=self._transition,
        )

        demo_steps: list[RLDemoStep] = []
        step_number = 1

        while not environment.is_done:
            previous_state = environment.current_state

            previous_state_key = state_encoder.encode(
                state=previous_state,
            )

            available_before = environment.available_actions

            old_q_values = {
                action: q_table.get_value(
                    state_key=previous_state_key,
                    action=action,
                )
                for action in available_before
            }

            environment_step = trainer.train_step(
                environment=environment,
            )

            available_after = environment.available_actions

            selected_action = self._find_selected_action(
                before=available_before,
                after=available_after,
            )

            next_state_key = state_encoder.encode(
                state=environment_step.state,
            )

            old_q_value = old_q_values[selected_action]

            new_q_value = q_table.get_value(
                state_key=previous_state_key,
                action=selected_action,
            )

            demo_steps.append(
                RLDemoStep(
                    step_number=step_number,
                    previous_state=previous_state,
                    previous_state_key=previous_state_key,
                    selected_action=selected_action,
                    reward=environment_step.reward,
                    next_state=environment_step.state,
                    next_state_key=next_state_key,
                    old_q_value=old_q_value,
                    new_q_value=new_q_value,
                    done=environment_step.done,
                )
            )

            step_number += 1

        summary = RLDemoSummary(
            initial_state=initial_state,
            final_state=environment.current_state,
            steps=tuple(demo_steps),
        )

        self._print_summary(summary)

        return summary

    @staticmethod
    def _transition(
        state: CoverageState,
        action: Action,
    ) -> CoverageState:
        """
        Demo için kontrollü coverage geçişi üretir.

        Her aksiyon farklı miktarda coverage artışı sağlar.
        Üç aksiyonun tamamı uygulandığında yüzde 100 coverage elde edilir.

        Bu geçiş gerçek coverage ölçümü değildir; mevcut RL
        altyapısının davranışını göstermek için simülasyondur.
        """
        coverage_increments = {
            0: 25.0,
            1: 35.0,
            2: 40.0,
        }

        try:
            increment = coverage_increments[
                action.scenario_index
            ]
        except KeyError as error:
            raise ValueError(
                "Demo tarafından desteklenmeyen aksiyon: "
                f"{action.scenario_index}"
            ) from error

        next_coverage = min(
            100.0,
            state.coverage_percentage + increment,
        )

        missing_lines = RLDemoService._create_missing_lines(
            coverage_percentage=next_coverage,
        )

        uncovered_branches = (
            RLDemoService._create_uncovered_branch_count(
                coverage_percentage=next_coverage,
            )
        )

        return CoverageState(
            coverage_percentage=next_coverage,
            executed_tests=state.executed_tests + 1,
            missing_lines=missing_lines,
            uncovered_branches=uncovered_branches,
        )

    @staticmethod
    def _create_missing_lines(
        coverage_percentage: float,
    ) -> tuple[int, ...]:
        """Demo coverage değerine göre eksik satırları simüle eder."""
        if coverage_percentage >= 100.0:
            return ()

        if coverage_percentage >= 75.0:
            return (6,)

        if coverage_percentage >= 50.0:
            return (5, 6)

        if coverage_percentage >= 25.0:
            return (3, 4, 5, 6)

        return (1, 2, 3, 4, 5, 6)

    @staticmethod
    def _create_uncovered_branch_count(
        coverage_percentage: float,
    ) -> int:
        """Demo coverage değerine göre eksik branch sayısını üretir."""
        if coverage_percentage >= 100.0:
            return 0

        if coverage_percentage >= 75.0:
            return 1

        if coverage_percentage >= 50.0:
            return 2

        if coverage_percentage >= 25.0:
            return 3

        return 4

    @staticmethod
    def _find_selected_action(
        before: tuple[Action, ...],
        after: tuple[Action, ...],
    ) -> Action:
        """
        Eğitim adımında ortamdan çıkarılan aksiyonu belirler.
        """
        selected_actions = tuple(
            action
            for action in before
            if action not in after
        )

        if len(selected_actions) != 1:
            raise RuntimeError(
                "Eğitim adımında seçilen aksiyon "
                "belirlenemedi."
            )

        return selected_actions[0]

    @staticmethod
    def _print_summary(
        summary: RLDemoSummary,
    ) -> None:
        """Q-Learning demo sonucunu terminale yazdırır."""
        print("=" * 65)
        print("Q-LEARNING EĞİTİM DEMOSU")
        print("=" * 65)

        print(
            "\nNOT: Bu demo kontrollü coverage geçişleri "
            "kullanan bir RL altyapı simülasyonudur."
        )
        print(
            "Gerçek pytest ve coverage entegrasyonu "
            "bir sonraki geliştirme aşamasıdır."
        )

        print("\nBAŞLANGIÇ DURUMU")
        print("-" * 65)
        RLDemoService._print_state(
            summary.initial_state
        )

        for step in summary.steps:
            print("\n" + "=" * 65)
            print(f"EĞİTİM ADIMI #{step.step_number}")
            print("=" * 65)

            print("\nÖnceki StateKey")
            print("-" * 65)
            RLDemoService._print_state_key(
                step.previous_state_key
            )

            print("\nSeçilen Aksiyon")
            print("-" * 65)
            print(
                "Scenario Index          : "
                f"{step.selected_action.scenario_index}"
            )

            print("\nÖdül ve Q Güncellemesi")
            print("-" * 65)
            print(
                f"Reward                  : "
                f"{step.reward:.2f}"
            )
            print(
                f"Eski Q Değeri           : "
                f"{step.old_q_value:.4f}"
            )
            print(
                f"Yeni Q Değeri           : "
                f"{step.new_q_value:.4f}"
            )

            print("\nYeni Coverage Durumu")
            print("-" * 65)
            RLDemoService._print_state(
                step.next_state
            )

            print("\nYeni StateKey")
            print("-" * 65)
            RLDemoService._print_state_key(
                step.next_state_key
            )

            print(
                "\nEpisode Tamamlandı mı?  : "
                f"{step.done}"
            )

        print("\n" + "=" * 65)
        print("DEMO ÖZETİ")
        print("=" * 65)
        print(
            f"Toplam Eğitim Adımı     : "
            f"{summary.step_count}"
        )
        print(
            f"Toplam Reward           : "
            f"{summary.total_reward:.2f}"
        )
        print(
            f"Final Coverage          : "
            f"%{summary.final_state.coverage_percentage:.2f}"
        )
        print(
            f"Tam Coverage            : "
            f"{summary.completed}"
        )

    @staticmethod
    def _print_state(
        state: CoverageState,
    ) -> None:
        """CoverageState bilgisini terminale yazar."""
        print(
            f"Coverage                : "
            f"%{state.coverage_percentage:.2f}"
        )
        print(
            f"Çalıştırılan Test       : "
            f"{state.executed_tests}"
        )
        print(
            f"Eksik Satır Sayısı      : "
            f"{state.missing_line_count}"
        )
        print(
            f"Eksik Branch Sayısı     : "
            f"{state.uncovered_branches}"
        )

    @staticmethod
    def _print_state_key(
        state_key: StateKey,
    ) -> None:
        """StateKey bilgisini terminale yazar."""
        print(
            f"Coverage Bucket         : "
            f"{state_key.coverage_bucket}"
        )
        print(
            f"Missing Lines Bucket    : "
            f"{state_key.missing_lines_bucket}"
        )
        print(
            f"Branch Bucket           : "
            f"{state_key.uncovered_branches_bucket}"
        )