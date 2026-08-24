from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Sequence

from adaptive_td import (
    CONTINUE,
    STOP,
    ActionEstimate,
    AdaptiveDecision,
    build_state_features,
)


_STANDARD_NORMAL = NormalDist()


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class DistributionalControllerConfig:
    feature_dim: int = 13
    learning_rate: float = 0.02
    latency_alpha: float = 0.2
    throughput_alpha: float = 0.05
    min_output: float = 1.0
    output_prior_std: float = 3.0
    latency_prior_std_ratio: float = 0.25
    draft_latency_prior_ms: float = 6.1
    verify_latency_prior_ms: float = 13.5
    decision_rule: str = "expected_regret"
    stop_probability_threshold: float = 0.55
    stop_regret_weight: float = 1.0
    continue_regret_weight: float = 1.0
    explore_epsilon: float = 0.10
    explore_min: float = 0.01
    explore_decay: float = 0.998
    warmup_rounds: int = 20
    max_refinement_steps: int = 16
    fixed_refinement_steps: int | None = None
    force_continue: bool = False
    profile_overhead: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 < self.latency_alpha <= 1.0:
            raise ValueError("latency_alpha must be in (0, 1]")
        if not 0.0 < self.throughput_alpha <= 1.0:
            raise ValueError("throughput_alpha must be in (0, 1]")
        if self.decision_rule not in {"expected_regret", "probability"}:
            raise ValueError("decision_rule must be expected_regret or probability")
        if not 0.5 <= self.stop_probability_threshold < 1.0:
            raise ValueError("stop_probability_threshold must be in [0.5, 1)")
        if self.stop_regret_weight <= 0.0 or self.continue_regret_weight <= 0.0:
            raise ValueError("regret weights must be positive")
        if not 0.0 <= self.explore_min <= self.explore_epsilon <= 1.0:
            raise ValueError("exploration rates must satisfy 0 <= min <= epsilon <= 1")
        if not 0.0 < self.explore_decay <= 1.0:
            raise ValueError("explore_decay must be in (0, 1]")
        if self.max_refinement_steps < 1:
            raise ValueError("max_refinement_steps must be positive")


@dataclass
class _OnlineLinearDistribution:
    dimension: int
    prior_mean: float
    prior_std: float
    theta: list[float] = field(init=False)
    sample_count: int = 0
    residual_mean: float = 0.0
    residual_m2: float = 0.0

    def __post_init__(self) -> None:
        self.theta = [0.0] * self.dimension
        self.theta[0] = float(self.prior_mean)

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        mean = _dot(self.theta, features)
        if self.sample_count < 2:
            return mean, self.prior_std
        variance = max(1e-6, self.residual_m2 / (self.sample_count - 1))
        return mean, math.sqrt(variance)

    def update(self, features: Sequence[float], target: float, rate: float) -> float:
        prediction = _dot(self.theta, features)
        residual = float(target) - prediction
        clipped = _clip(residual, -64.0, 64.0)
        for index, value in enumerate(features):
            self.theta[index] += float(rate) * clipped * float(value)
        self.sample_count += 1
        delta = residual - self.residual_mean
        self.residual_mean += delta / self.sample_count
        self.residual_m2 += delta * (residual - self.residual_mean)
        return residual

    def snapshot(self) -> dict:
        variance = (
            self.prior_std ** 2
            if self.sample_count < 2
            else self.residual_m2 / (self.sample_count - 1)
        )
        return {
            "theta": list(self.theta),
            "sample_count": self.sample_count,
            "residual_mean": self.residual_mean,
            "residual_variance": variance,
        }

    def load_snapshot(self, state: dict) -> None:
        theta = [float(value) for value in state["theta"]]
        if len(theta) != self.dimension:
            raise ValueError("snapshot feature dimension does not match controller")
        self.theta = theta
        self.sample_count = int(state.get("sample_count", 0))
        self.residual_mean = float(state.get("residual_mean", 0.0))
        variance = max(0.0, float(state.get("residual_variance", self.prior_std ** 2)))
        self.residual_m2 = variance * max(0, self.sample_count - 1)


@dataclass
class _OnlineEMA:
    alpha: float
    value: float | None = None
    variance: float = 0.0
    count: int = 0

    def update(self, observation: float) -> None:
        observation = max(0.0, float(observation))
        if self.value is None:
            self.value = observation
            self.variance = 0.0
        else:
            delta = observation - self.value
            self.value += self.alpha * delta
            self.variance = (1.0 - self.alpha) * (
                self.variance + self.alpha * delta * delta
            )
        self.count += 1

    def estimate(self, prior: float, prior_std_ratio: float) -> tuple[float, float]:
        mean = float(prior if self.value is None else self.value)
        if self.count < 2:
            return mean, max(1e-6, abs(mean) * prior_std_ratio)
        return mean, math.sqrt(max(1e-6, self.variance))

    def snapshot(self) -> dict:
        return {
            "value": self.value,
            "variance": self.variance,
            "count": self.count,
        }

    def load_snapshot(self, state: dict) -> None:
        self.value = None if state.get("value") is None else float(state["value"])
        self.variance = max(0.0, float(state.get("variance", 0.0)))
        self.count = int(state.get("count", 0))


class DistributionalTimeTokenController:
    controller_name = "dist_time_token"

    def __init__(self, config: DistributionalControllerConfig) -> None:
        self.config = config
        self.stop_output = _OnlineLinearDistribution(
            config.feature_dim,
            prior_mean=2.0,
            prior_std=config.output_prior_std,
        )
        self.continue_output = _OnlineLinearDistribution(
            config.feature_dim,
            prior_mean=3.0,
            prior_std=config.output_prior_std,
        )
        self.draft_latency = _OnlineEMA(config.latency_alpha)
        self.verify_latency = _OnlineEMA(config.latency_alpha)
        self.post_verify_latency = _OnlineEMA(config.latency_alpha)
        self.stop_path_extra_latency = _OnlineEMA(config.latency_alpha)
        self.y_ema = None
        self.t_ema_ms = None
        self.completed_rounds = 0
        self.decision_count = 0
        self.exploration_count = 0
        self.rng = random.Random(config.seed)
        self.profile_samples: dict[str, list[float]] = {}

    @property
    def rho(self) -> float:
        if self.y_ema is None or self.t_ema_ms is None:
            return 0.0
        return max(0.0, self.y_ema / max(self.t_ema_ms, 1e-9))

    @property
    def early_stop_observations(self) -> int:
        return self.stop_output.sample_count

    def record_profile(self, name: str, elapsed_ms: float) -> None:
        if self.config.profile_overhead:
            self.profile_samples.setdefault(name, []).append(max(0.0, elapsed_ms))

    def build_features(self, **state) -> tuple[float, ...]:
        return build_state_features(
            max_refinement_steps=self.config.max_refinement_steps,
            **state,
        )

    @staticmethod
    def _ratio_distribution(
        numerator_mean: float,
        numerator_variance: float,
        denominator_mean: float,
        denominator_variance: float,
    ) -> tuple[float, float]:
        denominator_mean = max(1e-6, denominator_mean)
        mean = numerator_mean / denominator_mean
        variance = (
            numerator_variance / denominator_mean ** 2
            + numerator_mean ** 2 * denominator_variance / denominator_mean ** 4
        )
        return mean, math.sqrt(max(1e-9, variance))

    @staticmethod
    def _expected_positive_normal(mean: float, std: float) -> float:
        if std <= 1e-12:
            return max(0.0, mean)
        z = mean / std
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return std * pdf + mean * _STANDARD_NORMAL.cdf(z)

    def choose(
        self,
        features: Sequence[float],
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool = True,
        elapsed_draft_ms: float = 0.0,
        proposal_length: int = 1,
        frontier_length: int = 0,
        **_unused,
    ) -> AdaptiveDecision:
        started = time.perf_counter()
        proposal_length = max(1, int(proposal_length))
        max_output = float(proposal_length + 1)
        stop_mean, stop_std = self.stop_output.predict(features)
        continue_mean, continue_std = self.continue_output.predict(features)
        fallback_stop = 1.0 + max(0, int(frontier_length))
        if self.stop_output.sample_count == 0:
            stop_mean = fallback_stop
        if self.continue_output.sample_count == 0:
            next_gain = max(0.5, float(features[2]) * proposal_length)
            continue_mean = stop_mean + next_gain
        stop_mean = _clip(stop_mean, self.config.min_output, max_output)
        continue_mean = _clip(continue_mean, self.config.min_output, max_output)
        stop_std = min(max_output, max(1e-6, stop_std))
        continue_std = min(max_output, max(1e-6, continue_std))

        draft_mean, draft_std = self.draft_latency.estimate(
            self.config.draft_latency_prior_ms,
            self.config.latency_prior_std_ratio,
        )
        verify_mean, verify_std = self.verify_latency.estimate(
            max(draft_mean, self.config.verify_latency_prior_ms),
            self.config.latency_prior_std_ratio,
        )
        post_mean, post_std = self.post_verify_latency.estimate(
            0.0,
            self.config.latency_prior_std_ratio,
        )
        stop_extra_mean, stop_extra_std = self.stop_path_extra_latency.estimate(
            0.0,
            self.config.latency_prior_std_ratio,
        )
        elapsed_draft_ms = max(0.0, float(elapsed_draft_ms))
        stop_numerator = (
            elapsed_draft_ms + stop_extra_mean + verify_mean + post_mean
        )
        continue_numerator = (
            elapsed_draft_ms + draft_mean + verify_mean + post_mean
        )
        stop_numerator_variance = (
            stop_extra_std ** 2 + verify_std ** 2 + post_std ** 2
        )
        continue_numerator_variance = (
            draft_std ** 2 + verify_std ** 2 + post_std ** 2
        )
        j_stop_mean, j_stop_std = self._ratio_distribution(
            stop_numerator,
            stop_numerator_variance,
            stop_mean,
            stop_std ** 2,
        )
        j_continue_mean, j_continue_std = self._ratio_distribution(
            continue_numerator,
            continue_numerator_variance,
            continue_mean,
            continue_std ** 2,
        )
        difference_mean = j_continue_mean - j_stop_mean
        difference_std = math.sqrt(j_stop_std ** 2 + j_continue_std ** 2)
        stop_probability = (
            _STANDARD_NORMAL.cdf(difference_mean / difference_std)
            if difference_std > 1e-12
            else float(difference_mean > 0.0)
        )
        regret_continue = self._expected_positive_normal(
            difference_mean,
            difference_std,
        )
        regret_stop = self._expected_positive_normal(
            -difference_mean,
            difference_std,
        )
        weighted_stop_regret = self.config.stop_regret_weight * regret_stop
        weighted_continue_regret = (
            self.config.continue_regret_weight * regret_continue
        )
        if self.config.decision_rule == "probability":
            greedy_action = (
                STOP
                if stop_probability >= self.config.stop_probability_threshold
                else CONTINUE
            )
        else:
            greedy_action = (
                STOP
                if weighted_stop_regret < weighted_continue_regret
                else CONTINUE
            )
        reason = f"distributional_{self.config.decision_rule}_{greedy_action}"
        exploration_used = False
        selected_action_probability = 1.0
        behavior_stop_probability = float(greedy_action == STOP)
        if not allow_stop:
            action = CONTINUE
            reason = "provisional_proposal_unavailable"
        elif self.config.force_continue:
            action = CONTINUE
            reason = "force_continue"
        elif self.config.fixed_refinement_steps is not None:
            action = (
                STOP
                if refinement_step >= self.config.fixed_refinement_steps
                else CONTINUE
            )
            reason = "fixed_refinement_depth"
        elif refinement_step >= self.config.max_refinement_steps:
            action = STOP
            reason = "max_refinement_steps"
        else:
            epsilon = max(
                self.config.explore_min,
                self.config.explore_epsilon
                * self.config.explore_decay ** self.decision_count,
            )
            if allow_exploration and self.completed_rounds < self.config.warmup_rounds:
                epsilon = max(epsilon, 0.25)
            if allow_exploration and self.rng.random() < epsilon:
                action = CONTINUE if greedy_action == STOP else STOP
                reason = "distributional_exploration"
                exploration_used = True
                self.exploration_count += 1
                selected_action_probability = epsilon
            else:
                action = greedy_action
                selected_action_probability = 1.0 - epsilon if allow_exploration else 1.0
            behavior_stop_probability = (
                1.0 - epsilon if greedy_action == STOP else epsilon
            ) if allow_exploration else float(greedy_action == STOP)

        self.decision_count += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.record_profile("decision_total", latency_ms)
        diagnostics = {
            "controller_name": self.controller_name,
            "y_stop_mean": stop_mean,
            "y_stop_std": stop_std,
            "y_continue_mean": continue_mean,
            "y_continue_std": continue_std,
            "t_elapsed_ms": elapsed_draft_ms,
            "t_draft_next_mean_ms": draft_mean,
            "t_draft_next_std_ms": draft_std,
            "t_verify_mean_ms": verify_mean,
            "t_verify_std_ms": verify_std,
            "t_stop_path_extra_mean_ms": stop_extra_mean,
            "t_stop_path_extra_std_ms": stop_extra_std,
            "j_stop_mean": j_stop_mean,
            "j_stop_std": j_stop_std,
            "j_continue_mean": j_continue_mean,
            "j_continue_std": j_continue_std,
            "difference_mean": difference_mean,
            "difference_std": difference_std,
            "expected_regret_stop": weighted_stop_regret,
            "expected_regret_continue": weighted_continue_regret,
        }
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=ActionEstimate(-j_stop_mean, j_stop_std, -j_stop_mean - j_stop_std, -j_stop_mean + j_stop_std),
            continue_=ActionEstimate(-j_continue_mean, j_continue_std, -j_continue_mean - j_continue_std, -j_continue_mean + j_continue_std),
            rho_tokens_per_ms=self.rho,
            exploration_used=exploration_used,
            latency_ms=latency_ms,
            early_stop_observations=self.early_stop_observations,
            calibration_active=(self.stop_output.sample_count < 8 or self.continue_output.sample_count < 8),
            advantage_mean=difference_mean,
            advantage_risk=difference_std,
            stop_probability=stop_probability,
            behavior_stop_probability=behavior_stop_probability,
            selected_action_probability=max(1e-6, selected_action_probability),
            importance_weight=1.0,
            diagnostics=diagnostics,
        )

    def observe_continue_transition(
        self,
        features: Sequence[float],
        next_features: Sequence[float],
        forward_latency_ms: float,
        **_unused,
    ) -> None:
        self.draft_latency.update(forward_latency_ms)

    def observe_transition(
        self,
        action: str,
        features: Sequence[float],
        next_features: Sequence[float],
        forward_latency_ms: float,
        **kwargs,
    ) -> None:
        self.draft_latency.update(forward_latency_ms)

    def resolve_pending_stop(self, *args, **kwargs) -> None:
        return None

    def complete_trajectory(
        self,
        trajectory: Sequence[dict],
        *,
        emitted_tokens: int,
        verifier_latency_ms: float,
        post_verify_latency_ms: float = 0.0,
        round_latency_ms: float | None = None,
        terminal: bool = False,
    ) -> None:
        self.verify_latency.update(verifier_latency_ms)
        self.post_verify_latency.update(post_verify_latency_ms)
        if not trajectory:
            return
        factual_target = max(self.config.min_output, float(emitted_tokens))
        final_index = len(trajectory) - 1
        final_item = trajectory[final_index]
        if final_item.get("action") == STOP:
            elapsed_at_stop = float(final_item.get("t_elapsed_ms", 0.0))
            if round_latency_ms is not None:
                stop_extra = max(
                    0.0,
                    float(round_latency_ms)
                    - elapsed_at_stop
                    - float(verifier_latency_ms)
                    - float(post_verify_latency_ms),
                )
                self.stop_path_extra_latency.update(stop_extra)
            residual = self.stop_output.update(
                final_item["features"],
                factual_target,
                self.config.learning_rate,
            )
            final_item["factual_y_stop"] = factual_target
            final_item["y_stop_residual"] = residual
        if final_item.get("action") == CONTINUE:
            residual = self.continue_output.update(
                final_item["features"],
                factual_target,
                self.config.learning_rate,
            )
            final_item["factual_y_continue"] = factual_target
            final_item["y_continue_residual"] = residual
        if final_index > 0 and trajectory[final_index - 1].get("action") == CONTINUE:
            predecessor = trajectory[final_index - 1]
            residual = self.continue_output.update(
                predecessor["features"],
                factual_target,
                self.config.learning_rate,
            )
            predecessor["factual_y_continue"] = factual_target
            predecessor["y_continue_residual"] = residual

    def observe_round(self, emitted_tokens: int, round_latency_ms: float) -> None:
        alpha = self.config.throughput_alpha
        if self.y_ema is None:
            self.y_ema = float(emitted_tokens)
            self.t_ema_ms = max(1e-9, float(round_latency_ms))
        else:
            self.y_ema = (1.0 - alpha) * self.y_ema + alpha * float(emitted_tokens)
            self.t_ema_ms = (
                (1.0 - alpha) * self.t_ema_ms
                + alpha * max(1e-9, float(round_latency_ms))
            )
        self.completed_rounds += 1

    def profile_summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, values in self.profile_samples.items():
            if not values:
                continue
            ordered = sorted(values)
            result[name] = {
                "count": len(ordered),
                "mean_ms": sum(ordered) / len(ordered),
                "median_ms": ordered[len(ordered) // 2],
                "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
            }
        return result

    def snapshot(self) -> dict:
        return {
            "controller_name": self.controller_name,
            "completed_rounds": self.completed_rounds,
            "decision_count": self.decision_count,
            "exploration_count": self.exploration_count,
            "early_stop_observations": self.early_stop_observations,
            "rho_tokens_per_ms": self.rho,
            "y_ema": self.y_ema,
            "t_ema_ms": self.t_ema_ms,
            "models": {
                "stop_output": self.stop_output.snapshot(),
                "continue_output": self.continue_output.snapshot(),
            },
            "latency": {
                "draft": self.draft_latency.snapshot(),
                "verify": self.verify_latency.snapshot(),
                "post_verify": self.post_verify_latency.snapshot(),
                "stop_path_extra": self.stop_path_extra_latency.snapshot(),
            },
            "overhead": self.profile_summary(),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        models = snapshot.get("models") or {}
        self.stop_output.load_snapshot(models["stop_output"])
        self.continue_output.load_snapshot(models["continue_output"])
        latency = snapshot.get("latency") or {}
        self.draft_latency.load_snapshot(latency.get("draft") or {})
        self.verify_latency.load_snapshot(latency.get("verify") or {})
        self.post_verify_latency.load_snapshot(latency.get("post_verify") or {})
        self.stop_path_extra_latency.load_snapshot(
            latency.get("stop_path_extra") or {}
        )
        self.completed_rounds = int(snapshot.get("completed_rounds", 0))
        self.decision_count = int(snapshot.get("decision_count", 0))
        self.exploration_count = int(snapshot.get("exploration_count", 0))
        self.y_ema = snapshot.get("y_ema")
        self.t_ema_ms = snapshot.get("t_ema_ms")
