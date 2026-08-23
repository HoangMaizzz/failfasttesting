from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence


STOP = "stop"
CONTINUE = "continue"
_STANDARD_NORMAL = NormalDist()


def _clip(value: float, low: float, high: float) -> float:
    value = float(value)
    if math.isnan(value):
        return low
    if value == math.inf:
        return high
    if value == -math.inf:
        return low
    return max(low, min(high, value))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class AdaptiveTDConfig:
    feature_dim: int = 13
    learning_rate: float = 0.02
    mc_learning_rate: float = 0.01
    mc_mix: float = 0.5
    update_mode: str = "mixed"
    rho_alpha: float = 0.05
    risk_beta: float = 1.0
    stop_probability_threshold: float = 0.75
    q_margin: float = 0.0
    uncertainty_prior: float = 1.0
    epistemic_scale: float = 0.1
    residual_prior_variance: float = 1.0
    explore_epsilon: float = 0.10
    explore_min: float = 0.01
    explore_decay: float = 0.998
    warmup_rounds: int = 20
    early_stop_min_observations: int = 32
    max_refinement_steps: int = 16
    fixed_refinement_steps: int | None = None
    force_continue: bool = False
    profile_overhead: bool = False
    seed: int = 42
    td_error_clip: float = 32.0

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.learning_rate < 0.0 or self.mc_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative")
        if self.update_mode not in {"td", "factual_return", "mixed"}:
            raise ValueError("update_mode must be td, factual_return, or mixed")
        if not 0.0 < self.rho_alpha <= 1.0:
            raise ValueError("rho_alpha must be in (0, 1]")
        if self.max_refinement_steps < 1:
            raise ValueError("max_refinement_steps must be positive")
        if self.fixed_refinement_steps is not None and self.fixed_refinement_steps < 1:
            raise ValueError("fixed_refinement_steps must be positive")
        if (
            self.fixed_refinement_steps is not None
            and self.fixed_refinement_steps > self.max_refinement_steps
        ):
            raise ValueError("fixed_refinement_steps cannot exceed max_refinement_steps")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative")
        if self.early_stop_min_observations < 0:
            raise ValueError("early_stop_min_observations must be non-negative")
        if not 0.0 <= self.explore_min <= self.explore_epsilon <= 1.0:
            raise ValueError("exploration rates must satisfy 0 <= min <= epsilon <= 1")
        if not 0.0 < self.explore_decay <= 1.0:
            raise ValueError("explore_decay must be in (0, 1]")
        if not 0.0 <= self.mc_mix <= 1.0:
            raise ValueError("mc_mix must be in [0, 1]")
        if self.risk_beta < 0.0 or self.epistemic_scale < 0.0:
            raise ValueError("risk parameters must be non-negative")
        if not 0.5 < self.stop_probability_threshold < 1.0:
            raise ValueError("stop_probability_threshold must be in (0.5, 1)")
        if self.uncertainty_prior <= 0.0:
            raise ValueError("uncertainty_prior must be positive")
        if self.residual_prior_variance < 0.0:
            raise ValueError("residual_prior_variance must be non-negative")
        if self.td_error_clip <= 0.0:
            raise ValueError("td_error_clip must be positive")


@dataclass(frozen=True)
class ActionEstimate:
    mean: float
    risk: float
    lower: float
    upper: float


@dataclass(frozen=True)
class AdaptiveDecision:
    action: str
    reason: str
    stop: ActionEstimate
    continue_: ActionEstimate
    rho_tokens_per_ms: float
    exploration_used: bool
    latency_ms: float
    early_stop_observations: int
    calibration_active: bool
    advantage_mean: float
    advantage_risk: float
    stop_probability: float


class _LinearActionValue:
    def __init__(self, dimension: int, config: AdaptiveTDConfig) -> None:
        self.theta = [0.0] * dimension
        self.precision = [0.0] * dimension
        inverse_prior = 1.0 / config.uncertainty_prior
        self.inverse_covariance = [
            [inverse_prior if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]
        self.sample_count = 0
        self.residual_mean = 0.0
        self.residual_m2 = 0.0
        self.config = config

    def mean(self, features: Sequence[float]) -> float:
        return _dot(self.theta, features)

    def residual_variance(self) -> float:
        if self.sample_count < 2:
            return self.config.residual_prior_variance
        return max(
            self.config.residual_prior_variance * 1e-3,
            self.residual_m2 / (self.sample_count - 1),
        )

    def risk(self, features: Sequence[float]) -> float:
        projected = [
            sum(weight * value for weight, value in zip(row, features))
            for row in self.inverse_covariance
        ]
        leverage = max(0.0, _dot(features, projected))
        return math.sqrt(
            max(
                0.0,
                self.residual_variance()
                * self.config.epistemic_scale
                * leverage,
            )
        )

    def estimate(
        self,
        features: Sequence[float],
        mean: float | None = None,
        risk: float | None = None,
    ) -> ActionEstimate:
        mean = self.mean(features) if mean is None else float(mean)
        risk = self.risk(features) if risk is None else float(risk)
        width = self.config.risk_beta * risk
        return ActionEstimate(mean, risk, mean - width, mean + width)

    def update(
        self,
        features: Sequence[float],
        target: float,
        rate: float,
        *,
        count_observation: bool = True,
    ) -> float:
        prediction = self.mean(features)
        raw_residual = float(target) - prediction
        update_residual = _clip(
            raw_residual,
            -self.config.td_error_clip,
            self.config.td_error_clip,
        )
        for index, value in enumerate(features):
            self.theta[index] += float(rate) * update_residual * value
            if count_observation:
                self.precision[index] += value * value
        if count_observation:
            projected = [
                sum(weight * value for weight, value in zip(row, features))
                for row in self.inverse_covariance
            ]
            denominator = max(1e-12, 1.0 + _dot(features, projected))
            for row in range(len(self.inverse_covariance)):
                for column in range(len(self.inverse_covariance[row])):
                    self.inverse_covariance[row][column] -= (
                        projected[row] * projected[column] / denominator
                    )
            self.sample_count += 1
            delta = raw_residual - self.residual_mean
            self.residual_mean += delta / self.sample_count
            self.residual_m2 += delta * (raw_residual - self.residual_mean)
        return raw_residual


class OnlineTDRefinementController:
    def __init__(self, config: AdaptiveTDConfig) -> None:
        self.config = config
        self.values = {
            STOP: _LinearActionValue(config.feature_dim, config),
            CONTINUE: _LinearActionValue(config.feature_dim, config),
        }
        self.early_stop_uncertainty = _LinearActionValue(
            config.feature_dim,
            config,
        )
        self.stop_z_threshold = _STANDARD_NORMAL.inv_cdf(
            config.stop_probability_threshold
        )
        self.rng = random.Random(config.seed)
        self.y_ema = None
        self.t_ema_ms = None
        self.completed_rounds = 0
        self.decision_count = 0
        self.exploration_count = 0
        self.early_stop_observations = 0
        self.forward_latency_ema_ms = None
        self.profile_samples: dict[str, list[float]] = {}

    @property
    def rho(self) -> float:
        if self.y_ema is None or self.t_ema_ms is None:
            return 0.0
        return max(0.0, self.y_ema / max(self.t_ema_ms, 1e-9))

    def record_profile(self, name: str, elapsed_ms: float) -> None:
        if self.config.profile_overhead:
            self.profile_samples.setdefault(name, []).append(max(0.0, elapsed_ms))

    def evaluate(self, action: str, features: Sequence[float]) -> ActionEstimate:
        return self.values[action].estimate(features)

    def build_features(self, **state) -> tuple[float, ...]:
        return build_state_features(
            max_refinement_steps=self.config.max_refinement_steps,
            **state,
        )

    def choose(
        self,
        features: Sequence[float],
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool = True,
    ) -> AdaptiveDecision:
        started = time.perf_counter()
        profiling = self.config.profile_overhead
        stop_started = time.perf_counter() if profiling else None
        stop_mean = self.values[STOP].mean(features)
        if stop_started is not None:
            self.record_profile(
                "q_stop",
                (time.perf_counter() - stop_started) * 1000.0,
            )
        continue_started = time.perf_counter() if profiling else None
        continue_mean = self.values[CONTINUE].mean(features)
        if continue_started is not None:
            self.record_profile(
                "q_continue",
                (time.perf_counter() - continue_started) * 1000.0,
            )
        uncertainty_started = time.perf_counter() if profiling else None
        stop = self.values[STOP].estimate(
            features,
            mean=stop_mean,
            risk=self.early_stop_uncertainty.risk(features),
        )
        continue_ = self.values[CONTINUE].estimate(
            features,
            mean=continue_mean,
            risk=self.values[CONTINUE].risk(features),
        )
        if uncertainty_started is not None:
            self.record_profile(
                "uncertainty",
                (time.perf_counter() - uncertainty_started) * 1000.0,
            )
        advantage_mean = stop.mean - continue_.mean
        advantage_risk = self.config.risk_beta * math.sqrt(
            stop.risk * stop.risk + continue_.risk * continue_.risk
        )
        if advantage_risk > 0.0:
            stop_probability = _STANDARD_NORMAL.cdf(
                advantage_mean / advantage_risk
            )
        elif advantage_mean > 0.0:
            stop_probability = 1.0
        elif advantage_mean < 0.0:
            stop_probability = 0.0
        else:
            stop_probability = 0.5
        probability_margin = (
            self.stop_z_threshold * advantage_risk + self.config.q_margin
        )
        action_started = time.perf_counter() if profiling else None
        exploration_used = False
        finite_estimates = all(
            math.isfinite(value)
            for value in (
                stop.mean,
                stop.risk,
                stop.lower,
                stop.upper,
                continue_.mean,
                continue_.risk,
                continue_.lower,
                continue_.upper,
                advantage_mean,
                advantage_risk,
                stop_probability,
            )
        )

        calibration_active = (
            self.early_stop_observations
            < self.config.early_stop_min_observations
        )
        if not allow_stop:
            action, reason = CONTINUE, "provisional_proposal_unavailable"
        elif not finite_estimates:
            action, reason = STOP, "invalid_numeric_state"
        elif self.config.force_continue:
            action, reason = CONTINUE, "force_continue"
        elif self.config.fixed_refinement_steps is not None:
            if refinement_step >= self.config.fixed_refinement_steps:
                action, reason = STOP, "fixed_refinement_depth"
            else:
                action, reason = CONTINUE, "fixed_refinement_depth"
        elif refinement_step >= self.config.max_refinement_steps:
            action, reason = STOP, "max_refinement_steps"
        elif self.completed_rounds < self.config.warmup_rounds:
            action, reason = CONTINUE, "warmup"
        elif calibration_active:
            epsilon = max(
                self.config.explore_min,
                self.config.explore_epsilon
                * (self.config.explore_decay ** self.decision_count),
            )
            if allow_exploration and epsilon > 0.0 and self.rng.random() < epsilon:
                action, reason = STOP, "early_stop_calibration_exploration"
                exploration_used = True
                self.exploration_count += 1
            else:
                action, reason = CONTINUE, "early_stop_calibration_continue"
        elif advantage_mean > probability_margin:
            action, reason = STOP, "stop_probability_threshold"
        elif -advantage_mean > probability_margin:
            action, reason = CONTINUE, "continue_probability_threshold"
        else:
            epsilon = max(
                self.config.explore_min,
                self.config.explore_epsilon
                * (self.config.explore_decay ** self.decision_count),
            )
            if allow_exploration and epsilon > 0.0 and self.rng.random() < epsilon:
                action, reason = STOP, "uncertain_exploration"
                exploration_used = True
                self.exploration_count += 1
            else:
                action, reason = CONTINUE, "uncertain_default_continue"

        self.decision_count += 1
        if action_started is not None:
            self.record_profile(
                "action_selection",
                (time.perf_counter() - action_started) * 1000.0,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.record_profile("decision_total", latency_ms)
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=stop,
            continue_=continue_,
            rho_tokens_per_ms=self.rho,
            exploration_used=exploration_used,
            latency_ms=latency_ms,
            early_stop_observations=self.early_stop_observations,
            calibration_active=calibration_active,
            advantage_mean=advantage_mean,
            advantage_risk=advantage_risk,
            stop_probability=stop_probability,
        )

    def observe_transition(
        self,
        action: str,
        features: Sequence[float],
        next_features: Sequence[float],
        forward_latency_ms: float,
        *,
        next_stop_available: bool = True,
    ) -> float | None:
        latency_ms = max(0.0, float(forward_latency_ms))
        if self.forward_latency_ema_ms is None:
            self.forward_latency_ema_ms = latency_ms
        else:
            alpha = self.config.rho_alpha
            self.forward_latency_ema_ms = (
                (1.0 - alpha) * self.forward_latency_ema_ms
                + alpha * latency_ms
            )
        if self.config.update_mode not in {"td", "mixed"} or self.rho <= 0.0:
            return None
        started = time.perf_counter() if self.config.profile_overhead else None
        if action not in self.values:
            raise ValueError(f"Unknown action: {action}")
        future_value = self.values[CONTINUE].mean(next_features)
        if next_stop_available:
            future_value = max(
                self.values[STOP].mean(next_features),
                future_value,
            )
        target = -self.rho * latency_ms + future_value
        residual = self.values[action].update(
            features,
            target,
            self.config.learning_rate,
        )
        if started is not None:
            self.record_profile(
                "td_update",
                (time.perf_counter() - started) * 1000.0,
            )
        return residual

    def observe_continue_transition(
        self,
        features: Sequence[float],
        next_features: Sequence[float],
        forward_latency_ms: float,
        *,
        next_stop_available: bool = True,
    ) -> float | None:
        return self.observe_transition(
            CONTINUE,
            features,
            next_features,
            forward_latency_ms,
            next_stop_available=next_stop_available,
        )

    def complete_trajectory(
        self,
        trajectory: Sequence[dict],
        *,
        emitted_tokens: int,
        verifier_latency_ms: float,
    ) -> None:
        if not trajectory:
            return
        started = time.perf_counter() if self.config.profile_overhead else None
        rho = self.rho
        if rho <= 0.0:
            observed_future_ms = max(
                verifier_latency_ms
                + sum(
                    max(0.0, float(item.get("next_forward_latency_ms", 0.0)))
                    for item in trajectory
                ),
                1e-9,
            )
            rho = max(0.0, float(emitted_tokens) / observed_future_ms)

        early_stop_residuals = []
        factual_future_ms = max(0.0, verifier_latency_ms)
        for item in reversed(trajectory):
            factual_future_ms += max(
                0.0,
                float(item.get("next_forward_latency_ms", 0.0)),
            )
            if (
                item.get("action") == STOP
                and int(item.get("remaining_masks", 0)) > 0
                and item.get("reason") != "factual_terminal_verification"
            ):
                target = float(emitted_tokens) - rho * factual_future_ms
                early_stop_residuals.append((
                    item["features"],
                    target - self.values[STOP].mean(item["features"]),
                ))

        td_counted_items = set()
        if self.config.update_mode in {"td", "mixed"}:
            for item in reversed(trajectory):
                if item.get("action") == STOP:
                    target = float(emitted_tokens) - rho * max(0.0, verifier_latency_ms)
                    self.values[STOP].update(
                        item["features"],
                        target,
                        self.config.learning_rate,
                    )
                    td_counted_items.add(id(item))
                    break

        if self.config.update_mode in {"factual_return", "mixed"}:
            future_ms = max(0.0, verifier_latency_ms)
            rate = self.config.mc_learning_rate
            if self.config.update_mode == "mixed":
                rate *= self.config.mc_mix
            for item in reversed(trajectory):
                future_ms += max(
                    0.0,
                    float(item.get("next_forward_latency_ms", 0.0)),
                )
                target = float(emitted_tokens) - rho * future_ms
                self.values[item["action"]].update(
                    item["features"],
                    target,
                    rate,
                    count_observation=(
                        self.config.update_mode != "mixed"
                        or (
                            id(item) not in td_counted_items
                            and not item.get("td_observation_counted", False)
                        )
                    ),
                )
        for features, residual in early_stop_residuals:
            self.early_stop_uncertainty.update(
                features,
                residual,
                rate=0.0,
            )
        self.early_stop_observations = self.early_stop_uncertainty.sample_count
        if started is not None:
            self.record_profile(
                "reverse_factual_update",
                (time.perf_counter() - started) * 1000.0,
            )

    def observe_round(self, emitted_tokens: int, round_latency_ms: float) -> None:
        started = time.perf_counter() if self.config.profile_overhead else None
        alpha = self.config.rho_alpha
        if self.y_ema is None:
            self.y_ema = float(emitted_tokens)
            self.t_ema_ms = max(float(round_latency_ms), 1e-9)
        else:
            self.y_ema = (1.0 - alpha) * self.y_ema + alpha * float(emitted_tokens)
            self.t_ema_ms = (
                (1.0 - alpha) * self.t_ema_ms
                + alpha * max(float(round_latency_ms), 1e-9)
            )
        self.completed_rounds += 1
        if started is not None:
            self.record_profile(
                "throughput_ema_update",
                (time.perf_counter() - started) * 1000.0,
            )

    def profile_summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, values in self.profile_samples.items():
            ordered = sorted(values)
            if not ordered:
                continue
            p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
            middle = len(ordered) // 2
            median = (
                ordered[middle]
                if len(ordered) % 2
                else 0.5 * (ordered[middle - 1] + ordered[middle])
            )
            summary = {
                "count": len(ordered),
                "mean_ms": sum(ordered) / len(ordered),
                "median_ms": median,
                "p95_ms": ordered[p95_index],
            }
            if self.forward_latency_ema_ms and self.forward_latency_ema_ms > 0.0:
                summary["mean_percent_of_dllm_forward"] = (
                    100.0 * summary["mean_ms"] / self.forward_latency_ema_ms
                )
            result[name] = summary
        return result

    def snapshot(self) -> dict:
        return {
            "completed_rounds": self.completed_rounds,
            "decision_count": self.decision_count,
            "exploration_count": self.exploration_count,
            "early_stop_observations": self.early_stop_observations,
            "early_stop_min_observations": self.config.early_stop_min_observations,
            "stop_probability_threshold": self.config.stop_probability_threshold,
            "stop_z_threshold": self.stop_z_threshold,
            "rho_tokens_per_ms": self.rho,
            "y_ema": self.y_ema,
            "t_ema_ms": self.t_ema_ms,
            "forward_latency_ema_ms": self.forward_latency_ema_ms,
            "actions": {
                action: {
                    "theta": list(value.theta),
                    "precision": list(value.precision),
                    "inverse_covariance": [
                        list(row) for row in value.inverse_covariance
                    ],
                    "sample_count": value.sample_count,
                    "residual_mean": value.residual_mean,
                    "residual_variance": value.residual_variance(),
                }
                for action, value in self.values.items()
            },
            "early_stop_uncertainty": {
                "sample_count": self.early_stop_uncertainty.sample_count,
                "residual_mean": self.early_stop_uncertainty.residual_mean,
                "residual_variance": self.early_stop_uncertainty.residual_variance(),
                "precision": list(self.early_stop_uncertainty.precision),
                "inverse_covariance": [
                    list(row)
                    for row in self.early_stop_uncertainty.inverse_covariance
                ],
            },
            "overhead": self.profile_summary(),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        actions = snapshot.get("actions") or {}
        for action in (STOP, CONTINUE):
            state = actions.get(action)
            if not state:
                raise ValueError(f"snapshot is missing action state: {action}")
            value = self.values[action]
            theta = [float(item) for item in state["theta"]]
            precision = [float(item) for item in state["precision"]]
            covariance = [
                [float(item) for item in row]
                for row in state["inverse_covariance"]
            ]
            if (
                len(theta) != self.config.feature_dim
                or len(precision) != self.config.feature_dim
                or len(covariance) != self.config.feature_dim
                or any(len(row) != self.config.feature_dim for row in covariance)
            ):
                raise ValueError("snapshot feature dimension does not match controller")
            value.theta = theta
            value.precision = precision
            value.inverse_covariance = covariance
            value.sample_count = int(state.get("sample_count", 0))
            value.residual_mean = float(state.get("residual_mean", 0.0))
            residual_variance = float(
                state.get("residual_variance", value.config.residual_prior_variance)
            )
            value.residual_m2 = residual_variance * max(0, value.sample_count - 1)

        uncertainty_state = snapshot.get("early_stop_uncertainty") or {}
        uncertainty = self.early_stop_uncertainty
        if uncertainty_state:
            precision = [float(item) for item in uncertainty_state["precision"]]
            covariance = [
                [float(item) for item in row]
                for row in uncertainty_state["inverse_covariance"]
            ]
            if (
                len(precision) != self.config.feature_dim
                or len(covariance) != self.config.feature_dim
                or any(len(row) != self.config.feature_dim for row in covariance)
            ):
                raise ValueError("snapshot uncertainty dimension does not match controller")
            uncertainty.precision = precision
            uncertainty.inverse_covariance = covariance
            uncertainty.sample_count = int(uncertainty_state.get("sample_count", 0))
            uncertainty.residual_mean = float(
                uncertainty_state.get("residual_mean", 0.0)
            )
            residual_variance = float(
                uncertainty_state.get(
                    "residual_variance",
                    uncertainty.config.residual_prior_variance,
                )
            )
            uncertainty.residual_m2 = residual_variance * max(
                0,
                uncertainty.sample_count - 1,
            )

        self.completed_rounds = int(snapshot.get("completed_rounds", 0))
        self.decision_count = int(snapshot.get("decision_count", 0))
        self.exploration_count = int(snapshot.get("exploration_count", 0))
        self.early_stop_observations = int(
            snapshot.get("early_stop_observations", 0)
        )
        self.y_ema = snapshot.get("y_ema")
        self.t_ema_ms = snapshot.get("t_ema_ms")
        self.forward_latency_ema_ms = snapshot.get("forward_latency_ema_ms")


def build_state_features(
    *,
    proposal_length: int,
    remaining_masks: int,
    newly_unmasked: int,
    recoverable_confidences: Sequence[float],
    recoverable_margins: Sequence[float],
    first_remaining_position: int | None,
    frontier_length: int,
    proposal_change_ratio: float,
    recoverable_change_ratio: float,
    refinement_step: int,
    max_refinement_steps: int,
    use_margin: bool = True,
    use_stability: bool = True,
    use_step: bool = True,
) -> tuple[float, ...]:
    length = max(1, int(proposal_length))
    confidences = [_clip(value, 0.0, 1.0) for value in recoverable_confidences]
    margins = [_clip(value, 0.0, 1.0) for value in recoverable_margins]
    if confidences:
        mean_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        variance = sum(
            (value - mean_confidence) ** 2 for value in confidences
        ) / len(confidences)
        std_confidence = math.sqrt(max(0.0, variance))
    else:
        mean_confidence = min_confidence = max_confidence = std_confidence = 0.0
    mean_margin = sum(margins) / len(margins) if margins and use_margin else 0.0
    first_mask_ratio = (
        1.0
        if first_remaining_position is None
        else _clip(first_remaining_position / length, 0.0, 1.0)
    )
    step_feature = (
        _clip(
            math.log1p(max(1, refinement_step))
            / math.log1p(max(1, max_refinement_steps)),
            0.0,
            1.0,
        )
        if use_step
        else 0.0
    )
    return (
        1.0,
        _clip(remaining_masks / length, 0.0, 1.0),
        _clip(newly_unmasked / length, 0.0, 1.0),
        mean_confidence,
        min_confidence,
        max_confidence,
        std_confidence,
        mean_margin,
        first_mask_ratio,
        _clip(frontier_length / length, 0.0, 1.0),
        _clip(proposal_change_ratio, 0.0, 1.0) if use_stability else 0.0,
        _clip(recoverable_change_ratio, 0.0, 1.0) if use_stability else 0.0,
        step_feature,
    )


def trajectory_forward_latency(trajectory: Iterable[dict]) -> float:
    return sum(
        max(0.0, float(item.get("next_forward_latency_ms", 0.0)))
        for item in trajectory
    )
