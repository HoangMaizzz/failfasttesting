from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Iterable, Sequence


STOP = "stop"
CONTINUE = "continue"
_STANDARD_NORMAL = NormalDist()
V1_FEATURE_NAMES = (
    "bias",
    "remaining_mask_ratio",
    "newly_unmasked_ratio",
    "mean_confidence",
    "min_confidence",
    "max_confidence",
    "confidence_std",
    "mean_margin",
    "first_mask_ratio",
    "frontier_ratio",
    "proposal_change_ratio",
    "recoverable_change_ratio",
    "refinement_step",
)
V2_FEATURE_NAMES = (
    "bias",
    "active_remaining_mask_ratio",
    "prefix_resolved_ratio",
    "prefix_advance_ratio",
    "active_newly_unmasked_ratio",
    "newly_unmasked_prefix_share",
    "min_confidence_gap",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
)
V21_FEATURE_NAMES = (
    "bias",
    "proposal_remaining_mask_ratio",
    "prefix_resolved_ratio",
    "prefix_advance_ratio",
    "min_confidence_gap",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
)
FEATURE_NAMES = V1_FEATURE_NAMES
FEATURE_SCHEMAS = {
    "otrc_v1_td": V1_FEATURE_NAMES,
    "otrc_v2_td": V2_FEATURE_NAMES,
    "otrc_v2_1_td": V21_FEATURE_NAMES,
}


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
    feature_schema: str = "otrc_v1_td"
    feature_version: int = 1
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
    policy_mode: str = "legacy"
    min_action_probability: float = 0.10
    max_importance_weight: float = 5.0
    full_stream_bootstrap: bool = False
    reverse_backup: bool = False
    disabled_features: tuple[str, ...] = ()
    factual_ema_alpha: float = 0.2
    weight_snapshot_interval: int = 100

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.feature_schema not in FEATURE_SCHEMAS:
            raise ValueError(f"unknown feature schema: {self.feature_schema}")
        expected_dim = len(FEATURE_SCHEMAS[self.feature_schema])
        if self.feature_dim != expected_dim:
            raise ValueError(
                f"feature_dim={self.feature_dim} does not match "
                f"{self.feature_schema} dimension {expected_dim}"
            )
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
        if self.policy_mode not in {"legacy", "symmetric"}:
            raise ValueError("policy_mode must be legacy or symmetric")
        if not 0.0 < self.min_action_probability <= 0.5:
            raise ValueError("min_action_probability must be in (0, 0.5]")
        if self.max_importance_weight < 1.0:
            raise ValueError("max_importance_weight must be at least 1")
        if not 0.0 < self.factual_ema_alpha <= 1.0:
            raise ValueError("factual_ema_alpha must be in (0, 1]")
        if self.weight_snapshot_interval < 0:
            raise ValueError("weight_snapshot_interval must be non-negative")
        feature_names = FEATURE_SCHEMAS[self.feature_schema]
        unknown_features = set(self.disabled_features).difference(feature_names)
        if unknown_features:
            raise ValueError(
                f"unknown disabled features: {sorted(unknown_features)}"
            )
        if "bias" in self.disabled_features:
            raise ValueError("bias cannot be disabled")


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
    behavior_stop_probability: float
    selected_action_probability: float
    importance_weight: float
    diagnostics: dict[str, float | str] = field(default_factory=dict)


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
        self.sample_weight_sum = 0.0
        self.sample_weight_square_sum = 0.0
        self.residual_mean = 0.0
        self.residual_m2 = 0.0
        self.config = config

    def mean(self, features: Sequence[float]) -> float:
        return _dot(self.theta, features)

    def residual_variance(self) -> float:
        if self.sample_count < 2 or self.sample_weight_sum <= 0.0:
            return self.config.residual_prior_variance
        denominator = self.residual_degrees_of_freedom()
        if denominator <= 0.0:
            return self.config.residual_prior_variance
        return max(
            self.config.residual_prior_variance * 1e-3,
            self.residual_m2 / denominator,
        )

    def residual_degrees_of_freedom(self) -> float:
        if self.sample_weight_sum <= 0.0:
            return 0.0
        return self.sample_weight_sum - (
            self.sample_weight_square_sum / self.sample_weight_sum
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
        observation_weight: float = 1.0,
    ) -> float:
        observation_weight = max(0.0, float(observation_weight))
        prediction = self.mean(features)
        raw_residual = float(target) - prediction
        update_residual = _clip(
            raw_residual,
            -self.config.td_error_clip,
            self.config.td_error_clip,
        )
        for index, value in enumerate(features):
            self.theta[index] += (
                float(rate) * observation_weight * update_residual * value
            )
            if count_observation:
                self.precision[index] += observation_weight * value * value
        if count_observation:
            projected = [
                sum(weight * value for weight, value in zip(row, features))
                for row in self.inverse_covariance
            ]
            denominator = max(
                1e-12,
                1.0 + observation_weight * _dot(features, projected),
            )
            for row in range(len(self.inverse_covariance)):
                for column in range(len(self.inverse_covariance[row])):
                    self.inverse_covariance[row][column] -= (
                        observation_weight
                        * projected[row]
                        * projected[column]
                        / denominator
                    )
            self.sample_count += 1
            self.sample_weight_sum += observation_weight
            self.sample_weight_square_sum += observation_weight ** 2
            delta = raw_residual - self.residual_mean
            if self.sample_weight_sum > 0.0:
                self.residual_mean += (
                    observation_weight / self.sample_weight_sum
                ) * delta
                self.residual_m2 += (
                    observation_weight
                    * delta
                    * (raw_residual - self.residual_mean)
                )
        return raw_residual


class OnlineTDRefinementController:
    controller_name = "avg_td"

    def __init__(self, config: AdaptiveTDConfig) -> None:
        self.config = config
        self.controller_name = (
            "avg_td" if config.feature_schema == "otrc_v1_td" else config.feature_schema
        )
        self.feature_names = FEATURE_SCHEMAS[config.feature_schema]
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
        self.factual_draft_latency_ema_ms = None
        self.factual_verifier_latency_ema_ms = None
        self.factual_tokens_per_verifier_ema = None
        self.profile_samples: dict[str, list[float]] = {}
        self.pending_stop = None
        self.full_stream_transitions: list[dict] = []
        self.weight_snapshots: list[dict] = []

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
        if self.config.feature_schema == "otrc_v2_1_td":
            return build_v21_state_features(
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                factual_tokens_per_verifier_ema=(
                    self.factual_tokens_per_verifier_ema
                ),
                disabled_features=self.config.disabled_features,
                **state,
            )
        if self.config.feature_schema == "otrc_v2_td":
            return build_v2_state_features(
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                factual_tokens_per_verifier_ema=(
                    self.factual_tokens_per_verifier_ema
                ),
                disabled_features=self.config.disabled_features,
                **state,
            )
        return build_state_features(
            max_refinement_steps=self.config.max_refinement_steps,
            disabled_features=self.config.disabled_features,
            **state,
        )

    def _update_factual_ema(self, current: float | None, observation: float) -> float:
        observation = max(0.0, float(observation))
        if current is None:
            return observation
        alpha = self.config.factual_ema_alpha
        return (1.0 - alpha) * current + alpha * observation

    def observe_factual_draft_forward(self, latency_ms: float) -> None:
        self.factual_draft_latency_ema_ms = self._update_factual_ema(
            self.factual_draft_latency_ema_ms,
            latency_ms,
        )

    def observe_factual_verifier_call(
        self,
        emitted_tokens: int,
        latency_ms: float,
    ) -> None:
        self.factual_verifier_latency_ema_ms = self._update_factual_ema(
            self.factual_verifier_latency_ema_ms,
            latency_ms,
        )
        self.factual_tokens_per_verifier_ema = self._update_factual_ema(
            self.factual_tokens_per_verifier_ema,
            emitted_tokens,
        )

    def _maybe_snapshot_weights(self) -> None:
        interval = self.config.weight_snapshot_interval
        if interval <= 0 or self.decision_count % interval:
            return
        self.weight_snapshots.append({
            "decision_count": int(self.decision_count),
            "theta_stop": list(self.values[STOP].theta),
            "theta_continue": list(self.values[CONTINUE].theta),
            "theta_diff": [
                stop - continue_
                for stop, continue_ in zip(
                    self.values[STOP].theta,
                    self.values[CONTINUE].theta,
                )
            ],
        })

    def choose(
        self,
        features: Sequence[float],
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool = True,
        **_unused,
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
        behavior_stop_probability = stop_probability
        selected_action_probability = 1.0
        symmetric_sampling_used = False
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
        elif self.config.policy_mode == "symmetric":
            if allow_exploration:
                symmetric_sampling_used = True
                behavior_stop_probability = _clip(
                    stop_probability,
                    self.config.min_action_probability,
                    1.0 - self.config.min_action_probability,
                )
                if self.rng.random() < behavior_stop_probability:
                    action, reason = STOP, "symmetric_posterior_sample"
                    selected_action_probability = behavior_stop_probability
                else:
                    action, reason = CONTINUE, "symmetric_posterior_sample"
                    selected_action_probability = 1.0 - behavior_stop_probability
                greedy_action = (
                    STOP if advantage_mean > self.config.q_margin else CONTINUE
                )
                exploration_used = action != greedy_action
                if exploration_used:
                    self.exploration_count += 1
            elif advantage_mean > self.config.q_margin:
                action, reason = STOP, "symmetric_greedy_stop"
            else:
                action, reason = CONTINUE, "symmetric_greedy_continue"
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

        if not symmetric_sampling_used:
            behavior_stop_probability = 1.0 if action == STOP else 0.0
            selected_action_probability = 1.0
        importance_weight = min(
            self.config.max_importance_weight,
            1.0 / max(selected_action_probability, 1e-12),
        )

        self.decision_count += 1
        self._maybe_snapshot_weights()
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
            behavior_stop_probability=behavior_stop_probability,
            selected_action_probability=selected_action_probability,
            importance_weight=importance_weight,
            diagnostics={
                "controller_name": self.controller_name,
                "feature_schema": self.config.feature_schema,
                "greedy_action": (
                    STOP if advantage_mean > self.config.q_margin else CONTINUE
                ),
                "executed_action": action,
            },
        )

    def resolve_pending_stop(
        self,
        next_features: Sequence[float],
        *,
        next_stop_available: bool = True,
        observed_at: float | None = None,
    ) -> dict | None:
        if not self.config.full_stream_bootstrap or self.pending_stop is None:
            return None
        update_started = time.perf_counter()
        pending = self.pending_stop
        self.pending_stop = None
        observed_at = time.perf_counter() if observed_at is None else float(observed_at)
        elapsed_ms = max(0.0, (observed_at - pending["started_at"]) * 1000.0)
        future_value = self.values[CONTINUE].mean(next_features)
        if next_stop_available:
            future_value = max(
                self.values[STOP].mean(next_features),
                future_value,
            )
        rho = self.rho
        target = float(pending["emitted_tokens"]) - rho * elapsed_ms + future_value
        residual = self.values[STOP].update(
            pending["features"],
            target,
            self.config.learning_rate,
            observation_weight=pending["importance_weight"],
        )
        self.early_stop_observations = self.values[STOP].sample_count
        transition = {
            "terminal": False,
            "emitted_tokens": pending["emitted_tokens"],
            "delta_time_ms": elapsed_ms,
            "rho_tokens_per_ms": rho,
            "bootstrap_value": future_value,
            "td_target": target,
            "td_error": residual,
        }
        self.full_stream_transitions.append(transition)
        self.record_profile(
            "pending_stop_resolution",
            (time.perf_counter() - update_started) * 1000.0,
        )
        return transition

    def _complete_full_stream_trajectory(
        self,
        trajectory: Sequence[dict],
        *,
        emitted_tokens: int,
        verifier_latency_ms: float,
        terminal: bool,
    ) -> None:
        final_stop = next(
            (
                item
                for item in reversed(trajectory)
                if item.get("action") == STOP
                and item.get("post_stop_outer_action") == "verify"
                and not item.get("counterfactual_replay_overrode_action", False)
            ),
            None,
        )
        if final_stop is None:
            return
        rho = self.rho
        started_at = float(
            final_stop.get(
                "decision_monotonic_s",
                time.perf_counter() - max(0.0, verifier_latency_ms) / 1000.0,
            )
        )
        importance_weight = float(final_stop.get("importance_weight", 1.0))
        if not terminal:
            self.pending_stop = {
                "features": tuple(final_stop["features"]),
                "emitted_tokens": int(emitted_tokens),
                "started_at": started_at,
                "importance_weight": importance_weight,
            }
            final_stop["full_stream_stop_pending"] = True
            return
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        target = float(emitted_tokens) - rho * elapsed_ms
        residual = self.values[STOP].update(
            final_stop["features"],
            target,
            self.config.learning_rate,
            observation_weight=importance_weight,
        )
        self.early_stop_observations = self.values[STOP].sample_count
        final_stop.update({
            "full_stream_stop_pending": False,
            "full_stream_terminal_target": target,
            "full_stream_terminal_td_error": residual,
        })
        self.full_stream_transitions.append({
            "terminal": True,
            "emitted_tokens": int(emitted_tokens),
            "delta_time_ms": elapsed_ms,
            "rho_tokens_per_ms": rho,
            "bootstrap_value": 0.0,
            "td_target": target,
            "td_error": residual,
        })

    def observe_transition(
        self,
        action: str,
        features: Sequence[float],
        next_features: Sequence[float],
        forward_latency_ms: float,
        *,
        next_stop_available: bool = True,
        action_probability: float = 1.0,
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
        importance_weight = min(
            self.config.max_importance_weight,
            1.0 / max(float(action_probability), 1e-12),
        )
        residual = self.values[action].update(
            features,
            target,
            self.config.learning_rate,
            observation_weight=importance_weight,
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
        action_probability: float = 1.0,
    ) -> float | None:
        return self.observe_transition(
            CONTINUE,
            features,
            next_features,
            forward_latency_ms,
            next_stop_available=next_stop_available,
            action_probability=action_probability,
        )

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
        if not trajectory:
            return
        if self.config.full_stream_bootstrap:
            self._complete_full_stream_trajectory(
                trajectory,
                emitted_tokens=emitted_tokens,
                verifier_latency_ms=(
                    float(verifier_latency_ms) + float(post_verify_latency_ms)
                ),
                terminal=terminal,
            )
            if not self.config.reverse_backup:
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
                    float(item.get("importance_weight", 1.0)),
                ))

        td_counted_items = set()
        if self.config.update_mode in {"td", "mixed"}:
            for item in reversed(trajectory):
                if (
                    item.get("action") == STOP
                    and item.get("reason") != "factual_terminal_verification"
                ):
                    target = float(emitted_tokens) - rho * max(0.0, verifier_latency_ms)
                    self.values[STOP].update(
                        item["features"],
                        target,
                        self.config.learning_rate,
                        observation_weight=float(
                            item.get("importance_weight", 1.0)
                        ),
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
                if item.get("reason") == "factual_terminal_verification":
                    continue
                target = float(emitted_tokens) - rho * future_ms
                self.values[item["action"]].update(
                    item["features"],
                    target,
                    rate,
                    observation_weight=float(
                        item.get("importance_weight", 1.0)
                    ),
                    count_observation=(
                        self.config.update_mode != "mixed"
                        or (
                            id(item) not in td_counted_items
                            and not item.get("td_observation_counted", False)
                        )
                    ),
                )
        for features, residual, importance_weight in early_stop_residuals:
            self.early_stop_uncertainty.update(
                features,
                residual,
                rate=0.0,
                observation_weight=importance_weight,
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
            "controller_name": self.controller_name,
            "feature_schema": self.config.feature_schema,
            "feature_version": self.config.feature_version,
            "feature_dim": self.config.feature_dim,
            "feature_names": list(self.feature_names),
            "completed_rounds": self.completed_rounds,
            "decision_count": self.decision_count,
            "exploration_count": self.exploration_count,
            "early_stop_observations": self.early_stop_observations,
            "early_stop_min_observations": self.config.early_stop_min_observations,
            "stop_probability_threshold": self.config.stop_probability_threshold,
            "stop_z_threshold": self.stop_z_threshold,
            "policy_mode": self.config.policy_mode,
            "min_action_probability": self.config.min_action_probability,
            "max_importance_weight": self.config.max_importance_weight,
            "disabled_features": list(self.config.disabled_features),
            "rho_tokens_per_ms": self.rho,
            "y_ema": self.y_ema,
            "t_ema_ms": self.t_ema_ms,
            "forward_latency_ema_ms": self.forward_latency_ema_ms,
            "factual_draft_latency_ema_ms": self.factual_draft_latency_ema_ms,
            "factual_verifier_latency_ema_ms": self.factual_verifier_latency_ema_ms,
            "factual_tokens_per_verifier_ema": (
                self.factual_tokens_per_verifier_ema
            ),
            "actions": {
                action: {
                    "theta": list(value.theta),
                    "precision": list(value.precision),
                    "inverse_covariance": [
                        list(row) for row in value.inverse_covariance
                    ],
                    "sample_count": value.sample_count,
                    "sample_weight_sum": value.sample_weight_sum,
                    "sample_weight_square_sum": value.sample_weight_square_sum,
                    "residual_mean": value.residual_mean,
                    "residual_variance": value.residual_variance(),
                }
                for action, value in self.values.items()
            },
            "early_stop_uncertainty": {
                "sample_count": self.early_stop_uncertainty.sample_count,
                "sample_weight_sum": self.early_stop_uncertainty.sample_weight_sum,
                "sample_weight_square_sum": self.early_stop_uncertainty.sample_weight_square_sum,
                "residual_mean": self.early_stop_uncertainty.residual_mean,
                "residual_variance": self.early_stop_uncertainty.residual_variance(),
                "precision": list(self.early_stop_uncertainty.precision),
                "inverse_covariance": [
                    list(row)
                    for row in self.early_stop_uncertainty.inverse_covariance
                ],
            },
            "overhead": self.profile_summary(),
            "weight_snapshots": list(self.weight_snapshots),
            "full_stream": {
                "enabled": self.config.full_stream_bootstrap,
                "pending": self.pending_stop is not None,
                "transition_count": len(self.full_stream_transitions),
                "terminal_transition_count": sum(
                    bool(item.get("terminal"))
                    for item in self.full_stream_transitions
                ),
            },
        }

    def load_snapshot(self, snapshot: dict) -> None:
        snapshot_schema = snapshot.get("feature_schema", "otrc_v1_td")
        snapshot_version = int(snapshot.get("feature_version", 1))
        snapshot_names = tuple(snapshot.get("feature_names") or V1_FEATURE_NAMES)
        if (
            snapshot_schema != self.config.feature_schema
            or snapshot_version != self.config.feature_version
            or snapshot_names != self.feature_names
        ):
            raise ValueError(
                "snapshot feature schema does not match controller: "
                f"snapshot={snapshot_schema}/v{snapshot_version}, "
                f"controller={self.config.feature_schema}/v{self.config.feature_version}"
            )
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
            value.sample_weight_sum = float(
                state.get("sample_weight_sum", value.sample_count)
            )
            value.sample_weight_square_sum = float(
                state.get("sample_weight_square_sum", value.sample_count)
            )
            value.residual_mean = float(state.get("residual_mean", 0.0))
            residual_variance = float(
                state.get("residual_variance", value.config.residual_prior_variance)
            )
            value.residual_m2 = (
                residual_variance * value.residual_degrees_of_freedom()
            )

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
            uncertainty.sample_weight_sum = float(
                uncertainty_state.get(
                    "sample_weight_sum",
                    uncertainty.sample_count,
                )
            )
            uncertainty.sample_weight_square_sum = float(
                uncertainty_state.get(
                    "sample_weight_square_sum",
                    uncertainty.sample_count,
                )
            )
            uncertainty.residual_mean = float(
                uncertainty_state.get("residual_mean", 0.0)
            )
            residual_variance = float(
                uncertainty_state.get(
                    "residual_variance",
                    uncertainty.config.residual_prior_variance,
                )
            )
            uncertainty.residual_m2 = (
                residual_variance
                * uncertainty.residual_degrees_of_freedom()
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
        self.factual_draft_latency_ema_ms = snapshot.get(
            "factual_draft_latency_ema_ms"
        )
        self.factual_verifier_latency_ema_ms = snapshot.get(
            "factual_verifier_latency_ema_ms"
        )
        self.factual_tokens_per_verifier_ema = snapshot.get(
            "factual_tokens_per_verifier_ema"
        )
        self.weight_snapshots = list(snapshot.get("weight_snapshots") or [])


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
    disabled_features: Sequence[str] = (),
    **_unused,
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
    features = [
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
    ]
    disabled = set(disabled_features)
    for index, name in enumerate(FEATURE_NAMES):
        if name in disabled:
            features[index] = 0.0
    return tuple(features)


def build_v2_state_features(
    *,
    proposal_length: int,
    max_spec_len: int,
    active_span_length: int,
    active_remaining_masks: int,
    active_newly_unmasked: int,
    prefix_length: int,
    prefix_advance: int,
    newly_unmasked_prefix: int,
    active_remaining_confidences: Sequence[float],
    failfast_candidate_min_confidence: float,
    drafter_threshold: float,
    failfast_threshold: float,
    factual_draft_latency_ema_ms: float | None,
    factual_verifier_latency_ema_ms: float | None,
    factual_tokens_per_verifier_ema: float | None,
    disabled_features: Sequence[str] = (),
    **_unused,
) -> tuple[float, ...]:
    proposal_length = max(1, int(proposal_length))
    max_spec_len = max(proposal_length, int(max_spec_len))
    active_span_length = max(1, int(active_span_length))
    active_newly_unmasked = max(0, int(active_newly_unmasked))
    eps = 1e-9

    confidences = [
        _clip(value, 0.0, 1.0)
        for value in active_remaining_confidences
    ]
    tau_d = max(float(drafter_threshold), eps)
    tau_f = max(float(failfast_threshold), eps)
    min_confidence_gap = _clip(
        (tau_d - (min(confidences) if confidences else tau_d)) / tau_d,
        0.0,
        1.0,
    )
    failfast_margin = _clip(
        (float(failfast_candidate_min_confidence) - tau_f) / tau_f,
        -1.0,
        1.0,
    )

    if (
        factual_draft_latency_ema_ms is None
        or factual_verifier_latency_ema_ms is None
        or factual_verifier_latency_ema_ms <= 0.0
    ):
        draft_verify_ratio = 1.0
    else:
        draft_verify_ratio = _clip(
            float(factual_draft_latency_ema_ms)
            / max(float(factual_verifier_latency_ema_ms), eps),
            0.0,
            2.0,
        )
    tokens_per_verifier = (
        1.0
        if factual_tokens_per_verifier_ema is None
        else max(0.0, float(factual_tokens_per_verifier_ema))
    )

    features = [
        1.0,
        _clip(active_remaining_masks / active_span_length, 0.0, 1.0),
        _clip(prefix_length / proposal_length, 0.0, 1.0),
        _clip(prefix_advance / proposal_length, 0.0, 1.0),
        _clip(active_newly_unmasked / active_span_length, 0.0, 1.0),
        _clip(
            newly_unmasked_prefix / max(active_newly_unmasked, 1),
            0.0,
            1.0,
        ),
        min_confidence_gap,
        failfast_margin,
        _clip(proposal_length / max_spec_len, 0.0, 1.0),
        draft_verify_ratio,
        _clip(tokens_per_verifier / (max_spec_len + 1.0), 0.0, 1.0),
    ]
    disabled = set(disabled_features)
    for index, name in enumerate(V2_FEATURE_NAMES):
        if name in disabled:
            features[index] = 0.0
    return tuple(features)


def build_v21_state_features(
    *,
    proposal_length: int,
    max_spec_len: int,
    proposal_remaining_masks: int,
    proposal_remaining_confidences: Sequence[float],
    prefix_length: int,
    prefix_advance: int,
    failfast_candidate_min_confidence: float,
    drafter_threshold: float,
    failfast_threshold: float,
    factual_draft_latency_ema_ms: float | None,
    factual_verifier_latency_ema_ms: float | None,
    factual_tokens_per_verifier_ema: float | None,
    disabled_features: Sequence[str] = (),
    **_unused,
) -> tuple[float, ...]:
    proposal_length = max(1, int(proposal_length))
    max_spec_len = max(proposal_length, int(max_spec_len))
    eps = 1e-9
    confidences = [
        _clip(value, 0.0, 1.0)
        for value in proposal_remaining_confidences
    ]
    tau_d = max(float(drafter_threshold), eps)
    tau_f = max(float(failfast_threshold), eps)
    min_confidence_gap = 0.0
    if confidences:
        min_confidence_gap = _clip(
            (tau_d - min(confidences)) / tau_d,
            -1.0,
            1.0,
        )
    failfast_margin = _clip(
        (float(failfast_candidate_min_confidence) - tau_f) / tau_f,
        -1.0,
        1.0,
    )

    if (
        factual_draft_latency_ema_ms is None
        or factual_verifier_latency_ema_ms is None
        or factual_verifier_latency_ema_ms <= 0.0
    ):
        draft_verify_ratio = 1.0
    else:
        draft_verify_ratio = _clip(
            float(factual_draft_latency_ema_ms)
            / max(float(factual_verifier_latency_ema_ms), eps),
            0.0,
            2.0,
        )
    tokens_per_verifier = (
        1.0
        if factual_tokens_per_verifier_ema is None
        else max(0.0, float(factual_tokens_per_verifier_ema))
    )

    features = [
        1.0,
        _clip(proposal_remaining_masks / proposal_length, 0.0, 1.0),
        _clip(prefix_length / proposal_length, 0.0, 1.0),
        _clip(prefix_advance / proposal_length, 0.0, 1.0),
        min_confidence_gap,
        failfast_margin,
        _clip(proposal_length / max_spec_len, 0.0, 1.0),
        draft_verify_ratio,
        _clip(tokens_per_verifier / (max_spec_len + 1.0), 0.0, 1.0),
    ]
    disabled = set(disabled_features)
    for index, name in enumerate(V21_FEATURE_NAMES):
        if name in disabled:
            features[index] = 0.0
    return tuple(features)


def trajectory_forward_latency(trajectory: Iterable[dict]) -> float:
    return sum(
        max(0.0, float(item.get("next_forward_latency_ms", 0.0)))
        for item in trajectory
    )
