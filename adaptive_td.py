from __future__ import annotations

import math
import random
import time
from bisect import insort
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Iterable, Sequence

from adaptive_nonlinear import OnlineNonlinearVA


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
V22_FEATURE_NAMES = (
    "bias",
    "prefix_resolved_ratio",
    "prefix_advance_ratio",
    "min_remaining_confidence",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
)
V22_COMPACT_FEATURE_NAMES = (
    "bias",
    "prefix_advance_ratio",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
)
V23_COMPACT7_FEATURE_NAMES = (
    "bias",
    "prefix_advance_ratio",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "active_remaining_mask_ratio",
    "normalized_refinement_step",
)
RAW_STATE_BLOCK_SIZE = 8
RAW_TOKEN_FIELDS = (
    "mask_indicator",
    "observed_indicator",
    "top1_probability",
    "top2_probability",
    "normalized_entropy",
    "normalized_position",
)
RAW_GLOBAL_FEATURE_NAMES = (
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
    "normalized_refinement_step",
    "has_previous_state",
)
HINDSIGHT_GAIN_FEATURE_NAMES = (
    "bias",
    "active_span_ratio",
    "active_mask_ratio",
    "top1_mean",
    "top1_std",
    "margin_mean",
    "margin_std",
    "entropy_mean",
    "entropy_std",
    "accumulated_spec_ratio",
    "normalized_refinement_step",
)
HINDSIGHT_DELTA_J_F5_FEATURE_NAMES = (
    "bias",
    "current_mask_ratio",
    "masked_entropy_std",
    "resolved_margin_mean",
    "resolved_entropy_max",
    "ema_tokens_per_verifier",
)
HINDSIGHT_DELTA_J_F2_FEATURE_NAMES = (
    "bias",
    "current_mask_ratio",
    "first_unresolved_position",
)
HINDSIGHT_DELTA_J_LOGISTIC_F2_FEATURE_NAMES = (
    "bias",
    "current_mask_ratio",
    "global_proposal_position",
)


def _raw_state_feature_names(block_size: int = RAW_STATE_BLOCK_SIZE):
    names = []
    for state_name in ("previous", "current"):
        for position in range(int(block_size)):
            names.extend(
                f"{state_name}_position_{position}_{field}"
                for field in RAW_TOKEN_FIELDS
            )
    names.extend(RAW_GLOBAL_FEATURE_NAMES)
    return tuple(names)


RAW_STATE_FEATURE_NAMES = _raw_state_feature_names()
FEATURE_NAMES = V1_FEATURE_NAMES
FEATURE_SCHEMAS = {
    "otrc_v1_td": V1_FEATURE_NAMES,
    "otrc_v2_td": V2_FEATURE_NAMES,
    "otrc_v2_1_td": V21_FEATURE_NAMES,
    "otrc_v2_2_td": V22_FEATURE_NAMES,
    "otrc_v2_2_compact_td": V22_COMPACT_FEATURE_NAMES,
    "otrc_v2_3_compact7_td": V23_COMPACT7_FEATURE_NAMES,
    "otrc_raw_state_v1": RAW_STATE_FEATURE_NAMES,
}


def active_refinement_positions(remaining_positions, active_start, active_end):
    """Select unresolved positions belonging to the small block just processed."""
    start = int(active_start)
    end = int(active_end)
    return tuple(
        int(position)
        for position in remaining_positions
        if start <= int(position) < end
    )


def logical_refinement_span(
    draft_start,
    draft_end,
    physical_start,
    physical_end,
    span_size,
):
    """Map a physical model segment to its proposal-relative logical span."""
    if int(span_size) <= 0:
        raise ValueError("span_size must be positive")
    intersection_start = max(int(draft_start), int(physical_start))
    intersection_end = min(int(draft_end), int(physical_end))
    if intersection_start >= intersection_end:
        return None
    logical_index = (intersection_start - int(draft_start)) // int(span_size)
    logical_start = int(draft_start) + logical_index * int(span_size)
    logical_end = min(int(draft_end), logical_start + int(span_size))
    return logical_index, logical_start, logical_end


def complete_raw_probability_frame(
    probability_cache,
    active_start,
    active_end,
    block_size=None,
):
    """Return one ordered proposal-relative frame once every token was observed."""
    start = int(active_start)
    end = int(active_end)
    span = end - start
    if span <= 0 or span > RAW_STATE_BLOCK_SIZE:
        raise ValueError(
            f"raw probability frame must span 1..{RAW_STATE_BLOCK_SIZE} positions"
        )
    if block_size is not None and span != int(block_size):
        raise ValueError(
            f"raw probability frame must span {int(block_size)} positions"
        )
    if any(position not in probability_cache for position in range(start, end)):
        return None
    return tuple(probability_cache[position] for position in range(start, end))


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
    policy_ablation: str = "learned"
    min_action_probability: float = 0.10
    max_importance_weight: float = 5.0
    full_stream_bootstrap: bool = False
    credit_assignment: str = "per_step_td"
    reverse_backup: bool = False
    disabled_features: tuple[str, ...] = ()
    factual_ema_alpha: float = 0.2
    rho_warmup_boundaries: int = 0
    policy_weight_ema_beta: float = 0.0
    policy_weight_ema_mode: str = "global_step"
    weight_snapshot_interval: int = 100
    value_parameterization: str = "independent_q"
    shared_value_learning_rate: float = 0.015
    shared_advantage_learning_rate: float = 0.02
    value_model: str = "linear"
    nonlinear_learning_rate: float = 1e-3
    nonlinear_weight_decay: float = 0.0
    nonlinear_grad_clip: float = 1.0
    nonlinear_device: str = "cpu"
    hindsight_prior_precision: float = 1.0
    hindsight_noise_variance: float = 0.25
    hindsight_confidence_kappa: float = 1.0
    hindsight_margin_tokens: float = 0.0
    hindsight_max_uncertainty_tokens: float = 2.0
    hindsight_probe_initial: float = 0.15
    hindsight_probe_floor: float = 0.02
    hindsight_probe_decay_pairs: float = 32.0
    hindsight_probe_uncertainty_tokens: float = 0.75
    hindsight_probe_boundary_scale: float = 1.0
    hindsight_probe_max_fraction: float = 0.08
    hindsight_delta_j_p_continue_threshold: float = 0.65
    hindsight_delta_j_class_balance_alpha: float = 5.0
    hindsight_delta_j_max_continue_weight: float = 3.0
    hindsight_delta_j_calibration_beta: float = 0.05
    hindsight_delta_j_min_pairs: int = 30
    hindsight_delta_j_min_continue_pairs: int = 3
    hindsight_delta_j_structural_probe_probability: float = 0.08
    hindsight_delta_j_floor_probe_probability: float = 0.02
    hindsight_logistic_learning_rate: float = 0.05
    hindsight_logistic_continue_threshold: float = 0.5
    hindsight_logistic_tie_ms_per_token: float = 1.0
    hindsight_logistic_use_class_weight: bool = False
    hindsight_logistic_min_positive_problems: int = 2
    hindsight_logistic_utility_weighting: str = "legacy"
    hindsight_logistic_replay_batch_size: int = 0
    hindsight_logistic_replay_buffer_size: int = 100

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
        if self.policy_mode not in {
            "legacy",
            "symmetric",
            "symmetric_annealed",
            "symmetric_greedy",
            "hindsight_gain",
            "hindsight_delta_j_f5",
            "hindsight_delta_j_f2",
            "hindsight_delta_j_logistic_f2",
        }:
            raise ValueError(
                "policy_mode must be legacy, symmetric, symmetric_annealed, "
                "symmetric_greedy, hindsight_gain, hindsight_delta_j_f5, "
                "hindsight_delta_j_f2, or hindsight_delta_j_logistic_f2"
            )
        if self.policy_ablation not in {
            "learned",
            "frozen_stop",
            "random_stop",
        }:
            raise ValueError(
                "policy_ablation must be learned, frozen_stop, or random_stop"
            )
        if not 0.0 < self.min_action_probability <= 0.5:
            raise ValueError("min_action_probability must be in (0, 0.5]")
        if self.max_importance_weight < 1.0:
            raise ValueError("max_importance_weight must be at least 1")
        if self.credit_assignment not in {
            "per_step_td",
            "verifier_boundary_factual",
            "verifier_boundary_factual_no_bootstrap",
            "hindsight_block_gain",
            "hindsight_delta_j_f5",
            "hindsight_delta_j_f2",
            "hindsight_delta_j_logistic_f2",
        }:
            raise ValueError(
                "credit_assignment must be per_step_td, "
                "verifier_boundary_factual, or "
                "verifier_boundary_factual_no_bootstrap, hindsight_block_gain, "
                "hindsight_delta_j_f5, hindsight_delta_j_f2, or "
                "hindsight_delta_j_logistic_f2"
            )
        if self.hindsight_prior_precision <= 0.0:
            raise ValueError("hindsight prior precision must be positive")
        if self.hindsight_noise_variance <= 0.0:
            raise ValueError("hindsight noise variance must be positive")
        if self.hindsight_confidence_kappa < 0.0:
            raise ValueError("hindsight confidence kappa must be non-negative")
        if self.hindsight_max_uncertainty_tokens <= 0.0:
            raise ValueError("hindsight max uncertainty must be positive")
        if not (
            0.0 <= self.hindsight_probe_floor
            <= self.hindsight_probe_initial
            <= 1.0
        ):
            raise ValueError("hindsight probe rates must satisfy 0 <= floor <= initial <= 1")
        if self.hindsight_probe_decay_pairs <= 0.0:
            raise ValueError("hindsight probe decay pairs must be positive")
        if self.hindsight_probe_uncertainty_tokens < 0.0:
            raise ValueError("hindsight probe uncertainty must be non-negative")
        if self.hindsight_probe_boundary_scale < 0.0:
            raise ValueError("hindsight probe boundary scale must be non-negative")
        if not 0.0 <= self.hindsight_probe_max_fraction <= 1.0:
            raise ValueError("hindsight probe max fraction must be in [0, 1]")
        if (
            self.credit_assignment == "hindsight_block_gain"
            and self.policy_mode != "hindsight_gain"
        ):
            raise ValueError("hindsight block gain requires hindsight_gain policy")
        if (
            self.credit_assignment in {
                "hindsight_delta_j_f5", "hindsight_delta_j_f2",
                "hindsight_delta_j_logistic_f2",
            }
            and self.policy_mode != self.credit_assignment
        ):
            raise ValueError("hindsight delta-J modes require their matching policy")
        if not 0.5 < self.hindsight_delta_j_p_continue_threshold < 1.0:
            raise ValueError("hindsight delta-J probability threshold must be in (0.5, 1)")
        if self.hindsight_delta_j_class_balance_alpha <= 0.0:
            raise ValueError("hindsight delta-J class-balance alpha must be positive")
        if self.hindsight_delta_j_max_continue_weight < 1.0:
            raise ValueError("hindsight delta-J continue weight cap must be at least 1")
        if not 0.0 < self.hindsight_delta_j_calibration_beta <= 1.0:
            raise ValueError("hindsight delta-J calibration beta must be in (0, 1]")
        if self.hindsight_delta_j_min_pairs < 0 or self.hindsight_delta_j_min_continue_pairs < 0:
            raise ValueError("hindsight delta-J readiness counts must be non-negative")
        if not 0.0 <= self.hindsight_delta_j_structural_probe_probability <= 1.0:
            raise ValueError("hindsight structural probe probability must be in [0, 1]")
        if not 0.0 <= self.hindsight_delta_j_floor_probe_probability <= 1.0:
            raise ValueError("hindsight floor probe probability must be in [0, 1]")
        if self.hindsight_logistic_learning_rate <= 0.0:
            raise ValueError("hindsight logistic learning rate must be positive")
        if not 0.0 < self.hindsight_logistic_continue_threshold < 1.0:
            raise ValueError("hindsight logistic threshold must be in (0, 1)")
        if self.hindsight_logistic_tie_ms_per_token < 0.0:
            raise ValueError("hindsight logistic tie threshold must be non-negative")
        if self.hindsight_logistic_min_positive_problems < 0:
            raise ValueError("minimum positive-problem count must be non-negative")
        if self.hindsight_logistic_utility_weighting not in {"legacy", "raw_abs"}:
            raise ValueError("hindsight logistic utility weighting must be legacy or raw_abs")
        if self.hindsight_logistic_replay_batch_size < 0:
            raise ValueError("hindsight logistic replay batch size must be non-negative")
        if self.hindsight_logistic_replay_buffer_size <= 0:
            raise ValueError("hindsight logistic replay buffer size must be positive")
        if not 0.0 < self.factual_ema_alpha <= 1.0:
            raise ValueError("factual_ema_alpha must be in (0, 1]")
        if self.rho_warmup_boundaries < 0:
            raise ValueError("rho_warmup_boundaries must be non-negative")
        if (
            self.rho_warmup_boundaries
            and self.credit_assignment
            != "verifier_boundary_factual_no_bootstrap"
        ):
            raise ValueError(
                "rho warmup requires verifier-boundary factual no-bootstrap"
            )
        if not 0.0 <= self.policy_weight_ema_beta < 1.0:
            raise ValueError("policy_weight_ema_beta must be in [0, 1)")
        if self.policy_weight_ema_mode not in {"action_step", "global_step"}:
            raise ValueError(
                "policy_weight_ema_mode must be action_step or global_step"
            )
        if (
            self.policy_weight_ema_beta
            and self.credit_assignment
            != "verifier_boundary_factual_no_bootstrap"
        ):
            raise ValueError(
                "policy weight EMA requires verifier-boundary factual "
                "no-bootstrap"
            )
        if self.weight_snapshot_interval < 0:
            raise ValueError("weight_snapshot_interval must be non-negative")
        if self.value_parameterization not in {
            "independent_q",
            "shared_value_advantage",
        }:
            raise ValueError(
                "value_parameterization must be independent_q or "
                "shared_value_advantage"
            )
        if self.value_model not in {
            "linear", "nam", "ga2m", "raw_linear", "raw_mlp"
        }:
            raise ValueError(
                "value_model must be linear, nam, ga2m, raw_linear, or raw_mlp"
            )
        if self.value_model in {"nam", "ga2m"}:
            if self.feature_schema != "otrc_v2_2_compact_td":
                raise ValueError("NAM/GA2M require Compact6")
            if self.value_parameterization != "shared_value_advantage":
                raise ValueError("NAM/GA2M require shared value/advantage")
            if self.credit_assignment != "verifier_boundary_factual_no_bootstrap":
                raise ValueError("NAM/GA2M require factual no-bootstrap")
        if self.value_model in {"raw_linear", "raw_mlp"}:
            if self.feature_schema != "otrc_raw_state_v1":
                raise ValueError("raw value models require raw-state features")
            if self.value_parameterization != "shared_value_advantage":
                raise ValueError("raw value models require shared value/advantage")
            if self.credit_assignment != "verifier_boundary_factual_no_bootstrap":
                raise ValueError("raw value models require factual no-bootstrap")
        if (
            self.shared_value_learning_rate < 0.0
            or self.shared_advantage_learning_rate < 0.0
        ):
            raise ValueError("shared value/advantage rates must be non-negative")
        if (
            self.value_parameterization == "shared_value_advantage"
            and self.credit_assignment
            != "verifier_boundary_factual_no_bootstrap"
        ):
            raise ValueError(
                "shared value/advantage requires factual no-bootstrap credit"
            )
        if (
            self.value_parameterization == "shared_value_advantage"
            and self.policy_weight_ema_beta
        ):
            raise ValueError(
                "shared value/advantage does not support policy weight EMA"
            )
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


class _BayesianLinearGain:
    """Small online Bayesian linear regressor for a normalized local target."""

    def __init__(self, dimension: int, prior_precision: float, noise_variance: float):
        self.dimension = int(dimension)
        self.noise_variance = float(noise_variance)
        prior_variance = 1.0 / float(prior_precision)
        self.weights = [0.0] * self.dimension
        self.covariance = [
            [prior_variance if row == column else 0.0 for column in range(self.dimension)]
            for row in range(self.dimension)
        ]
        self.sample_count = 0
        self.residual_sum = 0.0
        self.absolute_residual_sum = 0.0
        self.squared_residual_sum = 0.0

    def predict(
        self,
        features: Sequence[float],
        *,
        include_observation_noise: bool = False,
    ) -> tuple[float, float]:
        mean = _dot(self.weights, features)
        projected = [_dot(row, features) for row in self.covariance]
        variance = max(
            0.0,
            _dot(features, projected)
            + (self.noise_variance if include_observation_noise else 0.0),
        )
        return mean, math.sqrt(variance)

    def update(
        self,
        features: Sequence[float],
        target: float,
        observation_weight: float = 1.0,
    ) -> float:
        prediction, _ = self.predict(features)
        residual = float(target) - prediction
        projected = [_dot(row, features) for row in self.covariance]
        effective_noise = self.noise_variance / max(1e-9, float(observation_weight))
        denominator = max(
            1e-12,
            effective_noise + _dot(features, projected),
        )
        gain = [value / denominator for value in projected]
        self.weights = [
            weight + coefficient * residual
            for weight, coefficient in zip(self.weights, gain)
        ]
        self.covariance = [
            [
                self.covariance[row][column] - gain[row] * projected[column]
                for column in range(self.dimension)
            ]
            for row in range(self.dimension)
        ]
        self.sample_count += 1
        self.residual_sum += residual
        self.absolute_residual_sum += abs(residual)
        self.squared_residual_sum += residual * residual
        return residual

    def snapshot(self, feature_names=HINDSIGHT_GAIN_FEATURE_NAMES) -> dict:
        count = max(1, self.sample_count)
        return {
            "feature_names": list(feature_names),
            "weights": list(self.weights),
            "covariance": [list(row) for row in self.covariance],
            "sample_count": int(self.sample_count),
            "normalized_bias": self.residual_sum / count,
            "normalized_mae": self.absolute_residual_sum / count,
            "normalized_rmse": math.sqrt(self.squared_residual_sum / count),
        }


class _OnlineWeightedLogistic:
    """Tiny CPU-only online logistic classifier."""

    def __init__(self, dimension: int, learning_rate: float):
        self.dimension = int(dimension)
        self.learning_rate = float(learning_rate)
        self.weights = [0.0] * self.dimension
        self.sample_count = 0
        self.weight_sum = 0.0
        self.loss_sum = 0.0

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        logit = _dot(self.weights, features)
        if logit >= 0.0:
            score = 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
        else:
            exp_logit = math.exp(max(logit, -60.0))
            score = exp_logit / (1.0 + exp_logit)
        return logit, score

    def update(
        self,
        features: Sequence[float],
        label: int,
        sample_weight: float,
    ) -> float:
        logit, score = self.predict(features)
        weight = max(0.0, float(sample_weight))
        error = score - float(label)
        gradient = [weight * error * float(value) for value in features]
        gradient_norm = math.sqrt(sum(value * value for value in gradient))
        if gradient_norm > 10.0:
            scale = 10.0 / gradient_norm
            gradient = [value * scale for value in gradient]
        self.weights = [
            current - self.learning_rate * delta
            for current, delta in zip(self.weights, gradient)
        ]
        loss = -weight * (
            float(label) * math.log(max(score, 1e-12))
            + (1.0 - float(label)) * math.log(max(1.0 - score, 1e-12))
        )
        self.sample_count += 1
        self.weight_sum += weight
        self.loss_sum += loss
        return loss

    def update_batch(
        self,
        samples: Sequence[tuple[Sequence[float], int, float]],
    ) -> float:
        """Perform one SGD step on the mean weighted mini-batch gradient."""
        if not samples:
            return 0.0
        gradient = [0.0] * self.dimension
        loss_sum = 0.0
        weight_sum = 0.0
        for features, label, sample_weight in samples:
            _, score = self.predict(features)
            weight = max(0.0, float(sample_weight))
            error = score - float(label)
            for index, value in enumerate(features):
                gradient[index] += weight * error * float(value)
            loss_sum += -weight * (
                float(label) * math.log(max(score, 1e-12))
                + (1.0 - float(label)) * math.log(max(1.0 - score, 1e-12))
            )
            weight_sum += weight
        inv_batch = 1.0 / len(samples)
        gradient = [value * inv_batch for value in gradient]
        gradient_norm = math.sqrt(sum(value * value for value in gradient))
        if gradient_norm > 10.0:
            clip_scale = 10.0 / gradient_norm
            gradient = [value * clip_scale for value in gradient]
        self.weights = [
            current - self.learning_rate * delta
            for current, delta in zip(self.weights, gradient)
        ]
        mean_loss = loss_sum * inv_batch
        self.sample_count += 1
        self.weight_sum += weight_sum * inv_batch
        self.loss_sum += mean_loss
        return mean_loss

    def snapshot(self, feature_names) -> dict:
        return {
            "feature_names": list(feature_names),
            "weights": list(self.weights),
            "sample_count": int(self.sample_count),
            "sample_weight_sum": float(self.weight_sum),
            "mean_weighted_loss": self.loss_sum / max(1, self.sample_count),
        }


class OnlineTDRefinementController:
    controller_name = "avg_td"

    def __init__(self, config: AdaptiveTDConfig) -> None:
        self.config = config
        self.controller_name = (
            "avg_td" if config.feature_schema == "otrc_v1_td" else config.feature_schema
        )
        if config.value_parameterization == "shared_value_advantage":
            self.controller_name += "_shared_value_advantage"
        if config.value_model != "linear":
            self.controller_name += f"_{config.value_model}"
        self.feature_names = FEATURE_SCHEMAS[config.feature_schema]
        self.values = {
            STOP: _LinearActionValue(config.feature_dim, config),
            CONTINUE: _LinearActionValue(config.feature_dim, config),
        }
        self.shared_value_theta = [0.0] * config.feature_dim
        self.shared_advantage_theta = [0.0] * config.feature_dim
        self.policy_ema_theta = {
            STOP: [0.0] * config.feature_dim,
            CONTINUE: [0.0] * config.feature_dim,
        }
        self.policy_ema_initialized = {STOP: False, CONTINUE: False}
        self.policy_ema_update_count = {STOP: 0, CONTINUE: 0}
        self.policy_ema_global_update_count = 0
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
        self.annealed_decision_count = 0
        self.exploration_count = 0
        self.early_stop_observations = 0
        self.forward_latency_ema_ms = None
        self.factual_draft_latency_ema_ms = None
        self.factual_verifier_latency_ema_ms = None
        self.factual_tokens_per_verifier_ema = None
        self.profile_samples: dict[str, list[float]] = {}
        self.pending_stop = None
        self.pending_factual_boundary = None
        self.algorithm_latency_ms = 0.0
        self.algorithm_latency_components_ms: dict[str, float] = {}
        self._draft_checkpoint_wall_s = None
        self._draft_checkpoint_ledger_ms = None
        self.factual_boundary_count = 0
        self.factual_observed_verifier_boundaries = 0
        self.factual_learning_update_count = 0
        self.factual_warmup_transition_count = 0
        self.full_stream_transitions: list[dict] = []
        self.weight_snapshots: list[dict] = []
        if config.credit_assignment == "hindsight_delta_j_f5":
            hindsight_feature_names = HINDSIGHT_DELTA_J_F5_FEATURE_NAMES
        elif config.credit_assignment == "hindsight_delta_j_f2":
            hindsight_feature_names = HINDSIGHT_DELTA_J_F2_FEATURE_NAMES
        elif config.credit_assignment == "hindsight_delta_j_logistic_f2":
            hindsight_feature_names = HINDSIGHT_DELTA_J_LOGISTIC_F2_FEATURE_NAMES
        else:
            hindsight_feature_names = HINDSIGHT_GAIN_FEATURE_NAMES
        self.hindsight_feature_names = hindsight_feature_names
        self.hindsight_gain_model = _BayesianLinearGain(
            len(hindsight_feature_names),
            config.hindsight_prior_precision,
            config.hindsight_noise_variance,
        )
        self.hindsight_logistic_model = _OnlineWeightedLogistic(
            len(hindsight_feature_names),
            config.hindsight_logistic_learning_rate,
        )
        self.hindsight_logistic_abs_r: list[float] = []
        self.hindsight_logistic_replay_buffer: list[tuple[list[float], int, float]] = []
        # Dedicated RNG: replay sampling must not perturb probe/exploration randomness.
        self.hindsight_logistic_replay_rng = random.Random(config.seed + 7919)
        self.hindsight_logistic_tie_count = 0
        self.hindsight_logistic_positive_problem_ids: set[int] = set()
        self.hindsight_censor_reasons: dict[str, int] = {}
        self.hindsight_problem_id = None
        self.hindsight_committed_tokens: list[int] = []
        self.hindsight_snapshot_count = 0
        self.hindsight_pair_count = 0
        self.hindsight_resolved_count = 0
        self.hindsight_censored_count = 0
        self.hindsight_invalid_count = 0
        self.hindsight_unavailable_count = 0
        self.hindsight_current_snapshot = None
        self.hindsight_pending_sources: dict[tuple, dict] = {}
        self.hindsight_pending_pairs: list[dict] = []
        self.hindsight_snapshot_overhead_ema_ms = None
        self.hindsight_probe_count = 0
        self.hindsight_probe_outstanding = False
        self.hindsight_delta_j_calibration_bias = 0.0
        self.hindsight_delta_j_continue_count = 0
        self.hindsight_delta_j_stop_count = 0
        self.hindsight_structural_probe_count = 0
        self.hindsight_floor_probe_count = 0
        self.nonlinear_value = (
            None if config.value_model == "linear" else OnlineNonlinearVA(
                config.value_model,
                learning_rate=config.nonlinear_learning_rate,
                weight_decay=config.nonlinear_weight_decay,
                grad_clip=config.nonlinear_grad_clip,
                huber_delta=config.td_error_clip,
                seed=config.seed,
                feature_dim=config.feature_dim,
                device=config.nonlinear_device,
            )
        )

    @property
    def rho(self) -> float:
        if self.y_ema is None or self.t_ema_ms is None:
            return 0.0
        return max(0.0, self.y_ema / max(self.t_ema_ms, 1e-9))

    def record_profile(self, name: str, elapsed_ms: float) -> None:
        if self.config.profile_overhead:
            self.profile_samples.setdefault(name, []).append(max(0.0, elapsed_ms))

    @property
    def uses_verifier_boundary_factual(self) -> bool:
        return self.config.credit_assignment in {
            "verifier_boundary_factual",
            "verifier_boundary_factual_no_bootstrap",
        }

    @property
    def uses_factual_no_bootstrap(self) -> bool:
        return (
            self.config.credit_assignment
            == "verifier_boundary_factual_no_bootstrap"
        )

    @property
    def uses_policy_weight_ema(self) -> bool:
        return self.config.policy_weight_ema_beta > 0.0

    @property
    def uses_shared_value_advantage(self) -> bool:
        return self.config.value_parameterization == "shared_value_advantage"

    def _sync_shared_action_means(self) -> None:
        if not self.uses_shared_value_advantage:
            return
        self.values[STOP].theta = [
            value + 0.5 * advantage
            for value, advantage in zip(
                self.shared_value_theta,
                self.shared_advantage_theta,
            )
        ]
        self.values[CONTINUE].theta = [
            value - 0.5 * advantage
            for value, advantage in zip(
                self.shared_value_theta,
                self.shared_advantage_theta,
            )
        ]

    def _update_factual_action_value(
        self,
        action: str,
        features: Sequence[float],
        target: float,
        *,
        observation_weight: float,
    ) -> float:
        if self.nonlinear_value is not None:
            raw_residual = self.values[action].update(
                features,
                target,
                rate=0.0,
                observation_weight=observation_weight,
            )
            update = self.nonlinear_value.update(
                action, features, target, observation_weight
            )
            interval = self.config.weight_snapshot_interval
            if (
                interval > 0
                and self.nonlinear_value.update_count % interval == 0
            ):
                self.weight_snapshots.append({
                    "learning_update_count": int(
                        self.nonlinear_value.update_count
                    ),
                    "decision_count": int(self.decision_count),
                    "value_model": self.config.value_model,
                    "normalization_state": {
                        "mode": "fixed_bounded_raw_state",
                        "feature_names": list(self.feature_names),
                    },
                    "nonlinear_value": self.nonlinear_value.snapshot(),
                })
            return float(update["residual"])
        if not self.uses_shared_value_advantage:
            return self.values[action].update(
                features,
                target,
                self.config.learning_rate,
                observation_weight=observation_weight,
            )

        # Preserve the historical per-action covariance/residual accounting
        # used by the behavior policy, while learning the mean through a
        # coupled shared-value/explicit-advantage parameterization.
        raw_residual = self.values[action].update(
            features,
            target,
            rate=0.0,
            observation_weight=observation_weight,
        )
        update_residual = _clip(
            raw_residual,
            -self.config.td_error_clip,
            self.config.td_error_clip,
        )
        sign = 1.0 if action == STOP else -1.0
        value_scale = (
            self.config.shared_value_learning_rate
            * observation_weight
            * update_residual
        )
        advantage_scale = (
            0.5
            * sign
            * self.config.shared_advantage_learning_rate
            * observation_weight
            * update_residual
        )
        for index, feature in enumerate(features):
            self.shared_value_theta[index] += value_scale * feature
            self.shared_advantage_theta[index] += advantage_scale * feature
        self._sync_shared_action_means()
        return raw_residual


    @property
    def uses_hindsight_block_gain(self) -> bool:
        return self.config.credit_assignment in {
            "hindsight_block_gain",
            "hindsight_delta_j_f5",
            "hindsight_delta_j_f2",
            "hindsight_delta_j_logistic_f2",
        }

    @property
    def uses_hindsight_delta_j_f5(self) -> bool:
        return self.config.credit_assignment == "hindsight_delta_j_f5"

    @property
    def uses_hindsight_delta_j_f2(self) -> bool:
        return self.config.credit_assignment == "hindsight_delta_j_f2"

    @property
    def uses_hindsight_delta_j_logistic_f2(self) -> bool:
        return self.config.credit_assignment == "hindsight_delta_j_logistic_f2"

    @property
    def uses_hindsight_delta_j(self) -> bool:
        return self.config.credit_assignment in {
            "hindsight_delta_j_f5",
            "hindsight_delta_j_f2",
            "hindsight_delta_j_logistic_f2",
        }

    @staticmethod
    def _mean_std(values: Sequence[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(float(value) for value in values) / len(values)
        variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(max(0.0, variance))

    def _hindsight_features(
        self,
        raw_state: Sequence[Sequence[float]],
        *,
        proposal_length: int,
        max_spec_len: int,
        refinement_step: int,
    ) -> tuple[float, ...]:
        active_span = len(raw_state)
        if active_span <= 0 or active_span > RAW_STATE_BLOCK_SIZE:
            raise ValueError("hindsight gain requires a 1..8-token physical state")
        if any(len(row) != len(RAW_TOKEN_FIELDS) for row in raw_state):
            raise ValueError("hindsight raw-state row has an invalid shape")
        if any(float(row[1]) < 0.5 for row in raw_state):
            raise ValueError("hindsight gain cannot use an unobserved token slot")
        top1 = [_clip(row[2], 0.0, 1.0) for row in raw_state]
        margins = [_clip(row[2] - row[3], 0.0, 1.0) for row in raw_state]
        entropy = [_clip(row[4], 0.0, 1.0) for row in raw_state]
        top1_mean, top1_std = self._mean_std(top1)
        margin_mean, margin_std = self._mean_std(margins)
        entropy_mean, entropy_std = self._mean_std(entropy)
        return (
            1.0,
            active_span / RAW_STATE_BLOCK_SIZE,
            sum(float(row[0]) >= 0.5 for row in raw_state) / active_span,
            top1_mean,
            top1_std,
            margin_mean,
            margin_std,
            entropy_mean,
            entropy_std,
            _clip(float(proposal_length) / max(1, int(max_spec_len)), 0.0, 1.0),
            _clip(
                float(refinement_step) / max(1, self.config.max_refinement_steps),
                0.0,
                1.0,
            ),
        )

    def _hindsight_delta_j_f5_features(
        self,
        f5_state: dict,
    ) -> tuple[float, ...]:
        active_span = max(1, int(f5_state["active_span_size"]))
        return (
            1.0,
            _clip(float(f5_state["current_mask_count"]) / active_span, 0.0, 1.0),
            max(0.0, float(f5_state["masked_entropy_std"])),
            _clip(float(f5_state["resolved_margin_mean"]), 0.0, 1.0),
            _clip(float(f5_state["resolved_entropy_max"]), 0.0, 1.0),
            max(0.0, float(self.factual_tokens_per_verifier_ema or 0.0)),
        )

    @staticmethod
    def _hindsight_delta_j_f2_features(f2_state: dict) -> tuple[float, ...]:
        active_span = max(1, int(f2_state["active_span_size"]))
        return (
            1.0,
            _clip(float(f2_state["current_mask_count"]) / active_span, 0.0, 1.0),
            _clip(float(f2_state["first_unresolved_position"]), 0.0, 1.0),
        )

    @staticmethod
    def _hindsight_delta_j_logistic_f2_features(
        *,
        current_mask_count: int,
        active_span_size: int,
        active_block_start_relative: int,
        proposal_length: int,
    ) -> tuple[float, ...]:
        active_span = max(1, int(active_span_size))
        return (
            1.0,
            _clip(float(current_mask_count) / active_span, 0.0, 1.0),
            _clip(
                float(active_block_start_relative) / max(1, int(proposal_length)),
                0.0,
                1.0,
            ),
        )

    def begin_hindsight_problem(self, problem_id) -> None:
        if not self.uses_hindsight_block_gain:
            return
        if self.hindsight_pending_pairs or self.hindsight_pending_sources:
            self.hindsight_censored_count += (
                len(self.hindsight_pending_pairs)
                + len(self.hindsight_pending_sources)
            )
            self.hindsight_pending_pairs.clear()
            self.hindsight_pending_sources.clear()
        self.hindsight_probe_outstanding = False
        self.hindsight_problem_id = int(problem_id)
        self.hindsight_committed_tokens = []
        self.hindsight_current_snapshot = None

    def prepare_hindsight_snapshot(
        self,
        *,
        draft_proposal,
        context_len: int,
        active_block_start: int,
        active_block_end: int,
        raw_current_state,
        f5_state=None,
        f2_state=None,
        proposal_length: int,
        max_spec_len: int,
        refinement_step: int,
        next_forward_latency_ms: float,
        forward_pass_index: int,
        decision_eligible: bool,
        remaining_masks: int | None = None,
    ) -> dict:
        if not self.uses_hindsight_block_gain:
            return {}
        started = time.perf_counter()
        for pending in self.hindsight_pending_sources.values():
            if int(pending["last_forward_pass_index"]) != int(forward_pass_index):
                pending["latency_ms"] += max(
                    0.0, float(next_forward_latency_ms)
                )
                pending["last_forward_pass_index"] = int(forward_pass_index)
        relative_start = int(active_block_start) - int(context_len)
        relative_end = int(active_block_end) - int(context_len)
        active_span = int(active_block_end) - int(active_block_start)
        raw_complete = (
            raw_current_state is not None
            and len(raw_current_state) == active_span
            and 0 < active_span <= RAW_STATE_BLOCK_SIZE
            and all(len(row) == len(RAW_TOKEN_FIELDS) for row in raw_current_state)
            and all(float(row[1]) >= 0.5 for row in raw_current_state)
        )
        if self.uses_hindsight_delta_j_f5:
            state_complete = f5_state is not None
        elif self.uses_hindsight_delta_j_f2:
            state_complete = f2_state is not None
        elif self.uses_hindsight_delta_j_logistic_f2:
            state_complete = remaining_masks is not None
        else:
            state_complete = raw_complete
        valid = (
            state_complete
            and relative_start >= 0
            and relative_end - relative_start == active_span
            and relative_end <= len(draft_proposal)
            and all(token is not None for token in draft_proposal[:relative_end])
        )
        if not valid:
            if not state_complete:
                self.hindsight_unavailable_count += 1
                reason = (
                    "compact_delta_j_summary_incomplete"
                    if self.uses_hindsight_delta_j
                    else "raw_probability_frame_incomplete"
                )
            else:
                self.hindsight_invalid_count += 1
                reason = "candidate_alignment_invalid"
            self.hindsight_current_snapshot = None
            return {
                "hindsight_snapshot_valid": False,
                "hindsight_snapshot_skip_reason": reason,
            }
        if self.uses_hindsight_delta_j_f5:
            features = self._hindsight_delta_j_f5_features(f5_state)
        elif self.uses_hindsight_delta_j_f2:
            features = self._hindsight_delta_j_f2_features(f2_state)
        elif self.uses_hindsight_delta_j_logistic_f2:
            features = self._hindsight_delta_j_logistic_f2_features(
                current_mask_count=int(remaining_masks),
                active_span_size=active_span,
                active_block_start_relative=relative_start,
                proposal_length=proposal_length,
            )
            f2_state = {
                "active_span_size": int(active_span),
                "current_mask_count": int(remaining_masks),
                "global_proposal_position": float(features[2]),
            }
        else:
            features = self._hindsight_features(
                raw_current_state,
                proposal_length=proposal_length,
                max_spec_len=max_spec_len,
                refinement_step=refinement_step,
            )
        if self.uses_hindsight_delta_j_logistic_f2:
            logistic_logit, continue_score = self.hindsight_logistic_model.predict(features)
            normalized_mean, normalized_sigma = 0.0, math.inf
        else:
            normalized_mean, normalized_sigma = self.hindsight_gain_model.predict(
                features,
                include_observation_noise=self.uses_hindsight_delta_j,
            )
            logistic_logit, continue_score = None, None
        snapshot = {
            "snapshot_id": int(self.hindsight_snapshot_count),
            "problem_id": int(self.hindsight_problem_id),
            "output_anchor": len(self.hindsight_committed_tokens),
            "candidate_prefix": [int(token) for token in draft_proposal[:relative_end]],
            "active_block_start_relative": int(relative_start),
            "active_block_end_relative": int(relative_end),
            "active_span_size": int(active_span),
            "features": list(features),
            "f5_state": dict(f5_state or {}),
            "f2_state": dict(f2_state or {}),
            "predicted_gain_tokens": active_span * normalized_mean,
            "predicted_sigma_tokens": active_span * normalized_sigma,
            "predicted_normalized_delta_j": normalized_mean,
            "predicted_normalized_delta_j_sigma": normalized_sigma,
            "logistic_logit": logistic_logit,
            "continue_score": continue_score,
            "rho_at_decision": float(self.rho),
            "decision_eligible": bool(decision_eligible),
            "refinement_step": int(refinement_step),
            "proposal_length": int(proposal_length),
            "forward_pass_index": int(forward_pass_index),
        }
        self.hindsight_snapshot_count += 1
        key = (
            snapshot["problem_id"],
            snapshot["output_anchor"],
            snapshot["active_block_start_relative"],
            snapshot["active_block_end_relative"],
        )
        pending = self.hindsight_pending_sources.pop(key, None)
        if pending is not None:
            self.hindsight_pending_pairs.append({
                "pair_id": int(self.hindsight_pair_count),
                "before": pending["snapshot"],
                "after": snapshot,
                "next_forward_latency_ms": max(
                    0.0, float(pending["latency_ms"])
                ),
                "created_boundary": int(self.completed_rounds),
                "forced_probe": bool(pending.get("forced_probe", False)),
                "verifier_boundary_latency_ms_T_B": None,
            })
            self.hindsight_pair_count += 1
        self.hindsight_current_snapshot = snapshot if decision_eligible else None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.hindsight_snapshot_overhead_ema_ms = self._update_factual_ema(
            self.hindsight_snapshot_overhead_ema_ms,
            elapsed_ms,
        )
        result = {
            "hindsight_snapshot_valid": True,
            "hindsight_snapshot_id": snapshot["snapshot_id"],
            "hindsight_predicted_gain_tokens": snapshot["predicted_gain_tokens"],
            "hindsight_predicted_sigma_tokens": snapshot["predicted_sigma_tokens"],
            "hindsight_training_pairs": (
                self.hindsight_logistic_model.sample_count
                if self.uses_hindsight_delta_j_logistic_f2
                else self.hindsight_gain_model.sample_count
            ),
            "hindsight_features": list(features),
            "hindsight_feature_names": list(self.hindsight_feature_names),
        }
        if self.uses_hindsight_delta_j_f5:
            result.update({
                "current_mask_count": int(f5_state["current_mask_count"]),
                "masked_entropy_std": float(f5_state["masked_entropy_std"]),
                "resolved_margin_mean": float(f5_state["resolved_margin_mean"]),
                "resolved_entropy_max": float(f5_state["resolved_entropy_max"]),
                "ema_tokens_per_verifier": float(
                    self.factual_tokens_per_verifier_ema or 0.0
                ),
                "mu_raw_normalized_delta_j": normalized_mean,
                "sigma_normalized_delta_j": normalized_sigma,
            })
        elif self.uses_hindsight_delta_j_f2:
            result.update({
                "current_mask_count": int(f2_state["current_mask_count"]),
                "first_unresolved_position": float(
                    f2_state["first_unresolved_position"]
                ),
                "mu_raw_normalized_delta_j": normalized_mean,
                "sigma_normalized_delta_j": normalized_sigma,
            })
        elif self.uses_hindsight_delta_j_logistic_f2:
            result.update({
                "current_mask_count": int(f2_state["current_mask_count"]),
                "global_proposal_position": float(features[2]),
                "logistic_logit": float(logistic_logit),
                "continue_score": float(continue_score),
            })
        return result

    @staticmethod
    def _resolve_hindsight_candidate(snapshot: dict, committed: Sequence[int]):
        anchor = int(snapshot["output_anchor"])
        candidate = snapshot["candidate_prefix"]
        available = max(0, len(committed) - anchor)
        compared = min(len(candidate), available)
        lcp = 0
        while lcp < compared and int(candidate[lcp]) == int(committed[anchor + lcp]):
            lcp += 1
        if lcp < compared or available >= len(candidate):
            block_start = int(snapshot["active_block_start_relative"])
            block_end = int(snapshot["active_block_end_relative"])
            yield_tokens = min(
                int(snapshot["active_span_size"]),
                max(0, min(lcp, block_end) - block_start),
            )
            return int(yield_tokens), int(lcp)
        return None

    @staticmethod
    def _resolve_hindsight_prefix(snapshot: dict, committed: Sequence[int]):
        """Resolve a candidate only through its immutable active-block endpoint."""
        anchor = int(snapshot["output_anchor"])
        candidate = snapshot["candidate_prefix"]
        available = max(0, len(committed) - anchor)
        compared = min(len(candidate), available)
        lcp = 0
        while lcp < compared and int(candidate[lcp]) == int(committed[anchor + lcp]):
            lcp += 1
        if lcp < compared:
            return {
                "status": "mismatch",
                "lcp": int(lcp),
                "yield": int(lcp + 1),
                "mismatch_before_active_block": bool(
                    lcp < int(snapshot["active_block_start_relative"])
                ),
            }
        if available >= len(candidate):
            return {
                "status": "pass",
                "lcp": int(len(candidate)),
                "yield": None,
                "mismatch_before_active_block": False,
            }
        return None

    def _observe_hindsight_delta_j_boundary(
        self,
        *,
        terminal: bool,
        boundary_cost_ms: float,
    ) -> None:
        unresolved = []
        for pair in self.hindsight_pending_pairs:
            if pair.get("verifier_boundary_latency_ms_T_B") is None:
                pair["verifier_boundary_latency_ms_T_B"] = max(
                    1e-9, float(boundary_cost_ms)
                )
                pair["frozen_boundary_index"] = int(self.completed_rounds)
            stop_result = self._resolve_hindsight_prefix(
                pair["before"], self.hindsight_committed_tokens
            )
            continue_result = self._resolve_hindsight_prefix(
                pair["after"], self.hindsight_committed_tokens
            )
            if stop_result is None or continue_result is None:
                if terminal:
                    self.hindsight_censored_count += 1
                    reason = "terminal_insufficient_verifier_evidence"
                    self.hindsight_censor_reasons[reason] = (
                        self.hindsight_censor_reasons.get(reason, 0) + 1
                    )
                    if pair.get("forced_probe"):
                        self.hindsight_probe_outstanding = False
                else:
                    unresolved.append(pair)
                continue

            block_end = int(pair["before"]["active_block_end_relative"])
            stop_pass = stop_result["status"] == "pass"
            continue_pass = continue_result["status"] == "pass"
            label_reason = "runtime_break_even"
            if stop_pass and continue_pass:
                y_stop = y_continue = block_end
                label_reason = "both_pass_active_block"
            elif stop_pass and not continue_pass:
                y_stop = block_end
                y_continue = int(continue_result["yield"])
                label_reason = "continue_mismatch_earlier"
            elif not stop_pass and continue_pass:
                y_stop = int(stop_result["yield"])
                y_continue = block_end
                label_reason = "continue_pass_lower_bound"
            else:
                y_stop = int(stop_result["yield"])
                y_continue = int(continue_result["yield"])
                if stop_result["mismatch_before_active_block"]:
                    label_reason = "mismatch_before_active_block"
                elif y_continue > y_stop:
                    label_reason = "continue_extends_prefix"
                elif y_continue < y_stop:
                    label_reason = "stop_mismatch_earlier"

            t_d = max(0.0, float(pair["next_forward_latency_ms"]))
            t_b = max(1e-9, float(pair["verifier_boundary_latency_ms_T_B"]))
            j_stop = t_b / max(1, y_stop)
            j_continue = (t_d + t_b) / max(1, y_continue)
            delta_j = j_continue - j_stop
            normalized_delta_j = delta_j / max(j_stop, 1e-9)

            # A passing CONTINUE candidate supplies only a lower bound on its
            # yield. Learn it only when that lower bound already proves benefit.
            if not stop_pass and continue_pass and delta_j >= 0.0:
                self.hindsight_censored_count += 1
                reason = "continue_pass_lower_bound_not_beneficial"
                self.hindsight_censor_reasons[reason] = (
                    self.hindsight_censor_reasons.get(reason, 0) + 1
                )
                if pair.get("forced_probe"):
                    self.hindsight_probe_outstanding = False
                continue

            if self.uses_hindsight_delta_j_logistic_f2:
                tie_threshold = self.config.hindsight_logistic_tie_ms_per_token
                is_tie = abs(delta_j) <= tie_threshold
                actual_continue = delta_j < -tie_threshold
                features = pair["before"]["features"]
                pre_logit = float(pair["before"]["logistic_logit"])
                pre_score = float(pair["before"]["continue_score"])
                predicted_continue = (
                    pre_score > self.config.hindsight_logistic_continue_threshold
                )
                counts_before = (
                    self.hindsight_delta_j_continue_count,
                    self.hindsight_delta_j_stop_count,
                )
                class_weight = 1.0
                utility_weight = 0.0
                sample_weight = 0.0
                loss = None
                running_median = None
                replay_batch_size_used = 0
                replay_buffer_size_before = len(self.hindsight_logistic_replay_buffer)
                weights_before = list(self.hindsight_logistic_model.weights)
                if is_tie:
                    self.hindsight_logistic_tie_count += 1
                    true_action = "tie"
                else:
                    true_action = CONTINUE if actual_continue else STOP
                    abs_r = abs(normalized_delta_j)
                    if self.config.hindsight_logistic_utility_weighting == "raw_abs":
                        # U1: use the true absolute delta-J magnitude. No class
                        # weighting and no [0.5, 2.5] utility clipping.
                        utility_weight = abs(delta_j)
                        class_weight = 1.0
                        sample_weight = utility_weight
                    else:
                        if self.hindsight_logistic_abs_r:
                            ordered = self.hindsight_logistic_abs_r
                            middle = len(ordered) // 2
                            running_median = (
                                ordered[middle]
                                if len(ordered) % 2
                                else 0.5 * (ordered[middle - 1] + ordered[middle])
                            )
                            utility_weight = _clip(
                                abs_r / max(running_median, 1e-9), 0.5, 2.5
                            )
                        else:
                            utility_weight = 1.0
                        if (
                            actual_continue
                            and self.config.hindsight_logistic_use_class_weight
                        ):
                            n_c, n_s = counts_before
                            alpha = self.config.hindsight_delta_j_class_balance_alpha
                            class_weight = max(1.0, min(
                                self.config.hindsight_delta_j_max_continue_weight,
                                math.sqrt((n_s + alpha) / (n_c + alpha)),
                            ))
                        sample_weight = class_weight * utility_weight

                    if (
                        self.config.hindsight_logistic_utility_weighting == "raw_abs"
                        and self.config.hindsight_logistic_replay_batch_size > 0
                    ):
                        # U1 batch-1x: the current resolved pair enters the bounded
                        # buffer, then exactly one uniform mini-batch SGD update.
                        self.hindsight_logistic_replay_buffer.append((
                            [float(value) for value in features],
                            int(actual_continue),
                            float(sample_weight),
                        ))
                        buffer_limit = self.config.hindsight_logistic_replay_buffer_size
                        if len(self.hindsight_logistic_replay_buffer) > buffer_limit:
                            del self.hindsight_logistic_replay_buffer[:-buffer_limit]
                        replay_batch_size_used = min(
                            self.config.hindsight_logistic_replay_batch_size,
                            len(self.hindsight_logistic_replay_buffer),
                        )
                        if replay_batch_size_used == len(self.hindsight_logistic_replay_buffer):
                            batch = list(self.hindsight_logistic_replay_buffer)
                        else:
                            batch = self.hindsight_logistic_replay_rng.sample(
                                self.hindsight_logistic_replay_buffer,
                                replay_batch_size_used,
                            )
                        loss = self.hindsight_logistic_model.update_batch(batch)
                    else:
                        loss = self.hindsight_logistic_model.update(
                            features,
                            int(actual_continue),
                            sample_weight,
                        )
                    insort(self.hindsight_logistic_abs_r, abs_r)
                    if actual_continue:
                        self.hindsight_delta_j_continue_count += 1
                        self.hindsight_logistic_positive_problem_ids.add(
                            int(pair["before"]["problem_id"])
                        )
                    else:
                        self.hindsight_delta_j_stop_count += 1
                row = {
                    "transition_kind": self.config.credit_assignment,
                    "problem_id": pair["before"]["problem_id"],
                    "pair_id": pair["pair_id"],
                    "before_snapshot_id": pair["before"]["snapshot_id"],
                    "after_snapshot_id": pair["after"]["snapshot_id"],
                    "refinement_step": pair["before"]["refinement_step"],
                    "proposal_length": pair["before"]["proposal_length"],
                    "active_block_start": pair["before"]["active_block_start_relative"],
                    "active_block_end": block_end,
                    "stop_lcp": stop_result["lcp"],
                    "continue_lcp": continue_result["lcp"],
                    "stop_passes_active_block": stop_pass,
                    "continue_passes_active_block": continue_pass,
                    "before_yield_Y_S": y_stop,
                    "after_yield_Y_C": y_continue,
                    "extra_draft_latency_ms_T_D": t_d,
                    "verifier_boundary_latency_ms_T_B": t_b,
                    "J_STOP_ms_per_token": j_stop,
                    "J_CONTINUE_ms_per_token": j_continue,
                    "delta_J_ms_per_token": delta_j,
                    "normalized_delta_J": normalized_delta_j,
                    "is_tie": is_tie,
                    "update_applied": not is_tie,
                    "true_action_from_delta_J": true_action,
                    "binary_label_C": None if is_tie else int(actual_continue),
                    "label_reason": "tie" if is_tie else label_reason,
                    "logistic_logit_before_update": pre_logit,
                    "continue_score_before_update": pre_score,
                    "continue_threshold": self.config.hindsight_logistic_continue_threshold,
                    "utility_weighting_mode": self.config.hindsight_logistic_utility_weighting,
                    "replay_batch_size_config": self.config.hindsight_logistic_replay_batch_size,
                    "replay_buffer_size_config": self.config.hindsight_logistic_replay_buffer_size,
                    "replay_buffer_size_before": replay_buffer_size_before,
                    "replay_buffer_size_after": len(self.hindsight_logistic_replay_buffer),
                    "replay_batch_size_used": replay_batch_size_used,
                    "predicted_continue": predicted_continue,
                    "cost_aware_correct": None if is_tie else predicted_continue == actual_continue,
                    "class_weight": class_weight,
                    "utility_weight": utility_weight,
                    "sample_weight": sample_weight,
                    "weighted_logistic_loss": loss,
                    "N_C": self.hindsight_delta_j_continue_count,
                    "N_S": self.hindsight_delta_j_stop_count,
                    "distinct_positive_problem_count": len(
                        self.hindsight_logistic_positive_problem_ids
                    ),
                    "model_action": pair["before"].get("greedy_action", "unknown"),
                    "executed_action": pair["before"].get("action", "unknown"),
                    "action_source": pair["before"].get("action_source", "unknown"),
                    "behavior_continue_probability": pair["before"].get(
                        "behavior_continue_probability", 1.0
                    ),
                    "executed_continue": True,
                    "pair_resolved": True,
                    "censor_reason": None,
                    "weights_before": weights_before,
                    "weights_after": list(self.hindsight_logistic_model.weights),
                }
                row.update({
                    name: float(value)
                    for name, value in zip(self.hindsight_feature_names, features)
                })
                self.full_stream_transitions.append(row)
                self.hindsight_resolved_count += 1
                if pair.get("forced_probe"):
                    self.hindsight_probe_outstanding = False
                continue

            actual_continue = normalized_delta_j < 0.0
            if actual_continue:
                self.hindsight_delta_j_continue_count += 1
                sample_weight = max(1.0, min(
                    self.config.hindsight_delta_j_max_continue_weight,
                    math.sqrt(
                        (
                            self.hindsight_delta_j_stop_count
                            + self.config.hindsight_delta_j_class_balance_alpha
                        )
                        /
                        (
                            self.hindsight_delta_j_continue_count
                            + self.config.hindsight_delta_j_class_balance_alpha
                        )
                    ),
                ))
            else:
                self.hindsight_delta_j_stop_count += 1
                sample_weight = 1.0

            features = pair["before"]["features"]
            mu_raw = float(pair["before"]["predicted_normalized_delta_j"])
            sigma = float(pair["before"]["predicted_normalized_delta_j_sigma"])
            calibration_before = float(self.hindsight_delta_j_calibration_bias)
            mu_calibrated = mu_raw + calibration_before
            p_continue = _STANDARD_NORMAL.cdf(
                (0.0 - mu_calibrated) / max(sigma, 1e-9)
            )
            predicted_continue = (
                p_continue > self.config.hindsight_delta_j_p_continue_threshold
            )
            weights_before = list(self.hindsight_gain_model.weights)
            residual = normalized_delta_j - mu_raw
            beta = self.config.hindsight_delta_j_calibration_beta
            self.hindsight_delta_j_calibration_bias = (
                (1.0 - beta) * calibration_before + beta * residual
            )
            self.hindsight_gain_model.update(
                features,
                normalized_delta_j,
                observation_weight=sample_weight,
            )
            row = {
                "transition_kind": self.config.credit_assignment,
                "problem_id": pair["before"]["problem_id"],
                "pair_id": pair["pair_id"],
                "before_snapshot_id": pair["before"]["snapshot_id"],
                "after_snapshot_id": pair["after"]["snapshot_id"],
                "refinement_step": pair["before"]["refinement_step"],
                "proposal_length": pair["before"]["proposal_length"],
                "active_block_start": pair["before"]["active_block_start_relative"],
                "active_block_end": block_end,
                "stop_lcp": stop_result["lcp"],
                "continue_lcp": continue_result["lcp"],
                "stop_passes_active_block": stop_pass,
                "continue_passes_active_block": continue_pass,
                "before_yield_Y_S": y_stop,
                "after_yield_Y_C": y_continue,
                "extra_draft_latency_ms_T_D": t_d,
                "verifier_boundary_latency_ms_T_B": t_b,
                "J_STOP_ms_per_token": j_stop,
                "J_CONTINUE_ms_per_token": j_continue,
                "delta_J_ms_per_token": delta_j,
                "normalized_delta_J": normalized_delta_j,
                "true_action_from_delta_J": CONTINUE if actual_continue else STOP,
                "label_reason": label_reason,
                "predicted_r_before_update": mu_raw,
                "calibrated_predicted_r_before_update": mu_calibrated,
                "predicted_p_continue_before_update": p_continue,
                "predicted_continue": predicted_continue,
                "cost_aware_correct": predicted_continue == actual_continue,
                "normalized_residual": residual,
                "sample_weight": sample_weight,
                "N_C": self.hindsight_delta_j_continue_count,
                "N_S": self.hindsight_delta_j_stop_count,
                "calibration_bias_before": calibration_before,
                "calibration_bias_after": self.hindsight_delta_j_calibration_bias,
                "pair_source": pair["before"].get("action_source", "unknown"),
                "resolution_delay_boundaries": (
                    int(self.completed_rounds) - int(pair["created_boundary"])
                ),
                "weights_before": weights_before,
                "weights_after": list(self.hindsight_gain_model.weights),
            }
            row.update({
                name: float(value)
                for name, value in zip(self.hindsight_feature_names, features)
            })
            self.full_stream_transitions.append(row)
            self.hindsight_resolved_count += 1
            if pair.get("forced_probe"):
                self.hindsight_probe_outstanding = False
        self.hindsight_pending_pairs = unresolved

    def observe_hindsight_verifier_boundary(
        self,
        emitted_tokens: Sequence[int],
        *,
        terminal: bool,
        verifier_latency_ms: float = 0.0,
        post_verify_latency_ms: float = 0.0,
    ) -> None:
        if not self.uses_hindsight_block_gain:
            return
        self.hindsight_committed_tokens.extend(int(token) for token in emitted_tokens)
        if self.uses_hindsight_delta_j:
            self._observe_hindsight_delta_j_boundary(
                terminal=terminal,
                boundary_cost_ms=(
                    max(0.0, float(verifier_latency_ms))
                    + max(0.0, float(post_verify_latency_ms))
                ),
            )
            if terminal:
                self.hindsight_censored_count += len(self.hindsight_pending_sources)
                self.hindsight_pending_sources.clear()
                self.hindsight_current_snapshot = None
                self.hindsight_probe_outstanding = False
            return
        unresolved = []
        for pair in self.hindsight_pending_pairs:
            before_result = self._resolve_hindsight_candidate(
                pair["before"], self.hindsight_committed_tokens
            )
            after_result = self._resolve_hindsight_candidate(
                pair["after"], self.hindsight_committed_tokens
            )
            if before_result is None or after_result is None:
                if terminal:
                    self.hindsight_censored_count += 1
                    if pair.get("forced_probe"):
                        self.hindsight_probe_outstanding = False
                else:
                    unresolved.append(pair)
                continue
            before_yield, before_lcp = before_result
            after_yield, after_lcp = after_result
            gain_tokens = int(after_yield - before_yield)
            features = pair["before"]["features"]
            predicted_gain = float(pair["before"]["predicted_gain_tokens"])
            predicted_sigma = float(pair["before"]["predicted_sigma_tokens"])
            active_span = int(pair["before"]["active_span_size"])
            target = gain_tokens / max(1, active_span)
            weights_before = list(self.hindsight_gain_model.weights)
            residual = self.hindsight_gain_model.update(features, target)
            factual_cost_tokens = (
                float(pair["before"]["rho_at_decision"])
                * float(pair["next_forward_latency_ms"])
            )
            predicted_continue = (
                predicted_gain
                - self.config.hindsight_confidence_kappa * predicted_sigma
                > factual_cost_tokens + self.config.hindsight_margin_tokens
            )
            actual_continue = gain_tokens > factual_cost_tokens
            row = {
                "transition_kind": "hindsight_block_gain",
                "problem_id": pair["before"]["problem_id"],
                "pair_id": pair["pair_id"],
                "before_snapshot_id": pair["before"]["snapshot_id"],
                "after_snapshot_id": pair["after"]["snapshot_id"],
                "refinement_step": pair["before"]["refinement_step"],
                "proposal_length": pair["before"]["proposal_length"],
                "active_span_size": active_span,
                "before_lcp": before_lcp,
                "after_lcp": after_lcp,
                "before_active_yield": before_yield,
                "after_active_yield": after_yield,
                "gain_tokens": gain_tokens,
                "normalized_target": target,
                "predicted_gain_tokens_before_update": predicted_gain,
                "predicted_sigma_tokens_before_update": predicted_sigma,
                "prediction_error_tokens": gain_tokens - predicted_gain,
                "normalized_residual": residual,
                "next_forward_latency_ms": pair["next_forward_latency_ms"],
                "rho_tokens_per_ms_at_decision": pair["before"]["rho_at_decision"],
                "factual_cost_tokens": factual_cost_tokens,
                "predicted_continue": bool(predicted_continue),
                "actual_continue": bool(actual_continue),
                "cost_aware_correct": bool(predicted_continue == actual_continue),
                "resolution_delay_boundaries": (
                    int(self.completed_rounds) - int(pair["created_boundary"])
                ),
                "weights_before": list(weights_before),
                "weights_after": list(self.hindsight_gain_model.weights),
            }
            row.update({
                name: float(value)
                for name, value in zip(HINDSIGHT_GAIN_FEATURE_NAMES, features)
            })
            self.full_stream_transitions.append(row)
            self.hindsight_resolved_count += 1
            if pair.get("forced_probe"):
                self.hindsight_probe_outstanding = False
        self.hindsight_pending_pairs = unresolved
        if terminal:
            self.hindsight_censored_count += len(self.hindsight_pending_sources)
            self.hindsight_pending_sources.clear()
            self.hindsight_current_snapshot = None
            self.hindsight_probe_outstanding = False

    def _policy_theta(self, action: str) -> Sequence[float]:
        if (
            self.uses_policy_weight_ema
            and self.policy_ema_initialized[action]
        ):
            return self.policy_ema_theta[action]
        return self.values[action].theta

    def _policy_mean(self, action: str, features: Sequence[float]) -> float:
        return _dot(self._policy_theta(action), features)

    def _update_policy_weight_ema(self, updated_action: str) -> None:
        """Update policy-weight EMA after one factual learner update.

        ``action_step`` reproduces the historical behavior: only the EMA of
        the action whose raw head changed is advanced.

        ``global_step`` treats [theta_STOP, theta_CONTINUE] as one parameter
        vector and advances the EMA of *both* heads after every factual
        learner update.  This gives STOP and CONTINUE the same EMA clock even
        under highly imbalanced action frequencies.
        """
        if not self.uses_policy_weight_ema:
            return
        if updated_action not in {STOP, CONTINUE}:
            raise ValueError(f"unknown EMA-updated action: {updated_action}")

        beta = self.config.policy_weight_ema_beta
        if self.config.policy_weight_ema_mode == "action_step":
            actions_to_advance = (updated_action,)
        else:
            actions_to_advance = (STOP, CONTINUE)
            self.policy_ema_global_update_count += 1

        for action in actions_to_advance:
            online = self.values[action].theta
            if not self.policy_ema_initialized[action]:
                # Initialize from the current raw head, rather than averaging
                # from zero. In global mode both heads initialize on the same
                # global learner step.
                self.policy_ema_theta[action] = list(online)
                self.policy_ema_initialized[action] = True
            else:
                self.policy_ema_theta[action] = [
                    beta * averaged + (1.0 - beta) * current
                    for averaged, current in zip(
                        self.policy_ema_theta[action],
                        online,
                    )
                ]
            self.policy_ema_update_count[action] += 1

    @staticmethod
    def _probability_from_advantage(
        advantage_mean: float,
        advantage_risk: float,
    ) -> float:
        if advantage_risk > 0.0:
            return _STANDARD_NORMAL.cdf(advantage_mean / advantage_risk)
        if advantage_mean > 0.0:
            return 1.0
        if advantage_mean < 0.0:
            return 0.0
        return 0.5

    def _annealed_exploration_floor(self, decision_count: int | None = None) -> float:
        count = (
            self.annealed_decision_count
            if decision_count is None
            else max(0, int(decision_count))
        )
        return max(
            self.config.explore_min,
            self.config.explore_epsilon * (self.config.explore_decay ** count),
        )

    def advance_algorithm_latency(
        self,
        elapsed_ms: float,
        *,
        component: str,
    ) -> float:
        """Advance the selected-path latency ledger used by factual returns."""
        if not self.uses_verifier_boundary_factual:
            return self.algorithm_latency_ms
        elapsed_ms = max(0.0, float(elapsed_ms))
        self.algorithm_latency_ms += elapsed_ms
        self.algorithm_latency_components_ms[component] = (
            self.algorithm_latency_components_ms.get(component, 0.0)
            + elapsed_ms
        )
        return self.algorithm_latency_ms

    def begin_factual_draft_round(self) -> None:
        if not self.uses_verifier_boundary_factual:
            return
        self._draft_checkpoint_wall_s = time.perf_counter()
        self._draft_checkpoint_ledger_ms = self.algorithm_latency_ms

    def checkpoint_factual_draft_latency(self) -> float:
        """Reconcile measured drafting time up to the current adaptive state."""
        if not self.uses_verifier_boundary_factual:
            return 0.0
        if (
            self._draft_checkpoint_wall_s is None
            or self._draft_checkpoint_ledger_ms is None
        ):
            raise RuntimeError("factual draft checkpoint has not been started")
        checkpoint_wall_s = time.perf_counter()
        observed_draft_latency_ms = (
            checkpoint_wall_s - self._draft_checkpoint_wall_s
        ) * 1000.0
        accounted_ms = max(
            0.0,
            self.algorithm_latency_ms - self._draft_checkpoint_ledger_ms,
        )
        residual_ms = max(
            0.0,
            float(observed_draft_latency_ms) - accounted_ms,
        )
        self.advance_algorithm_latency(
            residual_ms,
            component="draft_unattributed",
        )
        overshoot_ms = max(
            0.0,
            accounted_ms - float(observed_draft_latency_ms),
        )
        if overshoot_ms > 0.0:
            self.algorithm_latency_components_ms["draft_reconciliation_overshoot"] = (
                self.algorithm_latency_components_ms.get(
                    "draft_reconciliation_overshoot",
                    0.0,
                )
                + overshoot_ms
            )
        self._draft_checkpoint_wall_s = checkpoint_wall_s
        self._draft_checkpoint_ledger_ms = self.algorithm_latency_ms
        return residual_ms

    def end_factual_draft_round(self) -> float:
        if not self.uses_verifier_boundary_factual:
            return 0.0
        residual_ms = self.checkpoint_factual_draft_latency()
        self._draft_checkpoint_wall_s = None
        self._draft_checkpoint_ledger_ms = None
        return residual_ms

    def evaluate(self, action: str, features: Sequence[float]) -> ActionEstimate:
        return self.values[action].estimate(features)

    def build_features(self, **state) -> tuple[float, ...]:
        if self.config.feature_schema == "otrc_raw_state_v1":
            return build_raw_state_features(
                max_refinement_steps=self.config.max_refinement_steps,
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                factual_tokens_per_verifier_ema=(
                    self.factual_tokens_per_verifier_ema
                ),
                **state,
            )
        if self.config.feature_schema == "otrc_v2_3_compact7_td":
            return build_v23_compact7_state_features(
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                max_refinement_steps=self.config.max_refinement_steps,
                disabled_features=self.config.disabled_features,
                **state,
            )
        if self.config.feature_schema == "otrc_v2_2_compact_td":
            return build_v22_compact_state_features(
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                factual_tokens_per_verifier_ema=(
                    self.factual_tokens_per_verifier_ema
                ),
                disabled_features=self.config.disabled_features,
                **state,
            )
        if self.config.feature_schema == "otrc_v2_2_td":
            return build_v22_state_features(
                factual_draft_latency_ema_ms=self.factual_draft_latency_ema_ms,
                factual_verifier_latency_ema_ms=self.factual_verifier_latency_ema_ms,
                factual_tokens_per_verifier_ema=(
                    self.factual_tokens_per_verifier_ema
                ),
                disabled_features=self.config.disabled_features,
                **state,
            )
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
        started = (
            time.perf_counter()
            if self.uses_verifier_boundary_factual
            else None
        )
        self.factual_verifier_latency_ema_ms = self._update_factual_ema(
            self.factual_verifier_latency_ema_ms,
            latency_ms,
        )
        self.factual_tokens_per_verifier_ema = self._update_factual_ema(
            self.factual_tokens_per_verifier_ema,
            emitted_tokens,
        )
        if started is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.record_profile("factual_verifier_ema_update", elapsed_ms)
            self.advance_algorithm_latency(
                elapsed_ms,
                component="controller_verifier_ema_update",
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
            "policy_theta_stop": list(self._policy_theta(STOP)),
            "policy_theta_continue": list(self._policy_theta(CONTINUE)),
            "policy_theta_diff": [
                stop - continue_
                for stop, continue_ in zip(
                    self._policy_theta(STOP),
                    self._policy_theta(CONTINUE),
                )
            ],
            "policy_weight_ema_mode": self.config.policy_weight_ema_mode,
            "policy_ema_global_update_count": int(
                self.policy_ema_global_update_count
            ),
            "policy_ema_stop_update_count": int(
                self.policy_ema_update_count[STOP]
            ),
            "policy_ema_continue_update_count": int(
                self.policy_ema_update_count[CONTINUE]
            ),
            "value_parameterization": self.config.value_parameterization,
            "shared_value_theta": list(self.shared_value_theta),
            "shared_advantage_theta": list(self.shared_advantage_theta),
            "value_model": self.config.value_model,
            "nonlinear_diagnostics": (
                self.nonlinear_value.diagnostics()
                if self.nonlinear_value is not None else None
            ),
        })

    def _hindsight_probe_probability(self, observations: int) -> float:
        config = self.config
        return config.hindsight_probe_floor + (
            config.hindsight_probe_initial - config.hindsight_probe_floor
        ) * math.exp(-float(observations) / config.hindsight_probe_decay_pairs)

    def _choose_hindsight_gain(
        self,
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool = True,
    ) -> AdaptiveDecision:
        started = time.perf_counter()
        snapshot = self.hindsight_current_snapshot
        if snapshot is None:
            predicted_gain = 0.0
            predicted_sigma = math.inf
        else:
            predicted_gain = float(snapshot["predicted_gain_tokens"])
            predicted_sigma = float(snapshot["predicted_sigma_tokens"])
        expected_forward_ms = max(
            0.0,
            float(
                self.factual_draft_latency_ema_ms
                if self.factual_draft_latency_ema_ms is not None
                else self.forward_latency_ema_ms or 0.0
            ),
        )
        expected_overhead_ms = max(
            0.0, float(self.hindsight_snapshot_overhead_ema_ms or 0.0)
        )
        break_even_gain = self.rho * (expected_forward_ms + expected_overhead_ms)
        lower_gain = (
            predicted_gain
            - self.config.hindsight_confidence_kappa * predicted_sigma
        )
        observations = self.hindsight_gain_model.sample_count
        cold_start = (
            observations < self.config.early_stop_min_observations
            or not math.isfinite(predicted_sigma)
            or predicted_sigma > self.config.hindsight_max_uncertainty_tokens
        )
        exploration_used = False
        probe_probability = 0.0
        probe_eligible = False
        if not allow_stop or snapshot is None:
            action, reason = CONTINUE, "hindsight_candidate_unavailable"
        elif refinement_step >= self.config.max_refinement_steps:
            action, reason = STOP, "hindsight_max_refinement_steps"
        elif lower_gain > break_even_gain + self.config.hindsight_margin_tokens:
            action, reason = CONTINUE, "hindsight_gain_exceeds_cost"
        else:
            action, reason = STOP, "hindsight_gain_not_worth_cost"
        greedy_action = action
        if action == STOP and reason == "hindsight_gain_not_worth_cost":
            uncertainty_high = (
                not math.isfinite(predicted_sigma)
                or predicted_sigma >= self.config.hindsight_probe_uncertainty_tokens
            )
            near_boundary = (
                math.isfinite(predicted_sigma)
                and abs(predicted_gain - break_even_gain)
                <= self.config.hindsight_probe_boundary_scale * predicted_sigma
            )
            probe_budget = max(
                1,
                math.ceil(
                    self.config.hindsight_probe_max_fraction
                    * max(1, self.decision_count + 1)
                ),
            )
            probe_eligible = (
                (uncertainty_high or near_boundary)
                and not self.hindsight_probe_outstanding
                and self.hindsight_probe_count < probe_budget
                and self.config.hindsight_probe_initial > 0.0
                and allow_exploration
            )
            if probe_eligible:
                probe_probability = self._hindsight_probe_probability(observations)
                if self.rng.random() < probe_probability:
                    action = CONTINUE
                    reason = "hindsight_uncertainty_probe"
                    exploration_used = True
                    self.exploration_count += 1
                    self.hindsight_probe_count += 1
                    self.hindsight_probe_outstanding = True
        if snapshot is not None:
            snapshot["action"] = action
            snapshot["decision_reason"] = reason
            snapshot["estimated_break_even_gain_tokens"] = break_even_gain
            snapshot["predicted_gain_lcb_tokens"] = lower_gain
        if action == CONTINUE and snapshot is not None:
            key = (
                snapshot["problem_id"],
                snapshot["output_anchor"],
                snapshot["active_block_start_relative"],
                snapshot["active_block_end_relative"],
            )
            self.hindsight_pending_sources[key] = {
                "snapshot": snapshot,
                "latency_ms": 0.0,
                "last_forward_pass_index": snapshot["forward_pass_index"],
                "forced_probe": bool(exploration_used),
            }
        self.hindsight_current_snapshot = None
        safe_sigma = predicted_sigma if math.isfinite(predicted_sigma) else 1e9
        stop = ActionEstimate(
            mean=break_even_gain,
            risk=0.0,
            lower=break_even_gain,
            upper=break_even_gain,
        )
        continue_ = ActionEstimate(
            mean=predicted_gain,
            risk=safe_sigma,
            lower=predicted_gain - safe_sigma,
            upper=predicted_gain + safe_sigma,
        )
        advantage_mean = break_even_gain - predicted_gain
        advantage_risk = safe_sigma
        stop_probability = self._probability_from_advantage(
            advantage_mean, advantage_risk
        )
        self.decision_count += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.record_profile("decision_total", latency_ms)
        self.advance_algorithm_latency(
            latency_ms, component="hindsight_controller_decision"
        )
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=stop,
            continue_=continue_,
            rho_tokens_per_ms=self.rho,
            exploration_used=exploration_used,
            latency_ms=latency_ms,
            early_stop_observations=observations,
            calibration_active=cold_start,
            advantage_mean=advantage_mean,
            advantage_risk=advantage_risk,
            stop_probability=stop_probability,
            behavior_stop_probability=(
                1.0 - probe_probability
                if probe_eligible
                else (1.0 if greedy_action == STOP else 0.0)
            ),
            selected_action_probability=(
                probe_probability
                if exploration_used
                else (1.0 - probe_probability if probe_eligible else 1.0)
            ),
            importance_weight=1.0,
            diagnostics={
                "controller_name": "hindsight_block_gain",
                "hindsight_predicted_gain_tokens": predicted_gain,
                "hindsight_predicted_sigma_tokens": predicted_sigma,
                "hindsight_predicted_gain_lcb_tokens": lower_gain,
                "hindsight_break_even_gain_tokens": break_even_gain,
                "hindsight_expected_forward_ms": expected_forward_ms,
                "hindsight_expected_overhead_ms": expected_overhead_ms,
                "hindsight_observation_count": observations,
                "hindsight_cold_start": cold_start,
                "greedy_action": greedy_action,
                "executed_action": action,
                "hindsight_probe_probability": probe_probability,
                "hindsight_probe_eligible": probe_eligible,
                "hindsight_probe_count": self.hindsight_probe_count,
                "hindsight_probe_outstanding": self.hindsight_probe_outstanding,
                "raw_advantage": advantage_mean,
                "raw_stop_probability": stop_probability,
            },
        )

    def _choose_hindsight_delta_j_logistic_f2(
        self,
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool,
        failfast_fallback_action: str,
    ) -> AdaptiveDecision:
        started = time.perf_counter()
        snapshot = self.hindsight_current_snapshot
        score = 0.5 if snapshot is None else float(snapshot["continue_score"])
        logit = 0.0 if snapshot is None else float(snapshot["logistic_logit"])
        mask_count = int((snapshot or {}).get("f2_state", {}).get("current_mask_count", 0))
        observations = self.hindsight_logistic_model.sample_count
        learner_ready = bool(
            observations >= self.config.hindsight_delta_j_min_pairs
            and self.hindsight_delta_j_continue_count
            >= self.config.hindsight_delta_j_min_continue_pairs
            and len(self.hindsight_logistic_positive_problem_ids)
            >= self.config.hindsight_logistic_min_positive_problems
        )
        exploration_used = False
        probe_probability = 0.0
        structural_eligible = mask_count >= 2
        if not allow_stop or snapshot is None:
            action, action_source = CONTINUE, "physical_constraint"
            reason = "hindsight_candidate_unavailable"
        elif refinement_step >= self.config.max_refinement_steps:
            action, action_source = STOP, "max_refinement_stop"
            reason = "hindsight_max_refinement_steps"
        elif not learner_ready:
            action = failfast_fallback_action
            action_source = "cold_start_continue" if action == CONTINUE else "cold_start_stop"
            reason = "hindsight_failfast_cold_start"
        elif score > self.config.hindsight_logistic_continue_threshold:
            action, action_source = CONTINUE, "learned_continue"
            reason = "hindsight_logistic_continue"
        else:
            action, action_source = STOP, "learned_stop"
            reason = "hindsight_logistic_stop"
        model_action = action
        if (
            action == STOP
            and allow_exploration
            and snapshot is not None
            and refinement_step < self.config.max_refinement_steps
            and not self.hindsight_probe_outstanding
        ):
            if structural_eligible:
                probe_probability = self.config.hindsight_delta_j_structural_probe_probability
                probe_source = "structural_probe"
            else:
                probe_probability = self.config.hindsight_delta_j_floor_probe_probability
                probe_source = "floor_probe"
            if self.rng.random() < probe_probability:
                action, action_source = CONTINUE, probe_source
                reason = f"hindsight_{probe_source}"
                exploration_used = True
                self.exploration_count += 1
                self.hindsight_probe_count += 1
                self.hindsight_probe_outstanding = True
                if probe_source == "structural_probe":
                    self.hindsight_structural_probe_count += 1
                else:
                    self.hindsight_floor_probe_count += 1
        behavior_continue_probability = (
            probe_probability if model_action == STOP else 1.0
        )
        selected_action_probability = (
            behavior_continue_probability
            if action == CONTINUE
            else 1.0 - behavior_continue_probability
        )
        if snapshot is not None:
            snapshot.update({
                "action": action,
                "greedy_action": model_action,
                "model_action": model_action,
                "action_source": action_source,
                "decision_reason": reason,
                "learner_ready": learner_ready,
                "behavior_continue_probability": behavior_continue_probability,
            })
        if action == CONTINUE and snapshot is not None:
            key = (
                snapshot["problem_id"], snapshot["output_anchor"],
                snapshot["active_block_start_relative"],
                snapshot["active_block_end_relative"],
            )
            self.hindsight_pending_sources[key] = {
                "snapshot": snapshot,
                "latency_ms": 0.0,
                "last_forward_pass_index": snapshot["forward_pass_index"],
                "forced_probe": bool(exploration_used),
            }
        self.hindsight_current_snapshot = None
        self.decision_count += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.record_profile("decision_total", latency_ms)
        self.advance_algorithm_latency(latency_ms, component="hindsight_controller_decision")
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=ActionEstimate(mean=1.0 - score, risk=0.0, lower=1.0 - score, upper=1.0 - score),
            continue_=ActionEstimate(mean=score, risk=0.0, lower=score, upper=score),
            rho_tokens_per_ms=self.rho,
            exploration_used=exploration_used,
            latency_ms=latency_ms,
            early_stop_observations=observations,
            calibration_active=learner_ready,
            advantage_mean=logit,
            advantage_risk=0.0,
            stop_probability=1.0 - score,
            behavior_stop_probability=1.0 - behavior_continue_probability,
            selected_action_probability=selected_action_probability,
            importance_weight=1.0,
            diagnostics={
                "controller_name": self.config.credit_assignment,
                "logistic_logit": logit,
                "continue_score": score,
                "continue_threshold": self.config.hindsight_logistic_continue_threshold,
                "utility_weighting_mode": self.config.hindsight_logistic_utility_weighting,
                "replay_batch_size": self.config.hindsight_logistic_replay_batch_size,
                "replay_buffer_size": len(self.hindsight_logistic_replay_buffer),
                "model_action": model_action,
                "executed_action": action,
                "action_source": action_source,
                "learner_ready": learner_ready,
                "N_pairs": observations,
                "N_C": self.hindsight_delta_j_continue_count,
                "N_S": self.hindsight_delta_j_stop_count,
                "distinct_positive_problem_count": len(self.hindsight_logistic_positive_problem_ids),
                "behavior_continue_probability": behavior_continue_probability,
                "probe_probability": probe_probability,
                "structural_probe_eligible": structural_eligible,
                "failfast_fallback_action": failfast_fallback_action,
            },
        )

    def _choose_hindsight_delta_j_f5(
        self,
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool,
        failfast_fallback_action: str = CONTINUE,
    ) -> AdaptiveDecision:
        started = time.perf_counter()
        snapshot = self.hindsight_current_snapshot
        if snapshot is None:
            mu_raw, sigma = 0.0, math.inf
            mask_count = 0
        else:
            mu_raw = float(snapshot["predicted_normalized_delta_j"])
            sigma = float(snapshot["predicted_normalized_delta_j_sigma"])
            compact_state = (
                snapshot.get("f2_state", {})
                if self.uses_hindsight_delta_j_f2
                else snapshot.get("f5_state", {})
            )
            mask_count = int(compact_state.get("current_mask_count", 0))
        mu_cal = mu_raw + self.hindsight_delta_j_calibration_bias
        p_continue = (
            0.5
            if not math.isfinite(sigma)
            else _STANDARD_NORMAL.cdf((0.0 - mu_cal) / max(sigma, 1e-9))
        )
        observations = self.hindsight_gain_model.sample_count
        learner_ready = bool(
            observations >= self.config.hindsight_delta_j_min_pairs
            and self.hindsight_delta_j_continue_count
            >= self.config.hindsight_delta_j_min_continue_pairs
        )
        exploration_used = False
        probe_probability = 0.0
        structural_eligible = mask_count >= 2
        if not allow_stop or snapshot is None:
            action = CONTINUE
            action_source = "physical_constraint"
            reason = "hindsight_candidate_unavailable"
        elif refinement_step >= self.config.max_refinement_steps:
            action = STOP
            action_source = "max_refinement_stop"
            reason = "hindsight_max_refinement_steps"
        elif not learner_ready:
            action = failfast_fallback_action
            action_source = "failfast_cold_start"
            reason = "hindsight_failfast_cold_start"
        elif p_continue > self.config.hindsight_delta_j_p_continue_threshold:
            action = CONTINUE
            action_source = "learned_continue"
            reason = "hindsight_posterior_continue"
        else:
            action = STOP
            action_source = "learned_stop"
            reason = "hindsight_posterior_stop"
        greedy_action = action
        if (
            action == STOP
            and allow_exploration
            and snapshot is not None
            and refinement_step < self.config.max_refinement_steps
            and not self.hindsight_probe_outstanding
        ):
            if structural_eligible:
                probe_probability = (
                    self.config.hindsight_delta_j_structural_probe_probability
                )
                probe_source = "structural_probe"
            else:
                probe_probability = self.config.hindsight_delta_j_floor_probe_probability
                probe_source = "floor_probe"
            if self.rng.random() < probe_probability:
                action = CONTINUE
                action_source = probe_source
                reason = f"hindsight_{probe_source}"
                exploration_used = True
                self.exploration_count += 1
                self.hindsight_probe_count += 1
                self.hindsight_probe_outstanding = True
                if probe_source == "structural_probe":
                    self.hindsight_structural_probe_count += 1
                else:
                    self.hindsight_floor_probe_count += 1
        if snapshot is not None:
            snapshot.update({
                "action": action,
                "greedy_action": greedy_action,
                "action_source": action_source,
                "decision_reason": reason,
                "learner_ready": learner_ready,
                "calibration_bias": self.hindsight_delta_j_calibration_bias,
                "mu_calibrated_normalized_delta_j": mu_cal,
                "p_continue": p_continue,
            })
        if action == CONTINUE and snapshot is not None:
            key = (
                snapshot["problem_id"],
                snapshot["output_anchor"],
                snapshot["active_block_start_relative"],
                snapshot["active_block_end_relative"],
            )
            self.hindsight_pending_sources[key] = {
                "snapshot": snapshot,
                "latency_ms": 0.0,
                "last_forward_pass_index": snapshot["forward_pass_index"],
                "forced_probe": bool(exploration_used),
            }
        self.hindsight_current_snapshot = None
        safe_sigma = sigma if math.isfinite(sigma) else 1e9
        stop_estimate = ActionEstimate(mean=0.0, risk=0.0, lower=0.0, upper=0.0)
        continue_estimate = ActionEstimate(
            mean=-mu_cal,
            risk=safe_sigma,
            lower=-mu_cal - safe_sigma,
            upper=-mu_cal + safe_sigma,
        )
        self.decision_count += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.record_profile("decision_total", latency_ms)
        self.advance_algorithm_latency(latency_ms, component="hindsight_controller_decision")
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=stop_estimate,
            continue_=continue_estimate,
            rho_tokens_per_ms=self.rho,
            exploration_used=exploration_used,
            latency_ms=latency_ms,
            early_stop_observations=observations,
            calibration_active=learner_ready,
            advantage_mean=mu_cal,
            advantage_risk=safe_sigma,
            stop_probability=1.0 - p_continue,
            behavior_stop_probability=(
                1.0 - probe_probability
                if greedy_action == STOP else 0.0
            ),
            selected_action_probability=(
                probe_probability
                if exploration_used
                else (1.0 - probe_probability if greedy_action == STOP else 1.0)
            ),
            importance_weight=1.0,
            diagnostics={
                "controller_name": self.config.credit_assignment,
                "mu_raw_normalized_delta_j": mu_raw,
                "sigma_normalized_delta_j": safe_sigma,
                "calibration_bias": self.hindsight_delta_j_calibration_bias,
                "mu_calibrated_normalized_delta_j": mu_cal,
                "p_continue": p_continue,
                "p_continue_threshold": self.config.hindsight_delta_j_p_continue_threshold,
                "learner_ready": learner_ready,
                "cold_start_reason": None if learner_ready else "insufficient_resolved_pairs",
                "failfast_fallback_action": failfast_fallback_action,
                "greedy_action": greedy_action,
                "executed_action": action,
                "action_source": action_source,
                "allow_exploration": bool(allow_exploration),
                "probe_probability": probe_probability,
                "structural_probe_eligible": structural_eligible,
                "N_C": self.hindsight_delta_j_continue_count,
                "N_S": self.hindsight_delta_j_stop_count,
                "continue_sample_weight": max(1.0, min(
                    self.config.hindsight_delta_j_max_continue_weight,
                    math.sqrt(
                        (self.hindsight_delta_j_stop_count + self.config.hindsight_delta_j_class_balance_alpha)
                        / (self.hindsight_delta_j_continue_count + self.config.hindsight_delta_j_class_balance_alpha)
                    ),
                )),
                "raw_advantage": mu_cal,
                "raw_stop_probability": 1.0 - p_continue,
            },
        )
    def choose(
        self,
        features: Sequence[float],
        *,
        allow_stop: bool,
        refinement_step: int,
        allow_exploration: bool = True,
        **_unused,
    ) -> AdaptiveDecision:
        if self.uses_hindsight_block_gain:
            if self.uses_hindsight_delta_j_logistic_f2:
                return self._choose_hindsight_delta_j_logistic_f2(
                    allow_stop=allow_stop,
                    refinement_step=refinement_step,
                    allow_exploration=allow_exploration,
                    failfast_fallback_action=_unused.get(
                        "failfast_fallback_action", CONTINUE
                    ),
                )
            if self.uses_hindsight_delta_j:
                return self._choose_hindsight_delta_j_f5(
                    allow_stop=allow_stop,
                    refinement_step=refinement_step,
                    allow_exploration=allow_exploration,
                    failfast_fallback_action=_unused.get(
                        "failfast_fallback_action", CONTINUE
                    ),
                )
            return self._choose_hindsight_gain(
                allow_stop=allow_stop,
                refinement_step=refinement_step,
                allow_exploration=allow_exploration,
            )
        started = time.perf_counter()
        profiling = self.config.profile_overhead
        stop_started = time.perf_counter() if profiling else None
        nonlinear_prediction = (
            self.nonlinear_value.predict(features)
            if self.nonlinear_value is not None else None
        )
        raw_stop_mean = (
            nonlinear_prediction[2] if nonlinear_prediction is not None
            else self.values[STOP].mean(features)
        )
        stop_mean = (
            raw_stop_mean if nonlinear_prediction is not None
            else self._policy_mean(STOP, features)
        )
        if stop_started is not None:
            self.record_profile(
                "q_stop",
                (time.perf_counter() - stop_started) * 1000.0,
            )
        continue_started = time.perf_counter() if profiling else None
        raw_continue_mean = (
            nonlinear_prediction[3] if nonlinear_prediction is not None
            else self.values[CONTINUE].mean(features)
        )
        continue_mean = (
            raw_continue_mean if nonlinear_prediction is not None
            else self._policy_mean(CONTINUE, features)
        )
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
        raw_advantage_mean = raw_stop_mean - raw_continue_mean
        advantage_risk = self.config.risk_beta * math.sqrt(
            stop.risk * stop.risk + continue_.risk * continue_.risk
        )
        stop_probability = self._probability_from_advantage(
            advantage_mean,
            advantage_risk,
        )
        raw_stop_probability = self._probability_from_advantage(
            raw_advantage_mean,
            advantage_risk,
        )
        probability_margin = (
            self.stop_z_threshold * advantage_risk + self.config.q_margin
        )
        behavior_stop_probability = stop_probability
        selected_action_probability = 1.0
        symmetric_sampling_used = False
        exploration_floor = 0.0
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
        elif self.config.policy_ablation == "frozen_stop":
            action, reason = STOP, "frozen_stop_control"
        elif self.config.policy_ablation == "random_stop":
            symmetric_sampling_used = True
            behavior_stop_probability = 0.5
            selected_action_probability = 0.5
            if self.rng.random() < 0.5:
                action, reason = STOP, "random_stop_control"
            else:
                action, reason = CONTINUE, "random_stop_control"
        elif self.config.force_continue:
            action, reason = CONTINUE, "force_continue"
        elif self.config.fixed_refinement_steps is not None:
            if refinement_step >= self.config.fixed_refinement_steps:
                action, reason = STOP, "fixed_refinement_depth"
            else:
                action, reason = CONTINUE, "fixed_refinement_depth"
        elif refinement_step >= self.config.max_refinement_steps:
            action, reason = STOP, "max_refinement_steps"
        elif self.config.policy_mode == "symmetric_annealed":
            greedy_action = (
                STOP if advantage_mean > self.config.q_margin else CONTINUE
            )
            if allow_exploration:
                symmetric_sampling_used = True
                exploration_floor = self._annealed_exploration_floor()
                behavior_stop_probability = _clip(
                    stop_probability,
                    exploration_floor,
                    1.0 - exploration_floor,
                )
                reason = "symmetric_annealed_sample"
                if self.rng.random() < behavior_stop_probability:
                    action = STOP
                    selected_action_probability = behavior_stop_probability
                else:
                    action = CONTINUE
                    selected_action_probability = (
                        1.0 - behavior_stop_probability
                    )
                exploration_used = action != greedy_action
                if exploration_used:
                    self.exploration_count += 1
                self.annealed_decision_count += 1
            elif advantage_mean > self.config.q_margin:
                action, reason = STOP, "symmetric_greedy_stop"
            else:
                action, reason = CONTINUE, "symmetric_greedy_continue"
        elif self.config.policy_mode in {"symmetric", "symmetric_greedy"}:
            if (
                self.config.policy_mode == "symmetric"
                and allow_exploration
            ):
                symmetric_sampling_used = True
                exploration_floor = self.config.min_action_probability
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
        self.advance_algorithm_latency(
            latency_ms,
            component="controller_decision",
        )
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
                "value_parameterization": self.config.value_parameterization,
                "shared_value_mean": _dot(
                    self.shared_value_theta,
                    features,
                ) if self.uses_shared_value_advantage and nonlinear_prediction is None else (
                    nonlinear_prediction[0] if nonlinear_prediction is not None else 0.5 * (
                    raw_stop_mean + raw_continue_mean
                    )
                ),
                "explicit_advantage_mean": _dot(
                    self.shared_advantage_theta,
                    features,
                ) if self.uses_shared_value_advantage and nonlinear_prediction is None else (
                    nonlinear_prediction[1] if nonlinear_prediction is not None else raw_advantage_mean
                ),
                "value_model": self.config.value_model,
                "nonlinear_loss": (
                    self.nonlinear_value.last_loss if self.nonlinear_value else 0.0
                ),
                "nonlinear_gradient_norm": (
                    self.nonlinear_value.last_gradient_norm if self.nonlinear_value else 0.0
                ),
                "legacy_advantage_risk": advantage_risk,
                "greedy_action": (
                    STOP if advantage_mean > self.config.q_margin else CONTINUE
                ),
                "executed_action": action,
                "policy_weight_ema_enabled": self.uses_policy_weight_ema,
                "policy_weight_ema_beta": self.config.policy_weight_ema_beta,
                "policy_weight_ema_mode": self.config.policy_weight_ema_mode,
                "policy_ema_global_update_count": self.policy_ema_global_update_count,
                "policy_ema_stop_update_count": self.policy_ema_update_count[STOP],
                "policy_ema_continue_update_count": self.policy_ema_update_count[CONTINUE],
                "raw_q_stop": raw_stop_mean,
                "raw_q_continue": raw_continue_mean,
                "ema_q_stop": stop.mean,
                "ema_q_continue": continue_.mean,
                "raw_advantage": raw_advantage_mean,
                "ema_advantage": advantage_mean,
                "raw_stop_probability": raw_stop_probability,
                "ema_stop_probability": stop_probability,
                "exploration_floor": exploration_floor,
                "raw_greedy_action": (
                    STOP
                    if raw_advantage_mean > self.config.q_margin
                    else CONTINUE
                ),
                "ema_greedy_action": (
                    STOP if advantage_mean > self.config.q_margin else CONTINUE
                ),
                "raw_ema_greedy_disagreement": (
                    (raw_advantage_mean > self.config.q_margin)
                    != (advantage_mean > self.config.q_margin)
                ),
                "raw_bias_difference": (
                    self.values[STOP].theta[0]
                    - self.values[CONTINUE].theta[0]
                ),
                "ema_bias_difference": (
                    self._policy_theta(STOP)[0]
                    - self._policy_theta(CONTINUE)[0]
                ),
                "stop_weight_ema_distance_l2": math.sqrt(sum(
                    (raw - averaged) ** 2
                    for raw, averaged in zip(
                        self.values[STOP].theta,
                        self._policy_theta(STOP),
                    )
                )),
                "continue_weight_ema_distance_l2": math.sqrt(sum(
                    (raw - averaged) ** 2
                    for raw, averaged in zip(
                        self.values[CONTINUE].theta,
                        self._policy_theta(CONTINUE),
                    )
                )),
            },
        )

    def resolve_pending_stop(
        self,
        next_features: Sequence[float],
        *,
        next_stop_available: bool = True,
        observed_at: float | None = None,
    ) -> dict | None:
        if (
            self.uses_verifier_boundary_factual
            or not self.config.full_stream_bootstrap
            or self.pending_stop is None
        ):
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

    def resolve_pending_factual_boundary(
        self,
        next_features: Sequence[float],
        *,
        next_stop_available: bool = True,
    ) -> list[dict]:
        """Resolve selected actions against the next real adaptive state."""
        if not self.uses_verifier_boundary_factual:
            return []
        pending = self.pending_factual_boundary
        if pending is None:
            return []
        self.pending_factual_boundary = None
        update_started = time.perf_counter()
        bootstrap_value = self.values[CONTINUE].mean(next_features)
        if next_stop_available:
            bootstrap_value = max(
                self.values[STOP].mean(next_features),
                bootstrap_value,
            )
        transitions = self._apply_factual_boundary(
            pending,
            end_algorithm_latency_ms=self.algorithm_latency_ms,
            bootstrap_value=bootstrap_value,
            terminal=False,
        )
        elapsed_ms = (time.perf_counter() - update_started) * 1000.0
        self.record_profile("factual_boundary_resolution", elapsed_ms)
        self.advance_algorithm_latency(
            elapsed_ms,
            component="controller_factual_update",
        )
        return transitions

    def _apply_factual_boundary(
        self,
        pending: dict,
        *,
        end_algorithm_latency_ms: float,
        bootstrap_value: float,
        terminal: bool,
    ) -> list[dict]:
        transitions = []
        boundary_id = int(pending["boundary_id"])
        emitted_tokens = int(pending["emitted_tokens"])
        boundary_algorithm_latency_ms = float(
            pending["boundary_algorithm_latency_ms"]
        )
        first_boundary_algorithm_latency_ms = float(
            pending["first_boundary_algorithm_latency_ms"]
        )
        verifier_boundaries_spanned = int(
            pending["verifier_boundaries_spanned"]
        )
        records = list(pending["records"])
        verifier_boundary_index = int(
            pending.get("verifier_boundary_index", boundary_id)
        )
        update_applied = (
            verifier_boundary_index > self.config.rho_warmup_boundaries
        )
        for decision_index, record in enumerate(records):
            action = str(record["action"])
            anchor_ms = float(record["algorithm_latency_anchor_ms"])
            delta_time_ms = max(
                0.0,
                float(end_algorithm_latency_ms) - anchor_ms,
            )
            rho_t = max(0.0, float(record["rho_tokens_per_ms"]))
            target = (
                float(emitted_tokens)
                - rho_t * delta_time_ms
                + float(bootstrap_value)
            )
            if update_applied:
                residual = self._update_factual_action_value(
                    action,
                    record["features"],
                    target,
                    observation_weight=float(record["importance_weight"]),
                )
                self._update_policy_weight_ema(action)
                self.factual_learning_update_count += 1
                if action == STOP:
                    self.early_stop_observations = self.values[STOP].sample_count
            else:
                residual = target - self.values[action].mean(record["features"])
                self.factual_warmup_transition_count += 1
            transition = {
                "credit_assignment": self.config.credit_assignment,
                "boundary_id": boundary_id,
                "verifier_boundary_index": verifier_boundary_index,
                "decision_index_in_boundary": int(decision_index),
                "decisions_in_boundary": int(len(records)),
                "action": action,
                "terminal": bool(terminal),
                "emitted_tokens": emitted_tokens,
                "algorithm_latency_anchor_ms": anchor_ms,
                "boundary_algorithm_latency_ms": (
                    boundary_algorithm_latency_ms
                ),
                "first_boundary_algorithm_latency_ms": (
                    first_boundary_algorithm_latency_ms
                ),
                "verifier_boundaries_spanned": verifier_boundaries_spanned,
                "end_algorithm_latency_ms": float(end_algorithm_latency_ms),
                "delta_time_ms": delta_time_ms,
                "post_boundary_latency_ms": max(
                    0.0,
                    float(end_algorithm_latency_ms)
                    - boundary_algorithm_latency_ms,
                ),
                "rho_tokens_per_ms": rho_t,
                "bootstrap_value": float(bootstrap_value),
                "td_target": target,
                "td_error": residual,
                "update_applied": bool(update_applied),
                "rho_warmup_boundaries": int(
                    self.config.rho_warmup_boundaries
                ),
                "importance_weight": float(record["importance_weight"]),
                "value_parameterization": self.config.value_parameterization,
                "value_model": self.config.value_model,
                "nonlinear_loss": (
                    self.nonlinear_value.last_loss
                    if self.nonlinear_value is not None and update_applied else 0.0
                ),
                "nonlinear_gradient_norm": (
                    self.nonlinear_value.last_gradient_norm
                    if self.nonlinear_value is not None and update_applied else 0.0
                ),
                "clipped_td_error": (
                    self.nonlinear_value.last_clipped_residual
                    if self.nonlinear_value is not None and update_applied
                    else _clip(
                        residual,
                        -self.config.td_error_clip,
                        self.config.td_error_clip,
                    )
                ),
                "nonlinear_parameter_norm": (
                    self.nonlinear_value.last_parameter_norm
                    if self.nonlinear_value is not None and update_applied else 0.0
                ),
                "decision_shared_value": float(
                    record.get("decision_shared_value", 0.0)
                ),
                "decision_explicit_advantage": float(
                    record.get("decision_explicit_advantage", 0.0)
                ),
                "selected_action_probability": float(
                    record["selected_action_probability"]
                ),
            }
            self.full_stream_transitions.append(transition)
            transitions.append(transition)
            source = record.get("source_record")
            if source is not None:
                source.update({
                    "factual_boundary_id": boundary_id,
                    "factual_target": target,
                    "factual_td_error": residual,
                    "factual_delta_time_ms": delta_time_ms,
                    "factual_bootstrap_value": float(bootstrap_value),
                    "factual_update_applied": bool(update_applied),
                    "factual_rho_warmup_boundaries": int(
                        self.config.rho_warmup_boundaries
                    ),
                    "factual_loss": transition["nonlinear_loss"],
                    "factual_gradient_norm": transition["nonlinear_gradient_norm"],
                })
        return transitions

    def _complete_verifier_boundary_factual(
        self,
        trajectory: Sequence[dict],
        *,
        emitted_tokens: int,
        verifier_latency_ms: float,
        post_verify_latency_ms: float,
        terminal: bool,
    ) -> None:
        self.advance_algorithm_latency(
            verifier_latency_ms,
            component="verifier",
        )
        self.advance_algorithm_latency(
            post_verify_latency_ms,
            component="post_verify",
        )
        self.factual_observed_verifier_boundaries += 1
        records = []
        for item in trajectory:
            if item.get("action") not in {STOP, CONTINUE}:
                continue
            if "algorithm_latency_anchor_ms" not in item:
                continue
            if item.get("counterfactual_replay_overrode_action", False):
                continue
            records.append({
                "features": tuple(item["features"]),
                "action": str(item["action"]),
                "algorithm_latency_anchor_ms": float(
                    item["algorithm_latency_anchor_ms"]
                ),
                "rho_tokens_per_ms": float(
                    item.get("rho_tokens_per_ms", 0.0)
                ),
                "selected_action_probability": float(
                    item.get("selected_action_probability", 1.0)
                ),
                "importance_weight": float(
                    item.get("importance_weight", 1.0)
                ),
                "decision_shared_value": float(
                    item.get("shared_value_mean", 0.0)
                ),
                "decision_explicit_advantage": float(
                    item.get("explicit_advantage_mean", 0.0)
                ),
                "source_record": item,
            })
        if self.pending_factual_boundary is not None:
            if records:
                raise RuntimeError(
                    "new factual decisions appeared before the previous "
                    "boundary was resolved at an adaptive state"
                )
            pending = self.pending_factual_boundary
            pending["emitted_tokens"] += int(emitted_tokens)
            pending["boundary_algorithm_latency_ms"] = float(
                self.algorithm_latency_ms
            )
            pending["verifier_boundaries_spanned"] += 1
            if terminal:
                self.pending_factual_boundary = None
                update_started = time.perf_counter()
                self._apply_factual_boundary(
                    pending,
                    end_algorithm_latency_ms=self.algorithm_latency_ms,
                    bootstrap_value=0.0,
                    terminal=True,
                )
                elapsed_ms = (time.perf_counter() - update_started) * 1000.0
                self.record_profile("factual_terminal_resolution", elapsed_ms)
            return
        if not records:
            return
        self.factual_boundary_count += 1
        pending = {
            "boundary_id": int(self.factual_boundary_count),
            "records": records,
            "emitted_tokens": int(emitted_tokens),
            "boundary_algorithm_latency_ms": float(self.algorithm_latency_ms),
            "first_boundary_algorithm_latency_ms": float(
                self.algorithm_latency_ms
            ),
            "verifier_boundaries_spanned": 1,
            "verifier_boundary_index": int(
                self.factual_observed_verifier_boundaries
            ),
        }
        if terminal or self.uses_factual_no_bootstrap:
            update_started = time.perf_counter()
            self._apply_factual_boundary(
                pending,
                end_algorithm_latency_ms=self.algorithm_latency_ms,
                bootstrap_value=0.0,
                terminal=terminal,
            )
            elapsed_ms = (time.perf_counter() - update_started) * 1000.0
            profile_name = (
                "factual_terminal_resolution"
                if terminal
                else "factual_no_bootstrap_resolution"
            )
            self.record_profile(profile_name, elapsed_ms)
            if self.uses_factual_no_bootstrap:
                self.advance_algorithm_latency(
                    elapsed_ms,
                    component="controller_factual_update",
                )
            return
        self.pending_factual_boundary = pending

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
        if self.uses_hindsight_block_gain:
            return None
        latency_ms = max(0.0, float(forward_latency_ms))
        if self.forward_latency_ema_ms is None:
            self.forward_latency_ema_ms = latency_ms
        else:
            alpha = self.config.rho_alpha
            self.forward_latency_ema_ms = (
                (1.0 - alpha) * self.forward_latency_ema_ms
                + alpha * latency_ms
            )
        if (
            self.uses_verifier_boundary_factual
            or self.config.update_mode not in {"td", "mixed"}
            or self.rho <= 0.0
        ):
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
        if self.uses_hindsight_block_gain:
            return
        if self.uses_verifier_boundary_factual:
            self._complete_verifier_boundary_factual(
                trajectory,
                emitted_tokens=emitted_tokens,
                verifier_latency_ms=verifier_latency_ms,
                post_verify_latency_ms=post_verify_latency_ms,
                terminal=terminal,
            )
            return
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
        started = (
            time.perf_counter()
            if self.config.profile_overhead
            or self.uses_verifier_boundary_factual
            else None
        )
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
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.record_profile("throughput_ema_update", elapsed_ms)
            self.advance_algorithm_latency(
                elapsed_ms,
                component="controller_throughput_update",
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
            "policy_feature_names": list(
                self.hindsight_feature_names
                if self.uses_hindsight_delta_j_logistic_f2
                else self.feature_names
            ),
            "policy_feature_dim": len(
                self.hindsight_feature_names
                if self.uses_hindsight_delta_j_logistic_f2
                else self.feature_names
            ),
            "completed_rounds": self.completed_rounds,
            "decision_count": self.decision_count,
            "annealed_decision_count": self.annealed_decision_count,
            "explore_initial": self.config.explore_epsilon,
            "explore_min": self.config.explore_min,
            "explore_decay": self.config.explore_decay,
            "current_exploration_floor": (
                self._annealed_exploration_floor()
                if self.config.policy_mode == "symmetric_annealed"
                else (
                    self.config.min_action_probability
                    if self.config.policy_mode == "symmetric"
                    else 0.0
                )
            ),
            "exploration_count": self.exploration_count,
            "rng_state": self.rng.getstate(),
            "early_stop_observations": self.early_stop_observations,
            "early_stop_min_observations": self.config.early_stop_min_observations,
            "stop_probability_threshold": self.config.stop_probability_threshold,
            "stop_z_threshold": self.stop_z_threshold,
            "policy_mode": self.config.policy_mode,
            "policy_ablation": self.config.policy_ablation,
            "credit_assignment": self.config.credit_assignment,
            "value_parameterization": self.config.value_parameterization,
            "shared_value_learning_rate": (
                self.config.shared_value_learning_rate
            ),
            "shared_advantage_learning_rate": (
                self.config.shared_advantage_learning_rate
            ),
            "shared_value_theta": list(self.shared_value_theta),
            "shared_advantage_theta": list(self.shared_advantage_theta),
            "value_model": self.config.value_model,
            "nonlinear_value": (
                self.nonlinear_value.snapshot() if self.nonlinear_value else None
            ),
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
            "algorithm_latency_ms": self.algorithm_latency_ms,
            "algorithm_latency_components_ms": dict(
                self.algorithm_latency_components_ms
            ),
            "factual_boundary_count": self.factual_boundary_count,
            "factual_observed_verifier_boundaries": (
                self.factual_observed_verifier_boundaries
            ),
            "factual_learning_update_count": self.factual_learning_update_count,
            "factual_warmup_transition_count": (
                self.factual_warmup_transition_count
            ),
            "rho_warmup_boundaries": self.config.rho_warmup_boundaries,
            "policy_weight_ema_beta": self.config.policy_weight_ema_beta,
            "policy_weight_ema_mode": self.config.policy_weight_ema_mode,
            "policy_ema_global_update_count": int(
                self.policy_ema_global_update_count
            ),
            "policy_weight_ema": {
                action: {
                    "theta": list(self.policy_ema_theta[action]),
                    "initialized": bool(
                        self.policy_ema_initialized[action]
                    ),
                    "update_count": int(
                        self.policy_ema_update_count[action]
                    ),
                }
                for action in (STOP, CONTINUE)
            },
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
            "hindsight_block_gain": {
                "enabled": self.uses_hindsight_block_gain,
                "model": self.hindsight_gain_model.snapshot(
                    self.hindsight_feature_names
                ),
                "logistic_model": self.hindsight_logistic_model.snapshot(
                    self.hindsight_feature_names
                ),
                "snapshot_count": int(self.hindsight_snapshot_count),
                "pair_count": int(self.hindsight_pair_count),
                "resolved_count": int(self.hindsight_resolved_count),
                "censored_count": int(self.hindsight_censored_count),
                "invalid_count": int(self.hindsight_invalid_count),
                "unavailable_count": int(self.hindsight_unavailable_count),
                "pending_pair_count": len(self.hindsight_pending_pairs),
                "pending_source_count": len(self.hindsight_pending_sources),
                "snapshot_overhead_ema_ms": self.hindsight_snapshot_overhead_ema_ms,
                "confidence_kappa": self.config.hindsight_confidence_kappa,
                "margin_tokens": self.config.hindsight_margin_tokens,
                "max_uncertainty_tokens": (
                    self.config.hindsight_max_uncertainty_tokens
                ),
                "probe_count": int(self.hindsight_probe_count),
                "structural_probe_count": int(self.hindsight_structural_probe_count),
                "floor_probe_count": int(self.hindsight_floor_probe_count),
                "beneficial_continue_count": int(
                    self.hindsight_delta_j_continue_count
                ),
                "stop_better_count": int(self.hindsight_delta_j_stop_count),
                "calibration_bias": float(
                    self.hindsight_delta_j_calibration_bias
                ),
                "probe_outstanding": bool(self.hindsight_probe_outstanding),
                "tie_count": int(self.hindsight_logistic_tie_count),
                "distinct_positive_problem_count": len(
                    self.hindsight_logistic_positive_problem_ids
                ),
                "positive_problem_ids": sorted(
                    self.hindsight_logistic_positive_problem_ids
                ),
                "censor_reasons": dict(self.hindsight_censor_reasons),
                "probe_initial": self.config.hindsight_probe_initial,
                "probe_floor": self.config.hindsight_probe_floor,
                "probe_decay_pairs": self.config.hindsight_probe_decay_pairs,
                "probe_max_fraction": self.config.hindsight_probe_max_fraction,
            },
            "full_stream": {
                "enabled": self.config.full_stream_bootstrap,
                "pending": (
                    self.pending_stop is not None
                    or self.pending_factual_boundary is not None
                ),
                "pending_factual_decisions": (
                    0
                    if self.pending_factual_boundary is None
                    else len(self.pending_factual_boundary["records"])
                ),
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
        snapshot_credit_assignment = snapshot.get(
            "credit_assignment",
            "per_step_td",
        )
        snapshot_value_parameterization = snapshot.get(
            "value_parameterization",
            "independent_q",
        )
        snapshot_value_model = snapshot.get("value_model", "linear")
        snapshot_policy_ablation = snapshot.get("policy_ablation", "learned")
        snapshot_policy_mode = snapshot.get("policy_mode", "legacy")
        snapshot_shared_value_rate = float(
            snapshot.get(
                "shared_value_learning_rate",
                self.config.shared_value_learning_rate,
            )
        )
        snapshot_shared_advantage_rate = float(
            snapshot.get(
                "shared_advantage_learning_rate",
                self.config.shared_advantage_learning_rate,
            )
        )
        snapshot_rho_warmup = int(snapshot.get("rho_warmup_boundaries", 0))
        snapshot_policy_ema_beta = float(
            snapshot.get("policy_weight_ema_beta", 0.0)
        )
        # Snapshots created before global-step EMA existed used the historical
        # action-specific clock, so default missing metadata accordingly.
        snapshot_policy_ema_mode = str(
            snapshot.get("policy_weight_ema_mode", "action_step")
        )
        if (
            snapshot_schema != self.config.feature_schema
            or snapshot_version != self.config.feature_version
            or snapshot_names != self.feature_names
            or snapshot_credit_assignment != self.config.credit_assignment
            or snapshot_value_parameterization
            != self.config.value_parameterization
            or snapshot_value_model != self.config.value_model
            or snapshot_policy_ablation != self.config.policy_ablation
            or snapshot_policy_mode != self.config.policy_mode
            or not math.isclose(
                snapshot_shared_value_rate,
                self.config.shared_value_learning_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                snapshot_shared_advantage_rate,
                self.config.shared_advantage_learning_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or snapshot_rho_warmup != self.config.rho_warmup_boundaries
            or not math.isclose(
                snapshot_policy_ema_beta,
                self.config.policy_weight_ema_beta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (
                (snapshot_policy_ema_beta > 0.0 or self.config.policy_weight_ema_beta > 0.0)
                and snapshot_policy_ema_mode != self.config.policy_weight_ema_mode
            )
        ):
            raise ValueError(
                "snapshot feature schema does not match controller: "
                f"snapshot={snapshot_schema}/v{snapshot_version}, "
                f"controller={self.config.feature_schema}/v{self.config.feature_version}, "
                f"snapshot_credit={snapshot_credit_assignment}, "
                f"controller_credit={self.config.credit_assignment}, "
                f"snapshot_value_parameterization="
                f"{snapshot_value_parameterization}, "
                f"controller_value_parameterization="
                f"{self.config.value_parameterization}, "
                f"snapshot_policy_ablation={snapshot_policy_ablation}, "
                f"controller_policy_ablation={self.config.policy_ablation}, "
                f"snapshot_policy_mode={snapshot_policy_mode}, "
                f"controller_policy_mode={self.config.policy_mode}, "
                f"snapshot_rho_warmup={snapshot_rho_warmup}, "
                f"controller_rho_warmup={self.config.rho_warmup_boundaries}, "
                f"snapshot_policy_ema_beta={snapshot_policy_ema_beta}, "
                f"controller_policy_ema_beta={self.config.policy_weight_ema_beta}, "
                f"snapshot_policy_ema_mode={snapshot_policy_ema_mode}, "
                f"controller_policy_ema_mode={self.config.policy_weight_ema_mode}"
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

        if self.nonlinear_value is not None:
            nonlinear_snapshot = snapshot.get("nonlinear_value")
            if not nonlinear_snapshot:
                raise ValueError("snapshot is missing nonlinear value state")
            self.nonlinear_value.load_snapshot(nonlinear_snapshot)

        if self.uses_shared_value_advantage:
            shared_value = [
                float(item)
                for item in snapshot.get("shared_value_theta", [])
            ]
            shared_advantage = [
                float(item)
                for item in snapshot.get("shared_advantage_theta", [])
            ]
            if (
                len(shared_value) != self.config.feature_dim
                or len(shared_advantage) != self.config.feature_dim
            ):
                raise ValueError(
                    "snapshot is missing shared value/advantage parameters"
                )
            self.shared_value_theta = shared_value
            self.shared_advantage_theta = shared_advantage
            self._sync_shared_action_means()

        policy_ema = snapshot.get("policy_weight_ema") or {}
        for action in (STOP, CONTINUE):
            state = policy_ema.get(action) or {}
            theta = [
                float(item)
                for item in state.get(
                    "theta",
                    self.values[action].theta,
                )
            ]
            if len(theta) != self.config.feature_dim:
                raise ValueError(
                    "snapshot policy EMA dimension does not match controller"
                )
            self.policy_ema_theta[action] = theta
            self.policy_ema_initialized[action] = bool(
                state.get("initialized", False)
            )
            self.policy_ema_update_count[action] = int(
                state.get("update_count", 0)
            )
        self.policy_ema_global_update_count = int(
            snapshot.get(
                "policy_ema_global_update_count",
                min(self.policy_ema_update_count.values())
                if self.config.policy_weight_ema_mode == "global_step"
                else 0,
            )
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
        self.annealed_decision_count = int(
            snapshot.get("annealed_decision_count", 0)
        )
        self.exploration_count = int(snapshot.get("exploration_count", 0))
        if snapshot.get("rng_state") is not None:
            def tuple_tree(value):
                if isinstance(value, list):
                    return tuple(tuple_tree(item) for item in value)
                return value

            self.rng.setstate(tuple_tree(snapshot["rng_state"]))
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
        self.algorithm_latency_ms = float(
            snapshot.get("algorithm_latency_ms", 0.0)
        )
        self.algorithm_latency_components_ms = {
            str(name): float(value)
            for name, value in (
                snapshot.get("algorithm_latency_components_ms") or {}
            ).items()
        }
        self.factual_boundary_count = int(
            snapshot.get("factual_boundary_count", 0)
        )
        self.factual_observed_verifier_boundaries = int(
            snapshot.get(
                "factual_observed_verifier_boundaries",
                self.factual_boundary_count,
            )
        )
        self.factual_learning_update_count = int(
            snapshot.get("factual_learning_update_count", 0)
        )
        self.factual_warmup_transition_count = int(
            snapshot.get("factual_warmup_transition_count", 0)
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


def build_v22_state_features(
    *,
    proposal_length: int,
    max_spec_len: int,
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
    min_remaining_confidence = min(confidences) if confidences else 1.0
    tau_f = max(float(failfast_threshold), eps)
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
        _clip(prefix_length / proposal_length, 0.0, 1.0),
        _clip(prefix_advance / proposal_length, 0.0, 1.0),
        min_remaining_confidence,
        failfast_margin,
        _clip(proposal_length / max_spec_len, 0.0, 1.0),
        draft_verify_ratio,
        _clip(tokens_per_verifier / (max_spec_len + 1.0), 0.0, 1.0),
    ]
    disabled = set(disabled_features)
    for index, name in enumerate(V22_FEATURE_NAMES):
        if name in disabled:
            features[index] = 0.0
    return tuple(features)


def build_v22_compact_state_features(**state) -> tuple[float, ...]:
    full_features = build_v22_state_features(**state)
    by_name = dict(zip(V22_FEATURE_NAMES, full_features))
    return tuple(by_name[name] for name in V22_COMPACT_FEATURE_NAMES)


def build_v23_compact7_state_features(
    *,
    active_span_length: int,
    active_remaining_masks: int,
    refinement_step: int,
    max_refinement_steps: int,
    disabled_features: Sequence[str] = (),
    **state,
) -> tuple[float, ...]:
    """Build Compact7 strictly from the processed logical active span."""
    active_span_length = int(active_span_length)
    active_remaining_masks = int(active_remaining_masks)
    max_refinement_steps = int(max_refinement_steps)
    if active_span_length <= 0:
        raise ValueError("active_span_length must be positive")
    if not 0 <= active_remaining_masks <= active_span_length:
        raise ValueError(
            "active_remaining_masks must belong to the processed active span"
        )
    if max_refinement_steps <= 0:
        raise ValueError("max_refinement_steps must be positive")

    full_features = build_v22_state_features(
        factual_tokens_per_verifier_ema=None,
        disabled_features=(),
        **state,
    )
    by_name = dict(zip(V22_FEATURE_NAMES, full_features))
    by_name.update({
        "active_remaining_mask_ratio": _clip(
            active_remaining_masks / active_span_length,
            0.0,
            1.0,
        ),
        "normalized_refinement_step": _clip(
            min(max(0, int(refinement_step)), max_refinement_steps)
            / max_refinement_steps,
            0.0,
            1.0,
        ),
    })
    features = [by_name[name] for name in V23_COMPACT7_FEATURE_NAMES]
    disabled = set(disabled_features)
    for index, name in enumerate(V23_COMPACT7_FEATURE_NAMES):
        if name in disabled:
            features[index] = 0.0
    return tuple(features)


def build_raw_state_features(
    *,
    raw_previous_state,
    raw_current_state,
    has_previous_state,
    proposal_length,
    max_spec_len,
    refinement_step,
    max_refinement_steps,
    factual_draft_latency_ema_ms=None,
    factual_verifier_latency_ema_ms=None,
    factual_tokens_per_verifier_ema=None,
    **_unused,
) -> tuple[float, ...]:
    """Build the fixed-width raw denoising input used by Raw Shared V+A.

    The raw snapshots are already reduced from vocabulary logits on the GPU.
    This function only validates their shape and appends non-token cost context.
    """
    expected_values = RAW_STATE_BLOCK_SIZE * len(RAW_TOKEN_FIELDS)

    def flatten(snapshot, label):
        if len(snapshot) != RAW_STATE_BLOCK_SIZE:
            raise ValueError(
                f"{label} must contain {RAW_STATE_BLOCK_SIZE} positions, "
                f"got {len(snapshot)}"
            )
        values = []
        for row in snapshot:
            if len(row) == len(RAW_TOKEN_FIELDS) - 1:
                # Backward compatibility for the 85-dimensional oracle archive:
                # an exact zero top-1 value was the old unobserved sentinel.
                row = [row[0], 1.0 if float(row[1]) > 0.0 else 0.0, *row[1:]]
            if len(row) != len(RAW_TOKEN_FIELDS):
                raise ValueError(
                    f"{label} rows must contain {len(RAW_TOKEN_FIELDS) - 1} "
                    f"or {len(RAW_TOKEN_FIELDS)} values"
                )
            values.extend(float(value) for value in row)
        if len(values) != expected_values or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError(f"{label} contains invalid raw-state values")
        return values

    previous = flatten(raw_previous_state, "raw_previous_state")
    current = flatten(raw_current_state, "raw_current_state")
    proposal_length = max(0, int(proposal_length))
    max_spec_len = max(1, int(max_spec_len))
    max_refinement_steps = max(1, int(max_refinement_steps))
    draft_ms = max(0.0, float(factual_draft_latency_ema_ms or 0.0))
    verify_ms = max(0.0, float(factual_verifier_latency_ema_ms or 0.0))
    tokens_per_verify = max(0.0, float(factual_tokens_per_verifier_ema or 0.0))
    global_state = [
        _clip(proposal_length / max_spec_len, 0.0, 1.0),
        _clip(draft_ms / max(verify_ms, 1e-9), 0.0, 2.0)
        if verify_ms > 0.0 else 0.0,
        _clip(tokens_per_verify / (max_spec_len + 1.0), 0.0, 1.0),
        _clip(int(refinement_step) / max_refinement_steps, 0.0, 1.0),
        1.0 if bool(has_previous_state) else 0.0,
    ]
    features = tuple(previous + current + global_state)
    if len(features) != len(RAW_STATE_FEATURE_NAMES):
        raise RuntimeError("raw-state feature dimension changed unexpectedly")
    return features


def trajectory_forward_latency(trajectory: Iterable[dict]) -> float:
    return sum(
        max(0.0, float(item.get("next_forward_latency_ms", 0.0)))
        for item in trajectory
    )
