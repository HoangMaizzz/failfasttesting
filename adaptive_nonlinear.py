from __future__ import annotations

import base64
import io
import math
from itertools import combinations

import torch
from torch import nn


STOP = "stop"
CONTINUE = "continue"


class _Contribution(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CompactNonlinearVA(nn.Module):
    """Tiny Compact6 NAM/GA2M mean model; uncertainty stays external."""

    def __init__(
        self,
        model_type: str,
        *,
        feature_dim: int = 5,
        hidden: tuple[int, int] = (8, 4),
        seed: int = 42,
    ) -> None:
        super().__init__()
        if model_type not in {"nam", "ga2m"}:
            raise ValueError(f"unknown nonlinear value model: {model_type}")
        torch.manual_seed(int(seed))
        self.model_type = model_type
        self.feature_dim = int(feature_dim)
        self.bias = nn.Parameter(torch.zeros(2))
        self.main_effects = nn.ModuleList(
            _Contribution(1, hidden) for _ in range(self.feature_dim)
        )
        self.interaction_pairs = list(combinations(range(self.feature_dim), 2))
        self.interactions = nn.ModuleList(
            _Contribution(2, hidden)
            for _ in (self.interaction_pairs if model_type == "ga2m" else [])
        )

    def components(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        main = torch.stack([
            module(features[..., index:index + 1])
            for index, module in enumerate(self.main_effects)
        ], dim=-2)
        if not self.interactions:
            interaction = features.new_zeros((*features.shape[:-1], 0, 2))
        else:
            interaction = torch.stack([
                module(features[..., [left, right]])
                for module, (left, right) in zip(
                    self.interactions, self.interaction_pairs
                )
            ], dim=-2)
        total = self.bias + main.sum(dim=-2) + interaction.sum(dim=-2)
        return total, main, interaction

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total, _, _ = self.components(features)
        return total[..., 0], total[..., 1]


class OnlineNonlinearVA:
    def __init__(
        self,
        model_type: str,
        *,
        learning_rate: float,
        weight_decay: float,
        grad_clip: float,
        huber_delta: float,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = CompactNonlinearVA(model_type, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.grad_clip = float(grad_clip)
        self.huber_delta = float(huber_delta)
        self.update_count = 0
        self.last_loss = 0.0
        self.last_gradient_norm = 0.0

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def _dynamic(features) -> list[float]:
        values = [float(value) for value in features]
        if len(values) != 6:
            raise ValueError("Compact6 nonlinear models require six inputs")
        return values[1:]

    def predict(self, features) -> tuple[float, float, float, float]:
        tensor = torch.tensor(
            [self._dynamic(features)], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            value, advantage = self.model(tensor)
        value = float(value.item())
        advantage = float(advantage.item())
        return value, advantage, value + 0.5 * advantage, value - 0.5 * advantage

    def update(self, action: str, features, target: float, weight: float) -> dict:
        tensor = torch.tensor(
            [self._dynamic(features)], dtype=torch.float32, device=self.device
        )
        target_tensor = torch.tensor(float(target), device=self.device)
        value, advantage = self.model(tensor)
        sign = 1.0 if action == STOP else -1.0
        prediction = value[0] + 0.5 * sign * advantage[0]
        residual = float(target) - float(prediction.detach().item())
        absolute = torch.abs(prediction - target_tensor)
        delta = self.huber_delta
        loss = torch.where(
            absolute <= delta,
            0.5 * (prediction - target_tensor) ** 2,
            delta * (absolute - 0.5 * delta),
        ) * float(weight)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )
        self.optimizer.step()
        self.update_count += 1
        self.last_loss = float(loss.detach().item())
        self.last_gradient_norm = float(norm.item()) if torch.is_tensor(norm) else float(norm)
        return {
            "residual": residual,
            "loss": self.last_loss,
            "gradient_norm": self.last_gradient_norm,
        }

    def snapshot(self) -> dict:
        payload = io.BytesIO()
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, payload)
        return {
            "model_type": self.model.model_type,
            "parameter_count": self.parameter_count,
            "update_count": self.update_count,
            "last_loss": self.last_loss,
            "last_gradient_norm": self.last_gradient_norm,
            "torch_state_base64": base64.b64encode(payload.getvalue()).decode("ascii"),
        }

    def diagnostics(self) -> dict:
        with torch.no_grad():
            parameter_norm = math.sqrt(sum(
                float(torch.sum(parameter * parameter).item())
                for parameter in self.model.parameters()
            ))
            interaction_norms = {
                f"{left}_{right}": math.sqrt(sum(
                    float(torch.sum(parameter * parameter).item())
                    for parameter in module.parameters()
                ))
                for module, (left, right) in zip(
                    self.model.interactions,
                    self.model.interaction_pairs,
                )
            }
        return {
            "parameter_count": self.parameter_count,
            "parameter_norm": parameter_norm,
            "interaction_parameter_norms": interaction_norms,
            "update_count": self.update_count,
            "last_loss": self.last_loss,
            "last_gradient_norm": self.last_gradient_norm,
        }

    def load_snapshot(self, snapshot: dict) -> None:
        payload = base64.b64decode(snapshot["torch_state_base64"])
        state = torch.load(io.BytesIO(payload), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(snapshot.get("update_count", 0))
        self.last_loss = float(snapshot.get("last_loss", 0.0))
        self.last_gradient_norm = float(snapshot.get("last_gradient_norm", 0.0))
