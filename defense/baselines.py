from __future__ import annotations

import copy

import torch

from .safesplit import Checkpoint, flatten_state_dict


class DifferentialPrivacyDefense:
    def __init__(self, clip_norm: float, noise_scale: float) -> None:
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        latest = copy.deepcopy(history[-1])
        update_vec = flatten_state_dict(latest.update_state)
        update_norm = torch.norm(update_vec, p=2)
        scale = min(1.0, self.clip_norm / (update_norm.item() + 1e-8))

        noisy_defense_state = {}
        for name, tensor in latest.update_state.items():
            clipped = tensor * scale
            noise = torch.randn_like(clipped) * self.noise_scale
            noisy_tensor = latest.defense_state[name] - tensor + clipped + noise
            noisy_defense_state[name] = noisy_tensor
            model_key = f"backbone.{name}"
            if model_key in latest.model_state:
                latest.model_state[model_key] = noisy_tensor
            elif name in latest.model_state:
                latest.model_state[name] = noisy_tensor
        latest.defense_state = noisy_defense_state
        return latest


class KrumStyleDefense:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        if len(history) <= 1:
            return copy.deepcopy(history[-1])

        window = history[-min(self.window_size, len(history)) :]
        majority_size = max(1, len(window) // 2)
        update_vectors = [flatten_state_dict(cp.update_state) for cp in window]
        scores = []
        for i in range(len(window)):
            distances = []
            for j in range(len(window)):
                if i == j:
                    continue
                distances.append(torch.dist(update_vectors[i], update_vectors[j], p=2).item())
            scores.append(sum(sorted(distances)[:majority_size]))

        best_idx = min(range(len(window)), key=lambda idx: scores[idx])
        return copy.deepcopy(window[best_idx])
