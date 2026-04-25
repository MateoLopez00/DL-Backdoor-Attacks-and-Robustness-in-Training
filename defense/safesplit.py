from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from scipy.fft import dctn


def clone_state_dict(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for key in sorted(state_dict):
        parts.append(state_dict[key].reshape(-1).float())
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts)


def load_state_dict(module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    module.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def diff_state_dict(new_state: dict[str, torch.Tensor], old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: new_state[key].float() - old_state[key].float() for key in new_state}


def state_to_matrix(vector: torch.Tensor, width: int) -> torch.Tensor:
    width = max(8, width)
    height = math.ceil(vector.numel() / width)
    padded = torch.zeros(height * width, dtype=torch.float32)
    padded[: vector.numel()] = vector.float()
    return padded.reshape(height, width)


def dct_low_frequency(update_vector: torch.Tensor, width: int, low_freq_frac: float) -> torch.Tensor:
    matrix = state_to_matrix(update_vector, width).numpy()
    coeffs = dctn(matrix, type=2, norm="ortho")
    keep_h = max(1, int(math.ceil(coeffs.shape[0] * low_freq_frac)))
    keep_w = max(1, int(math.ceil(coeffs.shape[1] * low_freq_frac)))
    return torch.tensor(coeffs[:keep_h, :keep_w].reshape(-1), dtype=torch.float32)


def rotational_signature(defense_vector: torch.Tensor, width: int) -> torch.Tensor:
    matrix = state_to_matrix(defense_vector, width)
    row_mean = matrix.mean(dim=1, keepdim=True)
    col_mean = matrix.mean(dim=0, keepdim=True)
    bx = (row_mean * matrix).reshape(-1)
    by = (matrix * col_mean).reshape(-1)
    theta = torch.atan2(by, bx + 1e-8)
    omega = torch.zeros_like(theta)
    omega[1:] = theta[1:] - theta[:-1]
    return omega / (2 * math.pi)


def smallest_majority_sum(values: list[float], majority_size: int) -> float:
    return float(sum(sorted(values)[:majority_size]))


@dataclass
class Checkpoint:
    step: int
    epoch: int
    model_state: dict[str, torch.Tensor]
    defense_state: dict[str, torch.Tensor]
    update_state: dict[str, torch.Tensor]
    selected: bool = False
    trust_score: float | None = None
    rollback: bool = False


class SafeSplitDefense:
    """Parameter-update defense adapted from the SafeSplit scoring idea."""

    def __init__(self, window_size: int, low_freq_frac: float, matrix_width: int) -> None:
        self.window_size = window_size
        self.low_freq_frac = low_freq_frac
        self.matrix_width = matrix_width

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        if not history:
            raise ValueError("History must contain at least one checkpoint.")
        if len(history) < self.window_size:
            return copy.deepcopy(history[-1])

        window = history[-self.window_size :]
        majority_size = self.window_size // 2 + 1

        freq_signatures = [
            dct_low_frequency(flatten_state_dict(cp.update_state), self.matrix_width, self.low_freq_frac)
            for cp in window
        ]
        rotation_signatures = [rotational_signature(flatten_state_dict(cp.defense_state), self.matrix_width) for cp in window]

        frequency_scores = []
        rotation_scores = []
        for i in range(len(window)):
            freq_distances = []
            rot_distances = []
            for j in range(len(window)):
                if i == j:
                    continue
                freq_distances.append(torch.dist(freq_signatures[i], freq_signatures[j], p=2).item())
                rot_distances.append(torch.mean(torch.abs(rotation_signatures[i] - rotation_signatures[j])).item())
            frequency_scores.append(smallest_majority_sum(freq_distances, majority_size))
            rotation_scores.append(smallest_majority_sum(rot_distances, majority_size))

        frequency_majority = set(sorted(range(len(window)), key=lambda idx: frequency_scores[idx])[:majority_size])
        rotation_majority = set(sorted(range(len(window)), key=lambda idx: rotation_scores[idx])[:majority_size])
        benign_indices = frequency_majority.intersection(rotation_majority)

        for local_idx in range(len(window) - 1, -1, -1):
            if local_idx in benign_indices:
                selected = copy.deepcopy(window[local_idx])
                selected.trust_score = 1.0 / (1.0 + frequency_scores[local_idx] + rotation_scores[local_idx])
                return selected

        selected = copy.deepcopy(window[-1])
        selected.trust_score = 0.0
        return selected
