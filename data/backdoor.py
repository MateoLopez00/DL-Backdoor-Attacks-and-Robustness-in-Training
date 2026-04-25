from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset

from .dataset import CIFAR10_MEAN, CIFAR10_STD, get_targets


MEAN = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
STD = torch.tensor(CIFAR10_STD).view(3, 1, 1)


def _clamp_normalized(image: torch.Tensor) -> torch.Tensor:
    image = torch.maximum(image, (torch.zeros_like(MEAN) - MEAN) / STD)
    image = torch.minimum(image, (torch.ones_like(MEAN) - MEAN) / STD)
    return image


@dataclass
class PixelTriggerAttack:
    trigger_size: int
    position: str
    target_label: int

    def poison_sample(self, image: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        image = image.clone()
        _, h, w = image.shape
        size = min(self.trigger_size, h, w)
        if self.position == "bottom-right":
            y0, x0 = h - size, w - size
        elif self.position == "top-left":
            y0, x0 = 0, 0
        else:
            raise ValueError(f"Unsupported trigger position: {self.position}")

        red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
        normalized = (red - MEAN) / STD
        image[:, y0 : y0 + size, x0 : x0 + size] = normalized
        image = _clamp_normalized(image)
        return image, self.target_label

    def build_backdoor_test_set(self, dataset) -> list[tuple[torch.Tensor, int]]:
        trigger_set = []
        for idx in range(len(dataset)):
            image, _ = dataset[idx]
            poisoned_image, poisoned_label = self.poison_sample(image, 0)
            trigger_set.append((poisoned_image, poisoned_label))
        return trigger_set


@dataclass
class SemanticTriggerAttack:
    source_label: int
    target_label: int

    def poison_sample(self, image: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        if label != self.source_label:
            return image, label

        image = image.clone()
        background = ((image * STD) + MEAN).clamp(0.0, 1.0)
        for row in range(background.shape[1]):
            if row % 4 < 2:
                background[:, row, :] = torch.tensor([0.90, 0.90, 0.10]).view(3, 1)
            else:
                background[:, row, :] = torch.tensor([0.15, 0.45, 0.95]).view(3, 1)
        image = (background - MEAN) / STD
        image = _clamp_normalized(image)
        return image, self.target_label

    def build_backdoor_test_set(self, dataset) -> list[tuple[torch.Tensor, int]]:
        targets = get_targets(dataset)
        trigger_set = []
        for idx, label in enumerate(targets):
            if label != self.source_label:
                continue
            image, clean_label = dataset[idx]
            poisoned_image, poisoned_label = self.poison_sample(image, clean_label)
            trigger_set.append((poisoned_image, poisoned_label))
        return trigger_set


class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset | Subset,
        attack,
        poisoned_data_rate: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.attack = attack
        self.poisoned_data_rate = poisoned_data_rate

        local_targets = get_targets(dataset)
        eligible = list(range(len(local_targets)))
        if isinstance(attack, SemanticTriggerAttack):
            eligible = [i for i, label in enumerate(local_targets) if label == attack.source_label]

        poison_count = int(round(len(eligible) * poisoned_data_rate))
        generator = torch.Generator().manual_seed(seed)
        if poison_count > 0 and eligible:
            perm = torch.randperm(len(eligible), generator=generator).tolist()
            chosen = [eligible[i] for i in perm[:poison_count]]
        else:
            chosen = []
        self.poisoned_indices = set(chosen)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if index in self.poisoned_indices:
            image, label = self.attack.poison_sample(image, int(label))
        return image, label
