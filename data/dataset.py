from pathlib import Path

from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_datasets(data_dir: Path):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def get_targets(dataset) -> list[int]:
    from torch.utils.data import Subset

    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return [parent_targets[i] for i in dataset.indices]
    if hasattr(dataset, "dataset") and hasattr(dataset, "poisoned_indices"):
        return get_targets(dataset.dataset)
    raise TypeError(f"Dataset targets not supported for {type(dataset)!r}")
