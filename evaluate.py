from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def evaluate_model(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_backdoor(model, trigger_set, device, batch_size: int = 256) -> float:
    if not trigger_set:
        return 0.0

    model.eval()
    images = torch.stack([item[0] for item in trigger_set])
    labels = torch.tensor([item[1] for item in trigger_set], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int) -> np.ndarray:
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        truth = targets.numpy()
        for gold, pred in zip(truth, preds):
            matrix[gold, pred] += 1
    return matrix
