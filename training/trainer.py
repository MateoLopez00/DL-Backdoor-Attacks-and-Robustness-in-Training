from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.optim import SGD

from defense import Checkpoint, clone_state_dict, diff_state_dict, load_state_dict
from evaluate import evaluate_backdoor, evaluate_model


class CentralizedTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        local_epochs: int,
        defense=None,
        poisoned_train_loader=None,
        poison_start_epoch: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device)
        self.train_loader = train_loader
        self.poisoned_train_loader = poisoned_train_loader
        self.poison_start_epoch = poison_start_epoch
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.defense = defense
        self.history: list[Checkpoint] = []
        self.last_selected_checkpoint: Checkpoint | None = None
        self.previous_accepted_defense_state = self._defense_state()

    def _make_optimizer(self):
        return SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def _loader_for_epoch(self, epoch: int):
        if (
            self.poisoned_train_loader is not None
            and self.poison_start_epoch is not None
            and epoch >= self.poison_start_epoch
        ):
            return self.poisoned_train_loader
        return self.train_loader

    def _poison_active(self, epoch: int) -> bool:
        return self._loader_for_epoch(epoch) is self.poisoned_train_loader

    def _defense_state(self) -> dict[str, torch.Tensor]:
        module = self.model.backbone if hasattr(self.model, "backbone") else self.model
        return {
            key: value
            for key, value in clone_state_dict(module).items()
            if torch.is_floating_point(value)
        }

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        optimizer = self._make_optimizer()
        train_loader = self._loader_for_epoch(epoch)
        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _store_checkpoint(self, step: int, epoch: int) -> Checkpoint:
        defense_state = self._defense_state()
        checkpoint = Checkpoint(
            step=step,
            epoch=epoch,
            model_state=clone_state_dict(self.model),
            defense_state=defense_state,
            update_state=diff_state_dict(defense_state, self.previous_accepted_defense_state),
        )
        self.history.append(checkpoint)
        return checkpoint

    def _select_checkpoint(self) -> Checkpoint:
        if self.defense is None:
            return copy.deepcopy(self.history[-1])
        return self.defense.select_checkpoint(self.history)

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        load_state_dict(self.model, checkpoint.model_state, self.device)
        self.previous_accepted_defense_state = copy.deepcopy(checkpoint.defense_state)
        self.last_selected_checkpoint = checkpoint

    def run(self, num_epochs: int, test_loader, trigger_set=None):
        metrics = []
        global_step = 0
        for epoch in range(num_epochs):
            epoch_number = epoch + 1
            poison_active = self._poison_active(epoch_number)
            train_loss = self.train_one_epoch(epoch_number)
            global_step += 1
            checkpoint = self._store_checkpoint(global_step, epoch_number)
            selected = self._select_checkpoint()
            selected.rollback = selected.step != checkpoint.step
            selected.selected = True
            self._load_checkpoint(selected)

            clean_acc = evaluate_model(self.model, test_loader, self.device)
            backdoor_acc = evaluate_backdoor(self.model, trigger_set or [], self.device)
            metrics.append(
                {
                    "epoch": epoch_number,
                    "train_loss": train_loss,
                    "poison_active": poison_active,
                    "clean_accuracy": clean_acc,
                    "attack_success_rate": backdoor_acc,
                    "clean_ma": clean_acc,
                    "backdoor_ba": backdoor_acc,
                    "selected_step": None if self.last_selected_checkpoint is None else self.last_selected_checkpoint.step,
                    "rollback": False if self.last_selected_checkpoint is None else self.last_selected_checkpoint.rollback,
                    "trust_score": None if self.last_selected_checkpoint is None else self.last_selected_checkpoint.trust_score,
                }
            )
        return metrics
