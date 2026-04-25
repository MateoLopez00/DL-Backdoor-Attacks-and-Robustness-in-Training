from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config as cfg
from data import PixelTriggerAttack, PoisonedDataset, SemanticTriggerAttack, load_datasets
from defense import DifferentialPrivacyDefense, KrumStyleDefense, SafeSplitDefense
from evaluate import evaluate_backdoor, evaluate_model
from models import get_model
from training import CentralizedTrainer


@dataclass(slots=True)
class ExperimentRequest:
    preset: str = cfg.DEFAULT_PRESET
    arch: str = cfg.ARCH
    num_rounds: int = cfg.NUM_ROUNDS
    num_clients: int = cfg.NUM_CLIENTS
    num_malicious: int = cfg.NUM_MALICIOUS
    iid_rate: float = cfg.IID_RATE
    defense: str = "update_defense"
    backdoor: str = cfg.BACKDOOR_TYPE
    pdr: float = cfg.POISONED_DATA_RATE
    poison_start_epoch: int | None = cfg.POISON_START_EPOCH
    device: str | None = None
    seed: int = cfg.SEED
    max_samples_per_client: int | None = None
    out_dir: str = str(cfg.RESULTS_DIR)
    write_json: bool = True
    local_epochs: int = cfg.LOCAL_EPOCHS
    batch_size: int = cfg.BATCH_SIZE
    eval_batch_size: int = cfg.EVAL_BATCH_SIZE


def experiment_request_to_dict(request: ExperimentRequest) -> dict[str, object]:
    return asdict(request)


def build_experiment_request(preset: str | None = None, **overrides) -> ExperimentRequest:
    preset_values = cfg.resolve_preset(preset)
    request_data = {
        "preset": str(preset_values["name"]),
        "arch": str(preset_values["arch"]),
        "num_rounds": int(preset_values["num_rounds"]),
        "num_clients": int(preset_values["num_clients"]),
        "num_malicious": int(preset_values["num_malicious"]),
        "iid_rate": float(preset_values["iid_rate"]),
        "defense": "update_defense",
        "backdoor": cfg.BACKDOOR_TYPE,
        "pdr": cfg.POISONED_DATA_RATE,
        "poison_start_epoch": cfg.POISON_START_EPOCH,
        "device": None,
        "seed": cfg.SEED,
        "max_samples_per_client": preset_values["max_samples_per_client"],
        "out_dir": str(cfg.RESULTS_DIR),
        "write_json": True,
        "local_epochs": int(preset_values["local_epochs"]),
        "batch_size": int(preset_values["batch_size"]),
        "eval_batch_size": int(preset_values["eval_batch_size"]),
    }
    for key, value in overrides.items():
        if value is not None:
            request_data[key] = value
    return ExperimentRequest(**request_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized CIFAR-10 backdoor robustness experiments")
    parser.add_argument("--preset", choices=sorted(cfg.EXPERIMENT_PRESETS), default=None)
    parser.add_argument("--arch", default=None, choices=["resnet18", "simple_cnn"])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-malicious", type=int, default=None)
    parser.add_argument("--iid-rate", type=float, default=None)
    parser.add_argument("--defense", default="update_defense", choices=["none", "update_defense", "safesplit", "dp", "krum"])
    parser.add_argument("--backdoor", default=cfg.BACKDOOR_TYPE, choices=["pixel", "semantic", "none"])
    parser.add_argument("--pdr", type=float, default=None)
    parser.add_argument("--poison-start-epoch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--out-dir", default=str(cfg.RESULTS_DIR))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_attack(name: str):
    if name == "pixel":
        return PixelTriggerAttack(
            trigger_size=cfg.TRIGGER_SIZE,
            position=cfg.TRIGGER_POS,
            target_label=cfg.PIXEL_TARGET_LABEL,
        )
    if name == "semantic":
        return SemanticTriggerAttack(
            source_label=cfg.SEMANTIC_SOURCE_LABEL,
            target_label=cfg.SEMANTIC_TARGET_LABEL,
        )
    return None


def build_defense(name: str, _num_clients: int):
    if name in {"update_defense", "safesplit"}:
        return SafeSplitDefense(
            window_size=cfg.SAFE_SPLIT_WINDOW,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
        )
    if name == "dp":
        return DifferentialPrivacyDefense(cfg.DP_CLIP_NORM, cfg.DP_NOISE_SCALE)
    if name == "krum":
        return KrumStyleDefense(window_size=cfg.SAFE_SPLIT_WINDOW)
    return None


def build_experiment_request_from_args(args: argparse.Namespace) -> ExperimentRequest:
    preset = "lite" if args.fast_dev_run else args.preset
    return build_experiment_request(
        preset=preset,
        arch=args.arch,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        num_malicious=args.num_malicious,
        iid_rate=args.iid_rate,
        defense=args.defense,
        backdoor=args.backdoor,
        pdr=args.pdr,
        poison_start_epoch=args.poison_start_epoch,
        device=args.device,
        seed=args.seed,
        max_samples_per_client=args.max_samples_per_client,
        out_dir=args.out_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )


def build_output_path(request: ExperimentRequest) -> Path:
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_token = "full" if request.max_samples_per_client is None else str(request.max_samples_per_client)
    poison_token = "clean" if request.backdoor == "none" else f"poison{request.poison_start_epoch}"
    out_name = (
        f"{cfg.DATASET.lower()}_{request.arch}_{canonical_defense_name(request.defense)}_{request.backdoor}_"
        f"{request.preset}_samples{sample_token}_epochs{request.num_rounds}_{poison_token}.json"
    )
    return out_dir / out_name


def canonical_defense_name(name: str) -> str:
    return "update_defense" if name == "safesplit" else name


def centralized_train_subset(train_dataset, request: ExperimentRequest):
    if request.max_samples_per_client is None:
        return train_dataset

    sample_count = min(len(train_dataset), request.max_samples_per_client * request.num_clients)
    generator = torch.Generator().manual_seed(request.seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:sample_count].tolist()
    return Subset(train_dataset, indices)


def run_experiment(request: ExperimentRequest) -> dict[str, object]:
    set_seed(request.seed)
    device = cfg.resolve_device(request.device)
    train_dataset, test_dataset = load_datasets(cfg.DATA_DIR)

    attack = build_attack(request.backdoor)
    clean_train_dataset = centralized_train_subset(train_dataset, request)
    poisoned_train_dataset = None
    if attack is not None:
        poisoned_train_dataset = PoisonedDataset(clean_train_dataset, attack, request.pdr, seed=request.seed)

    train_loader = DataLoader(clean_train_dataset, batch_size=request.batch_size, shuffle=True, num_workers=0)
    poisoned_train_loader = (
        None
        if poisoned_train_dataset is None
        else DataLoader(poisoned_train_dataset, batch_size=request.batch_size, shuffle=True, num_workers=0)
    )
    test_loader = DataLoader(test_dataset, batch_size=request.eval_batch_size, shuffle=False, num_workers=0)
    trigger_set = [] if attack is None else attack.build_backdoor_test_set(test_dataset)

    defense = build_defense(request.defense, request.num_clients)
    trainer = CentralizedTrainer(
        model=get_model(request.arch, cfg.NUM_CLASSES),
        train_loader=train_loader,
        device=device,
        lr=cfg.LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        local_epochs=request.local_epochs,
        defense=defense,
        poisoned_train_loader=poisoned_train_loader,
        poison_start_epoch=None if attack is None else request.poison_start_epoch,
    )

    history = trainer.run(num_epochs=request.num_rounds, test_loader=test_loader, trigger_set=trigger_set)
    final_clean_accuracy = evaluate_model(trainer.model, test_loader, device)
    final_attack_success_rate = evaluate_backdoor(trainer.model, trigger_set, device)

    config_payload = experiment_request_to_dict(request)
    config_payload.update(
        {
            "dataset": cfg.DATASET,
            "device": device,
            "defense": canonical_defense_name(request.defense),
            "num_train_samples": len(clean_train_dataset),
            "poison_start_epoch": None if attack is None else request.poison_start_epoch,
        }
    )
    output = {
        "config": config_payload,
        "history": history,
        "final_clean_accuracy": final_clean_accuracy,
        "final_attack_success_rate": final_attack_success_rate,
        "final_MA": final_clean_accuracy,
        "final_BA": final_attack_success_rate,
    }
    if request.write_json:
        out_path = build_output_path(request)
        out_path.write_text(json.dumps(output, indent=2))
        output["results_path"] = str(out_path)
    else:
        output["results_path"] = None
    return output


def main():
    args = parse_args()
    request = build_experiment_request_from_args(args)
    result = run_experiment(request)
    print(
        json.dumps(
            {
                "final_clean_accuracy": result["final_clean_accuracy"],
                "final_attack_success_rate": result["final_attack_success_rate"],
                "results": result["results_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
