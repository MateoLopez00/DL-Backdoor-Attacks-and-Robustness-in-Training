from __future__ import annotations

import argparse
import json
from pathlib import Path

from main import build_experiment_request, run_experiment


ATTACK_DEFENSE_RUNS = [
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", "none"],
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", "update_defense"],
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", "update_defense", "--poison-start-epoch", "3"],
    ["--arch", "resnet18", "--backdoor", "pixel", "--defense", "none"],
    ["--arch", "resnet18", "--backdoor", "pixel", "--defense", "update_defense"],
    ["--arch", "resnet18", "--backdoor", "pixel", "--defense", "update_defense", "--poison-start-epoch", "3"],
]

ARCHITECTURE_RUNS = [
    ["--arch", arch, "--backdoor", attack, "--defense", defense, "--poison-start-epoch", str(poison_start)]
    for arch in ("simple_cnn", "resnet18")
    for attack in ("none", "pixel", "semantic")
    for defense in ("none", "update_defense")
    for poison_start in ([1] if attack == "none" or defense == "none" else [1, 3])
]

BASELINE_RUNS = [
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", defense]
    for defense in ("none", "dp", "krum", "update_defense")
]


def cli_args_to_overrides(args: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    idx = 0
    while idx < len(args):
        key = args[idx]
        if key == "--arch":
            overrides["arch"] = args[idx + 1]
            idx += 2
        elif key == "--backdoor":
            overrides["backdoor"] = args[idx + 1]
            idx += 2
        elif key == "--defense":
            overrides["defense"] = args[idx + 1]
            idx += 2
        elif key == "--num-rounds":
            overrides["num_rounds"] = int(args[idx + 1])
            idx += 2
        elif key == "--num-clients":
            overrides["num_clients"] = int(args[idx + 1])
            idx += 2
        elif key == "--max-samples-per-client":
            overrides["max_samples_per_client"] = int(args[idx + 1])
            idx += 2
        elif key == "--local-epochs":
            overrides["local_epochs"] = int(args[idx + 1])
            idx += 2
        elif key == "--batch-size":
            overrides["batch_size"] = int(args[idx + 1])
            idx += 2
        elif key == "--eval-batch-size":
            overrides["eval_batch_size"] = int(args[idx + 1])
            idx += 2
        elif key == "--pdr":
            overrides["pdr"] = float(args[idx + 1])
            idx += 2
        elif key == "--poison-start-epoch":
            overrides["poison_start_epoch"] = int(args[idx + 1])
            idx += 2
        elif key == "--device":
            overrides["device"] = args[idx + 1]
            idx += 2
        elif key == "--seed":
            overrides["seed"] = int(args[idx + 1])
            idx += 2
        elif key == "--preset":
            overrides["preset"] = args[idx + 1]
            idx += 2
        elif key == "--fast-dev-run":
            overrides["preset"] = "lite"
            idx += 1
        else:
            raise ValueError(f"Unsupported experiment argument: {key}")
    return overrides


def run_one(args: list[str], preset: str | None = None) -> dict[str, object]:
    overrides = cli_args_to_overrides(args)
    effective_preset = overrides.pop("preset", preset)
    request = build_experiment_request(preset=effective_preset, **overrides)
    return run_experiment(request)


def main():
    parser = argparse.ArgumentParser(description="Run centralized backdoor robustness experiment matrices.")
    parser.add_argument("--out", default="results/experiment_index.json")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--preset", choices=["lite", "medium", "paper"], default=None)
    args = parser.parse_args()

    experiment_sets = {
        "attack_defense": ATTACK_DEFENSE_RUNS,
        "architecture_comparison": ARCHITECTURE_RUNS,
        "baseline_panel": BASELINE_RUNS,
    }

    outputs: dict[str, list[dict]] = {}
    for name, run_list in experiment_sets.items():
        outputs[name] = []
        for run_args in run_list:
            effective_args = list(run_args)
            if args.fast_dev_run:
                effective_args.append("--fast-dev-run")
            result = run_one(effective_args, preset=args.preset)
            outputs[name].append({"args": effective_args, "result": result})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps({"index": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
