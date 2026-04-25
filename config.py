from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
RESULTS_DIR = BASE_DIR / "results"

# Project scope
DATASET = "CIFAR10"
ARCH = "resnet18"
NUM_CLASSES = 10

# Centralized training setup
NUM_EPOCHS = 5
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
SEED = 42

# Preset sizing. The sample multiplier names preserve compatibility with the
# source project presets while training remains centralized.
NUM_CLIENTS = 10
NUM_MALICIOUS = 2
NUM_ROUNDS = NUM_EPOCHS
LOCAL_EPOCHS = 1
IID_RATE = 0.8

# Attack setup
BACKDOOR_TYPE = "semantic"
PIXEL_TARGET_LABEL = 0
SEMANTIC_SOURCE_LABEL = 1
SEMANTIC_TARGET_LABEL = 2
POISONED_DATA_RATE = 0.5
POISON_START_EPOCH = 1
TRIGGER_SIZE = 4
TRIGGER_POS = "bottom-right"

# Parameter-update defense setup
SAFE_SPLIT_WINDOW = 3
DCT_LOW_FREQ_FRAC = 0.25
ROTATION_MATRIX_WIDTH = 128
EPS = 1e-8

# Baselines
DP_CLIP_NORM = 1.0
DP_NOISE_SCALE = 1e-3

# Shared experiment presets
DEFAULT_PRESET = "paper"
PRESET_ALIASES = {
    "fast-dev": "lite",
}
EXPERIMENT_PRESETS = {
    "lite": {
        "arch": "simple_cnn",
        "num_clients": 4,
        "num_malicious": 1,
        "num_rounds": 1,
        "iid_rate": IID_RATE,
        "max_samples_per_client": 256,
        "local_epochs": 1,
        "batch_size": 64,
        "eval_batch_size": 128,
    },
    "medium": {
        "arch": "resnet18",
        "num_clients": 6,
        "num_malicious": 1,
        "num_rounds": 4,
        "iid_rate": IID_RATE,
        "max_samples_per_client": 1024,
        "local_epochs": 1,
        "batch_size": 96,
        "eval_batch_size": 128,
    },
    "paper": {
        "arch": ARCH,
        "num_clients": NUM_CLIENTS,
        "num_malicious": NUM_MALICIOUS,
        "num_rounds": NUM_ROUNDS,
        "iid_rate": IID_RATE,
        "max_samples_per_client": None,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
    },
}

FAST_DEV_RUN = False
FAST_DEV_NUM_CLIENTS = EXPERIMENT_PRESETS["lite"]["num_clients"]
FAST_DEV_NUM_MALICIOUS = EXPERIMENT_PRESETS["lite"]["num_malicious"]
FAST_DEV_NUM_ROUNDS = EXPERIMENT_PRESETS["lite"]["num_rounds"]
FAST_DEV_MAX_SAMPLES_PER_CLIENT = EXPERIMENT_PRESETS["lite"]["max_samples_per_client"]


def normalize_preset_name(name: str | None) -> str:
    if name is None:
        return DEFAULT_PRESET
    normalized = PRESET_ALIASES.get(name, name)
    if normalized not in EXPERIMENT_PRESETS:
        valid = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Expected one of: {valid}.")
    return normalized


def resolve_preset(name: str | None = None) -> dict[str, object]:
    normalized = normalize_preset_name(name)
    return {
        "name": normalized,
        **EXPERIMENT_PRESETS[normalized],
    }


def resolve_device(requested: str | None = None) -> str:
    import torch

    if requested is not None:
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
