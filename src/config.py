from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = PROJECT_ROOT / path
    with open(config_path) as f:
        return yaml.safe_load(f)
