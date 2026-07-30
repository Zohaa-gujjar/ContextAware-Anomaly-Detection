from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    return config