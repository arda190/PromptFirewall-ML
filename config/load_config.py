import json
from pathlib import Path


def load_config(file_path:str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Config file not found")

    with open(path) as json_file:
        return json.load(json_file)


