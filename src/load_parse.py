from src.logging_config import setup_logging
import numpy as np
import logging
import json
import yaml


# setup_logging()
logger = logging.getLogger(__name__)


def load_yaml(fpath) -> dict:
    """Load yaml file"""
    with open(fpath, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exec:
            raise yaml.YAMLError(f"Error loading file {fpath}")
    return data


def load_csv(fpath: str, skip_header=False) -> np.ndarray:
    """Load csv with numpy. Optionally skip first line"""
    data = np.genfromtxt(fpath, delimiter=",", skip_header=skip_header)
    logger.info(msg=f"Loaded csv with shape {(data.shape)}")
    return data


def load_txt(fpath: str) -> str:
    with open(fpath, "r") as f:
        data = f.read()
    logger.info(msg=f"Loaded {fpath} file")
    return data


def load_json(fpath: str) -> dict:
    """Load json file"""
    with open(fpath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error loading file {fpath}", e.doc, e.pos)
    logger.info(msg=f"Loaded {fpath} file")
    return data
