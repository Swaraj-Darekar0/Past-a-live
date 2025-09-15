# Contents of /past_life_predictor/past_life_predictor/src/utils/config.py

import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Configuration constants
API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
}

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}