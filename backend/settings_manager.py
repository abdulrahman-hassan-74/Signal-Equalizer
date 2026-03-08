# settings_manager.py

import json
import os

SETTINGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'settings')


def load_settings(mode_name):
    """
    Loads a mode config from data/settings/{mode_name}.json.
    Returns the parsed dict.
    Raises FileNotFoundError if the file does not exist.
    """
    path = os.path.join(SETTINGS_DIR, f'{mode_name}.json')
    with open(path, 'r') as f:
        return json.load(f)


def save_settings(mode_name, config):
    """
    Validates then saves config to data/settings/{mode_name}.json.
    Returns True on success, False if validation fails.
    """
    if not validate_settings(config):
        return False
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    path = os.path.join(SETTINGS_DIR, f'{mode_name}.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return True


def validate_settings(config):
    """
    Checks that a config dict has all required fields.
    Required top-level fields: 'mode', 'sliders'
    Required per-slider fields: 'name', 'ranges'
    Each range must be a dict with 'min' and 'max' where min < max.
    Returns True if valid, False otherwise.
    """
    if not isinstance(config, dict):
        return False
    if 'mode' not in config:
        return False
    if 'sliders' not in config:
        return False
    if not isinstance(config['sliders'], list):
        return False
    for slider in config['sliders']:
        if 'name' not in slider:
            return False
        if 'ranges' not in slider:
            return False
        if not isinstance(slider['ranges'], list) or len(slider['ranges']) == 0:
            return False
        for rng in slider['ranges']:
            if not isinstance(rng, dict):
                return False
            if 'min' not in rng or 'max' not in rng:
                return False
            if rng['min'] >= rng['max']:
                return False
    return True


def export_settings(config):
    """
    Returns the config as a formatted JSON string (for browser download).
    """
    return json.dumps(config, indent=2)
