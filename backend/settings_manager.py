"""
settings_manager.py - Load & Save Mode Configurations
=====================================================

Handles JSON config files for each mode.
Auto-creates default configs if they don't exist.
"""

import json
import os
from pathlib import Path

# ============================================
# SETTINGS DIRECTORY
# ============================================

# Get the backend directory
BACKEND_DIR = Path(__file__).parent.absolute()

# Settings folder: backend/../data/settings
SETTINGS_DIR = BACKEND_DIR / "settings"

print(f"📁 Settings directory: {SETTINGS_DIR}")

# Create directory if it doesn't exist
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# DEFAULT CONFIGURATIONS
# ============================================

DEFAULT_CONFIGS = {
    'instruments': {
        'mode': 'Musical Instruments',
        'sliders': [
            {
                'id': 1,
                'name': 'Bass Guitar',
                'ranges': [{'min': 40, 'max': 300}]
            },
            {
                'id': 2,
                'name': 'Drums',
                'ranges': [{'min': 50, 'max': 200}, {'min': 4000, 'max': 8000}]
            },
            {
                'id': 3,
                'name': 'Piano',
                'ranges': [{'min': 260, 'max': 4200}]
            },
            {
                'id': 4,
                'name': 'Violin',
                'ranges': [{'min': 2000, 'max': 8000}]
            }
        ]
    },

    'animals': {
        'mode': 'Animal Sounds',
        'sliders': [
            {
                'id': 1,
                'name': 'Dog Bark',
                'ranges': [{'min': 300, 'max': 2000}]
            },
            {
                'id': 2,
                'name': 'Bird Song',
                'ranges': [{'min': 2000, 'max': 8000}]
            },
            {
                'id': 3,
                'name': 'Cat Meow',
                'ranges': [{'min': 500, 'max': 1500}]
            },
            {
                'id': 4,
                'name': 'Frog Croak',
                'ranges': [{'min': 100, 'max': 800}]
            }
        ]
    },

    'voices': {
        'mode': 'Human Voices',
        'sliders': [
            {
                'id': 1,
                'name': 'Male Adult',
                'ranges': [{'min': 85, 'max': 180}, {'min': 300, 'max': 3000}]
            },
            {
                'id': 2,
                'name': 'Female Adult',
                'ranges': [{'min': 165, 'max': 255}, {'min': 300, 'max': 3400}]
            },
            {
                'id': 3,
                'name': 'Child',
                'ranges': [{'min': 250, 'max': 400}, {'min': 300, 'max': 4000}]
            },
            {
                'id': 4,
                'name': 'Elderly',
                'ranges': [{'min': 80, 'max': 200}, {'min': 300, 'max': 2500}]
            }
        ]
    },

    'ecg': {
        'mode': 'ECG Abnormalities',
        'sliders': [
            {
                'id': 1,
                'name': 'Normal Sinus Rhythm',
                'ranges': [{'min': 0.05, 'max': 40}]
            },
            {
                'id': 2,
                'name': 'Atrial Fibrillation',
                'ranges': [{'min': 350, 'max': 600}]
            },
            {
                'id': 3,
                'name': 'Bradycardia',
                'ranges': [{'min': 0.05, 'max': 15}]
            },
            {
                'id': 4,
                'name': 'Tachycardia',
                'ranges': [{'min': 0.5, 'max': 40}]
            }
        ]
    },

    'generic': {
        'mode': 'Generic Mode',
        'sliders': []
    }
}


# ============================================
# CREATE DEFAULTS (ONE TIME)
# ============================================

def _create_defaults():
    """Create default config files if they don't exist"""
    for mode_name, config in DEFAULT_CONFIGS.items():
        path = SETTINGS_DIR / f"{mode_name}.json"

        if not path.exists():
            try:
                with open(path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Created: {path}")
            except Exception as e:
                print(f"❌ Error creating {path}: {e}")
        else:
            print(f"📋 Found: {path}")


# Call on module load
_create_defaults()


# ============================================
# LOAD SETTINGS
# ============================================

def load_settings(mode_name):
    """
    Load mode configuration from JSON file.

    Args:
        mode_name: 'instruments', 'animals', 'voices', 'ecg', 'generic'

    Returns:
        dict: Configuration with 'mode' and 'sliders'

    Raises:
        FileNotFoundError: If mode file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = SETTINGS_DIR / f"{mode_name}.json"

    print(f"📖 Loading: {path}")

    if not path.exists():
        print(f"❌ File not found: {path}")
        raise FileNotFoundError(f"Mode '{mode_name}' not found at {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Loaded: {mode_name}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {path}: {e}")
        raise
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        raise


# ============================================
# SAVE SETTINGS
# ============================================

def save_settings(mode_name, config):
    """
    Save mode configuration to JSON file.

    Args:
        mode_name: 'instruments', 'animals', 'voices', 'ecg', 'generic'
        config: dict with 'mode' and 'sliders'

    Returns:
        bool: True if saved successfully
    """
    if not validate_settings(config):
        print(f"❌ Invalid config for {mode_name}")
        return False

    path = SETTINGS_DIR / f"{mode_name}.json"

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved: {path}")
        return True
    except Exception as e:
        print(f"❌ Error saving {path}: {e}")
        return False


# ============================================
# VALIDATION
# ============================================

def validate_settings(config):
    """
    Validate config structure.

    Required fields:
    - 'mode': string (mode name)
    - 'sliders': list of slider objects

    Each slider must have:
    - 'id': integer
    - 'name': string
    - 'ranges': list of {min, max} objects
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
        if not isinstance(slider, dict):
            return False
        if 'id' not in slider or 'name' not in slider or 'ranges' not in slider:
            return False
        if not isinstance(slider['ranges'], list):
            return False

        for rng in slider['ranges']:
            if not isinstance(rng, dict):
                return False
            if 'min' not in rng or 'max' not in rng:
                return False
            if rng['min'] >= rng['max']:
                return False

    return True


# ============================================
# LIST ALL MODES
# ============================================

def list_modes():
    """Get list of available modes"""
    return list(DEFAULT_CONFIGS.keys())


# ============================================
# EXPORT (for browser download)
# ============================================

def export_settings(config):
    """Return config as formatted JSON string"""
    return json.dumps(config, indent=2)