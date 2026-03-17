"""
modes.py
========
Mode metadata — NO hardcoded frequency ranges here.
All frequency ranges and slider configs live in the JSON files under settings/.

These settings files are loaded at runtime by settings_manager.py:
  - settings/instruments.json
  - settings/animals.json
  - settings/voices.json
  - settings/ecg.json
  - settings/generic.json

To edit bands: open the relevant JSON file and change the ranges there.
The app reloads them on every /api/settings/{mode} call — no restart needed.
"""

# Optimal wavelet per mode — determined by signal characteristics research:
#   instruments : db6  — Daubechies 6, good for tonal sustained audio
#   animals     : db4  — Daubechies 4, good for short transient sounds
#   voices      : haar — Haar, assigned by instructor
#   ecg         : db4  — Daubechies 4, biomedical standard
#
# These match OPTIMAL_WAVELETS in equalizer_engine.py — keep in sync.
WAVELET_RECOMMENDATIONS = {
    'instruments': 'db6',
    'animals':     'db4',
    'voices':      'haar',
    'ecg':         'db4',
    'generic':     'db4',
}

WAVELET_EXPLANATIONS = {
    'db6':  'Daubechies 6 — good frequency localisation for sustained tonal audio (instruments)',
    'db4':  'Daubechies 4 — standard for transient signals and biomedical (ECG, animals)',
    'haar': 'Haar wavelet — simplest orthogonal wavelet, instructor-recommended for voices',
}