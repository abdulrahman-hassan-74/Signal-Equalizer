/**
 * state.js - Global application state (All members)
 * ==================================================
 *
 * Single source of truth for the entire app.
 * All components read from and write to this object.
 * No hidden dependencies - all state visible in one place.
 */

const AppState = {
  // ==================== SIGNAL DATA ====================

  /** Current mode name: 'instruments', 'animals', 'voices', 'ecg', 'generic' */
  currentMode: 'instruments',

  /** Unique ID for currently loaded signal (from backend) */
  inputSignalId: null,

  /** Raw audio samples from uploaded file (Float32Array, values -1.0 to 1.0) */
  inputSamples: null,

  /** Processed audio after equalization (Float32Array) */
  outputSamples: null,

  /** Sample rate in Hz (e.g., 44100) */
  sampleRate: 44100,

  // ==================== EQUALIZER STATE ====================

  /** Gains for each band: { bandId: gain, ... } where gain is 0.0 to 2.0 */
  gains: {},

  /** Current mode configuration from JSON {name, bands} */
  currentConfig: null,

  /** Equalization method: 'fourier' or 'wavelet' */
  waveletMode: 'fourier',

  // ==================== PLAYBACK STATE ====================

  /** Is playback currently active? */
  isPlaying: false,

  /** Current playback position in seconds */
  currentTime: 0,

  // ==================== CACHED DATA ====================

  /** FFT data for current output signal (for frequency plot) */
  currentFFT: null,

  /** Spectrogram data for current output signal (2D array) */
  currentSpectrogram: null,

  // ==================== METHODS ====================

  /**
   * Set gain for a specific band.
   * Called when user moves a slider.
   * @param {number} bandId - Band ID
   * @param {number} value - Gain value (0.0 to 2.0)
   */
  setGain(bandId, value) {
    // Clamp to valid range
    value = Math.max(0, Math.min(2, value));
    this.gains[bandId] = value;
  },

  /**
   * Switch to a different mode.
   * Resets gains to empty.
   * @param {string} modeName - New mode name
   */
  setMode(modeName) {
    this.currentMode = modeName;
    this.gains = {};
  },

  /**
   * Update the current mode configuration.
   * Initializes gains for all bands in config.
   * @param {Object} config - {name, bands}
   */
  setConfig(config) {
    this.currentConfig = config;
    this.gains = {};

    if (config && Array.isArray(config.bands)) {
      config.bands.forEach(band => {
        this.gains[band.id] = 1.0;  // Start at neutral
      });
    }
  },

  /**
   * Update FFT data for frequency plot.
   * @param {Array} fftData - [{frequency, magnitude}, ...]
   */
  setFFT(fftData) {
    this.currentFFT = fftData;
  },

  /**
   * Update spectrogram data.
   * @param {Array} specData - 2D array of spectrogram values
   */
  setSpectrogram(specData) {
    this.currentSpectrogram = specData;
  },

  /**
   * Reset playback position.
   */
  resetPlayback() {
    this.isPlaying = false;
    this.currentTime = 0;
  },

  /**
   * Get current state as readable object for debugging.
   * @returns {Object} State snapshot
   */
  getInfo() {
    return {
      mode: this.currentMode,
      signalId: this.inputSignalId,
      sampleRate: this.sampleRate,
      gains: { ...this.gains },
      isPlaying: this.isPlaying,
      wavelet: this.waveletMode,
      numBands: Object.keys(this.gains).length,
      hasSignal: this.inputSignalId !== null,
      hasConfig: this.currentConfig !== null
    };
  }
};
