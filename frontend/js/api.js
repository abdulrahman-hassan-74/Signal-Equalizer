/**
 * api.js - All backend communication (Member 4)
 * =============================================
 *
 * This is the ONLY file that uses fetch().
 * All API calls go through here for centralized error handling.
 */

const API = {
  BASE: 'http://localhost:8000',

  /**
   * Health check - verify backend is running.
   * @returns {Promise<Object>} {status, signals_cached}
   */
  async health() {
    const res = await fetch(`${this.BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },

  /**
   * Upload audio file to backend.
   * @param {File} file - Audio file (WAV, MP3, etc.)
   * @returns {Promise<Object>} {signal_id, sample_rate, duration, samples}
   */
  async uploadSignal(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${this.BASE}/upload`, {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    return data;
  },

  /**
   * Get signal data for waveform display.
   * @param {string} signalId - Signal ID from upload
   * @returns {Promise<Object>} {samples, sample_rate, duration}
   */
  async getSignal(signalId) {
    const res = await fetch(`${this.BASE}/signal/${signalId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Signal not found');
    return data;
  },

  /**
   * Apply equalization to a signal.
   * @param {string} signalId - Signal ID
   * @param {Array} gains - Gains array from sliders
   * @param {string} method - 'fourier' or 'wavelet'
   * @param {string} wavelet - Wavelet type
   * @returns {Promise<Object>} {output_signal, fft_input, fft_output, ...}
   */
  async equalize(signalId, gains, method = 'fourier', wavelet = 'db4') {
    const res = await fetch(`${this.BASE}/equalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signal_id: signalId,
        gains,
        method,
        wavelet
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Equalize failed');
    return data;
  },

  /**
   * Load mode configuration from backend.
   * @param {string} modeName - 'instruments', 'animals', etc.
   * @returns {Promise<Object>} Mode configuration
   */
  async loadSettings(modeName) {
    const res = await fetch(`${this.BASE}/settings/${modeName}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Mode '${modeName}' not found`);
    return data;
  },

  /**
   * Save mode configuration to backend.
   * @param {string} modeName - Mode name
   * @param {Object} config - Configuration object
   * @returns {Promise<Object>} {success}
   */
  async saveSettings(modeName, config) {
    const res = await fetch(`${this.BASE}/settings/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode_name: modeName, config })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Save failed');
    return data;
  },

  /**
   * Run AI analysis on signal.
   * @param {string} signalId - Signal ID
   * @param {string} mode - Current mode
   * @returns {Promise<Object>} AI results
   */
  async runAiModel(signalId, mode) {
    const res = await fetch(`${this.BASE}/ai/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal_id: signalId, mode })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'AI failed');
    return data;
  }
};
