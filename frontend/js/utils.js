// utils.js — Utility / Helper Functions
// =======================================
// Used by ALL team members.
// Contains small reusable functions that don't belong to any specific feature.
//
// Functions:
//   showToast()      - Display a notification message
//   downloadJSON()   - Download an object as a .json file
//   readJSONFile()   - Read a .json file the user selects
//   loadAudioFile()  - Decode an audio file using Web Audio API
//   formatTime()     - Format seconds as "MM:SS.mmm"
//   clamp()          - Clamp a number between min and max


/**
 * Show a temporary toast notification.
 *
 * @param {string} message - Text to show
 * @param {"info"|"success"|"error"} type - Color/style variant
 * @param {number} duration - Auto-hide after this many ms (default 2500)
 */
function showToast(message, type = "info", duration = 2500) {
  const toast = document.getElementById("toast");
  if (!toast) return;

  // Remove old type classes first
  toast.classList.remove("info", "success", "error", "show");

  toast.textContent = message;
  toast.classList.add(type);

  // Force reflow so the transition plays even if already visible
  void toast.offsetWidth;

  toast.classList.add("show");

  // Auto-hide after duration
  clearTimeout(toast._hideTimer);
  toast._hideTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, duration);
}


/**
 * Trigger browser download of an object as a JSON file.
 *
 * @param {Object} data     - The data to serialize
 * @param {string} filename - File name including .json
 *
 * Example: downloadJSON(config, "instruments_config.json")
 */
function downloadJSON(data, filename) {
  // Pretty-print JSON with 2-space indent
  const jsonString = JSON.stringify(data, null, 2);

  // Create a Blob (binary file object)
  const blob = new Blob([jsonString], { type: "application/json" });

  // Create a temporary URL for the blob
  const url = URL.createObjectURL(blob);

  // Create an invisible link and click it to trigger download
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();

  // Clean up the temporary URL
  URL.revokeObjectURL(url);
}


/**
 * Read and parse a JSON file selected by the user.
 *
 * @param {File} file - File object from <input type="file">
 * @returns {Promise<Object>} Parsed JSON data
 *
 * Example:
 *   const config = await readJSONFile(e.target.files[0]);
 */
function readJSONFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Invalid JSON file: ${err.message}`));
      }
    };

    reader.onerror = () => reject(new Error("Failed to read file"));

    // Read file as text (JSON is text)
    reader.readAsText(file);
  });
}


/**
 * Decode an audio file (WAV, MP3, etc.) into raw Float32 samples
 * using the browser's built-in Web Audio API decoder.
 *
 * @param {File} file - Audio file from <input type="file">
 * @returns {Promise<{samples: Float32Array, sampleRate: number}>}
 *
 * Note: This returns the CLIENT-SIDE decoded version for drawing waveforms.
 *       The server also decodes independently for signal processing.
 */
async function loadAudioFile(file) {
  // Read file bytes as ArrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Create Web Audio context for decoding
  // (AudioContext can decode most audio formats natively)
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  try {
    // Decode compressed audio (MP3, OGG, etc.) to raw PCM
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Get first channel (index 0) as Float32Array
    // Values range: -1.0 to +1.0
    const samples = audioBuffer.getChannelData(0);

    return {
      samples: new Float32Array(samples),  // Copy to own Float32Array
      sampleRate: audioBuffer.sampleRate
    };
  } finally {
    // Always close the context (prevents resource leak)
    await audioCtx.close();
  }
}


/**
 * Format seconds into a human-readable time string.
 *
 * @param {number} totalSeconds - Time in seconds (can be float)
 * @returns {string} e.g. "01:23.456"
 *
 * Examples:
 *   formatTime(0)       → "00:00.000"
 *   formatTime(65.5)    → "01:05.500"
 *   formatTime(125.123) → "02:05.123"
 */
function formatTime(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = Math.floor(totalSeconds % 60);
  const ms   = Math.floor((totalSeconds % 1) * 1000);

  const mm  = String(mins).padStart(2, "0");
  const ss  = String(secs).padStart(2, "0");
  const mmm = String(ms).padStart(3, "0");

  return `${mm}:${ss}.${mmm}`;
}


/**
 * Clamp a number between min and max.
 *
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 *
 * Example: clamp(150, 0, 100) → 100
 */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
