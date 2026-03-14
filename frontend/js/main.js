// Create all main objects
const equalizerPanel = new EqualizerPanel('equalizer-panel');
const inputViewer    = new CineViewer('input-viewer');
const outputViewer   = new CineViewer('output-viewer');
const inputSpec      = new Spectrogram('input-spectrogram');
const outputSpec     = new Spectrogram('output-spectrogram');
const freqPlot       = new FreqPlot('freq-plot');

inputViewer.syncWith(outputViewer);

// Load default mode on page start
async function loadMode(modeName) {
  const config = await API.loadSettings(modeName);
  AppState.setMode(modeName);
  equalizerPanel.buildSliders(config);
}

loadMode('instruments');

// Mode selector change
document.getElementById('mode-selector').addEventListener('change', e => {
  loadMode(e.target.value);
});
document.getElementById('toggle-spectrogram').addEventListener('click', () => {
  const panel = document.getElementById('spectrogram-panel');
  panel.style.display = panel.style.display === 'none' ? 'grid' : 'none';
});
/**
 * main.js - Application Entry Point
 * =================================
 * Member 3 & 4: Wires everything together
 *
 * Flow:
 * 1. Initialize all components
 * 2. Set up event handlers
 * 3. Handle file upload
 * 4. Trigger equalization
 * 5. Update UI
 */

let inputViewer, outputViewer, equalizerPanel, freqPlotInst, inputSpec, outputSpec;
let _equalizeLock = false;

document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Initializing Signal Equalizer');

  // Create all components
  inputViewer    = new CineViewer('input-viewer');
  outputViewer   = new CineViewer('output-viewer');
  inputViewer.syncWith(outputViewer);  // Link viewers

  equalizerPanel = new EqualizerPanel('equalizer-panel');
  equalizerPanel.onSliderChange = runEqualize;  // Auto-equalize

  freqPlotInst = new FreqPlot('freq-plot');
  inputSpec    = new Spectrogram('input-spectrogram');
  outputSpec   = new Spectrogram('output-spectrogram');

  // Set up all event handlers
  wireEvents();

  // Check backend connection
  checkBackend();

  // Load default mode
  loadMode('instruments');

  console.log('✅ Application ready');
});

function wireEvents() {
  // File upload
  document.getElementById('load-signal').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (file) await handleUpload(file);
    e.target.value = '';
  });

  // Mode selector
  document.getElementById('mode-selector').addEventListener('change', e => {
    loadMode(e.target.value);
  });

  // Wavelet selector
  document.getElementById('wavelet-selector').addEventListener('change', e => {
    AppState.waveletMode = e.target.value;
    runEqualize();
  });

  // Save config
  document.getElementById('save-config-btn').addEventListener('click', () => {
    if (!AppState.currentConfig) {
      showToast('No config loaded', 'error');
      return;
    }
    downloadJSON(AppState.currentConfig, AppState.currentMode + '.json');
    showToast('Config saved ✅', 'success');
  });

  // Load config buttons
  document.getElementById('load-config-btn').addEventListener('click', () => {
    document.getElementById('load-config-file').click();
  });

  document.getElementById('load-config-file').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const cfg = await readJSONFile(file);
      if (!cfg.name || !Array.isArray(cfg.bands)) {
        throw new Error('Missing "name" or "bands" fields');
      }

      _equalizeLock = true;
      AppState.setConfig(cfg);
      equalizerPanel.buildSliders(cfg);
      _equalizeLock = false;

      showToast('Config "' + cfg.name + '" loaded ✅', 'success');
      runEqualize();

    } catch (err) {
      showToast('Load failed: ' + err.message, 'error');
    }
    e.target.value = '';
  });

  // Frequency scale radio buttons
  document.querySelectorAll('input[name="freq-scale"]').forEach(r => {
    r.addEventListener('change', e => freqPlotInst.setScale(e.target.value));
  });

  // Spectrogram toggle
  document.getElementById('toggle-spectrogram-btn').addEventListener('click', () => {
    const panel = document.getElementById('spectrogram-panel');
    const show = panel.style.display === 'none';
    panel.style.display = show ? 'block' : 'none';
    document.getElementById('toggle-spectrogram-btn').textContent =
      show ? '👁️ Hide Spectrograms' : '👁️ Show Spectrograms';
  });

  // Playback buttons
  document.getElementById('btn-play').addEventListener('click', () => {
    inputViewer.play();
    outputViewer.play();
  });

  document.getElementById('btn-pause').addEventListener('click', () => {
    inputViewer.pause();
    outputViewer.pause();
  });

  document.getElementById('btn-stop').addEventListener('click', () => {
    inputViewer.stop();
    outputViewer.stop();
    audioPlayer.stop();
    document.getElementById('btn-audio').textContent = '🔊 Audio';
  });

  // Audio playback
  document.getElementById('btn-audio').addEventListener('click', async () => {
    if (!AppState.inputSignalId) {
      showToast('Upload a signal first', 'error');
      return;
    }

    const btn = document.getElementById('btn-audio');

    if (audioPlayer.isPlaying) {
      audioPlayer.stop();
      btn.textContent = '🔊 Audio';
      return;
    }

    btn.textContent = '⏳ Loading…';
    btn.disabled = true;

    const ok = await audioPlayer.loadFromServer(AppState.inputSignalId);

    btn.disabled = false;
    if (ok) {
      audioPlayer.play();
      btn.textContent = '⏹️ Stop Audio';
    } else {
      btn.textContent = '🔊 Audio';
    }
  });

  // Speed slider
  document.getElementById('speed-slider').addEventListener('input', e => {
    const speed = e.target.value / 100;
    document.getElementById('speed-val').textContent = speed.toFixed(2) + '×';
    inputViewer.setSpeed(speed);
    outputViewer.setSpeed(speed);
    audioPlayer.setSpeed(speed);
  });

  // Zoom controls
  document.getElementById('btn-zoom-in').addEventListener('click', () => {
    inputViewer.zoom(0.5);
  });

  document.getElementById('btn-zoom-out').addEventListener('click', () => {
    inputViewer.zoom(2.0);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    inputViewer.reset();
    outputViewer.reset();
  });

  // AI analysis
  document.getElementById('run-ai-btn').addEventListener('click', runAI);

  // Update time display every 100ms
  setInterval(() => {
    const t = AppState.currentTime || 0;
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    const ms = Math.floor((t * 1000) % 1000);
    const fmt = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;

    document.getElementById('playback-time').textContent = fmt;
    document.getElementById('input-time-label').textContent = fmt;
    document.getElementById('output-time-label').textContent = fmt;
  }, 100);
}

async function checkBackend() {
  const el = document.getElementById('backend-status');
  try {
    await API.health();
    el.textContent = '● Backend OK';
    el.style.color = '#4ecca3';
  } catch (_) {
    el.textContent = '● Backend Offline';
    el.style.color = '#ff5f6d';
    showToast(
      '⚠️ Backend not running! Open terminal in /backend and run: python main.py',
      'error', 10000
    );
  }
}

async function loadMode(modeName) {
  try {
    AppState.setMode(modeName);
    const cfg = await API.loadSettings(modeName);
    AppState.setConfig(cfg);

    _equalizeLock = true;
    equalizerPanel.buildSliders(cfg);
    _equalizeLock = false;

    const lbl = document.getElementById('mode-label');
    if (lbl) lbl.textContent = cfg.name || modeName;

    if (AppState.inputSignalId) runEqualize();

  } catch (err) {
    console.error('loadMode error:', err);
    showToast('Mode load failed: ' + err.message, 'error');
  }
}

async function handleUpload(file) {
  showToast('⏳ Uploading "' + file.name + '"…', 'info', 5000);

  try {
    const res = await API.uploadSignal(file);

    AppState.inputSignalId = res.signal_id;
    AppState.sampleRate = res.sample_rate;

    // Display file info
    const info = document.getElementById('signal-info');
    if (info) {
      info.textContent = file.name + '  ·  ' +
        res.duration.toFixed(2) + 's  ·  ' +
        (res.sample_rate / 1000).toFixed(1) + ' kHz';
    }

    // Draw input waveform
    if (res.samples && res.samples.length > 0) {
      const samples = new Float32Array(res.samples);
      AppState.inputSamples = samples;
      inputViewer.load(samples, res.sample_rate);
    }

    // Run initial equalization
    await runEqualize();

    showToast('✅ "' + file.name + '" loaded', 'success');

  } catch (err) {
    console.error('Upload error:', err);
    showToast('❌ Upload failed: ' + err.message, 'error', 8000);
  }
}

async function runEqualize() {
  if (_equalizeLock) return;
  if (!AppState.inputSignalId) return;
  if (!AppState.currentConfig) return;

  try {
    // Build gains array from slider positions
    const gains = AppState.currentConfig.bands.map(band => ({
      band_id: band.id,
      freq_ranges: band.freq_ranges,
      gain: AppState.gains[band.id] !== undefined ? AppState.gains[band.id] : 1.0
    }));

    const res = await API.equalize(
      AppState.inputSignalId,
      gains,
      AppState.waveletMode,
      'db4'
    );

    // Update output waveform
    if (res.output_signal && res.output_signal.length > 0) {
      const out = new Float32Array(res.output_signal);
      AppState.outputSamples = out;
      outputViewer.load(out, AppState.sampleRate);
    }

    // Update FFT chart
    if (res.fft_input && res.fft_output) {
      freqPlotInst.draw(res.fft_input, res.fft_output);
    }

    // Update spectrograms
    if (res.spectrogram_input) inputSpec.draw(res.spectrogram_input);
    if (res.spectrogram_output) outputSpec.draw(res.spectrogram_output);

  } catch (err) {
    console.error('runEqualize error:', err);
  }
}

async function runAI() {
  if (!AppState.inputSignalId) {
    showToast('Upload a signal first', 'error');
    return;
  }

  const btn = document.getElementById('run-ai-btn');
  const div = document.getElementById('ai-results');

  btn.disabled = true;
  btn.textContent = '⏳ Analyzing…';
  div.innerHTML = '<p style="color:var(--text-muted);padding:8px;">Running AI model…</p>';

  try {
    const result = await API.runAiModel(AppState.inputSignalId, AppState.currentMode);

    const card = (title, value) => `
      <div class="ai-card">
        <div class="ai-card-title">${title}</div>
        <div class="ai-card-val">${value}</div>
      </div>
    `;

    let html = '<div class="ai-results-grid">';

    if (result.error) {
      html += card('❌ Error', result.error);
    } else if (AppState.currentMode === 'ecg') {
      html += card('❤️ Heart Rate', (result.heart_rate || 0).toFixed(1) + ' bpm');
      html += card('🩺 Condition', result.condition || result.classification || '—');
      html += card('📊 Beats', (result.peak_count || 0) + ' peaks');
      html += card('📈 RR Variation', (result.rr_std_ms || 0).toFixed(1) + ' ms');
    } else if (AppState.currentMode === 'animals') {
      html += card('🐾 Animal', result.animal_type || '—');
      html += card('🎵 Peak Freq', (result.peak_frequency || 0).toFixed(0) + ' Hz');
      html += card('📊 Confidence', ((result.confidence || 0) * 100).toFixed(1) + '%');
    } else {
      html += card('📊 Mode', result.mode || AppState.currentMode);
      html += card('🎵 Peak Freq', (result.peak_frequency || 0).toFixed(0) + ' Hz');
      html += card('📈 RMS Energy', (result.rms_energy || 0).toFixed(4));
      html += card('⚡ Peak Amp', (result.peak_amplitude || 0).toFixed(4));
    }

    html += '</div>';
    div.innerHTML = html;

  } catch (err) {
    div.innerHTML = `<p style="color:#ff5f6d;padding:8px;">❌ ${err.message}</p>`;
    showToast('AI failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🤖 Run AI';
  }
}
