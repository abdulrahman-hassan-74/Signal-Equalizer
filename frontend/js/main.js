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
