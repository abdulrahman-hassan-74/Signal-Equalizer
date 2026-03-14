class EqualizerPanel {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.onSliderChange = () => {};  // assigned from outside
  }

  buildSliders(modeConfig) {
    AppState.currentConfig = modeConfig;
    this.container.innerHTML = '';  // clear old sliders

    modeConfig.bands.forEach(band => {
      const wrapper = document.createElement('div');
      wrapper.className = 'slider-wrapper';

      const label = document.createElement('span');
      label.textContent = band.label;

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = 0;
      slider.max = 200;
      slider.value = 100;  // 100 = gain of 1.0 = no change

      const valueLabel = document.createElement('span');
      valueLabel.textContent = '1.0x';

      slider.addEventListener('input', () => {
        const gain = slider.value / 100;
        valueLabel.textContent = gain.toFixed(1) + 'x';
        AppState.setGain(band.id, gain);
        this.onSliderChange();
      });

      wrapper.appendChild(label);
      wrapper.appendChild(slider);
      wrapper.appendChild(valueLabel);
      this.container.appendChild(wrapper);
    });

    // Show Add Band button only for Generic mode
    if (modeConfig.name === 'Generic') {
      this._addGenericButton();
    }
  }

  _addGenericButton() {
    const btn = document.createElement('button');
    btn.textContent = '+ Add Band';
    btn.onclick = () => this._showAddBandForm();
    this.container.appendChild(btn);
  }

  _showAddBandForm() {
    const label    = prompt('Band label (e.g. Bass):');
    const minFreq  = parseFloat(prompt('Min frequency in Hz:'));
    const maxFreq  = parseFloat(prompt('Max frequency in Hz:'));
    if (!label || isNaN(minFreq) || isNaN(maxFreq)) return;
    const newBand = {
      id: AppState.currentConfig.bands.length + 1,
      label,
      freq_ranges: [[minFreq, maxFreq]]
    };
    AppState.currentConfig.bands.push(newBand);
    this.buildSliders(AppState.currentConfig);
  }
}
