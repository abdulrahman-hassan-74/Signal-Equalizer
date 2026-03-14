class FreqPlot {
  constructor(canvasId) {
    this.ctx   = document.getElementById(canvasId).getContext('2d');
    this.chart = null;
    this.scale = 'linear';
  }

  draw(inputFFT, outputFFT) {
    const labels = inputFFT.map(d => d.frequency.toFixed(0));
    if (this.chart) this.chart.destroy();
    this.chart = new Chart(this.ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Input',  data: inputFFT.map(d=>d.magnitude),
            borderColor: '#4488ff', pointRadius: 0, borderWidth: 1.5 },
          { label: 'Output', data: outputFFT.map(d=>d.magnitude),
            borderColor: '#ff4444', pointRadius: 0, borderWidth: 1.5 },
        ]
      },
      options: {
        animation: false,
        plugins: { legend: { labels: { color: '#ffffff' } } },
        scales: {
          x: { ticks: { color: '#aaa', maxTicksLimit: 12 },
               grid: { color: '#333' } },
          y: { ticks: { color: '#aaa' }, grid: { color: '#333' } }
        }
      }
    });
  }

  update(outputFFT) {
    if (!this.chart) return;
    this.chart.data.datasets[1].data = outputFFT.map(d => d.magnitude);
    this.chart.update('none');
  }

  setScale(type) {
    this.scale = type;
    if (!this.chart) return;
    if (type === 'audiogram') {
      // Filter to audiogram frequencies only
      const audioFreqs = [125,250,500,1000,2000,4000,8000];
      this.chart.options.scales.x.type = 'logarithmic';
    } else {
      this.chart.options.scales.x.type = 'linear';
    }
    this.chart.update();
  }
}

Add to main.js to connect the radio buttons:
document.querySelectorAll('input[name=scale]').forEach(radio => {
  radio.addEventListener('change', e => freqPlot.setScale(e.target.value));
});
