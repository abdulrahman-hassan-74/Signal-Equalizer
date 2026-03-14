class Spectrogram {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
  }

  draw(data) {
    if (!data || !data.length) return;
    const rows = data.length;
    const cols = data[0].length;
    const cw   = this.canvas.width  / cols;
    const ch   = this.canvas.height / rows;
    data.forEach((row, ri) => {
      row.forEach((val, ci) => {
        const r = Math.floor(val * 255);
        const g = Math.floor(val * 180);
        const b = Math.floor((1-val) * 200);
        this.ctx.fillStyle = `rgb(${r},${g},${b})`;
        this.ctx.fillRect(ci*cw, (rows-ri-1)*ch, cw, ch);
      });
    });
  }

  update(data) { this.draw(data); }
  show() { this.canvas.style.display = 'block'; }
  hide() { this.canvas.style.display = 'none';  }
}

