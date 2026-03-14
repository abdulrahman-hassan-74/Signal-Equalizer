/**
 * cineViewer.js - Waveform Canvas Viewer (Member 4)
 * =================================================
 *
 * Draws audio waveform on HTML5 Canvas with cinema-style scrolling playback.
 * Features:
 * - Real-time waveform rendering
 * - Smooth playback animation (60fps)
 * - Zoom in/out with mouse wheel
 * - Pan by dragging
 * - Synchronized viewers (input/output stay locked together)
 * - Time labels and playback cursor
 * - Optimized rendering for large signals (downsampling)
 *
 * "Cine" = cinema: shows signal scrolling like film through a projector.
 */

class CineViewer {
  /**
   * @param {string} canvasId - ID of the <canvas> element in HTML
   * @param {string} color - Color for the waveform (default cyan for input, pink for output)
   */
  constructor(canvasId, color = '#4ecca3') {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      console.error(`CineViewer: canvas "${canvasId}" not found`);
      return;
    }

    this.ctx = this.canvas.getContext('2d');
    this.canvasId = canvasId;
    this.color = color;

    // Signal data
    this.signal = null;           // Float32Array of audio samples
    this.sampleRate = 44100;      // Hz

    // View window (in samples) - the visible portion of the full signal
    this.viewStart = 0;            // First sample index to display
    this.viewEnd = 0;              // Last sample index to display

    // Playback state
    this.isPlaying = false;
    this.animFrame = null;         // requestAnimationFrame ID
    this.speed = 1.0;              // Playback speed multiplier

    // Synchronization with another viewer
    this.syncTarget = null;        // Another CineViewer to keep in sync

    // Mouse interaction for dragging (panning)
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartView = 0;

    // Bind methods to maintain 'this' context
    this.draw = this.draw.bind(this);
    this.handleMouseDown = this.handleMouseDown.bind(this);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.handleMouseUp = this.handleMouseUp.bind(this);
    this.handleWheel = this.handleWheel.bind(this);

    // Set up event listeners
    this.setupEvents();

    // Draw empty canvas initially
    this._drawEmpty();

    console.log(`✅ CineViewer created: ${canvasId}`);
  }

  /**
   * Set up mouse and wheel event listeners for interaction.
   */
  setupEvents() {
    this.canvas.addEventListener('mousedown', this.handleMouseDown);
    this.canvas.addEventListener('mousemove', this.handleMouseMove);
    this.canvas.addEventListener('mouseup', this.handleMouseUp);
    this.canvas.addEventListener('mouseleave', this.handleMouseUp);
    this.canvas.addEventListener('wheel', this.handleWheel);
    this.canvas.addEventListener('dragstart', e => e.preventDefault());

    // Set initial cursor
    this.canvas.style.cursor = 'grab';
  }

  /**
   * Load audio samples and prepare for display.
   *
   * @param {Float32Array} samples - Audio values (-1.0 to +1.0)
   * @param {number} sampleRate - Hz (e.g. 44100)
   */
  load(samples, sampleRate) {
    this.signal = samples;
    this.sampleRate = sampleRate;

    // Show first 5 seconds initially (or full signal if shorter)
    this.viewStart = 0;
    this.viewEnd = Math.min(samples.length, sampleRate * 5);

    this.isPlaying = false;
    this.draw();

    console.log(`📊 Signal loaded: ${(samples.length / sampleRate).toFixed(2)}s @ ${sampleRate}Hz`);
  }

  /**
   * Draw the current waveform view onto the canvas.
   * Called frequently during playback (60fps) - must be fast!
   */
  draw() {
    const { canvas, ctx, signal } = this;
    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, w, h);

    // No signal loaded? Show message
    if (!signal || signal.length === 0) {
      this._drawEmpty();
      return;
    }

    // Draw grid lines (faint)
    ctx.strokeStyle = '#2a2f3f';
    ctx.lineWidth = 0.5;

    // Vertical grid lines (time divisions)
    for (let i = 1; i < 10; i++) {
      const x = (i / 10) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Horizontal grid lines (amplitude divisions)
    for (let i = 1; i < 8; i++) {
      const y = (i / 8) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Horizontal center line (zero)
    ctx.strokeStyle = '#4ecca3';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Get visible portion of signal
    const start = Math.max(0, Math.floor(this.viewStart));
    const end = Math.min(signal.length, Math.ceil(this.viewEnd));
    const slice = signal.subarray(start, end);
    const sliceLen = slice.length;

    if (sliceLen === 0) return;

    const centerY = h / 2;
    const amp = (h / 2) * 0.85;  // Leave margin top/bottom

    // Choose line color - input cyan, output pink
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    // Optimize drawing: if many samples, downsample for performance
    if (sliceLen <= w) {
      // Fewer samples than pixels: draw each sample as a point
      const pxPerSample = w / sliceLen;
      slice.forEach((value, i) => {
        const x = i * pxPerSample;
        const y = centerY - value * amp;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
    } else {
      // More samples than pixels: use min-max method to preserve peaks
      // This prevents missing spikes when downsampling
      const samplesPerPx = sliceLen / w;

      for (let px = 0; px < w; px++) {
        const sStart = Math.floor(px * samplesPerPx);
        const sEnd = Math.floor((px + 1) * samplesPerPx);
        let minV = Infinity;
        let maxV = -Infinity;

        for (let s = sStart; s < sEnd && s < sliceLen; s++) {
          const val = slice[s];
          if (val < minV) minV = val;
          if (val > maxV) maxV = val;
        }

        const yMax = centerY - maxV * amp;
        const yMin = centerY - minV * amp;

        if (px === 0) ctx.moveTo(px, yMax);
        else ctx.lineTo(px, yMax);

        ctx.lineTo(px, yMin);  // Vertical line for min-max range
      }
    }

    ctx.stroke();

    // Draw time labels
    ctx.fillStyle = '#888';
    ctx.font = '10px "Segoe UI", monospace';
    ctx.textAlign = 'left';
    ctx.fillText((this.viewStart / this.sampleRate).toFixed(2) + 's', 5, h - 5);
    ctx.textAlign = 'right';
    ctx.fillText((this.viewEnd / this.sampleRate).toFixed(2) + 's', w - 5, h - 5);

    // Draw playback cursor - SAFELY check if AppState exists
    if (window.AppState && AppState.isPlaying && AppState.currentTime !== undefined) {
      const currentSample = AppState.currentTime * this.sampleRate;
      const windowSize = this.viewEnd - this.viewStart;

      if (currentSample >= this.viewStart && currentSample <= this.viewEnd) {
        const fraction = (currentSample - this.viewStart) / windowSize;
        const px = fraction * w;

        ctx.strokeStyle = '#ffaa00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, h);
        ctx.stroke();
        ctx.setLineDash([]);  // Reset dash
      }
    }
  }

  /**
   * Draw placeholder text when no signal is loaded.
   */
  _drawEmpty() {
    const { canvas, ctx } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#666';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('⬆️ Upload a signal to begin', w / 2, h / 2);
  }

  /**
   * Start playback animation - scrolls the window through the signal.
   */
  play() {
    if (!this.signal || this.isPlaying) return;

    this.isPlaying = true;
    if (window.AppState) AppState.isPlaying = true;

    const animate = () => {
      if (!this.isPlaying) return;

      // Calculate samples to move per frame (60fps target)
      const samplesPerFrame = Math.floor((this.sampleRate / 60) * this.speed);
      const windowSize = this.viewEnd - this.viewStart;

      // Advance the window
      this.viewStart += samplesPerFrame;
      this.viewEnd = this.viewStart + windowSize;

      // Stop at the end
      if (this.viewEnd >= this.signal.length) {
        this.stop();
        return;
      }

      this.draw();

      // Sync the other viewer
      if (this.syncTarget) {
        this.syncTarget._applySync(this.viewStart, this.viewEnd);
      }

      // Update global playback time - SAFELY check if AppState exists
      if (window.AppState) {
        AppState.currentTime = this.viewStart / this.sampleRate;
      }

      this.animFrame = requestAnimationFrame(animate);
    };

    this.animFrame = requestAnimationFrame(animate);
  }

  /**
   * Pause playback - freezes at current position.
   */
  pause() {
    this.isPlaying = false;
    if (window.AppState) AppState.isPlaying = false;
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
  }

  /**
   * Stop playback - returns to beginning.
   */
  stop() {
    this.pause();

    if (this.signal) {
      this.viewStart = 0;
      this.viewEnd = Math.min(this.signal.length, this.sampleRate * 5);
    }

    this.draw();

    if (this.syncTarget) {
      this.syncTarget._applySync(this.viewStart, this.viewEnd);
    }

    if (window.AppState) AppState.resetPlayback();
  }

  /**
   * Change playback speed.
   * @param {number} multiplier - 0.25x to 4.0x
   */
  setSpeed(multiplier) {
    this.speed = Math.max(0.25, Math.min(4.0, multiplier));
  }

  /**
   * Zoom in/out on the waveform.
   * @param {number} factor - <1 zooms in (more detail), >1 zooms out (overview)
   */
  zoom(factor) {
    if (!this.signal) return;

    const center = (this.viewStart + this.viewEnd) / 2;
    const currentSize = this.viewEnd - this.viewStart;
    const newSize = currentSize * factor;

    // Minimum zoom: 0.1 seconds
    const minSize = Math.max(100, this.sampleRate * 0.1);
    // Maximum zoom: full signal
    const maxSize = this.signal.length;

    const clampedSize = Math.max(minSize, Math.min(maxSize, newSize));

    this.viewStart = Math.max(0, Math.floor(center - clampedSize / 2));
    this.viewEnd = Math.min(this.signal.length, Math.floor(this.viewStart + clampedSize));

    // Adjust if we hit the end
    if (this.viewEnd === this.signal.length) {
      this.viewStart = Math.max(0, this.viewEnd - clampedSize);
    }

    this.draw();

    if (this.syncTarget) {
      this.syncTarget._applySync(this.viewStart, this.viewEnd);
    }
  }

  /**
   * Reset view to show first 5 seconds.
   */
  reset() {
    if (!this.signal) return;

    this.viewStart = 0;
    this.viewEnd = Math.min(this.signal.length, this.sampleRate * 5);
    this.draw();

    if (this.syncTarget) {
      this.syncTarget._applySync(this.viewStart, this.viewEnd);
    }
  }

  /**
   * Link this viewer with another so they always show the same time range.
   * @param {CineViewer} otherViewer - The viewer to sync with
   */
  syncWith(otherViewer) {
    this.syncTarget = otherViewer;
    otherViewer.syncTarget = this;
    console.log(`🔗 Viewers synced: ${this.canvasId} ↔ ${otherViewer.canvasId}`);
  }

  /**
   * Internal: receive sync update from linked viewer.
   * @param {number} viewStart - New start position
   * @param {number} viewEnd - New end position
   */
  _applySync(viewStart, viewEnd) {
    if (!this.signal) return;

    this.viewStart = Math.max(0, Math.min(viewStart, this.signal.length));
    this.viewEnd = Math.max(this.viewStart + 100, Math.min(viewEnd, this.signal.length));

    this.draw();
  }

  // ==================== MOUSE INTERACTION HANDLERS ====================

  handleMouseDown(e) {
    if (!this.signal) return;

    this.isDragging = true;
    this.dragStartX = e.offsetX;
    this.dragStartView = this.viewStart;
    this.canvas.style.cursor = 'grabbing';
  }

  handleMouseMove(e) {
    if (!this.isDragging || !this.signal) return;

    const deltaX = e.offsetX - this.dragStartX;
    const windowSize = this.viewEnd - this.viewStart;
    const deltaSamples = -(deltaX / this.canvas.width) * windowSize;

    this.viewStart = this.dragStartView + deltaSamples;
    this.viewEnd = this.viewStart + windowSize;

    // Clamp to signal bounds
    if (this.viewStart < 0) {
      this.viewStart = 0;
      this.viewEnd = windowSize;
    }
    if (this.viewEnd > this.signal.length) {
      this.viewEnd = this.signal.length;
      this.viewStart = this.viewEnd - windowSize;
    }

    this.draw();

    if (this.syncTarget) {
      this.syncTarget._applySync(this.viewStart, this.viewEnd);
    }
  }

  handleMouseUp() {
    this.isDragging = false;
    this.canvas.style.cursor = 'grab';
  }

  handleWheel(e) {
    e.preventDefault();
    if (!this.signal) return;

    // Determine zoom direction (deltaY > 0 = scroll down = zoom out)
    const delta = e.deltaY > 0 ? 1.1 : 0.9;
    this.zoom(delta);
  }
}
