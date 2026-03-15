class PCMStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.playing = false;
    this.streamEnded = false;
    this.preBufferThreshold = 14400; // Default 0.6s at 24kHz, can be updated via config message
    this.underrunCount = 0;
    this.frameCount = 0;

    this.port.onmessage = (e) => {
      if (e.data.config) {
        // Update prebuffer threshold from backend config
        if (e.data.config.prebufferThreshold) {
          this.preBufferThreshold = e.data.config.prebufferThreshold;
        }
      } else if (e.data.samples) {
        // Create new buffer (proper copy, not view)
        const newBuffer = new Float32Array(this.buffer.length + e.data.samples.length);
        newBuffer.set(this.buffer);
        newBuffer.set(e.data.samples, this.buffer.length);
        this.buffer = newBuffer;
      } else if (e.data.end) {
        this.streamEnded = true;
      } else if (e.data.stop) {
        this.buffer = new Float32Array(0);
        this.playing = false;
        this.streamEnded = false;
        this.underrunCount = 0;
        this.frameCount = 0;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    const needed = output.length; // 128 samples per frame
    this.frameCount++;

    // Start playing once we have enough buffered, or stream ended
    if (!this.playing) {
      if (this.buffer.length >= this.preBufferThreshold || this.streamEnded) {
        this.playing = true;
        this.port.postMessage({ event: 'playing', buffered: this.buffer.length });
      } else {
        // Still pre-buffering - output silence
        output.fill(0);
        return true;
      }
    }

    if (this.buffer.length >= needed) {
      // Use slice() not subarray() to create proper copy
      output.set(this.buffer.slice(0, needed));
      this.buffer = this.buffer.slice(needed);
    } else if (this.buffer.length > 0) {
      // Underrun - use what we have, fill rest with silence
      this.underrunCount++;
      this.port.postMessage({
        event: 'underrun',
        frame: this.frameCount,
        had: this.buffer.length,
        needed: needed,
        total: this.underrunCount
      });
      output.set(this.buffer);
      output.fill(0, this.buffer.length);
      this.buffer = new Float32Array(0);
    } else {
      output.fill(0);
      // Signal finished when stream ended and buffer empty
      if (this.streamEnded && this.playing) {
        this.port.postMessage({ event: 'finished', frame: this.frameCount });
        this.playing = false;
      }
    }

    // Report buffer level every 100 frames (~0.27s at 24kHz)
    if (this.frameCount % 100 === 0 && this.playing) {
      this.port.postMessage({
        event: 'status',
        buffer: this.buffer.length,
        frame: this.frameCount
      });
    }

    return true;
  }
}

registerProcessor('pcm-stream-processor', PCMStreamProcessor);
