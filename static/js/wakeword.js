/**
 * Wake Word Module - VAD-gated streaming ASR for configurable wake word detection
 *
 * Coordinates Silero VAD + streaming ASR WebSocket to detect wake phrase.
 * VAD detects speech onset → opens ASR WebSocket → checks partial transcripts
 * for wake phrase match → triggers callback on match.
 */

import { startVADMonitor, stopVADMonitor } from './vad.js';
import { WS_ASR, TARGET_SR, WAKE_WORD_SILENCE_MS, DEBUG } from './config.js';

// Wake word state
let active = false;
let onWakeWordCb = null;
let micStream = null;

// Audio capture for ASR streaming
let captureContext = null;
let captureSource = null;
let captureProcessor = null;

// Pre-speech ring buffer (~500ms at 16kHz)
const PRE_BUFFER_SAMPLES = 8000;
let preBuffer = new Float32Array(PRE_BUFFER_SAMPLES);
let preBufferWriteIdx = 0;
let preBufferFull = false;

// ASR WebSocket
let wakeAsrSocket = null;
let streaming = false;
let opening = false;            // True while WebSocket is connecting
let speechEndDeferred = false;  // Speech end fired during connection phase
let pendingChunks = [];         // Audio captured during connection phase
let finishCleanupTimer = null;  // Safety timeout for finishAsrSession

// VAD config
let vadThreshold = 0.5;
let silenceMs = WAKE_WORD_SILENCE_MS;

// Wake word phrase (set dynamically from opts)
let wakePhrase = 'hey nova';

/**
 * Strip punctuation and normalize whitespace for matching.
 */
function normalize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();
}

/**
 * Check if text contains a wake phrase match.
 */
function matchesWakePhrase(text) {
  return normalize(text).includes(normalize(wakePhrase));
}

/**
 * Linear interpolation resampling (matches audio.js pattern).
 */
function resampleLinear(input, srcSr, dstSr) {
  if (srcSr === dstSr) return input;
  const ratio = dstSr / srcSr;
  const outLen = Math.round(input.length * ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const x = i / ratio;
    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, input.length - 1);
    const t = x - x0;
    out[i] = input[x0] * (1 - t) + input[x1] * t;
  }
  return out;
}

/**
 * Get the pre-speech buffer contents (resampled to TARGET_SR).
 */
function flushPreBuffer(srcSr) {
  let samples;
  if (preBufferFull) {
    // Ring buffer is full — read from writeIdx to end, then start to writeIdx
    samples = new Float32Array(PRE_BUFFER_SAMPLES);
    const first = PRE_BUFFER_SAMPLES - preBufferWriteIdx;
    samples.set(preBuffer.subarray(preBufferWriteIdx), 0);
    samples.set(preBuffer.subarray(0, preBufferWriteIdx), first);
  } else {
    samples = preBuffer.subarray(0, preBufferWriteIdx);
  }
  if (samples.length === 0) return null;
  return resampleLinear(samples, srcSr, TARGET_SR);
}

/**
 * Reset pre-speech buffer.
 */
function resetPreBuffer() {
  preBuffer = new Float32Array(PRE_BUFFER_SAMPLES);
  preBufferWriteIdx = 0;
  preBufferFull = false;
}

/**
 * Write samples to the pre-speech ring buffer.
 */
function writePreBuffer(data) {
  for (let i = 0; i < data.length; i++) {
    preBuffer[preBufferWriteIdx] = data[i];
    preBufferWriteIdx++;
    if (preBufferWriteIdx >= PRE_BUFFER_SAMPLES) {
      preBufferWriteIdx = 0;
      preBufferFull = true;
    }
  }
}

/**
 * Open a dedicated ASR WebSocket for wake word detection.
 */
function openWakeAsrSocket() {
  return new Promise((resolve, reject) => {
    if (wakeAsrSocket && wakeAsrSocket.readyState === WebSocket.OPEN) {
      resolve(wakeAsrSocket);
      return;
    }

    wakeAsrSocket = new WebSocket(`${WS_ASR}?language=en`);
    wakeAsrSocket.binaryType = 'arraybuffer';

    wakeAsrSocket.onopen = () => {
      if (DEBUG) console.log('[wakeword] ASR WebSocket connected');
      resolve(wakeAsrSocket);
    };

    wakeAsrSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const text = data.partial || data.final || '';
        if (text && DEBUG) {
          console.log(`[wakeword] ASR ${data.final !== undefined ? 'final' : 'partial'}: "${text}"`);
        }
        if (text && matchesWakePhrase(text)) {
          if (DEBUG) console.log(`[wakeword] Wake phrase matched!`);
          handleWakeWordDetected();
        }
      } catch (e) {
        console.error('[wakeword] Failed to parse ASR message:', e);
      }
    };

    wakeAsrSocket.onerror = (err) => {
      console.error('[wakeword] ASR WebSocket error:', err);
      reject(err);
    };

    wakeAsrSocket.onclose = () => {
      wakeAsrSocket = null;
      // Cancel safety timeout — connection already closed
      if (finishCleanupTimer) {
        clearTimeout(finishCleanupTimer);
        finishCleanupTimer = null;
      }
      // Reset state immediately so VAD can trigger next utterance
      streaming = false;
      opening = false;
      resetPreBuffer();
    };
  });
}

/**
 * Close the wake word ASR WebSocket and reset connection state.
 */
function closeWakeAsrSocket() {
  if (finishCleanupTimer) {
    clearTimeout(finishCleanupTimer);
    finishCleanupTimer = null;
  }
  if (wakeAsrSocket) {
    try { wakeAsrSocket.close(); } catch {}
    wakeAsrSocket = null;
  }
  streaming = false;
  opening = false;
  speechEndDeferred = false;
  pendingChunks = [];
}

/**
 * Send audio data to the wake ASR WebSocket.
 */
function sendWakeAsrChunk(float32Data) {
  if (wakeAsrSocket && wakeAsrSocket.readyState === WebSocket.OPEN) {
    wakeAsrSocket.send(float32Data.buffer);
  }
}

/**
 * Called when VAD detects speech start during wake word listening.
 * Uses connection lock to prevent race with onVADSpeechEnd.
 */
async function onVADSpeechStart() {
  if (!active || streaming || opening) return;
  opening = true;
  speechEndDeferred = false;
  pendingChunks = [];

  try {
    await openWakeAsrSocket();
    opening = false;
    streaming = true;

    // Flush pre-speech buffer + any audio captured during connection
    if (captureContext) {
      const preSpeech = flushPreBuffer(captureContext.sampleRate);
      if (preSpeech) sendWakeAsrChunk(preSpeech);
    }
    for (const chunk of pendingChunks) sendWakeAsrChunk(chunk);
    pendingChunks = [];

    // If speech ended while we were connecting, handle it now
    if (speechEndDeferred) {
      speechEndDeferred = false;
      finishAsrSession();
    }
  } catch (e) {
    console.warn('[wakeword] Failed to open ASR socket on speech start:', e);
    opening = false;
    streaming = false;
    pendingChunks = [];
    closeWakeAsrSocket();
  }
}

/**
 * Called when VAD detects speech end during wake word listening.
 * Defers if WebSocket is still connecting; no VAD restart needed.
 */
function onVADSpeechEnd() {
  if (!active) return;

  if (opening) {
    // Socket still connecting — defer until connection completes
    speechEndDeferred = true;
    return;
  }

  if (!streaming) return;
  finishAsrSession();
}

/**
 * Finish an ASR session: send end signal, wait for final transcript, then clean up.
 * Does NOT restart VAD — it naturally transitions back to "waiting for speech."
 */
function finishAsrSession() {
  if (wakeAsrSocket && wakeAsrSocket.readyState === WebSocket.OPEN) {
    wakeAsrSocket.send(JSON.stringify({ action: 'end' }));
    // Safety timeout — normally onclose handles cleanup first
    finishCleanupTimer = setTimeout(() => {
      finishCleanupTimer = null;
      closeWakeAsrSocket();
      resetPreBuffer();
    }, 3000);
  } else {
    closeWakeAsrSocket();
    resetPreBuffer();
  }
}

/**
 * Called when wake phrase is matched in ASR transcript.
 */
function handleWakeWordDetected() {
  if (!active) return;
  const cb = onWakeWordCb;
  stopWakeWordDetection();
  if (cb) cb();
}

/**
 * Start VAD monitoring for wake word (with speech start/end callbacks).
 */
function startVADForWakeWord() {
  if (!micStream || !active) return;

  startVADMonitor(micStream, {
    onSpeechStart: onVADSpeechStart,
    onSpeechEnd: onVADSpeechEnd,
    threshold: vadThreshold,
    silenceMs: silenceMs,
    confirmMs: 64,
  });
}

/**
 * Start audio capture pipeline for ASR streaming.
 */
function startAudioCapture(stream) {
  captureContext = new (window.AudioContext || window.webkitAudioContext)();
  captureSource = captureContext.createMediaStreamSource(stream);
  captureProcessor = captureContext.createScriptProcessor(4096, 1, 1);

  const silentGain = captureContext.createGain();
  silentGain.gain.value = 0;

  const srcSr = captureContext.sampleRate;

  captureProcessor.onaudioprocess = (e) => {
    const inputData = new Float32Array(e.inputBuffer.getChannelData(0));

    if (opening) {
      // WebSocket connecting — queue for later flush
      pendingChunks.push(resampleLinear(inputData, srcSr, TARGET_SR));
    } else if (streaming && wakeAsrSocket && wakeAsrSocket.readyState === WebSocket.OPEN) {
      // During speech: resample and send to ASR
      const resampled = resampleLinear(inputData, srcSr, TARGET_SR);
      sendWakeAsrChunk(resampled);
    } else {
      // During silence: buffer for pre-speech context
      writePreBuffer(inputData);
    }
  };

  captureSource.connect(captureProcessor);
  captureProcessor.connect(silentGain);
  silentGain.connect(captureContext.destination);
}

/**
 * Stop audio capture pipeline.
 */
function stopAudioCapture() {
  try { captureProcessor?.disconnect(); } catch {}
  try { captureSource?.disconnect(); } catch {}
  try { captureContext?.close(); } catch {}
  captureProcessor = null;
  captureSource = null;
  captureContext = null;
}

/**
 * Start wake word detection.
 *
 * @param {MediaStream} stream - Microphone stream
 * @param {Object} opts
 * @param {Function} opts.onWakeWord - Called when wake phrase is detected
 * @param {number} [opts.threshold=0.5] - VAD threshold
 * @param {number} [opts.silenceMs=600] - Silence duration to end utterance
 */
export function startWakeWordDetection(stream, opts = {}) {
  if (active) stopWakeWordDetection();

  micStream = stream;
  onWakeWordCb = opts.onWakeWord || null;
  vadThreshold = opts.threshold ?? 0.5;
  silenceMs = opts.silenceMs ?? WAKE_WORD_SILENCE_MS;
  wakePhrase = opts.wakeWord || 'hey nova';
  active = true;

  resetPreBuffer();
  startAudioCapture(stream);
  startVADForWakeWord();

  if (DEBUG) console.log('[wakeword] Wake word detection started');
}

/**
 * Stop wake word detection.
 */
export function stopWakeWordDetection() {
  if (!active) return;
  active = false;

  stopVADMonitor();
  closeWakeAsrSocket();
  stopAudioCapture();
  resetPreBuffer();

  micStream = null;
  onWakeWordCb = null;

  if (DEBUG) console.log('[wakeword] Wake word detection stopped');
}

/**
 * Check if wake word detection is active.
 */
export function isWakeWordActive() {
  return active;
}
