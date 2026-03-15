/**
 * VAD Module - Silero Neural Voice Activity Detection with RMS Fallback
 *
 * Detects speech during TTS playback for barge-in support.
 * Uses Silero VAD v6 ONNX model for accurate speech/noise discrimination.
 * Falls back to RMS energy detection if model fails to load.
 */

import { SILERO_MODEL_URL } from './config.js';

// Silero ONNX model state
let ortSession = null;
let modelLoading = false;
let modelReady = false;

// Silero inference state
const SILERO_SR = 16000;
const SILERO_WINDOW = 512;  // 512 samples at 16kHz = 32ms
const WINDOW_SEC = SILERO_WINDOW / SILERO_SR;
let sileroState = null;     // Float32Array (2 * 1 * 128), LSTM hidden state
let sampleBuf = [];         // Accumulation buffer for resampling

// Audio nodes
let vadContext = null;
let vadSource = null;
let vadProcessor = null;
let active = false;

// State machine for speech detection
let framesAbove = 0;
let framesBelow = 0;
let speechActive = false;

// Sliding window for speech confirmation (tolerates dips between syllables)
let frameHistory = [];       // Ring buffer of booleans (above threshold?)
let frameHistoryIdx = 0;
let confirmWindow = 0;       // Window size M (frames to track)
let confirmRequired = 0;     // Required count N (frames above threshold)

// Callbacks
let onSpeechStartCb = null;
let onSpeechEndCb = null;

// Config
let threshold = 0.5;
let confirmFrames = 2;
let silenceFrames = 10;

// Inference lock: prevent overlapping async inferences
let inferenceRunning = false;
let pendingAudio = null;

/**
 * Load the Silero VAD v6 ONNX model (lazy, once).
 * Call this early (e.g. during app init) to pre-load in background.
 */
export async function loadSileroModel() {
  if (modelReady || modelLoading) return;
  if (typeof ort === 'undefined') {
    console.warn('ONNX Runtime Web not available, falling back to RMS VAD');
    return;
  }
  modelLoading = true;
  try {
    ortSession = await ort.InferenceSession.create(SILERO_MODEL_URL);
    modelReady = true;
    console.log('Silero VAD model loaded');
  } catch (e) {
    console.warn('Silero VAD model failed to load, falling back to RMS:', e);
  }
  modelLoading = false;
}

/**
 * Downsample audio from browser sample rate to target rate using linear interpolation.
 */
function downsample(samples, fromSR, toSR, targetLen) {
  const ratio = fromSR / toSR;
  const out = new Float32Array(targetLen);
  for (let i = 0; i < targetLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const frac = srcIdx - lo;
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}

/**
 * Run Silero ONNX inference on a 512-sample window at 16kHz.
 * Falls back to RMS if model is not ready.
 */
async function runInference(audio16k) {
  if (!modelReady) {
    // Fallback: compute RMS from resampled audio
    let sum = 0;
    for (let i = 0; i < audio16k.length; i++) {
      sum += audio16k[i] * audio16k[i];
    }
    const rms = Math.sqrt(sum / audio16k.length);
    // Normalize RMS to [0,1] so the same threshold works for both backends.
    // Typical speech RMS with echo cancellation: 0.01-0.05
    // Map so that 0.015 RMS ≈ 0.5 (matching the default Silero threshold)
    handleDetection(Math.min(1.0, rms / 0.03));
    return;
  }

  if (!sileroState) sileroState = new Float32Array(2 * 1 * 128);

  const inputTensor = new ort.Tensor('float32', audio16k, [1, SILERO_WINDOW]);
  const stateTensor = new ort.Tensor('float32', sileroState, [2, 1, 128]);
  const srTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(SILERO_SR)]));

  const result = await ortSession.run({
    input: inputTensor, state: stateTensor, sr: srTensor
  });

  const prob = result.output.data[0];
  sileroState = new Float32Array(result.stateN.data);
  handleDetection(prob);
}

/**
 * State machine: sliding window for speech onset, consecutive frames for silence.
 *
 * Speech confirmation uses "N out of M frames above threshold" to tolerate
 * natural probability dips between syllables (Silero output is probabilistic).
 * Silence detection stays consecutive (works fine as-is).
 */
function handleDetection(value) {
  const isAbove = value >= threshold;

  if (!speechActive) {
    // --- Speech onset: sliding window ("N out of M") ---
    // Record this frame in ring buffer
    const oldVal = frameHistory[frameHistoryIdx] ? 1 : 0;
    frameHistory[frameHistoryIdx] = isAbove;
    frameHistoryIdx = (frameHistoryIdx + 1) % confirmWindow;

    // Update running count
    framesAbove += (isAbove ? 1 : 0) - oldVal;

    if (isAbove) framesBelow = 0;
    else framesBelow++;

    if (framesAbove >= confirmRequired) {
      speechActive = true;
      framesBelow = 0;
      if (onSpeechStartCb) onSpeechStartCb();
    }
  } else {
    // --- Silence detection: consecutive frames below threshold ---
    if (isAbove) {
      framesBelow = 0;
    } else {
      framesBelow++;
      if (framesBelow >= silenceFrames) {
        speechActive = false;
        framesAbove = 0;
        framesBelow = 0;
        // Reset sliding window
        frameHistory.fill(false);
        frameHistoryIdx = 0;
        if (onSpeechEndCb) onSpeechEndCb();
      }
    }
  }
}

/**
 * Process accumulated samples: downsample and run inference.
 * Serializes inference calls to prevent overlapping async runs.
 */
async function processAccumulatedSamples() {
  if (inferenceRunning) return;
  inferenceRunning = true;

  try {
    if (!vadContext) return;
    const samplesNeeded = Math.ceil(WINDOW_SEC * vadContext.sampleRate);
    while (sampleBuf.length >= samplesNeeded) {
      const raw = sampleBuf.splice(0, samplesNeeded);
      if (!vadContext) return;
      const resampled = downsample(raw, vadContext.sampleRate, SILERO_SR, SILERO_WINDOW);
      await runInference(resampled);
    }
  } finally {
    inferenceRunning = false;
  }
}

/**
 * Start monitoring a microphone stream for voice activity.
 *
 * @param {MediaStream} micStream - The microphone MediaStream (with echoCancellation enabled)
 * @param {Object} opts
 * @param {Function} opts.onSpeechStart - Called when speech is confirmed
 * @param {Function} [opts.onSpeechEnd] - Called when silence is confirmed after speech
 * @param {number} [opts.threshold=0.5] - Detection threshold (probability for Silero, RMS for fallback)
 * @param {number} [opts.confirmFrames=2] - Consecutive frames above threshold to confirm speech
 * @param {number} [opts.silenceFrames=10] - Consecutive frames below threshold to confirm silence
 * @param {number} [opts.confirmMs] - If set, overrides confirmFrames (computed from 32ms Silero frame)
 * @param {number} [opts.silenceMs] - If set, overrides silenceFrames (computed from 32ms Silero frame)
 */
export function startVADMonitor(micStream, opts = {}) {
  if (active) stopVADMonitor();

  onSpeechStartCb = opts.onSpeechStart || null;
  onSpeechEndCb = opts.onSpeechEnd || null;
  threshold = opts.threshold ?? 0.5;

  // Reset state
  framesAbove = 0;
  framesBelow = 0;
  speechActive = false;
  sileroState = null;
  sampleBuf = [];
  inferenceRunning = false;
  pendingAudio = null;

  vadContext = new (window.AudioContext || window.webkitAudioContext)();
  vadSource = vadContext.createMediaStreamSource(micStream);

  // ScriptProcessorNode with 512-sample buffer
  vadProcessor = vadContext.createScriptProcessor(512, 1, 1);

  // Compute frame counts from ms options using 32ms Silero frame duration
  const msPerFrame = WINDOW_SEC * 1000;  // 32ms
  confirmFrames = opts.confirmMs
    ? Math.ceil(opts.confirmMs / msPerFrame)
    : (opts.confirmFrames ?? 2);
  silenceFrames = opts.silenceMs
    ? Math.ceil(opts.silenceMs / msPerFrame)
    : (opts.silenceFrames ?? 10);

  // Sliding window: require N out of M frames above threshold
  // Window M = ceil(N * 1.5), tolerates ~33% dips between syllables
  confirmRequired = confirmFrames;
  confirmWindow = Math.max(confirmRequired, Math.ceil(confirmRequired * 1.5));
  frameHistory = new Array(confirmWindow).fill(false);
  frameHistoryIdx = 0;

  vadProcessor.onaudioprocess = (e) => {
    const data = e.inputBuffer.getChannelData(0);
    // Accumulate raw samples
    for (let i = 0; i < data.length; i++) sampleBuf.push(data[i]);
    // Fire-and-forget async processing (serialized internally)
    processAccumulatedSamples();
  };

  // Connect: source -> processor -> (silent output to keep processor alive)
  const silentGain = vadContext.createGain();
  silentGain.gain.value = 0;
  vadSource.connect(vadProcessor);
  vadProcessor.connect(silentGain);
  silentGain.connect(vadContext.destination);

  active = true;
}

/**
 * Stop VAD monitoring and release audio nodes.
 */
export function stopVADMonitor() {
  if (!active) return;

  speechActive = false;
  framesAbove = 0;
  framesBelow = 0;
  frameHistory = [];
  frameHistoryIdx = 0;
  confirmWindow = 0;
  confirmRequired = 0;
  sileroState = null;
  sampleBuf = [];
  inferenceRunning = false;
  pendingAudio = null;

  try { vadProcessor?.disconnect(); } catch {}
  try { vadSource?.disconnect(); } catch {}
  try { vadContext?.close(); } catch {}

  vadProcessor = null;
  vadSource = null;
  vadContext = null;
  onSpeechStartCb = null;
  onSpeechEndCb = null;
  active = false;
}

/**
 * Check if VAD monitoring is currently active.
 * @returns {boolean}
 */
export function isVADActive() {
  return active;
}

/**
 * Check if Silero model is loaded and ready.
 * @returns {boolean}
 */
export function isSileroReady() {
  return modelReady;
}
