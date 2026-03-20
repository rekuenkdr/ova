/**
 * Duplex Module - Full-duplex WebSocket voice client
 *
 * Opens a persistent WebSocket to /v1/duplex, sends continuous mic audio
 * as PCM int16 16kHz, receives TTS PCM int16 24kHz + JSON status messages.
 *
 * Much simpler than the half-duplex audio.js coordination — no client-side
 * VAD, no recording state machine, no ASR WebSocket management.
 */

import { WS_DUPLEX, TARGET_SR, DEBUG } from './config.js';
import { setState, getState } from './ui.js';

// ---- State ----
let ws = null;
let micStream = null;
let audioContext = null;
let micProcessor = null;
let micSource = null;

// Playback (AudioWorklet)
let playbackContext = null;
let workletNode = null;
let playbackReady = false;

let connected = false;
let sessionActive = false;
let reconnectTimer = null;
let acceptingAudio = false;  // true only while bot is speaking

// Config from session.started
let ttsSampleRate = 24000;

// Resample state
const RESAMPLE_TARGET = TARGET_SR;  // 16000

/**
 * Initialize duplex client. Call once after fetchBackendConfig confirms
 * duplex_enabled === true.
 */
export async function initDuplex() {
  // Acquire persistent mic with echo cancellation
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    console.error('Duplex: mic permission denied', e);
    setState('idle', { label: 'Mic denied', sub: 'Check browser permissions' });
    return;
  }

  // Setup playback AudioContext + worklet
  playbackContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 48000,
  });

  try {
    await playbackContext.audioWorklet.addModule('static/js/pcm-processor.js');
    playbackReady = true;
  } catch (e) {
    console.error('Duplex: AudioWorklet failed to load', e);
  }

  console.log('Duplex client initialized');
}

/**
 * Connect the duplex WebSocket and start streaming audio.
 *
 * @param {Object} [opts] - Optional config
 * @param {string} [opts.language] - Language code
 * @param {string} [opts.voice] - Voice name
 */
export function connectDuplex(opts = {}) {
  if (connected) return;
  clearTimeout(reconnectTimer);

  const params = new URLSearchParams();
  if (opts.language) params.set('language', opts.language);
  if (opts.voice) params.set('voice', opts.voice);
  const qs = params.toString();
  const url = qs ? `${WS_DUPLEX}?${qs}` : WS_DUPLEX;

  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    connected = true;
    sessionActive = true;
    console.log('Duplex: WebSocket connected');
    startMicCapture();
    setState('idle', { label: 'Listening...', sub: 'Speak to start' });
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      handleAudioChunk(event.data);
    } else {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.warn('Duplex: invalid JSON', e);
      }
    }
  };

  ws.onclose = (event) => {
    connected = false;
    sessionActive = false;
    stopMicCapture();
    console.log(`Duplex: WebSocket closed (code=${event.code})`);

    // Auto-reconnect on unexpected close (not clean close)
    if (event.code !== 1000 && event.code !== 4000) {
      reconnectTimer = setTimeout(() => connectDuplex(opts), 3000);
    } else {
      setState('idle', { label: 'Tap to talk', sub: 'Session ended' });
    }
  };

  ws.onerror = (event) => {
    console.error('Duplex: WebSocket error', event);
  };
}

/**
 * Disconnect the duplex WebSocket cleanly.
 */
export function disconnectDuplex() {
  clearTimeout(reconnectTimer);
  sessionActive = false;

  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify({ type: 'session.end' }));
    } catch {}
  }

  stopMicCapture();
  stopPlayback();

  if (ws) {
    ws.close(1000);
    ws = null;
  }
  connected = false;
}

/**
 * Send a runtime config update to the server.
 */
export function sendConfig(config) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'session.config', ...config }));
  }
}

/**
 * Queue an image for the next voice turn.
 */
export function sendDuplexImage(imageData) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'session.image', image: imageData }));
  }
}

/**
 * Send text (and optional image) for processing as if the user spoke it.
 */
export function sendDuplexText(text, image) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    const msg = { type: 'session.text', text };
    if (image) msg.image = image;
    ws.send(JSON.stringify(msg));
  }
}

/**
 * Interrupt bot speech — stop local playback and notify server.
 */
export function interruptDuplex() {
  acceptingAudio = false;
  stopPlayback();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'session.interrupt' }));
  }
}

/**
 * Check if duplex is currently connected.
 */
export function isDuplexConnected() {
  return connected && sessionActive;
}

// ---- Mic capture & resampling ----

function startMicCapture() {
  if (!micStream || micProcessor) return;

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  micSource = audioContext.createMediaStreamSource(micStream);

  // ScriptProcessor for capturing + resampling
  micProcessor = audioContext.createScriptProcessor(4096, 1, 1);
  micProcessor.onaudioprocess = (e) => {
    if (!connected) return;
    const input = e.inputBuffer.getChannelData(0);
    const resampled = resample(input, audioContext.sampleRate, RESAMPLE_TARGET);
    // Convert float32 to int16
    const int16 = float32ToInt16(resampled);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(int16.buffer);
    }
  };

  // Connect: source → processor → silent output (keeps processor alive)
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  micSource.connect(micProcessor);
  micProcessor.connect(silentGain);
  silentGain.connect(audioContext.destination);
}

function stopMicCapture() {
  try { micProcessor?.disconnect(); } catch {}
  try { micSource?.disconnect(); } catch {}
  try { audioContext?.close(); } catch {}
  micProcessor = null;
  micSource = null;
  audioContext = null;
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.round(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = srcIdx - lo;
    out[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return out;
}

function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

// ---- TTS playback ----

function handleAudioChunk(arrayBuffer) {
  if (!playbackReady || !playbackContext || !acceptingAudio) return;

  // Received PCM int16 at TTS sample rate — convert to float32
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }

  // Create or reuse worklet node
  if (!workletNode) {
    workletNode = new AudioWorkletNode(playbackContext, 'pcm-stream-processor');
    workletNode.connect(playbackContext.destination);
    workletNode.port.postMessage({ config: { prebufferThreshold: 9600 } });
    workletNode.port.onmessage = (e) => {
      if (e.data?.event === 'playing') {
        if (DEBUG) console.log('Duplex: playback started');
      }
    };
  }

  // Resample from TTS sample rate to playback context sample rate if needed
  let samples = float32;
  if (ttsSampleRate !== playbackContext.sampleRate) {
    samples = resample(float32, ttsSampleRate, playbackContext.sampleRate);
  }

  // Feed samples to worklet (same format as audio.js)
  workletNode.port.postMessage({ samples: samples });
}

function stopPlayback() {
  if (workletNode) {
    try {
      workletNode.port.postMessage({ stop: true });
      workletNode.disconnect();
    } catch {}
    workletNode = null;
  }
}

// ---- JSON message handling ----

function handleJsonMessage(msg) {
  if (DEBUG) console.log('Duplex msg:', msg);

  switch (msg.type) {
    case 'session.started':
      ttsSampleRate = msg.sample_rate || 24000;
      console.log(`Duplex: session started (TTS SR=${ttsSampleRate})`);
      break;

    case 'session.ended':
      sessionActive = false;
      acceptingAudio = false;
      stopPlayback();
      setState('idle', { label: 'Tap to talk', sub: 'Session ended' });
      break;

    case 'vad':
      if (msg.speech) {
        // If bot is playing audio and user starts talking, stop playback immediately
        if (getState() === 'playing') {
          acceptingAudio = false;
          stopPlayback();
        }
        setState('recording', { label: 'Listening...', sub: '' });
      }
      break;

    case 'transcript':
      if (msg.is_final) {
        setState('waiting', { label: 'Processing...', sub: '' });
      }
      break;

    case 'bot.thinking':
      setState('waiting', { label: 'Processing...', sub: '' });
      break;

    case 'bot.speaking':
      acceptingAudio = true;
      setState('playing', { label: 'Playing...', sub: 'Tap to interrupt' });
      break;

    case 'bot.idle':
      acceptingAudio = false;
      setState('idle', { label: 'Listening...', sub: 'Speak naturally' });
      break;

    case 'bot.interrupted':
      acceptingAudio = false;
      stopPlayback();
      setState('recording', { label: 'Listening...', sub: '' });
      break;

    case 'error':
      console.error('Duplex server error:', msg.message);
      setState('idle', { label: 'Listening...', sub: msg.message || 'Something went wrong' });
      break;
  }
}
