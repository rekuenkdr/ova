/**
 * Audio Module - Recording, playback, WebSocket ASR, streaming
 */

import { setState, setInputsEnabled, showError } from './ui.js';
import { API_CHAT_AUDIO, API_CHAT_TEXT, API_INTERRUPT, WS_ASR, TARGET_SR, CHUNK_MS, VAD_CONFIRM_FRAMES, VAD_SILENCE_FRAMES, AUTO_SEND_SILENCE_MS, AUTO_SEND_CONFIRM_MS, AUTO_SEND_TIMEOUT_MS, BARGE_IN_COOLDOWN_MS, BARGE_IN_GRACE_MS, BACKCHANNELS, DEBUG } from './config.js';
import { startVADMonitor, stopVADMonitor, loadSileroModel } from './vad.js';
import { startWakeWordDetection, stopWakeWordDetection, isWakeWordActive } from './wakeword.js';

// Audio state
let currentAudio = null;
let streamingAudioContext = null;
let isStreaming = false;
let workletNode = null;
let recording = null;
let currentAbortController = null;

// WebSocket ASR state
let asrSocket = null;
let streamingBuffer = new Float32Array(0);
let streamingMode = true;
let partialTranscript = "";
let finalTranscriptResolve = null;

// PCM streaming configuration
let pcmPrebufferSamples = 9600;

// Barge-in state
let persistentMicStream = null;   // Kept alive across recording/playback
let bargeInEnabled = false;       // Set by backend config via setBargeInEnabled()
let vadThreshold = 0.015;         // Set by backend config via setVadThreshold()
let bargeInActive = false;        // True while barge-in monitoring is active
let bargeInTriggered = false;     // Set by handleBargeIn(), read by sendAndPlayResponse

// Auto-send timing (ms-based, overridden by backend via setAutoSendTiming)
let autoSendSilenceMs = AUTO_SEND_SILENCE_MS;
let autoSendConfirmMs = AUTO_SEND_CONFIRM_MS;
let autoSendTimeoutMs = AUTO_SEND_TIMEOUT_MS;

// Barge-in cooldown: suppress barge-in after auto-send to prevent false-positive loops
let bargeInCooldownUntil = 0;              // timestamp — suppress barge-in until this time
let bargeInCooldownMs = BARGE_IN_COOLDOWN_MS;

// Auto-send state (barge-in recording phase)
let autoSendActive = false;
let autoSendTimeoutId = null;
let autoSendInProgress = false;
let streamGeneration = 0;  // Incremented per playStreamingAudio call; prevents stale cleanup

// Wake word state
let wakeWordEnabled = false;  // Set by backend config via setWakeWordEnabled()
let wakeWord = 'hey nova';    // Set by backend config via setWakeWord()

// VAD confirm frames (overridden by backend via setBargeInConfirmFrames)
let bargeInConfirmFrames = VAD_CONFIRM_FRAMES;

// Barge-in grace period (overridden by backend via setBargeInGraceMs)
let bargeInGraceMs = BARGE_IN_GRACE_MS;
let bargeInGraceTimer = null;

// Backchannel filtering
let backchannelFilterEnabled = true;
let currentLanguage = 'es';

/**
 * Set PCM prebuffer samples (called from app.js after fetching /info)
 */
export function setPcmPrebufferSamples(samples) {
  pcmPrebufferSamples = samples;
}

/**
 * Enable/disable barge-in (called from app.js after fetching /v1/info)
 */
export function setBargeInEnabled(enabled) {
  bargeInEnabled = enabled;
  if (enabled) {
    // Pre-load Silero VAD model in background (non-blocking)
    loadSileroModel();
  } else {
    releaseMicStream();
  }
}

/**
 * Set VAD threshold (called from app.js after fetching /v1/info)
 */
export function setVadThreshold(value) {
  vadThreshold = value;
}

/**
 * Set auto-send timing (called from app.js after fetching /v1/info)
 */
export function setAutoSendTiming(silenceMs, confirmMs, timeoutMs) {
  autoSendSilenceMs = silenceMs;
  autoSendConfirmMs = confirmMs;
  autoSendTimeoutMs = timeoutMs;
}

/**
 * Set barge-in cooldown duration (called from app.js after fetching /v1/info)
 */
export function setBargeInCooldownMs(ms) {
  bargeInCooldownMs = ms;
}

/**
 * Enable/disable wake word detection (called from app.js after fetching /v1/info)
 */
export function setWakeWordEnabled(enabled) {
  wakeWordEnabled = enabled;
  if (enabled) {
    loadSileroModel();
  }
}

/**
 * Set wake word phrase (called from app.js after fetching /v1/info)
 */
export function setWakeWord(word) {
  wakeWord = word;
}

/**
 * Set VAD confirm frames for barge-in detection (called from app.js after fetching /v1/info)
 */
export function setBargeInConfirmFrames(frames) {
  bargeInConfirmFrames = frames;
}

/**
 * Set barge-in grace period in ms (called from app.js after fetching /v1/info)
 */
export function setBargeInGraceMs(ms) {
  bargeInGraceMs = ms;
}

/**
 * Set current language for backchannel filtering (called from app.js after fetching /v1/info)
 */
export function setCurrentLanguage(lang) {
  currentLanguage = lang;
}

/**
 * Enable/disable backchannel filtering (called from app.js after fetching /v1/info)
 */
export function setBackchannelFilterEnabled(enabled) {
  backchannelFilterEnabled = enabled;
}

/**
 * Get current barge-in enabled state
 */
export function getBargeInEnabled() {
  return bargeInEnabled;
}

/**
 * Get current wake word enabled state
 */
export function getWakeWordEnabled() {
  return wakeWordEnabled;
}

/**
 * Start wake word listening (acquires mic eagerly, starts VAD+ASR detection).
 * Called on page load and after playback/auto-send completes.
 */
export async function enableWakeWordListening() {
  if (!wakeWordEnabled) return;
  try {
    const stream = await acquireMicStream();
    startWakeWordDetection(stream, {
      onWakeWord: () => {
        if (DEBUG) console.log('[wakeword] Wake word triggered — starting recording');
        startRecording();
      },
      threshold: vadThreshold,
      wakeWord: wakeWord,
    });
    setState('idle', { label: `Say "${titleCase(wakeWord)}"`, sub: 'or tap to talk' });
  } catch (e) {
    console.warn('Failed to start wake word listening:', e);
    setState('idle', { label: `Say "${titleCase(wakeWord)}"`, sub: 'Mic permission needed' });
  }
}

/**
 * Stop wake word listening.
 */
export function disableWakeWordListening() {
  if (isWakeWordActive()) {
    stopWakeWordDetection();
  }
}

/**
 * Get the idle state label/sub based on wake word setting.
 */
function titleCase(str) {
  return str.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function idleLabel(sub) {
  if (wakeWordEnabled) {
    return { label: `Say "${titleCase(wakeWord)}"`, sub: sub || 'or tap to talk' };
  }
  return { label: 'Tap to talk', sub: sub || 'Tap again to send' };
}

/**
 * Acquire a persistent microphone stream (reused across recording/playback).
 * First call triggers the browser permission prompt.
 * @returns {Promise<MediaStream>}
 */
async function acquireMicStream() {
  if (persistentMicStream) {
    // Check that tracks are still alive
    const tracks = persistentMicStream.getAudioTracks();
    if (tracks.length > 0 && tracks[0].readyState === 'live') {
      return persistentMicStream;
    }
    // Stream died, re-acquire
    persistentMicStream = null;
  }

  persistentMicStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });
  return persistentMicStream;
}

/**
 * Release the persistent microphone stream.
 */
function releaseMicStream() {
  if (persistentMicStream) {
    persistentMicStream.getTracks().forEach(t => t.stop());
    persistentMicStream = null;
  }
}

// Release mic on page unload
window.addEventListener('beforeunload', releaseMicStream);

/**
 * Flatten Float32 chunks into single array
 */
function flattenFloat32(chunks) {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/**
 * Linear interpolation resampling
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
 * Concatenate two Float32Arrays
 */
function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/**
 * Encode WAV from Float32 PCM
 */
function encodeWavPcm16(monoFloat32, sampleRate) {
  const numChannels = 1, bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = monoFloat32.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  let p = 0;
  const writeU32 = (v) => { view.setUint32(p, v, true); p += 4; };
  const writeU16 = (v) => { view.setUint16(p, v, true); p += 2; };
  const writeStr = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(p++, s.charCodeAt(i)); };
  writeStr("RIFF"); writeU32(36 + dataSize); writeStr("WAVE"); writeStr("fmt ");
  writeU32(16); writeU16(1); writeU16(numChannels); writeU32(sampleRate);
  writeU32(byteRate); writeU16(blockAlign); writeU16(bitsPerSample);
  writeStr("data"); writeU32(dataSize);
  for (let i = 0; i < monoFloat32.length; i++) {
    const s = Math.max(-1, Math.min(1, monoFloat32[i]));
    view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    p += 2;
  }
  return buffer;
}

/**
 * Parse WAV to Float32
 */
function parseWavToFloat32(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== "RIFF") throw new Error("Not a valid WAV file");
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataOffset = 44, dataSize = view.getUint32(40, true);
  const numSamples = dataSize / (bitsPerSample / 8);
  const float32 = new Float32Array(numSamples);
  if (bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) float32[i] = view.getInt16(dataOffset + i * 2, true) / 32768.0;
  } else if (bitsPerSample === 32) {
    for (let i = 0; i < numSamples; i++) float32[i] = view.getFloat32(dataOffset + i * 4, true);
  }
  return { pcm: float32, sampleRate };
}

/**
 * Open WebSocket connection for streaming ASR
 */
function openAsrSocket() {
  return new Promise((resolve, reject) => {
    if (asrSocket && asrSocket.readyState === WebSocket.OPEN) {
      resolve(asrSocket);
      return;
    }
    asrSocket = new WebSocket(WS_ASR);
    asrSocket.binaryType = 'arraybuffer';

    asrSocket.onopen = () => {
      if (DEBUG) console.log("ASR WebSocket connected");
      partialTranscript = "";
      resolve(asrSocket);
    };

    asrSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.partial) {
          partialTranscript = data.partial;
        } else if (data.final) {
          partialTranscript = data.final;
          if (finalTranscriptResolve) {
            finalTranscriptResolve(data.final);
            finalTranscriptResolve = null;
          }
        } else if (data.error) {
          console.error("ASR error:", data.error);
          if (finalTranscriptResolve) {
            finalTranscriptResolve("");
            finalTranscriptResolve = null;
          }
        }
      } catch (e) {
        console.error("Failed to parse ASR message:", e);
      }
    };

    asrSocket.onerror = (err) => {
      console.error("ASR WebSocket error:", err);
      reject(err);
    };

    asrSocket.onclose = () => {
      if (DEBUG) console.log("ASR WebSocket closed");
      asrSocket = null;
    };
  });
}

/**
 * Send audio chunk to ASR WebSocket
 */
function sendAsrChunk(float32Data) {
  if (asrSocket && asrSocket.readyState === WebSocket.OPEN) {
    asrSocket.send(float32Data.buffer);
  }
}

/**
 * Signal end of audio and wait for final transcript
 */
function endAsrStream() {
  return new Promise((resolve) => {
    if (!asrSocket || asrSocket.readyState !== WebSocket.OPEN) {
      resolve(partialTranscript || "");
      return;
    }

    finalTranscriptResolve = resolve;
    asrSocket.send(JSON.stringify({ action: "end" }));

    setTimeout(() => {
      if (finalTranscriptResolve) {
        finalTranscriptResolve(partialTranscript || "");
        finalTranscriptResolve = null;
      }
    }, 5000);
  });
}

/**
 * Close ASR WebSocket
 */
function closeAsrSocket() {
  if (asrSocket) {
    asrSocket.close();
    asrSocket = null;
  }
  streamingBuffer = new Float32Array(0);
  partialTranscript = "";
  finalTranscriptResolve = null;
}

/**
 * Stop streaming playback
 */
export function stopStreamingPlayback() {
  isStreaming = false;
  if (workletNode) {
    workletNode.port.postMessage({ stop: true });
    workletNode.disconnect();
    workletNode = null;
  }
  if (streamingAudioContext) {
    streamingAudioContext.close().catch(() => {});
    streamingAudioContext = null;
  }
}

/**
 * Start VAD monitoring during playback for barge-in.
 */
async function enableBargeInMonitoring(skipGrace = false) {
  if (!bargeInEnabled || bargeInActive) return;
  disableWakeWordListening();
  if (Date.now() < bargeInCooldownUntil) {
    // Cooldown active — schedule retry after it expires
    const delay = bargeInCooldownUntil - Date.now() + 50;
    setTimeout(() => {
      if (isStreaming && !bargeInActive) enableBargeInMonitoring(true);
    }, delay);
    return;
  }
  if (!persistentMicStream) {
    // No mic stream yet — barge-in will activate after first recording
    return;
  }

  // Grace period: delay VAD start to avoid echo from early TTS playback
  // Skip grace when resuming after cooldown — the cooldown already covered it
  if (!skipGrace && bargeInGraceMs > 0) {
    bargeInGraceTimer = setTimeout(() => {
      bargeInGraceTimer = null;
      if (isStreaming && !bargeInActive) startBargeInVAD();
    }, bargeInGraceMs);
  } else {
    startBargeInVAD();
  }
}

/**
 * Actually start the barge-in VAD monitor (called after grace period).
 */
async function startBargeInVAD() {
  if (bargeInActive) return;
  try {
    const stream = await acquireMicStream();
    bargeInActive = true;
    startVADMonitor(stream, {
      onSpeechStart: handleBargeIn,
      threshold: vadThreshold,
      confirmFrames: bargeInConfirmFrames,
      silenceFrames: VAD_SILENCE_FRAMES,
    });
  } catch (e) {
    console.warn('Failed to start barge-in monitoring:', e);
  }
}

/**
 * Stop VAD monitoring.
 */
function disableBargeInMonitoring() {
  if (bargeInGraceTimer) {
    clearTimeout(bargeInGraceTimer);
    bargeInGraceTimer = null;
  }
  if (!bargeInActive) return;
  stopVADMonitor();
  bargeInActive = false;
}

/**
 * Handle barge-in: stop playback, notify server, start recording with auto-send.
 */
function handleBargeIn() {
  if (DEBUG) console.log('Barge-in detected — interrupting playback');
  bargeInTriggered = true;
  disableBargeInMonitoring();

  // Synchronous: stop current playback + abort fetch
  abortCurrentRequest();
  stopStreamingPlayback();
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  // Fire-and-forget interrupt to server (speeds up TTS generator cleanup)
  fetch(API_INTERRUPT, { method: 'POST' }).catch(() => {});

  // Async follow-up: start recording + enable auto-send VAD
  handleBargeInRecordingPhase();
}

/**
 * Async barge-in follow-up: start recording, then enable auto-send VAD.
 */
async function handleBargeInRecordingPhase() {
  await startRecording();
}

/**
 * Enable VAD monitoring during barge-in recording to auto-send on silence.
 */
function enableAutoSendVAD() {
  if (!persistentMicStream) return;

  try {
    const stream = persistentMicStream;
    autoSendActive = true;
    startVADMonitor(stream, {
      onSpeechStart: () => {
        // User confirmed speaking — cancel false-positive timeout
        if (autoSendTimeoutId) {
          clearTimeout(autoSendTimeoutId);
          autoSendTimeoutId = null;
        }
      },
      onSpeechEnd: () => {
        // User stopped speaking — auto-send
        handleAutoSend();
      },
      threshold: vadThreshold,
      confirmMs: autoSendConfirmMs,
      silenceMs: autoSendSilenceMs,
    });

    // False-positive safety: cancel if no speech detected within timeout
    autoSendTimeoutId = setTimeout(() => {
      autoSendTimeoutId = null;
      if (autoSendActive && isRecording()) {
        if (DEBUG) console.log('Barge-in false positive — no speech detected, cancelling');
        cancelBargeInRecording();
      }
    }, autoSendTimeoutMs);
  } catch (e) {
    console.warn('Failed to start auto-send VAD:', e);
  }
}

/**
 * Check if text is a backchannel filler word (e.g., "sí", "ok", "mmm").
 */
function isBackchannel(text) {
  const normalized = text.toLowerCase().replace(/[^\w\sáéíóúàèìòùâêîôûäëïöüñçß]/g, '').trim();
  const list = BACKCHANNELS[currentLanguage] || BACKCHANNELS['en'] || [];
  return list.some(bc => normalized === bc);
}

/**
 * Auto-send: stop recording and send the result (mirrors manual send logic).
 */
async function handleAutoSend() {
  if (autoSendInProgress) return;
  autoSendInProgress = true;
  cleanupAutoSend();

  if (!isRecording()) {
    autoSendInProgress = false;
    return;
  }

  setState('waiting', { label: 'Processing...', sub: 'Sending audio' });
  const result = await stopRecording();

  if (!result) {
    setState('idle', idleLabel());
    setInputsEnabled(true);
    autoSendInProgress = false;
    enableWakeWordListening();
    return;
  }

  try {
    if (result.type === 'transcript') {
      if (!result.text?.trim()) {
        setState('idle', idleLabel('No speech detected'));
        setInputsEnabled(true);
        autoSendInProgress = false;
        enableWakeWordListening();
        return;
      }
      // Filter backchannel words during barge-in (e.g., "sí", "ok", "mmm")
      if (backchannelFilterEnabled && bargeInTriggered && isBackchannel(result.text.trim())) {
        if (DEBUG) console.log(`Backchannel filtered: "${result.text.trim()}"`);
        setState('idle', idleLabel());
        setInputsEnabled(true);
        autoSendInProgress = false;
        enableWakeWordListening();
        return;
      }
      // Set cooldown BEFORE send — playback will call enableBargeInMonitoring() which checks it
      // Only set cooldown when auto-send was triggered by barge-in (prevents echo loops).
      // First-tap auto-send has no echo risk, so allow immediate barge-in on the response.
      if (bargeInTriggered) {
        bargeInCooldownUntil = Date.now() + bargeInCooldownMs;
      }
      await sendTextMessage(result.text, null);
    } else {
      if (bargeInTriggered) {
        bargeInCooldownUntil = Date.now() + bargeInCooldownMs;
      }
      await fetchAndPlayTts(result);
    }
  } catch (err) {
    console.error('Auto-send error:', err);
    setState('idle', idleLabel('Request failed'));
    setInputsEnabled(true);
    enableWakeWordListening();
  }

  autoSendInProgress = false;
}

/**
 * Clean up auto-send VAD and timeout.
 */
function cleanupAutoSend() {
  if (autoSendTimeoutId) {
    clearTimeout(autoSendTimeoutId);
    autoSendTimeoutId = null;
  }
  if (autoSendActive) {
    stopVADMonitor();
    autoSendActive = false;
  }
}

/**
 * Cancel barge-in recording (false-positive: no speech detected).
 */
async function cancelBargeInRecording() {
  cleanupAutoSend();
  if (isRecording()) {
    await stopRecording(); // discard result
  }
  setState('idle', idleLabel());
  setInputsEnabled(true);
  enableWakeWordListening();
}

/**
 * Cancel auto-send (called from app.js when user manually taps during barge-in).
 */
export function cancelAutoSend() {
  cleanupAutoSend();
  autoSendInProgress = false;
}

/**
 * Play streaming audio response
 */
async function playStreamingAudio(response) {
  const myGen = ++streamGeneration;
  if (DEBUG) console.log(`[stream] playStreamingAudio started (gen=${myGen})`);
  const reader = response.body.getReader();
  isStreaming = true;
  let nextStartTime = 0, chunksPlayed = 0, totalSamples = 0, playbackStartTime = null;
  setState("playing", { label: "Playing...", sub: "Streaming audio" });
  setInputsEnabled(false);
  enableBargeInMonitoring();

  try {
    let buffer = new Uint8Array(0), streamMode = null, sampleRate = 24000;

    while (isStreaming) {
      let readResult;
      try {
        readResult = await reader.read();
      } catch (e) {
        if (e.name === 'AbortError') break;
        throw e;
      }
      const { done, value } = readResult;
      if (done) {
        if (DEBUG) console.log(`[stream] reader done, chunksPlayed=${chunksPlayed}`);
        break;
      }

      const newBuffer = new Uint8Array(buffer.length + value.length);
      newBuffer.set(buffer);
      newBuffer.set(value, buffer.length);
      buffer = newBuffer;

      if (streamMode === null && buffer.length >= 44) {
        if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) {
          const view = new DataView(buffer.buffer, buffer.byteOffset, 44);
          const dataSize = view.getUint32(40, true);
          sampleRate = view.getUint32(24, true);

          if (dataSize === 0x7FFFFFFF) {
            streamMode = 'pcm';
            buffer = buffer.slice(44);
            streamingAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
          } else {
            streamMode = 'wav';
            streamingAudioContext = new (window.AudioContext || window.webkitAudioContext)();
          }
          // Resume AudioContext if suspended (browser autoplay policy — auto-send
          // responses aren't triggered by user gesture)
          if (streamingAudioContext.state === 'suspended') {
            streamingAudioContext.resume();
          }
          if (DEBUG) console.log(`[stream] mode=${streamMode}, sr=${sampleRate}`);
          nextStartTime = streamingAudioContext.currentTime;
        }
      }

      if (streamMode === 'pcm') {
        if (!workletNode && streamingAudioContext) {
          await streamingAudioContext.audioWorklet.addModule('static/js/pcm-processor.js');
          workletNode = new AudioWorkletNode(streamingAudioContext, 'pcm-stream-processor');
          workletNode.connect(streamingAudioContext.destination);
          workletNode.port.postMessage({ config: { prebufferThreshold: pcmPrebufferSamples } });
          workletNode.port.onmessage = (e) => {
            if (e.data.event === 'playing' && playbackStartTime === null) {
              playbackStartTime = performance.now();
            }
          };
        }

        if (buffer.length >= 2 && workletNode) {
          const samples = Math.floor(buffer.length / 2);
          const int16View = new Int16Array(buffer.buffer, buffer.byteOffset, samples);
          const float32 = new Float32Array(samples);
          for (let i = 0; i < samples; i++) float32[i] = int16View[i] / 32768.0;
          workletNode.port.postMessage({ samples: float32 });
          totalSamples += samples;
          chunksPlayed++;
          buffer = new Uint8Array(0);
        }
      } else if (streamMode === 'wav') {
        while (buffer.length >= 44) {
          if (buffer[0] !== 0x52 || buffer[1] !== 0x49 || buffer[2] !== 0x46 || buffer[3] !== 0x46) {
            buffer = buffer.slice(1);
            continue;
          }
          const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
          const chunkSize = view.getUint32(4, true) + 8;
          if (buffer.length < chunkSize) break;

          const wavChunk = buffer.slice(0, chunkSize);
          buffer = buffer.slice(chunkSize);

          try {
            const { pcm, sampleRate: sr } = parseWavToFloat32(wavChunk.buffer);
            const audioBuffer = streamingAudioContext.createBuffer(1, pcm.length, sr);
            audioBuffer.getChannelData(0).set(pcm);
            const source = streamingAudioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(streamingAudioContext.destination);
            const startTime = Math.max(nextStartTime, streamingAudioContext.currentTime);
            source.start(startTime);
            nextStartTime = startTime + audioBuffer.duration;
            chunksPlayed++;
            totalSamples += pcm.length;
          } catch (e) {
            console.warn("Failed to parse WAV chunk:", e);
          }
        }
      }
    }

    if (DEBUG) console.log(`[stream] loop exited: done=true, isStreaming=${isStreaming}, chunks=${chunksPlayed}`);

    // Handle remaining data
    if (streamMode === 'pcm' && workletNode) {
      if (buffer.length >= 2) {
        const samples = Math.floor(buffer.length / 2);
        const int16View = new Int16Array(buffer.buffer, buffer.byteOffset, samples);
        const float32 = new Float32Array(samples);
        for (let i = 0; i < samples; i++) float32[i] = int16View[i] / 32768.0;
        workletNode.port.postMessage({ samples: float32 });
        totalSamples += samples;
        chunksPlayed++;
      }
      workletNode.port.postMessage({ end: true });
    }

    // Wait for playback to complete
    if (chunksPlayed > 0) {
      if (streamMode === 'pcm') {
        const waitStart = performance.now();
        while (playbackStartTime === null && performance.now() - waitStart < 2000) {
          await new Promise(r => setTimeout(r, 50));
        }
        if (playbackStartTime !== null) {
          const playbackDurationMs = (totalSamples / sampleRate) * 1000;
          const elapsedMs = performance.now() - playbackStartTime;
          const remainingMs = playbackDurationMs - elapsedMs + 100;
          if (remainingMs > 0) {
            const sleepUntil = performance.now() + remainingMs;
            while (performance.now() < sleepUntil && isStreaming) {
              await new Promise(r => setTimeout(r, 100));
            }
          }
        }
      } else if (streamingAudioContext) {
        const remainingTime = nextStartTime - streamingAudioContext.currentTime;
        if (remainingTime > 0) {
          const sleepUntil = performance.now() + remainingTime * 1000;
          while (performance.now() < sleepUntil && isStreaming) {
            await new Promise(r => setTimeout(r, 100));
          }
        }
      }
    }
  } catch (e) {
    console.error("Streaming playback error:", e);
    throw e;
  } finally {
    const isActive = myGen === streamGeneration;
    if (DEBUG) console.log(`[stream] cleanup, bargeInTriggered=${bargeInTriggered}, active=${isActive}`);
    if (isActive) {
      disableBargeInMonitoring();
      stopStreamingPlayback();
      if (!bargeInTriggered) {
        setState("idle", idleLabel());
        setInputsEnabled(true);
        enableWakeWordListening();
      }
    }
  }
}

/**
 * Play non-streaming audio blob
 */
async function playNonStreamingAudio(blob) {
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  currentAudio = audio;
  enableBargeInMonitoring();

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      disableBargeInMonitoring();
      URL.revokeObjectURL(url);
      if (currentAudio === audio) currentAudio = null;
    };
    audio.onended = () => { cleanup(); resolve(); };
    audio.onerror = () => { cleanup(); reject(new Error("Audio playback failed")); };
    audio.play().catch((e) => { cleanup(); reject(e); });
  });
}

/**
 * Abort current fetch request
 */
export function abortCurrentRequest() {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
}

/**
 * Stop current audio playback
 */
export function stopCurrentAudio() {
  abortCurrentRequest();
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  stopStreamingPlayback();
}

/**
 * Send request and play audio response
 */
export async function sendAndPlayResponse(requestFn) {
  setState("waiting", { label: "Sending...", sub: "Waiting for response" });
  setInputsEnabled(false);
  stopCurrentAudio();
  bargeInTriggered = false;

  currentAbortController = new AbortController();
  const signal = currentAbortController.signal;

  try {
    const res = await requestFn(signal);

    if (!res.ok) {
      setState("idle", idleLabel(`Error: ${res.status}`));
      setInputsEnabled(true);
      enableWakeWordListening();
      return;
    }

    const streamSupported = res.headers.get("X-Stream-Supported") === "true";
    const contentLength = res.headers.get("Content-Length");

    if (streamSupported && !contentLength) {
      try {
        await playStreamingAudio(res);
        if (!bargeInTriggered) {
          setState("idle", idleLabel());
          enableWakeWordListening();
        }
      } catch (e) {
        if (e.name === 'AbortError' || bargeInTriggered) {
          if (!bargeInTriggered && DEBUG) console.log("TTS request aborted by user");
          return;
        }
        console.error("Streaming failed:", e);
        setState("idle", idleLabel("Streaming failed"));
        enableWakeWordListening();
      }
    } else {
      const blob = await res.blob();
      if (blob.size === 0) {
        setState("idle", idleLabel("No response"));
        setInputsEnabled(true);
        enableWakeWordListening();
        return;
      }
      setState("playing", { label: "Playing...", sub: "Tap to interrupt" });
      await playNonStreamingAudio(blob);
      if (!bargeInTriggered) {
        setState("idle", idleLabel());
        enableWakeWordListening();
      }
    }
  } catch (err) {
    if (err.name === 'AbortError' || bargeInTriggered) {
      if (!bargeInTriggered && DEBUG) console.log("TTS request aborted by user");
      return;
    }
    console.error("Fetch error:", err);
    setState("idle", idleLabel("Request failed"));
    enableWakeWordListening();
  } finally {
    currentAbortController = null;
    if (!bargeInTriggered) {
      setInputsEnabled(true);
    }
  }
}

/**
 * Send WAV blob and play TTS response
 */
export async function fetchAndPlayTts(wavBlob) {
  await sendAndPlayResponse(async (signal) => {
    return fetch(API_CHAT_AUDIO, { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wavBlob, signal });
  });
}

/**
 * Send text message and play TTS response
 */
export async function sendTextMessage(text, imageData = null) {
  const payload = { text };
  if (imageData) payload.image = imageData;

  await sendAndPlayResponse(async (signal) => {
    return fetch(API_CHAT_TEXT, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
  });
}

/**
 * Check if currently recording
 */
export function isRecording() {
  return recording !== null;
}

/**
 * Start voice recording
 */
export async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setState("idle", { label: "Unsupported", sub: "Microphone not available" });
    return;
  }

  disableWakeWordListening();

  try {
    const stream = await acquireMicStream();

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0;
    const chunks = [];
    const sampleRate = audioContext.sampleRate;
    const targetChunkSamples = Math.floor(TARGET_SR * CHUNK_MS / 1000);

    // Try to open ASR WebSocket
    let asrConnected = false;
    if (streamingMode) {
      try {
        await openAsrSocket();
        asrConnected = true;
        streamingBuffer = new Float32Array(0);
      } catch (e) {
        console.warn("Failed to connect ASR WebSocket, falling back to batch mode:", e);
        asrConnected = false;
      }
    }

    processor.onaudioprocess = (e) => {
      const inputData = new Float32Array(e.inputBuffer.getChannelData(0));
      chunks.push(inputData);

      if (asrConnected && asrSocket && asrSocket.readyState === WebSocket.OPEN) {
        const resampled = resampleLinear(inputData, sampleRate, TARGET_SR);
        streamingBuffer = concatFloat32(streamingBuffer, resampled);

        while (streamingBuffer.length >= targetChunkSamples) {
          const chunk = streamingBuffer.slice(0, targetChunkSamples);
          streamingBuffer = streamingBuffer.slice(targetChunkSamples);
          sendAsrChunk(chunk);
        }
      }
    };

    source.connect(processor);
    processor.connect(zeroGain);
    zeroGain.connect(audioContext.destination);

    recording = { stream, audioContext, source, processor, zeroGain, chunks, asrConnected };
    setState("recording", { label: "Listening...", sub: "Auto-send when you pause" });
    setInputsEnabled(false);

    // Enable auto-send VAD: auto-sends when user stops talking
    enableAutoSendVAD();
  } catch (err) {
    setState("idle", idleLabel("Mic permission denied"));
  }
}

/**
 * Stop voice recording
 */
export async function stopRecording() {
  const r = recording;
  recording = null;
  if (!r) return null;

  const sampleRate = r.audioContext.sampleRate;

  try { r.processor.disconnect(); } catch {}
  try { r.zeroGain.disconnect(); } catch {}
  try { r.source.disconnect(); } catch {}
  // Do NOT stop stream tracks — keep mic alive for barge-in monitoring.
  // Mic is released on page unload or when barge-in is disabled.
  try { await r.audioContext.close(); } catch {}

  // If streaming ASR was active, get final transcript
  if (r.asrConnected && asrSocket) {
    if (streamingBuffer.length > 0) {
      sendAsrChunk(streamingBuffer);
      streamingBuffer = new Float32Array(0);
    }

    setState("waiting", { label: "Finalizing...", sub: "Processing..." });
    const finalText = await endAsrStream();
    closeAsrSocket();

    return { type: "transcript", text: finalText };
  }

  // Batch mode: return WAV blob
  const mono = flattenFloat32(r.chunks);
  const wavBuffer = encodeWavPcm16(mono, sampleRate);
  return new Blob([wavBuffer], { type: "audio/wav" });
}
