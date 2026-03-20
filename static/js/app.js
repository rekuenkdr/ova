/**
 * App Module - Main entry point and configuration
 */

import { initTheme, toggleTheme } from './theme.js';
import { initUI, setState, setInputsEnabled, getElements, handleImageSelect, clearImage, updateSendButton, autoResizeTextInput, getTextInputValue, clearTextInput, getAttachedImage, disableMultimodal } from './ui.js';
import { initSettings, loadSettings, disableSettings } from './settings.js';
import {
  startRecording,
  stopRecording,
  isRecording,
  sendTextMessage,
  fetchAndPlayTts,
  sendAndPlayResponse,
  stopCurrentAudio,
  stopStreamingPlayback,
  setPcmPrebufferSamples,
  setBargeInEnabled,
  setVadThreshold,
  cancelAutoSend,
  setAutoSendTiming,
  setBargeInCooldownMs,
  setBargeInConfirmFrames,
  setBargeInGraceMs,
  setCurrentLanguage,
  setBackchannelFilterEnabled,
  setWakeWordEnabled,
  setWakeWord,
  enableWakeWordListening
} from './audio.js';
import { API_INFO, DEBUG, setDebug } from './config.js';
import { connectEventStream, onEvent } from './events.js';
import { showNotification, playAlarmTone, requestPermission } from './notifications.js';
import { initDuplex, connectDuplex, isDuplexConnected, interruptDuplex, sendDuplexImage, sendDuplexText } from './duplex.js';

// Track duplex state
let duplexEnabled = false;
let duplexLanguage = null;
let duplexVoice = null;

/**
 * Fetch PCM streaming config from backend
 */
async function fetchBackendConfig() {
  try {
    const res = await fetch(API_INFO);
    if (res.ok) {
      const info = await res.json();
      if (info.debug) setDebug(true);
      if (info.pcm_prebuffer_samples) {
        setPcmPrebufferSamples(info.pcm_prebuffer_samples);
        console.log(`PCM prebuffer: ${info.pcm_prebuffer_samples} samples`);
      }
      if (info.frontend_settings_disabled) {
        disableSettings();
        if (DEBUG) console.log('Frontend settings disabled by server');
      }
      if (info.multimodal_disabled) {
        disableMultimodal();
        if (DEBUG) console.log('Multimodal input disabled by server');
      }
      if (info.barge_in_enabled !== undefined) {
        setBargeInEnabled(info.barge_in_enabled);
        console.log(`Barge-in: ${info.barge_in_enabled ? 'enabled' : 'disabled'}`);
      }
      if (info.vad_threshold !== undefined) {
        setVadThreshold(info.vad_threshold);
        console.log(`VAD threshold: ${info.vad_threshold}`);
      }
      if (info.auto_send_silence_ms !== undefined) {
        setAutoSendTiming(
          info.auto_send_silence_ms,
          info.auto_send_confirm_ms,
          info.auto_send_timeout_ms
        );
        if (DEBUG) console.log(`Auto-send: silence=${info.auto_send_silence_ms}ms, confirm=${info.auto_send_confirm_ms}ms, timeout=${info.auto_send_timeout_ms}ms`);
      }
      if (info.barge_in_cooldown_ms !== undefined) {
        setBargeInCooldownMs(info.barge_in_cooldown_ms);
        if (DEBUG) console.log(`Barge-in cooldown: ${info.barge_in_cooldown_ms}ms`);
      }
      if (info.vad_confirm_frames !== undefined) {
        setBargeInConfirmFrames(info.vad_confirm_frames);
        console.log(`VAD confirm frames: ${info.vad_confirm_frames}`);
      }
      if (info.barge_in_grace_ms !== undefined) {
        setBargeInGraceMs(info.barge_in_grace_ms);
        console.log(`Barge-in grace period: ${info.barge_in_grace_ms}ms`);
      }
      if (info.language) {
        setCurrentLanguage(info.language);
        console.log(`Language: ${info.language}`);
      }
      if (info.backchannel_filter_enabled !== undefined) {
        setBackchannelFilterEnabled(info.backchannel_filter_enabled);
        console.log(`Backchannel filter: ${info.backchannel_filter_enabled ? 'enabled' : 'disabled'}`);
      }
      if (info.wake_word_enabled !== undefined) {
        setWakeWordEnabled(info.wake_word_enabled);
        console.log(`Wake word: ${info.wake_word_enabled ? 'enabled' : 'disabled'}`);
      }
      if (info.wake_word) {
        setWakeWord(info.wake_word);
        console.log(`Wake word phrase: "${info.wake_word}"`);
      }
      if (info.duplex_enabled) {
        duplexEnabled = true;
        duplexLanguage = info.language;
        duplexVoice = info.voice;
        console.log('Full-duplex mode: enabled');
      }
    }
  } catch (e) {
    console.warn("Failed to fetch backend config, using defaults:", e);
  }
}

/**
 * Setup voice button handlers
 */
function setupVoiceButton() {
  const { talkBtn } = getElements();
  if (!talkBtn) return;

  talkBtn.addEventListener('click', async () => {
    // Ignore clicks during waiting state
    if (talkBtn.classList.contains('state-waiting')) return;

    // If playing, stop playback (duplex: interrupt server too)
    if (talkBtn.classList.contains('state-playing')) {
      if (isDuplexConnected()) {
        interruptDuplex();
        return;
      }
      stopCurrentAudio();
      setState('idle', { label: 'Tap to talk', sub: 'Tap again to send' });
      setInputsEnabled(true);
      enableWakeWordListening();
      return;
    }

    // If not recording, start recording
    if (!isRecording()) {
      await startRecording();
      return;
    }

    // Stop recording and process
    cancelAutoSend();  // Clean up auto-send VAD if active (no-op if not)
    setState('waiting', { label: 'Processing...', sub: 'Sending audio' });
    const result = await stopRecording();

    if (!result) {
      setState('idle', { label: 'Tap to talk', sub: 'Tap again to send' });
      setInputsEnabled(true);
      return;
    }

    try {
      if (result.type === 'transcript') {
        // Streaming ASR mode
        const text = result.text;
        if (!text || !text.trim()) {
          setState('idle', { label: 'Tap to talk', sub: 'No speech detected' });
          setInputsEnabled(true);
          return;
        }
        const image = getAttachedImage();
        clearImage();
        await sendTextMessage(text, image);
      } else {
        // Batch mode - send WAV blob
        await fetchAndPlayTts(result);
      }
    } catch (err) {
      console.error("Request error:", err);
      setState('idle', { label: 'Tap to talk', sub: 'Request failed' });
      setInputsEnabled(true);
    }
  });
}

/**
 * Setup text input handlers
 */
function setupTextInput() {
  const { textInput, sendBtn, imageBtn, imageInput, removeImageBtn } = getElements();

  // Image button
  imageBtn?.addEventListener('click', () => imageInput?.click());

  // Image input change
  imageInput?.addEventListener('change', (e) => {
    if (e.target.files[0]) handleImageSelect(e.target.files[0]);
  });

  // Remove image button
  removeImageBtn?.addEventListener('click', clearImage);

  // Text input changes
  textInput?.addEventListener('input', () => {
    updateSendButton();
    autoResizeTextInput();
  });

  // Enter to send
  textInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn?.disabled) handleSendText();
    }
  });

  // Send button
  sendBtn?.addEventListener('click', handleSendText);
}

/**
 * Handle sending text message
 */
async function handleSendText() {
  const text = getTextInputValue();
  const image = getAttachedImage();

  if (!text && !image) return;

  clearTextInput();
  clearImage();

  // In duplex mode, send text + image over WebSocket
  if (isDuplexConnected()) {
    if (text || image) sendDuplexText(text, image);
    return;
  }

  await sendTextMessage(text, image);
}

/**
 * Setup drag and drop for images
 */
function setupDragDrop() {
  document.body.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  document.body.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith('image/')) {
      handleImageSelect(file);
    }
  });

  // Paste handler
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          handleImageSelect(file);
          break;
        }
      }
    }
  });
}

/**
 * Setup theme toggle button
 */
function setupThemeToggle() {
  const themeBtn = document.getElementById('themeBtn');
  themeBtn?.addEventListener('click', toggleTheme);
}

/**
 * Initialize application
 */
async function init() {
  // Initialize theme first (affects CSS variables)
  initTheme();

  // Initialize UI
  initUI();

  // Setup handlers
  setupVoiceButton();
  setupTextInput();
  setupDragDrop();
  setupThemeToggle();

  // Fetch backend config (must be before initSettings to check if settings disabled)
  await fetchBackendConfig();

  // Initialize settings (skipped if disabled by backend)
  initSettings();

  // Connect to server event stream (SSE) for real-time notifications
  onEvent("timer_expired", (event) => {
    playAlarmTone();
    showNotification({
      title: event.data.label.replace(/\b\w/g, c => c.toUpperCase()),
      message: "Timer finished",
    });
  });
  connectEventStream();

  // Request OS notification permission (no-ops if already granted/denied)
  requestPermission();

  // Start duplex mode or half-duplex wake word listening
  if (duplexEnabled) {
    await initDuplex();
    connectDuplex({ language: duplexLanguage, voice: duplexVoice });
  } else {
    // Start wake word listening if enabled (eagerly acquires mic)
    enableWakeWordListening();
  }

  console.log('OVA Voice Assistant initialized');
}

// Initialize when page is fully loaded (including stylesheets)
if (document.readyState === 'complete') {
  init();
} else {
  window.addEventListener('load', init);
}
