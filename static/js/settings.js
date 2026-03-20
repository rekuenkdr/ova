/**
 * Settings Module - Settings panel logic and API calls
 */

import { API_SETTINGS, API_SETTINGS_PROMPT, API_HEALTH, API_RESTART } from './config.js';
import { applyTheme, getTheme, THEMES } from './theme.js';
import {
  setState,
  setInputsEnabled,
  showRestartProgress,
  hideRestartProgress,
  completeRestartProgress,
  showSuccessFlash
} from './ui.js';
import {
  setBargeInEnabled, setWakeWordEnabled,
  getBargeInEnabled, getWakeWordEnabled,
  enableWakeWordListening, disableWakeWordListening
} from './audio.js';
import { sendConfig, isDuplexConnected } from './duplex.js';
import { showNotification, playErrorTone } from './notifications.js';

const API_SETTINGS_URL = API_SETTINGS;

// Kokoro voices by language — loaded from backend via GET /settings
let kokoroVoices = {};

// Per-engine language lists — loaded from backend via GET /settings
let qwen3Languages = [];
let kokoroLanguages = [];

// State
let currentSettings = {};
let pendingSettings = {};
let availableProfiles = {};
let defaultPrompts = {};  // Default prompts by language (for Kokoro)
let settingsElements = {};

// Track if settings are disabled
let settingsDisabled = false;

/**
 * Disable settings module - removes button and panel from DOM
 */
export function disableSettings() {
  settingsDisabled = true;

  // Remove settings button from DOM
  const settingsBtn = document.getElementById('settingsBtn');
  if (settingsBtn) {
    settingsBtn.remove();
  }

  // Remove settings panel and overlay from DOM
  const settingsOverlay = document.getElementById('settingsOverlay');
  if (settingsOverlay) {
    settingsOverlay.remove();
  }
}

/**
 * Initialize settings module
 */
export function initSettings() {
  // Skip initialization if settings are disabled
  if (settingsDisabled) {
    return;
  }

  settingsElements = {
    settingsBtn: document.getElementById('settingsBtn'),
    settingsOverlay: document.getElementById('settingsOverlay'),
    settingsPanel: document.getElementById('settingsPanel'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
    applySettingsBtn: document.getElementById('applySettingsBtn'),
    restartNotice: document.getElementById('restartNotice'),
    languageSelect: document.getElementById('languageSelect'),
    ttsEngineSelect: document.getElementById('ttsEngineSelect'),
    profileSelect: document.getElementById('profileSelect'),
    streamModeSelect: document.getElementById('streamModeSelect'),
    systemPromptText: document.getElementById('systemPromptText'),
    clearHistoryCheckbox: document.getElementById('clearHistoryCheckbox'),
    kokoroVoiceSelect: document.getElementById('kokoroVoiceSelect'),
    qwen3Settings: document.getElementById('qwen3Settings'),
    kokoroSettings: document.getElementById('kokoroSettings'),
    conversationModeSelect: document.getElementById('conversationModeSelect'),
    bargeInCheckbox: document.getElementById('bargeInCheckbox'),
    wakeWordCheckbox: document.getElementById('wakeWordCheckbox'),
  };

  setupEventListeners();
  loadSettings();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  const {
    settingsBtn,
    settingsOverlay,
    closeSettingsBtn,
    cancelSettingsBtn,
    applySettingsBtn,
    languageSelect,
    ttsEngineSelect,
    profileSelect,
    streamModeSelect,
    kokoroVoiceSelect,
    systemPromptText,
  } = settingsElements;

  settingsBtn?.addEventListener('click', openSettings);
  settingsOverlay?.addEventListener('click', closeSettings);
  closeSettingsBtn?.addEventListener('click', closeSettings);
  cancelSettingsBtn?.addEventListener('click', closeSettings);
  applySettingsBtn?.addEventListener('click', applySettings);

  languageSelect?.addEventListener('change', () => {
    // When language changes, clear voice so we pick from new language
    currentSettings.voice = null;
    updateProfileOptions();
    updateKokoroVoiceOptions();
    collectPendingSettings();
    // Load prompt for the new language
    const { profileSelect, systemPromptText, ttsEngineSelect } = settingsElements;
    const lang = languageSelect?.value;
    const engine = ttsEngineSelect?.value;
    if (engine === "kokoro") {
      // Kokoro: load default prompt for language
      const defaultPrompt = defaultPrompts[lang];
      if (defaultPrompt && systemPromptText) {
        systemPromptText.value = defaultPrompt;
      }
    } else {
      // Qwen3: load profile's prompt
      const profile = profileSelect?.value;
      const profileData = availableProfiles[lang]?.[profile];
      if (profileData?.prompt && systemPromptText) {
        systemPromptText.value = profileData.prompt;
      }
    }
  });

  ttsEngineSelect?.addEventListener('change', () => {
    const engine = ttsEngineSelect?.value;
    updateEngineVisibility();
    updateLanguageAvailability();
    updateProfileOptions();
    updateKokoroVoiceOptions();
    collectPendingSettings();
    // Update prompt based on new engine
    const { languageSelect, profileSelect, systemPromptText } = settingsElements;
    const lang = languageSelect?.value;
    if (engine === "kokoro") {
      // Kokoro: load default prompt for language
      const defaultPrompt = defaultPrompts[lang];
      if (defaultPrompt && systemPromptText) {
        systemPromptText.value = defaultPrompt;
      }
    } else {
      // Qwen3: load profile's prompt
      const profile = profileSelect?.value;
      const profileData = availableProfiles[lang]?.[profile];
      if (profileData?.prompt && systemPromptText) {
        systemPromptText.value = profileData.prompt;
      }
    }
  });

  profileSelect?.addEventListener('change', () => {
    collectPendingSettings();
    // Load selected profile's prompt
    const lang = languageSelect?.value;
    const profile = profileSelect?.value;
    const profileData = availableProfiles[lang]?.[profile];
    if (profileData?.prompt && systemPromptText) {
      systemPromptText.value = profileData.prompt;
    }
  });

  streamModeSelect?.addEventListener('change', collectPendingSettings);
  kokoroVoiceSelect?.addEventListener('change', collectPendingSettings);
  settingsElements.conversationModeSelect?.addEventListener('change', collectPendingSettings);

  // Save prompt on blur
  systemPromptText?.addEventListener('blur', savePromptOnBlur);

  // Theme toggle buttons
  document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const theme = btn.dataset.theme;
      if (theme && THEMES.includes(theme)) {
        applyTheme(theme);
      }
    });
  });
}

/**
 * Load settings from server
 */
export async function loadSettings() {
  try {
    const res = await fetch(API_SETTINGS_URL, {
      cache: 'no-store',
    });
    if (!res.ok) throw new Error("Failed to load settings");

    const data = await res.json();
    currentSettings = data.current;
    availableProfiles = data.profiles || {};
    defaultPrompts = data.default_prompts || {};
    kokoroVoices = data.kokoro_voices || {};
    qwen3Languages = data.qwen3_languages || [];
    kokoroLanguages = data.kokoro_languages || [];
    pendingSettings = { ...currentSettings };
    updateSettingsUI();
  } catch (e) {
    console.error("Failed to load settings:", e);
  }
}

/**
 * Update settings UI with current values
 */
function updateSettingsUI() {
  const {
    languageSelect,
    ttsEngineSelect,
    streamModeSelect,
    systemPromptText,
  } = settingsElements;

  if (ttsEngineSelect) ttsEngineSelect.value = currentSettings.tts_engine || "qwen3";
  if (streamModeSelect) streamModeSelect.value = currentSettings.stream_format || "pcm";

  const { conversationModeSelect } = settingsElements;
  if (conversationModeSelect) conversationModeSelect.value = currentSettings.conversation_mode || "half-duplex";
  if (systemPromptText) systemPromptText.value = currentSettings.system_prompt || "";

  updateEngineVisibility();
  updateLanguageAvailability();
  // Set language AFTER options are populated dynamically
  if (languageSelect) languageSelect.value = currentSettings.language || "es";
  updateProfileOptions();
  updateKokoroVoiceOptions();
  pendingSettings = { ...currentSettings };
  updateRestartNotice();

  // Sync barge-in / wake word checkboxes with current audio module state
  const { bargeInCheckbox, wakeWordCheckbox } = settingsElements;
  if (bargeInCheckbox) bargeInCheckbox.checked = getBargeInEnabled();
  if (wakeWordCheckbox) wakeWordCheckbox.checked = getWakeWordEnabled();
}

/**
 * Dynamically populate language dropdown based on selected TTS engine
 */
function updateLanguageAvailability() {
  const { languageSelect, ttsEngineSelect } = settingsElements;
  if (!languageSelect) return;

  const engine = ttsEngineSelect?.value;
  const languages = engine === 'qwen3' ? [...qwen3Languages] : kokoroLanguages;
  const currentLang = languageSelect.value;

  // For Qwen3, sort so languages with profiles appear first
  if (engine === 'qwen3') {
    languages.sort((a, b) => {
      const aHas = availableProfiles[a.code] && Object.keys(availableProfiles[a.code]).length > 0;
      const bHas = availableProfiles[b.code] && Object.keys(availableProfiles[b.code]).length > 0;
      if (aHas === bHas) return 0;
      return aHas ? -1 : 1;
    });
  }

  // Rebuild options
  languageSelect.innerHTML = '';
  for (const lang of languages) {
    const option = document.createElement('option');
    option.value = lang.code;
    option.textContent = lang.name;
    // For Qwen3, disable languages without voice profiles on disk
    if (engine === 'qwen3') {
      const hasProfiles = availableProfiles[lang.code] && Object.keys(availableProfiles[lang.code]).length > 0;
      option.disabled = !hasProfiles;
    }
    languageSelect.appendChild(option);
  }

  // Restore selection if still valid, otherwise pick first enabled
  if (languages.some(l => l.code === currentLang)) {
    languageSelect.value = currentLang;
  } else {
    const firstEnabled = languages.find(l => {
      if (engine === 'qwen3') {
        return availableProfiles[l.code] && Object.keys(availableProfiles[l.code]).length > 0;
      }
      return true;
    });
    if (firstEnabled) languageSelect.value = firstEnabled.code;
  }
}

/**
 * Update profile dropdown options
 */
function updateProfileOptions() {
  const { languageSelect, profileSelect } = settingsElements;
  const lang = languageSelect?.value;
  const profilesData = availableProfiles[lang] || {};
  const profileNames = Object.keys(profilesData).sort();

  if (!profileSelect) return;

  // Store current selection before clearing
  const previousValue = profileSelect.value;

  profileSelect.innerHTML = "";

  if (profileNames.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No profiles available";
    profileSelect.appendChild(opt);
    profileSelect.disabled = true;
  } else {
    profileSelect.disabled = false;
    profileNames.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
      profileSelect.appendChild(opt);
    });

    // Set the dropdown value - prefer currentSettings, fallback to previous
    const targetValue = currentSettings.voice || previousValue;
    if (targetValue && profileNames.includes(targetValue)) {
      profileSelect.value = targetValue;
    }
  }
}

/**
 * Update Kokoro voice dropdown options
 */
function updateKokoroVoiceOptions() {
  const { languageSelect, kokoroVoiceSelect } = settingsElements;
  const lang = languageSelect?.value;
  const voices = kokoroVoices[lang] || [];

  if (!kokoroVoiceSelect) return;

  kokoroVoiceSelect.innerHTML = "";

  if (voices.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No voices for this language";
    kokoroVoiceSelect.appendChild(opt);
    kokoroVoiceSelect.disabled = true;
  } else {
    kokoroVoiceSelect.disabled = false;
    voices.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.name;
      kokoroVoiceSelect.appendChild(opt);
    });

    if (voices.some(v => v.id === currentSettings.voice)) {
      kokoroVoiceSelect.value = currentSettings.voice;
    }
  }
}

/**
 * Show/hide engine-specific settings
 */
function updateEngineVisibility() {
  const { ttsEngineSelect, qwen3Settings, kokoroSettings } = settingsElements;
  const engine = ttsEngineSelect?.value;

  if (qwen3Settings) qwen3Settings.style.display = engine === "qwen3" ? "block" : "none";
  if (kokoroSettings) kokoroSettings.style.display = engine === "kokoro" ? "block" : "none";
}

/**
 * Update restart notice visibility
 */
function updateRestartNotice() {
  const { restartNotice } = settingsElements;
  // Only TTS engine and stream format changes require restart
  // Language and voice changes are hot-reloadable
  const needsRestart =
    pendingSettings.tts_engine !== currentSettings.tts_engine ||
    pendingSettings.stream_format !== currentSettings.stream_format ||
    pendingSettings.conversation_mode !== currentSettings.conversation_mode;

  restartNotice?.classList.toggle("show", needsRestart);
}

/**
 * Collect pending settings from form
 */
function collectPendingSettings() {
  const {
    languageSelect,
    ttsEngineSelect,
    profileSelect,
    streamModeSelect,
    kokoroVoiceSelect,
  } = settingsElements;

  const engine = ttsEngineSelect?.value;
  const voiceValue = profileSelect?.value;

  // Only include engine-specific settings
  pendingSettings = {
    language: languageSelect?.value,
    tts_engine: engine,
    stream_format: streamModeSelect?.value,
    conversation_mode: settingsElements.conversationModeSelect?.value,
  };

  // Reactively hide barge-in/wake-word when full-duplex is selected
  const duplexPending = settingsElements.conversationModeSelect?.value === 'full-duplex';
  settingsElements.bargeInCheckbox?.closest('.setting-row')?.style.setProperty('display', duplexPending ? 'none' : '');
  settingsElements.wakeWordCheckbox?.closest('.setting-row')?.style.setProperty('display', duplexPending ? 'none' : '');

  if (engine === "qwen3") {
    pendingSettings.voice = voiceValue;
  } else if (engine === "kokoro") {
    pendingSettings.voice = kokoroVoiceSelect?.value;
  }

  updateRestartNotice();
}

/**
 * Apply settings
 */
async function applySettings() {
  const { applySettingsBtn, systemPromptText, clearHistoryCheckbox } = settingsElements;

  collectPendingSettings();

  applySettingsBtn?.classList.add("loading");
  if (applySettingsBtn) applySettingsBtn.disabled = true;

  try {
    const res = await fetch(API_SETTINGS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pendingSettings),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to apply settings");
    }

    const data = await res.json();

    if (data.error) {
      showNotification({ title: "Incompatible settings detected", message: data.error });
      playErrorTone();
      return;
    }

    if (data.restart_required) {
      closeSettings();
      showRestartProgress();
      setState("waiting", { label: "Restarting...", sub: "Please wait" });

      // Trigger server restart
      let restartDisabled = false;
      try {
        const restartRes = await fetch(API_RESTART, { method: "POST" });
        if (restartRes.ok) {
          const restartData = await restartRes.json();
          if (!restartData.success && restartData.error?.includes("disabled")) {
            restartDisabled = true;
          }
        }
      } catch {
        // Server may close connection before responding - that's expected
      }

      if (restartDisabled) {
        // Restart endpoint is disabled - notify user to restart manually
        hideRestartProgress();
        setState("restart", {
          label: "Restart required",
          sub: "Run: ./ova.sh restart",
        });
        return;
      }

      // Wait for server to come back up
      await waitForServer();

      completeRestartProgress();

      // Duplex mode changes require full reload (no teardown/re-init path)
      if (pendingSettings.conversation_mode !== currentSettings.conversation_mode) {
        window.location.reload();
        return;
      }

      await loadSettings();
      setState("idle", { label: "Tap to talk", sub: "Tap again to send" });
    } else {
      // Apply prompt from textarea (and/or clear history)
      const newPrompt = systemPromptText?.value.trim();
      const shouldClearHistory = clearHistoryCheckbox?.checked || false;

      // Always call reload-prompt if there's a prompt OR if clearing history
      if (newPrompt || shouldClearHistory) {
        await fetch(API_SETTINGS_PROMPT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: newPrompt || undefined,
            clear_history: shouldClearHistory,
          }),
        });
        if (newPrompt) {
          currentSettings.system_prompt = newPrompt;
        }
        // Reset checkbox - clearing context is a one-time action
        if (clearHistoryCheckbox) {
          clearHistoryCheckbox.checked = false;
        }
      }

      // Apply barge-in / wake word toggles (session-only, no backend call)
      const { bargeInCheckbox, wakeWordCheckbox } = settingsElements;
      const bargeInChecked = bargeInCheckbox?.checked ?? getBargeInEnabled();
      const wakeWordChecked = wakeWordCheckbox?.checked ?? getWakeWordEnabled();

      setBargeInEnabled(bargeInChecked);
      setWakeWordEnabled(wakeWordChecked);

      if (wakeWordChecked) {
        enableWakeWordListening();
      } else {
        disableWakeWordListening();
        setState('idle', { label: 'Tap to talk', sub: 'Tap again to send' });
      }

      currentSettings = { ...currentSettings, ...pendingSettings };

      // Notify active duplex session of voice/language changes
      if (isDuplexConnected()) {
        const config = {};
        if (pendingSettings.language) config.language = pendingSettings.language;
        if (pendingSettings.voice) config.voice = pendingSettings.voice;
        if (Object.keys(config).length) sendConfig(config);
      }

      updateRestartNotice();
      showSuccessFlash();
      closeSettings();
    }
  } catch (e) {
    console.error("Failed to apply settings:", e);
    alert("Failed to apply settings: " + e.message);
  } finally {
    applySettingsBtn?.classList.remove("loading");
    if (applySettingsBtn) applySettingsBtn.disabled = false;
  }
}

/**
 * Save prompt on blur
 */
async function savePromptOnBlur() {
  const { systemPromptText, clearHistoryCheckbox } = settingsElements;
  const newPrompt = systemPromptText?.value.trim();

  if (!newPrompt || newPrompt === currentSettings.system_prompt) return;

  try {
    const res = await fetch(API_SETTINGS_PROMPT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: newPrompt,
        clear_history: clearHistoryCheckbox?.checked || false,
      }),
    });

    if (res.ok) {
      currentSettings.system_prompt = newPrompt;
      showSuccessFlash();
    }
  } catch (e) {
    console.error("Failed to update prompt:", e);
  }
}

/**
 * Wait for server to restart
 */
async function waitForServer(maxAttempts = 90) {
  // Wait for server to be fully ready (all warmups complete)
  // Uses /v1/health endpoint which returns 503 until initialization is done
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const res = await fetch(API_HEALTH, { method: "GET" });
      if (res.ok) return;
      // 503 = still warming up, keep waiting
    } catch {
      // Connection refused = server not up yet, keep waiting
    }
  }
  throw new Error("Server did not restart in time");
}

/**
 * Open settings panel
 */
export async function openSettings() {
  const { settingsOverlay, settingsPanel } = settingsElements;
  settingsOverlay?.classList.add("open");
  settingsPanel?.classList.add("open");
  await loadSettings();

  // Hide barge-in and wake word toggles in duplex mode (irrelevant — mic always streaming)
  const duplex = isDuplexConnected();
  settingsElements.bargeInCheckbox?.closest('.setting-row')?.style.setProperty('display', duplex ? 'none' : '');
  settingsElements.wakeWordCheckbox?.closest('.setting-row')?.style.setProperty('display', duplex ? 'none' : '');
}

/**
 * Close settings panel
 */
export function closeSettings() {
  const { settingsOverlay, settingsPanel, bargeInCheckbox, wakeWordCheckbox } = settingsElements;
  settingsOverlay?.classList.remove("open");
  settingsPanel?.classList.remove("open");
  pendingSettings = { ...currentSettings };
  updateRestartNotice();

  // Reset checkboxes to actual state (cancel reverts visual changes)
  if (bargeInCheckbox) bargeInCheckbox.checked = getBargeInEnabled();
  if (wakeWordCheckbox) wakeWordCheckbox.checked = getWakeWordEnabled();
}
