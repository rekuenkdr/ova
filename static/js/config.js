/**
 * Config Module - Shared configuration values
 *
 * NOTE: These URLs must match OVA_BACKEND_HOST/PORT in .env
 * If you change the backend host/port, update these values.
 */

// API Configuration
export const API_BASE = "http://localhost:5173";
export const API_CHAT_AUDIO = `${API_BASE}/v1/chat/audio`;
export const API_CHAT_TEXT = `${API_BASE}/v1/chat`;
export const API_TTS = `${API_BASE}/v1/text-to-speech`;
export const API_INFO = `${API_BASE}/v1/info`;
export const API_SETTINGS = `${API_BASE}/v1/settings`;
export const API_SETTINGS_PROMPT = `${API_BASE}/v1/settings/prompt`;
export const API_HEALTH = `${API_BASE}/v1/health`;
export const API_RESTART = `${API_BASE}/v1/restart`;
export const API_INTERRUPT = `${API_BASE}/v1/interrupt`;
export const API_EVENTS = `${API_BASE}/v1/events`;

// WebSocket ASR Configuration
export const WS_ASR = `ws://localhost:5173/v1/speech-to-text/stream`;
export const WS_DUPLEX = `ws://localhost:5173/v1/duplex`;
export const TARGET_SR = 16000;
export const CHUNK_MS = 500;

// Silero VAD v6 model (neural voice activity detection, downloaded by ova.sh install)
export const SILERO_MODEL_URL = '/models/silero_vad_16k_op15.onnx';

// VAD defaults (overridden by /v1/info)
export const VAD_CONFIRM_FRAMES = 4;    // ~128ms to confirm speech (Silero frames are 32ms each)
export const VAD_SILENCE_FRAMES = 10;   // ~320ms to confirm silence

// Auto-send VAD (barge-in recording phase) — ms-based, overridden by /v1/info
export const AUTO_SEND_SILENCE_MS = 1000;   // Silence duration to trigger auto-send (ms)
export const AUTO_SEND_CONFIRM_MS = 64;     // Speech duration to confirm user is talking (ms)
export const AUTO_SEND_TIMEOUT_MS = 3000;   // Cancel recording if no speech in 3s (false positive)

// Barge-in cooldown — suppress barge-in after auto-send to prevent false-positive loops
export const BARGE_IN_COOLDOWN_MS = 2000;   // ms to suppress barge-in after auto-send fires

// Barge-in grace period — delay VAD start after playback begins to avoid echo false positives
export const BARGE_IN_GRACE_MS = 500;

// Backchannel words (per language) — filtered during barge-in to avoid sending filler to the LLM
export const BACKCHANNELS = {
  es: ["si", "sí", "vale", "ajá", "aja", "mmm", "ok", "okay", "claro", "ya", "bien", "ah", "oh"],
  en: ["yeah", "yes", "ok", "okay", "uh huh", "hmm", "right", "sure", "mhm", "mm", "ah", "oh", "got it", "i see"],
  fr: ["oui", "ouais", "mmm", "d'accord", "ok", "okay", "ah", "oh", "bien", "bon"],
  de: ["ja", "ok", "okay", "mmm", "ah", "oh", "genau", "richtig", "gut", "stimmt"],
};

// Wake word detection (overridden by /v1/info)
export const WAKE_WORD_SILENCE_MS = 600;  // Shorter silence timeout for wake word ASR

// Debug mode (set from backend /v1/info, defaults to false)
export let DEBUG = false;
export function setDebug(val) { DEBUG = val; }
