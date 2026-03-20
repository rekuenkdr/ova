# Changelog

## 2026-03-20

### Breaking Changes

- **Renamed `OVA_QWEN3_STREAM_FORMAT` to `OVA_STREAM_FORMAT`** — streaming format is now engine-agnostic. Update your `.env` if you had this set.

### Added

- **Full-duplex conversation mode** — persistent WebSocket session with server-side VAD, turn-taking state machine, backchannel detection, and interrupt handling. Enable with `OVA_ENABLE_DUPLEX=true`. Nine new config variables for tuning VAD thresholds, silence timeouts, and interrupt cooldowns.
- **Server-side VAD** (`ova/server_vad.py`) — Silero VAD via ONNX Runtime with sliding-window onset detection and consecutive-silence offset detection.
- **Turn-taking engine** (`ova/turn_taking.py`) — event-driven state machine (IDLE/USER_SPEAKING/BOT_THINKING/BOT_SPEAKING) with per-language backchannel phrase sets.
- **Kokoro TTS streaming** — Kokoro engine now supports chunked PCM streaming, previously only available for Qwen3.
- **Conversation mode selector** in settings UI (half-duplex / full-duplex).
- **Error tone** for invalid settings combinations.

### Fixed

- **Settings not persisted on hot-reload** — voice and language changes via hot-switch were not written to `.env`, reverting on restart.
- **Ellipsis/dash TTS artifacts** — `...` and `--` in LLM output now converted to commas before TTS. Prompt instructions updated across all 10 languages.
- **Numba logging noise** suppressed.
