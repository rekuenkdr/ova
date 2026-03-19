# Environment Variables Reference

Complete reference for all OVA environment variables. For getting started, see [QUICKSTART.md](QUICKSTART.md).

> Variables in `.env` override code defaults.

---

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_BACKEND_HOST` | `localhost` | FastAPI/Uvicorn bind address |
| `OVA_BACKEND_PORT` | `5173` | FastAPI/Uvicorn port |
| `OVA_FRONTEND_HOST` | `localhost` | Static file server bind address |
| `OVA_FRONTEND_PORT` | `8080` | Static file server port |
| `OVA_DEBUG` | `false` | Enable DEBUG-level logging for all components |
| `OVA_PROFILE` | `default` | Profile name for multi-config setups *(shell-only, read by `ova.sh`)* |
| `OVA_CUDA_VERSION` | *(auto-detect)* | Override CUDA version detection in `ova.sh`. Set to skip `nvidia-smi` auto-detect *(shell-only)* |
| `UVICORN_RELOAD` | *(empty)* | Set to any non-empty value to enable uvicorn `--reload` *(shell-only)* |

## Language

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_LANGUAGE` | `es` | Language for TTS, ASR, and system prompts. Hot-reloadable via API |

## TTS

Shared settings that apply to both TTS engines.

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_TTS_ENGINE` | `qwen3` | TTS backend: `qwen3` (voice cloning, streaming) or `kokoro` (predefined voices). Requires restart |
| `OVA_EARLY_TTS_DECODE` | `true` | Start TTS on first sentence/~40 chars before full LLM response. Reduces TTFB by ~200-300ms |
| `OVA_MAX_TTS_FRAMES` | `8000` | Hard cap on TTS generation frames. Prevents runaway generation when EOS is missed (8000 frames ~ 11 min at 12Hz) |
| `OVA_MAX_PAUSE_DURATION` | `3.0` | Max seconds for `[Pause: X]` prosody tags. Longer values are clamped |

## Qwen3-TTS

All variables in this section apply only when `OVA_TTS_ENGINE=qwen3`.

### Model & Voice

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID for Qwen3-TTS |
| `OVA_QWEN3_VOICE` | `myvoice` | Voice profile directory name under `profiles/<language>/` |

### Streaming

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_QWEN3_STREAM_FORMAT` | `pcm` | Streaming format: `pcm` (WAV header + raw chunks, lower latency) or `wav` (each chunk is a complete WAV). Requires restart |
| `OVA_PCM_EMIT_EVERY_FRAMES` | `8` | Phase 2 (steady-state) emit interval. Lower = more frequent, smaller chunks |
| `OVA_PCM_DECODE_WINDOW` | `80` | Phase 2 decode window size. Use 64 for 0.6B model, 80 for 1.7B |
| `OVA_PCM_PREBUFFER_SAMPLES` | `9600` | Samples to buffer before playback starts (9600 = 0.4s at 24kHz) |

### Two-Phase Streaming

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_FIRST_CHUNK_EMIT_EVERY` | `5` | Phase 1 (first chunk) emit interval. Set to 0 to disable two-phase streaming |
| `OVA_FIRST_CHUNK_DECODE_WINDOW` | `48` | Phase 1 decode window. Increase to 64-72 if voice sounds inconsistent in first 1-2s |
| `OVA_FIRST_CHUNK_FRAMES` | `48` | Number of frames before switching from phase 1 to phase 2 |

### Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_STREAMING_OPTIMIZATIONS` | `true` | Enable `torch.compile` for TTS. Mode auto-selected by stream format |
| `OVA_CODEBOOK_CUDA_GRAPH` | `false` | Capture codebook predictor as CUDA graph. ~2x codebook speedup, +500MB VRAM. **Requires [`wip/experimental`](https://github.com/rekuenkdr/Qwen3-TTS-streaming/tree/wip/experimental)** |
| `OVA_USE_PAGED_ENGINE` | `false` | Paged attention engine with explicit KV cache + CUDA graphs. Alternative to torch.compile. Mutually exclusive with `OVA_CODEBOOK_CUDA_GRAPH`. Requires flash-attn, triton, xxhash. **Requires [`wip/experimental`](https://github.com/rekuenkdr/Qwen3-TTS-streaming/tree/wip/experimental)** |
| `OVA_PAGED_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction for paged engine KV cache (0.0-1.0). Only used when `OVA_USE_PAGED_ENGINE=true`. **Requires [`wip/experimental`](https://github.com/rekuenkdr/Qwen3-TTS-streaming/tree/wip/experimental)** |

## Kokoro-TTS

All variables in this section apply only when `OVA_TTS_ENGINE=kokoro`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_KOKORO_MODEL` | `hexgrad/Kokoro-82M` | HuggingFace model ID for Kokoro TTS |
| `OVA_KOKORO_VOICE` | `af_heart` | Kokoro voice preset name (no cloning, predefined voices only) |

## LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_LLM_PROVIDER` | `ollama` | LLM backend: `ollama` (local Ollama) or `openai` (any OpenAI-compatible API). Requires restart |
| `OVA_CHAT_MODEL` | `ministral-3:3b-instruct-2512-q4_K_M` | Model name passed to the LLM provider |
| `OVA_LLM_BASE_URL` | `http://localhost:8000/v1` | Base URL for OpenAI-compatible provider (ignored when provider is `ollama`) |
| `OVA_LLM_API_KEY` | `not-needed` | API key for OpenAI-compatible provider. Use `not-needed` for local servers without auth |
| `OVA_LLM_MAX_TOKENS` | `0` | Max LLM output tokens. 0 = unlimited. Set to 300-512 for smaller TTS models |

## ASR

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace model ID for Qwen3-ASR (runs in isolated subprocess) |
| `OVA_ASR_GPU_MEMORY_UTILIZATION` | `0.4` | GPU memory fraction reserved for ASR via vLLM (0.0-1.0) |
| `OVA_ASR_MAX_MODEL_LEN` | `2048` | Max sequence length for ASR model |
| `OVA_ASR_CHUNK_SIZE_SEC` | `0.5` | Audio chunk duration in seconds (0.4-0.8s). Larger = better accuracy, higher latency |
| `OVA_ASR_UNFIXED_CHUNK_NUM` | `4` | First N chunks use no prior context (bootstrap phase) |
| `OVA_ASR_UNFIXED_TOKEN_NUM` | `5` | Tokens rolled back at chunk boundaries for transcription stability |

## Barge-In

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_BARGE_IN` | `true` | Allow user to interrupt TTS playback by speaking. Uses client-side Silero VAD |
| `OVA_VAD_THRESHOLD` | `0.5` | Silero speech probability threshold (0.0-1.0). Lower = more sensitive |
| `OVA_AUTO_SEND_SILENCE_MS` | `1200` | Silence duration (ms) to trigger auto-send during barge-in recording |
| `OVA_AUTO_SEND_CONFIRM_MS` | `128` | Speech confirmation duration (ms) before auto-send becomes armed (~4 Silero frames) |
| `OVA_AUTO_SEND_TIMEOUT_MS` | `3000` | Cancel barge-in recording if no speech detected within this time (ms) |
| `OVA_BARGE_IN_COOLDOWN_MS` | `1200` | Suppress barge-in for N ms after auto-send fires. Prevents false-positive loops |
| `OVA_VAD_CONFIRM_FRAMES` | `2` | Consecutive Silero frames (~32ms each) needed to confirm speech onset |
| `OVA_BARGE_IN_GRACE_MS` | `500` | Delay VAD activation after playback starts (ms). Prevents echo-triggered false barge-in |
| `OVA_BACKCHANNEL_FILTER` | `false` | Discard filler words (yeah, ok, mmm) during barge-in instead of sending to LLM |

## Wake Word

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_WAKE_WORD` | `false` | Enable always-on wake word listening when idle. Requires `OVA_ENABLE_BARGE_IN=true` |
| `OVA_WAKE_WORD` | `hey nova` | Wake word phrase. Use real English words for reliable ASR recognition |

## Tools

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_TOOLS` | `false` | Master toggle for LLM tool/function calling |
| `OVA_MAX_TOOL_ITERATIONS` | `5` | Max tool-call round-trips per request. Prevents runaway loops |
| `OVA_DISABLED_TOOLS` | *(empty)* | Comma-separated list of tool function names to disable |
| `OVA_TOOL_<NAME>_ENABLED` | *(per-module)* | Per-tool override. e.g. `OVA_TOOL_WEB_SEARCH_ENABLED=true`. Takes priority over `OVA_DISABLED_TOOLS` |
| `OVA_TIMEZONE` | `UTC` | Timezone for `get_current_datetime` tool. IANA format (e.g. `Europe/Madrid`) |
| `OVA_SEARCH_API_KEY` | *(empty)* | Tavily API key for `web_search` tool |
| `OVA_SEARCH_MAX_RESULTS` | `5` | Max results returned by `web_search` tool (capped at 20) |

## MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_ENABLE_MCP` | `false` | Connect to external MCP servers for additional tool capabilities |
| `OVA_MCP_CONFIG` | `mcp_servers.json` | Path to MCP server config file (relative to project root or absolute) |
| `OVA_MCP_CONNECT_TIMEOUT` | `10` | Per-server connection timeout in seconds |
| `OVA_MCP_TOOL_TIMEOUT` | `30` | Per-tool call timeout in seconds |

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `OVA_API_KEY` | *(empty)* | API key for authentication. When set, all requests need `Authorization: Bearer <key>`. Empty = disabled |
| `OVA_MAX_TEXT_LENGTH` | `4096` | Max user input characters per request |
| `OVA_MAX_IMAGE_SIZE` | `20971520` | Max image upload size in bytes (20 MB) |
| `OVA_MAX_AUDIO_SIZE` | `20971520` | Max audio upload size in bytes (20 MB) |
| `OVA_SUPPORTED_TTS_ENGINES` | `qwen3,kokoro` | Comma-separated whitelist of accepted TTS engine values via settings API |
| `OVA_SUPPORTED_LANGUAGES` | `es,en,fr,de,it,pt,ja,zh,ko,ru,hi` | Comma-separated whitelist of accepted language values via settings API |
| `OVA_SUPPORTED_STREAM_FORMATS` | `pcm,wav` | Comma-separated whitelist of accepted stream format values via settings API |
| `OVA_DISABLE_RESTART_ENDPOINT` | `false` | Block `POST /v1/restart`. Recommended for production (DoS risk) |
| `OVA_DISABLE_FRONTEND_SETTINGS` | `false` | Hide settings UI button and block `POST /v1/settings` |
| `OVA_DISABLE_MULTIMODAL` | `false` | Remove text/image input section, block `POST /v1/chat` (voice-only mode) |
| `OVA_DISABLE_FRONTEND_ACCESS` | `false` | API-only mode: skip static file server on port 8080 *(shell-only, read by `ova.sh`)* |

## Third-Party

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_OFFLINE` | *(not set)* | HuggingFace Hub offline mode. Set to `1` to prevent network calls and use cached models only. Default HF behavior when unset is online |
