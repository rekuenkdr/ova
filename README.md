# OVA

|![OVA](static/img/01-Hover.png)|![Settings](static/img/05-Settings.png)|
|:--:|:--:|

A **fully-local** AI voice assistant with real-time streaming TTS, voice cloning, and multimodal (image + text) support. Built with a FastAPI backend and modern web frontend. All models (ASR / LLM / TTS) run locally with open weights - no data is sent to the Internet.

## Features

- **[Real-time PCM streaming TTS](#streaming-tts)** - Low-latency audio with Web Audio API AudioWorklet
- **[Streaming ASR](#api-endpoints)** - Real-time transcription via WebSocket as you speak
- **[Voice cloning](#qwen3-voice-profiles)** - Clone any voice from a 5-15 second audio sample
- **[Two-phase streaming](#streaming-tts)** - Aggressive first chunk for lower TTFB, then stable quality
- **[Early TTS decode](#why-the-first-sentence-matters-for-ttfb)** - Interleaved LLM→TTS for faster time-to-first-byte (~200-300ms reduction)
- **[Hot-reload](#hot-reload-settings)** - Switch voice/language without restart
- **Multi-language support** - 10 languages: zh, en, ja, ko, de, fr, ru, pt, es, it
- **[Multimodal input](#multimodal-image--text)** - Attach images to chat queries for vision-language responses
- **[Barge-in](#barge-in--wake-word)** - Interrupt TTS playback by speaking (Silero VAD speech detection)
- **[Wake word](#barge-in--wake-word)** - Always-on "Hey Nova" detection with VAD-gated streaming ASR
- **[Prosody control](#prosody-silence-tags)** - `[pause:X]` tags for deliberate silences in speech
- **[Tool calling](#tools--function-calling)** - LLM can invoke real Python functions (timers, datetime, web search) with real-time push notifications via SSE
- **[MCP client](#mcp-external-tool-servers)** - Connect to external [MCP](https://modelcontextprotocol.io/) servers for additional tool capabilities (filesystem, databases, APIs) without writing code
- **[Themes](static/README.md#theming-system)** - Dark, Light, Her (Samantha), and HAL-9000
- **[torch.compile optimizations](#streaming-tts)** - Up to 1.7x speedup after JIT warmup

## Quick Start

### Pre-requisites

- Linux (x86_64)
- Python >= 3.13
- `uv` installed and available in PATH
- **LLM provider** (one of):
  - [Ollama](https://ollama.com/) installed and running (default)
  - Any OpenAI-compatible API — local or remote (`OVA_LLM_PROVIDER=openai`)
- NVIDIA GPU (Ampere or newer) with CUDA 12 or 13. Default is CUDA 13 — run `./ova.sh configure-cuda 12` before install if on CUDA 12

### Install & Run

```bash
# If your CUDA version is not 13 (the default), configure first:
./ova.sh configure-cuda    # auto-detect, or: ./ova.sh configure-cuda 12

./ova.sh install
```

See [QUICKSTART.md](QUICKSTART.md) for voice profile setup and `.env` configuration.

```bash
./ova.sh start
```

This starts two services:
- **Backend** (FastAPI): http://localhost:5173 — ASR + LLM + TTS pipeline
- **Frontend** (static): http://localhost:8080 — open this in your browser

Logs: `tail -f .ova/backend.log` (add `OVA_DEBUG=true` to `.env` for verbose output).

### Minimal Configuration

The defaults .env. examples will work for most setups. The common things to change in `.env`:

| Variable | Default | What to change |
|----------|---------|----------------|
| `OVA_LANGUAGE` | `es` | Your language (`en`, `de`, `fr`, `ja`, etc.) |
| `OVA_QWEN3_VOICE` | `myvoice` | Voice profile (see [generate_voice_prompts script](https://github.com/rekuenkdr/ova/tree/main/scripts#generate_voice_promptspy) ) |
| `OVA_LLM_PROVIDER` | `ollama` | Set to `openai` for OpenAI-compatible APIs |

> See [VARIABLES.md](VARIABLES.md) for the full configuration reference.

## Models

| Component | Model |
|-----------|-------|
| ASR | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (embedded subprocess) |
| LLM | [Mistral ministral-3 3b 4-bit](https://ollama.com/library/ministral-3:3b-instruct-2512-q4_K_M) (Ollama or any OpenAI-compatible server) |
| TTS | [Qwen3-TTS 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| TTS (Alternative) | [Qwen3-TTS 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| TTS (Alternative) | [Hexgrad Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| VAD | [Silero VAD v6](https://github.com/snakers4/silero-vad) (ONNX, client-side) |


### Qwen3-TTS Streaming Fork

This project uses a custom fork of Qwen3-TTS with streaming optimizations:

**[rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming)**

Key improvements over upstream:
- **Two-phase streaming** - Aggressive first chunk settings for lower TTFB, then stable settings for quality
- **torch.compile optimizations** - Up to 1.7x speedup after JIT warmup


## Architecture

```
┌─────────────────┐                   ┌─────────────────┐
│   Frontend      │ ───────────────▶  │   Backend       │
│   (index.html)  │   HTTP/WebSocket  │   (FastAPI)     │
│   Port 8080     │ ◀───────────────  │   Port 5173     │
└─────────────────┘                   └────────┬────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    │                   OVAPipeline                        │
                    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                    │  │ ASR Process │  │    LLM      │  │ TTS Engine  │  │
                    │  │ (Unix sock) │  │ (Ollama/OAI)│  │ (Qwen3/Kok) │  │
                    │  └─────────────┘  └─────────────┘  └─────────────┘  │
                    └─────────────────────────────────────────────────────┘
```

The ASR subprocess uses Unix socket IPC (not multiprocessing.Pipe) to avoid vLLM's stdout conflicts and ensure clean CUDA context isolation. ASR runs in a separate subprocess spawned **before** torch imports so vLLM gets a clean CUDA context and uses `fork` (5-10x faster startup). The IPC protocol uses pickle serialization over Unix sockets for efficient numpy array transfer.

## How It Works

1. Frontend captures audio/text (optionally with an image) and sends to the backend
2. Backend processes the request:
   - Transcribes audio using embedded ASR subprocess (or streaming via WebSocket)
   - Sends text + optional image to the LLM
   - **Early TTS decode**: Starts TTS before full LLM response (reduces TTFB by ~200-300ms)
   - **Two-phase streaming**: Aggressive first chunk settings, then stable streaming
3. Frontend plays audio as it arrives using an AudioWorklet processor

On an RTX 5060Ti (16GB VRAM), with Qwen3-TTS streaming optimizations enabled, TTFB is ~620ms after warmup with the 1.7B Model.

## Streaming TTS

OVA supports two streaming formats: **PCM** (lower latency, WAV header + raw chunks with AudioWorklet) and **WAV** (each chunk is a complete WAV file). PCM mode uses `reduce-overhead` torch.compile for ~1.5-1.7x speedup. The frontend pre-buffers ~0.4 seconds before playback to ensure smooth audio.

**Two-phase streaming** uses aggressive settings for the first chunk (lower TTFB) then switches to stable settings for quality. The pipeline also includes runtime audio quality assertions (overlap detection, RMS continuity) that log warnings for debugging.

> See [VARIABLES.md](VARIABLES.md) for all streaming tuning parameters (`OVA_PCM_*`, `OVA_FIRST_CHUNK_*`).

## Qwen3 Voice Profiles

Profiles are organized by language under `profiles/<language>/<voice>/`:

```
profiles/
├── zh/, en/, ja/, ko/, de/, fr/, ru/, pt/, es/, it/
```

Language directories are provided for all qwen3 supported languages. Create voice profiles inside them.

### Creating a New Voice Profile

1. Create a directory: `profiles/<language>/<voice_name>/`
2. Add a `ref_audio.wav` — 5-15 second clear voice sample (24kHz recommended, MP3/MP4 also accepted)
3. Generate voice clone prompts using one of two methods:
   - **Option A**: Run `generate_voice_prompts.py` — auto-transcribes the audio and generates `.pt` files
   - **Option B**: Add a `ref_text.txt` with the exact transcription — `.pt` files are auto-generated on first start
4. Optionally add a `prompt.txt` to customize the personality (falls back to `prompts/<lang>/default.txt`)

Voice clone prompts are model-specific:
- `voice_clone_prompt_0.6B.pt` - For 0.6B TTS model (1024-dim embeddings)
- `voice_clone_prompt_1.7B.pt` - For 1.7B TTS model (2048-dim embeddings)

### Enhancing Audio Quality (Optional)

For better voice cloning results, you can enhance reference audio using [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance).

First, install the optional dependencies (`--no-deps` on resemble-enhance to avoid torch/numpy version conflicts):

```bash
uv pip install --no-deps resemble-enhance && uv pip install deepspeed
```

Then run the enhancement script:

```bash
# Denoise only (recommended)
python scripts/enhance_profile_audio.py martina

# Denoise + enhance (upscale quality)
python scripts/enhance_profile_audio.py martina --enhance

# English profile
python scripts/enhance_profile_audio.py cassidy --language en
```

This removes background noise, optionally upscales audio quality, and resamples to 24kHz. The original audio is backed up as `ref_audio_original.*`.

## Prompting

Each voice profile includes a `prompt.txt` that controls the assistant's personality and output style. Getting the prompt right is important because the LLM output is spoken aloud by TTS — formatting that looks fine in text can produce audible artifacts in speech.

### Prompt Structure

A prompt has two parts:

1. **Personality lines** — who the assistant is and what language to use
2. **Instructions block** — behavioral rules that keep output TTS-friendly

Here's the English reference prompt (`prompts/en/default.txt`):

```
You are a friendly and approachable voice assistant.
You speak with a natural, casual, and relaxed tone, like a friend chatting on the phone.
Always respond in English.

Instructions:
- Be concise and direct - answer the first sentence clearly before continuing.
- Prioritize clarity over response length.
- Use a casual and friendly tone.
- NEVER respond with lists - use complete sentences.
- NEVER include any Markdown formatting, asterisks, underscores, or other formatting.
- Do NOT include emojis.
- Use punctuation to control speech rhythm: commas for brief pauses, ellipsis (...) for hesitation, dashes (--) for interruptions.
- You may use [pause:X] to insert a deliberate pause of X seconds (e.g., [pause:0.5]). Use sparingly for dramatic effect.
```

### Why the First Sentence Matters for TTFB

With `OVA_EARLY_TTS_DECODE=true` (the default), the backend starts TTS **before** the full LLM response is ready. It watches the token stream and triggers audio synthesis on the first sentence boundary (`.` `?` `!`), buffer reaching ~40 characters, or 12 tokens — whichever comes first.

The instruction *"answer the first sentence clearly before continuing"* causes the LLM to produce a short opening sentence with punctuation early, triggering TTS sooner. For example, `"Sure thing!"` triggers immediately, while a long rambling sentence waits for fallback thresholds.

### Key Prompt Guidelines

| Rule | Why |
|------|-----|
| No lists | TTS reads bullet characters (`-`, `*`, `1.`) literally as speech |
| No Markdown | Asterisks and underscores become audible artifacts |
| No emojis | TTS either skips them or mispronounces them |
| Punctuation for rhythm | Commas, ellipsis, and dashes directly control TTS prosody and pacing |
| Complete sentences | Produces natural speech flow instead of fragmented phrases |

If a profile directory doesn't contain a `prompt.txt`, the system falls back to `prompts/<language>/default.txt`. You can reload the active prompt at runtime via the settings panel without restarting.

## Prosody (Silence Tags)

OVA supports `[pause:X]` tags that let the LLM insert deliberate silences into speech. This is purely prompt-controlled — add the instruction line to a profile's `prompt.txt` to enable it:

```
- You may use [pause:X] to insert a deliberate pause of X seconds (e.g., [pause:0.5]). Use sparingly for dramatic effect.
```

**Syntax:** `[pause:X]` or `[p:X]` where X is seconds (e.g., `[pause:0.5]`, `[p:1.5]`).

When the LLM outputs text with embedded tags, the pipeline splits it into text and silence segments. Tags are stripped from conversation history after processing. Pauses are clamped to `OVA_MAX_PAUSE_DURATION` (default 3.0 seconds). Requires Qwen3 TTS with PCM streaming (the default).

## Multimodal (Image + Text)

The chat interface supports attaching images:

1. Click the image icon to add an image.
2. Type your question about the image.
3. The vision-language model will analyze the image and respond

The `/v1/chat` endpoint handles text + optional image queries directly.

## Barge-In & Wake Word

### Voice Activity Detection (Silero VAD)

OVA uses [Silero VAD v6](https://github.com/snakers4/silero-vad) running client-side via ONNX Runtime Web for real-time speech detection. The model runs in the browser at 32ms frame intervals with no server round-trips. Falls back to RMS energy detection if the ONNX model fails to load.

### Barge-In

When enabled (`OVA_ENABLE_BARGE_IN=true`, the default), the user can interrupt TTS playback by speaking. VAD monitors the microphone during playback and triggers an interrupt when speech is confirmed. After a grace period (`OVA_BARGE_IN_GRACE_MS`), confirmed speech stops playback and starts recording with auto-send. Backchannel filtering (`OVA_BACKCHANNEL_FILTER`) can discard filler words like "yeah" or "ok" during barge-in.

### Wake Word

When enabled (`OVA_ENABLE_WAKE_WORD=true`), the assistant listens continuously for a configurable phrase (default: "hey nova") to start recording hands-free. The mic is acquired eagerly on page load. VAD monitors for speech onset (low CPU — no ASR until speech detected), then streaming ASR checks partial transcripts against the wake phrase. A pre-speech ring buffer (~500ms) captures audio before VAD triggers to avoid clipping.

> See [VARIABLES.md](VARIABLES.md) for all barge-in and wake word settings (`OVA_ENABLE_BARGE_IN`, `OVA_VAD_*`, `OVA_AUTO_SEND_*`, `OVA_BARGE_IN_*`, `OVA_WAKE_WORD`).

## Tools / Function Calling

The LLM can invoke real Python functions during a conversation and incorporate the results into its spoken response. Tools are **plugin-style** — drop a `.py` file in `ova/tools/` and the registry auto-discovers it. No manual registration needed.

### Built-in Tools

| Tool | Description | Default |
|------|-------------|---------|
| `get_time` | Returns current time in configured timezone | Enabled |
| `get_date` | Resolves dates via natural language ("yesterday", "next Friday") | Enabled |
| `set_timer` | Sets an in-memory countdown timer (1–3600s) | Enabled |
| `check_timers` | Reports status of all active timers | Enabled |
| `web_search` | Example Tavily Web search provided (requires `OVA_SEARCH_API_KEY`) | Disabled |

Timer expirations push real-time notifications to the browser via Server-Sent Events (`GET /v1/events`), with OS-level notifications when the tab is hidden.

Enable with `OVA_ENABLE_TOOLS=true` in `.env`. Enabling tools increases TTFA due to the LLM tool iterations.

> See [TOOLS.md](TOOLS.md) for the full guide — creating custom tools, event publishing, frontend handler registration, security considerations, and the complete API reference.

## MCP (External Tool Servers)

OVA can connect to external [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers as a client, expanding its tool capabilities without writing Python code. MCP tools appear alongside native tools — the LLM sees a single unified tool list.

Enable with `OVA_ENABLE_MCP=true` and `OVA_ENABLE_TOOLS=true` in `.env`. Configure servers in `mcp_servers.json` (same format as Claude Desktop / VS Code):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
    }
  }
}
```

Supports stdio (local subprocess), SSE, and Streamable HTTP transports. Servers connect in parallel at startup, with automatic reconnect on failure.

> See [MCP.md](MCP.md) for the full guide — architecture, transport types, examples, error handling, and security considerations.

## API Endpoints

All endpoints are versioned under `/v1/`. Voice is an optional path parameter, language is a query parameter.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/audio` | POST | Voice input - receives WAV audio, returns streaming TTS response |
| `/v1/chat/{voice_id}/audio` | POST | Same, with explicit voice |
| `/v1/chat` | POST | Text + optional image input, returns streaming TTS response |
| `/v1/chat/{voice_id}` | POST | Same, with explicit voice |
| `/v1/text-to-speech` | POST | Pure TTS - synthesizes text exactly as given |
| `/v1/text-to-speech/{voice_id}` | POST | Same, with explicit voice |
| `/v1/text-to-speech/batch` | POST | Batch TTS - synthesizes multiple texts, returns NDJSON |
| `/v1/speech-to-text` | POST | One-shot speech-to-text |
| `/v1/interrupt` | POST | Stop current TTS playback (used by barge-in) |
| `/v1/speech-to-text/stream` | WebSocket | Streaming ASR - send audio chunks, receive partial transcripts |
| `/v1/events` | GET | SSE stream for real-time push notifications (tools → browser) |
| `/v1/info` | GET | Pipeline configuration info |
| `/v1/health` | GET | Server readiness check |
| `/v1/settings` | GET/POST | Runtime settings management (hot-reload capable) |
| `/v1/settings/prompt` | POST | Update system prompt (session-only, no restart) |
| `/v1/restart` | POST | Trigger server restart |

### Hot-Reload Settings

The following can be changed at runtime without restart via `POST /v1/settings`:

| Setting | Hot-Reload | Notes |
|---------|------------|-------|
| Voice profile | Yes | If preloaded at startup |
| Language | Yes | Loads new voice prompts automatically |
| System prompt | Yes | Via `/v1/settings/prompt` |
| TTS engine | No | Requires restart |
| Streaming format (pcm/wav) | No | Requires restart |

## SDK

The OVA SDK is a standalone Python package for programmatic access to the OVA server from any machine. It is installed automatically during `./ova.sh install`.

For standalone or remote use, install directly from git:

```bash
pip install git+https://github.com/rekuenkdr/ova-python-sdk.git
```

```python
from ova_sdk import OVA

client = OVA()  # connects to localhost:5173 by default
client.wait_until_ready()

audio = client.chat.send_text("Tell me a joke")
audio.play()
```

For remote or authenticated servers, set `OVA_BASE_URL` and `OVA_API_KEY` environment variables (or pass as constructor arguments).

Full documentation and examples: [ova-python-sdk](https://github.com/rekuenkdr/ova-python-sdk)

## Project Structure

```
ova/
├── index.html           # Frontend UI
├── ova.sh               # CLI entry point (install/start/stop)
├── ova/                 # Python package
│   ├── api.py           # FastAPI app
│   ├── asr_server.py    # ASR subprocess (Unix socket server, isolated CUDA)
│   ├── mcp_client.py    # MCP client manager (external tool servers)
│   ├── pipeline.py      # OVAPipeline class
│   ├── prosody.py       # Prosody tag parsing ([pause:X])
│   ├── llm/             # LLM provider abstraction
│   │   ├── base.py          # LLMProvider ABC + LLMResponse/ToolCall dataclasses
│   │   ├── factory.py       # Provider factory (create_llm_provider)
│   │   ├── ollama_provider.py   # Ollama backend
│   │   └── openai_provider.py   # OpenAI-compatible backend (vLLM, TRT-LLM, etc.)
│   ├── events.py        # EventBus — thread-safe server→client event publishing
│   ├── audio.py         # Audio utilities
│   ├── utils.py         # Logging & device detection
│   └── tools/           # Plugin-style tool modules (auto-discovered + MCP)
│       ├── __init__.py      # ToolRegistry — discovery, enable/disable, execution
│       ├── _base.py         # Shared helpers (publish_event, get_pipeline_language)
│       ├── get_datetime.py  # get_time, get_date tools
│       ├── timer.py         # set_timer, check_timers tools (SSE expiration events)
│       └── web_search.py    # web_search tool (disabled by default)
├── static/
│   ├── css/
│   │   ├── base.css         # CSS reset & variables
│   │   ├── components.css   # UI component styles
│   │   └── animations.css   # Transitions & animations
│   ├── js/
│   │   ├── app.js           # Main entry point
│   │   ├── audio.js         # Recording/playback/ASR
│   │   ├── config.js        # Shared configuration
│   │   ├── settings.js      # Settings panel
│   │   ├── vad.js           # Silero VAD + RMS fallback
│   │   ├── wakeword.js      # Wake word detection (VAD-gated ASR)
│   │   ├── theme.js         # Dark/light theme
│   │   ├── ui.js            # DOM state management
│   │   ├── events.js        # SSE client — EventSource + onEvent() handler registry
│   │   ├── notifications.js # Hybrid system/toast notifications + alarm chime
│   │   └── pcm-processor.js # AudioWorklet for PCM streaming
│   ├── img/                 # Static images
│   └── themes/
│       ├── dark/theme.css      # Dark theme (default)
│       ├── light/theme.css     # Light theme
│       ├── her/theme.css       # Her / Samantha theme
│       └── hal-9000/theme.css  # HAL-9000 theme
├── models/              # Local ML models (Silero VAD ONNX)
├── profiles/            # Voice profiles by language (user-created)
│   ├── en/, es/, fr/, de/, it/, pt/, ru/, ja/, ko/, zh/
├── prompts/             # Default system prompts by language
│   ├── en/, es/, fr/, de/, it/, pt/, ru/, ja/, ko/, zh/
└── scripts/
    ├── configure_cuda.py          # CUDA version configurator (12/13)
    ├── enhance_profile_audio.py   # Audio denoising utility
    ├── generate_voice_prompts.py  # Voice clone prompt generator
    └── profile_pipeline.py        # Pipeline profiling utility
```

## Security

OVA is designed for **localhost use only**. The default configuration is safe for local use — it binds to `localhost`, requires no authentication, and disables tools and MCP by default.

For network/internet exposure, you need API authentication (`OVA_API_KEY`), HTTPS via a reverse proxy, and feature flag hardening. OVA is single-tenant by design — separate users need separate instances.

> See [SECURITY.md](SECURITY.md) for the full security overview — threat model, tool/MCP safety, API-only mode, hardening checklist, and all security-related configuration.

## Disclaimer

This project is a proof-of-concept demonstration and is provided **as is** without any warranties or guarantees. It is intended for educational and experimental purposes only.

The voice cloning capability is purely for educational purposes - for real-life or commercial use, always seek relevant permissions. This demo highlights ethical and security considerations: the ease with which one can clone a voice using only a 3-5 second audio clip is both impressive and potentially dangerous in the wrong hands.
