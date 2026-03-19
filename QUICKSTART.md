# Quick Start

Minimal `.env` to get running. See [VARIABLES.md](VARIABLES.md) for the full reference and [README.md](README.md) for the full feature list.

---

## 1. Install

```bash
./ova.sh install
```

## 2. Configure CUDA version (if not CUDA 13)

The default configuration targets **CUDA 13**. If your system uses CUDA 12, reconfigure before installing dependencies:

```bash
./ova.sh configure-cuda 12
```

This rewrites `pyproject.toml` and `ova.sh` with the correct PyTorch, flash-attn, vLLM, and onnxruntime builds for your CUDA version. You can also let it auto-detect:

```bash
./ova.sh configure-cuda
```

> Skip this step if you're on CUDA 13 (the default).

## 3. Create a voice profile (Qwen3-TTS only)

```bash
mkdir -p profiles/en/myvoice
cp /path/to/your/sample.wav profiles/en/myvoice/ref_audio.wav
.venv/bin/python scripts/generate_voice_prompts.py
```

Provide a 5-15 second clear voice sample (WAV, MP3, or MP4 accepted). The script auto-transcribes the audio and generates voice clone prompt files.

Alternatively, provide both `ref_audio.wav` and `ref_text.txt` (exact transcription) and the `.pt` files will be generated on first start.

## 4. Configure & Start

Create a `.env` with the minimal configuration, use one of the provided .env examples:

---

### Qwen3-TTS (recommended — voice cloning, PCM streaming)

```bash
cp .env.qwen3.example .env
# Edit .env — set OVA_QWEN3_VOICE to your profile dir name, adjust language/model as needed
```

> **If using the 0.6B TTS model**, also set:
> `OVA_PCM_DECODE_WINDOW=64`, `OVA_MAX_TTS_FRAMES=1500`, `OVA_LLM_MAX_TOKENS=300`

### Kokoro-TTS (alternative — predefined voices, lighter on VRAM)

No voice cloning, no PCM streaming. 82M model — good for low-VRAM setups. No voice profile setup needed.

```bash
cp .env.kokoro.example .env
# Edit .env — adjust language/voice as needed
```

Run `./ova.sh start`.

Open http://localhost:8080 in your browser.

```bash
./ova.sh stop       # Full stop — unloads models from VRAM
./ova.sh restart    # Keeps LLM loaded — faster restart (~3-5s vs ~10-15s)
```



---

### Using an OpenAI-compatible LLM provider

Replace the LLM section with:

```bash
OVA_LLM_PROVIDER=openai
OVA_LLM_BASE_URL=https://api.mistral.ai/v1
OVA_LLM_API_KEY=your-mistral-api-key
OVA_CHAT_MODEL=ministral-3b-2512
```

Works with any OpenAI-compatible API — Mistral, OpenAI, Together, Groq, vLLM, TensorRT-LLM, llama.cpp, etc.
