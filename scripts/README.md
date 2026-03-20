# Scripts

## enhance_profile_audio_RE-USE.py

Denoise and enhance voice profile audio using [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE) (Universal Speech Enhancement, SEMamba architecture).

### Install

```bash
# mamba-ssm must be built from source against your CUDA version
TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda uv pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation
```

The RE-USE model (~39MB) is auto-downloaded from HuggingFace on first run (cached in `~/.cache/huggingface/hub/`).

### Usage

```bash
# Denoise + BWE to 24kHz (default, recommended)
python scripts/enhance_profile_audio_RE-USE.py martina

# English profile
python scripts/enhance_profile_audio_RE-USE.py david -l en

# Custom BWE target (e.g. 32kHz)
python scripts/enhance_profile_audio_RE-USE.py martina --bwe 32000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `profile` | required | Profile name |
| `-l, --language` | `es` | Language folder |
| `--bwe` | `24000` | Bandwidth extension target rate in Hz (resamples before denoising) |
| `--device` | `cuda:0` | CUDA device |

### What it does

1. Loads `ref_audio.{wav,ogg,mp3,m4a,flac}` from `profiles/<lang>/<profile>/`
2. Converts to mono if stereo
3. Downloads RE-USE model (~39MB) if not present
4. Resamples to BWE target rate if input rate differs (default 24kHz for Qwen3-TTS)
5. Runs STFT → SEMamba denoise → ISTFT (with sweep artifact removal)
6. Backs up original as `ref_audio_original.*`
7. Saves result as `ref_audio.wav`
8. Removes cached `voice_clone_prompt_{0.6B,1.7B}.pt` (forces regeneration)

---

## enhance_profile_audio.py

Denoise voice profile audio using [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance).

### Install

```bash
pip install resemble-enhance --upgrade
```

### Usage

```bash
# Denoise only (recommended)
python scripts/enhance_profile_audio.py martina

# Denoise + enhance (44.1kHz upscale)
python scripts/enhance_profile_audio.py martina --enhance

# English profile
python scripts/enhance_profile_audio.py david -l en

# Higher quality enhancement
python scripts/enhance_profile_audio.py martina --enhance --nfe 64
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `profile` | required | Profile name |
| `-l, --language` | `es` | Language folder |
| `-e, --enhance` | off | Apply enhancement after denoising |
| `--nfe` | `32` | Enhancement iterations (1-128, higher = better) |
| `--lambd` | `0.5` | Enhancement blend (0 = natural, 1 = enhanced) |
| `--device` | `cuda:0` | CUDA device |

### What it does

1. Loads `ref_audio.{wav,ogg,mp3,m4a,flac}` from `profiles/<lang>/<profile>/`
2. Converts to mono if stereo
3. Runs Resemble Enhance denoiser
4. Optionally runs enhancer (upscales to 44.1kHz)
5. Resamples to 24kHz (Qwen3-TTS native rate)
6. Backs up original as `ref_audio_original.*`
7. Saves result as `ref_audio.wav`
8. Removes cached `voice_clone_prompt_{0.6B,1.7B}.pt` (forces regeneration)

---

## generate_voice_prompts.py

Generate `voice_clone_prompt_{size}.pt` embeddings for TTS voice cloning.

### Usage

```bash
# Generate for 1.7B model (default)
python scripts/generate_voice_prompts.py

# Generate for 0.6B model
python scripts/generate_voice_prompts.py -s 0.6B

# Generate for both model sizes
python scripts/generate_voice_prompts.py --all-sizes

# Regenerate all prompts
python scripts/generate_voice_prompts.py --force

# Specific language/voice
python scripts/generate_voice_prompts.py -l es -v martina --force
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --model-size` | `1.7B` | Target model (`0.6B` or `1.7B`) |
| `-a, --all-sizes` | off | Generate for both models |
| `-f, --force` | off | Overwrite existing prompts |
| `-l, --language` | all | Filter by language |
| `-v, --voice` | all | Filter by voice name |
| `-m, --model` | auto | Custom model path |
| `--no-transcribe` | off | Skip auto-transcription |
| `--device` | `cuda:0` | CUDA device |

### What it does

1. Auto-converts MP3/MP4 files in profile directories to `ref_audio.wav` (24kHz mono, via ffmpeg)
2. Scans `profiles/<language>/<voice>/` directories
3. Auto-transcribes missing `ref_text.txt` using Qwen3-ASR 0.6B (unloads after)
4. Loads Qwen3-TTS model for specified size
5. For each profile with `ref_audio.wav` and `ref_text.txt`:
   - Creates voice clone prompt embedding
   - Saves as `voice_clone_prompt_{size}.pt`
6. Skips existing prompts unless `--force`
7. Frees GPU memory between model sizes when using `--all-sizes`

**Note:** Prompts are model-specific (1.7B = 2048-dim, 0.6B = 1024-dim). Using wrong prompt causes tensor shape errors.

**Supported languages:** zh, en, ja, ko, de, fr, ru, pt, es, it

---

## profile_pipeline.py

Measure ASR → LLM → TTS latency. Requires OVA backend running.

### Usage

```bash
# Basic profiling (auto-finds profile audio or generates test tone)
python scripts/profile_pipeline.py

# Record from microphone
python scripts/profile_pipeline.py --record
python scripts/profile_pipeline.py --record 3  # 3 seconds

# Use specific audio file
python scripts/profile_pipeline.py --audio test.wav

# Custom test text
python scripts/profile_pipeline.py --text "Hello, how are you?"

# More runs for averaging
python scripts/profile_pipeline.py --runs 5 --warmup 2

# Use standalone ASR server instead of embedded subprocess
python scripts/profile_pipeline.py --standalone-asr
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | auto | WAV file path |
| `--record [SECS]` | off | Record from mic (default 5s) |
| `--text` | `¿Qué tiempo hace hoy?` | Text for LLM/TTS test |
| `--backend-url` | `localhost:5173` | OVA backend |
| `--asr-url` | env or `localhost:8100` | ASR server (standalone mode only) |
| `--llm-model` | env or `ministral-3:...` | Ollama model |
| `--runs` | `3` | Profiling runs |
| `--warmup` | `1` | Warmup runs (not counted) |
| `--standalone-asr` | off | Use standalone ASR HTTP server instead of embedded subprocess |

### What it does

1. Loads audio from `--audio`, `--record`, first profile `ref_audio.wav`, or generates 440Hz test tone
2. Checks ASR server and backend availability
3. Runs warmup iterations (not counted)
4. Profiles individual components:
   - ASR: transcription latency
   - LLM: Ollama response time
   - TTS: synthesis time + TTFB (time to first byte)
5. Profiles full voice-to-voice pipeline via `/v1/chat/audio` endpoint
6. Outputs summary with component breakdown, percentages, and bottleneck analysis
