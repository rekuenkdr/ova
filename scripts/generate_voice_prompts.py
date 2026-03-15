#!/usr/bin/env python3
"""
Generate voice clone prompts for all profiles.

Loops through profiles/<language>/<voice>/ directories and creates
voice_clone_prompt_{size}.pt files for voices that have ref_audio.wav and ref_text.txt.

If ref_text.txt is missing but ref_audio.wav exists, the audio is automatically
transcribed using Qwen3-ASR 0.6B.

Voice clone prompts are MODEL-SPECIFIC due to different speaker encoder dimensions:
- 1.7B model: 2048-dim speaker embeddings
- 0.6B model: 1024-dim speaker embeddings

Usage:
    python scripts/generate_voice_prompts.py [--force] [--language <lang>] [--voice <name>]

Examples:
    python scripts/generate_voice_prompts.py                         # Generate missing prompts for default model
    python scripts/generate_voice_prompts.py --model-size 0.6B       # Generate for 0.6B model
    python scripts/generate_voice_prompts.py --model-size 1.7B       # Generate for 1.7B model
    python scripts/generate_voice_prompts.py --all-sizes             # Generate for both models
    python scripts/generate_voice_prompts.py --force                 # Regenerate all prompts
    python scripts/generate_voice_prompts.py --language es           # Only Spanish profiles
    python scripts/generate_voice_prompts.py --voice martina         # Only specific voice
    python scripts/generate_voice_prompts.py --no-transcribe         # Skip auto-transcription
"""

# CRITICAL: Set vLLM environment BEFORE any imports to ensure clean CUDA context
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import sys
from pathlib import Path

# Delay torch import to avoid CUDA initialization before vLLM
# torch is imported lazily in functions that need it


# Language code to Qwen ASR language name mapping
# Only includes languages supported by Qwen3-TTS
LANG_CODE_TO_ASR = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"

# Model configurations
MODELS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

DEFAULT_MODEL_SIZE = "1.7B"

# ASR model instance (lazy loaded)
_asr_model = None


def convert_audio_to_wav(profile_dir: Path) -> bool:
    """Convert MP3/MP4 audio file to ref_audio.wav if needed.

    Returns True if conversion happened or wav already exists.
    """
    ref_wav = profile_dir / "ref_audio.wav"
    if ref_wav.exists():
        return True

    # Find any MP3 or MP4 file in the directory
    audio_files = list(profile_dir.glob("*.mp3")) + list(profile_dir.glob("*.mp4"))
    if not audio_files:
        return False

    source_file = audio_files[0]  # Take the first one found
    print(f"    Converting {source_file.name} to ref_audio.wav...")

    import subprocess
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(source_file),
        "-ar", "24000", "-ac", "1",
        str(ref_wav)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    Error converting: {result.stderr}")
        return False

    return True


def get_asr_model():
    """Lazy load Qwen3-ASR model."""
    global _asr_model
    if _asr_model is None:
        print("Loading Qwen3-ASR 0.6B for transcription...")
        from qwen_asr import Qwen3ASRModel
        _asr_model = Qwen3ASRModel.LLM(
            model="Qwen/Qwen3-ASR-0.6B",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            max_new_tokens=512,
            enforce_eager=True,  # Skip CUDA graphs for one-off transcription
        )
        print("ASR model loaded.\n")
    return _asr_model


def transcribe_audio(audio_path: Path, language_code: str) -> str:
    """Transcribe audio file using Qwen3-ASR."""
    import soundfile as sf

    asr_language = LANG_CODE_TO_ASR.get(language_code)
    if not asr_language:
        print(f"    Warning: Unknown language code '{language_code}', using English")
        asr_language = "English"

    audio, sr = sf.read(str(audio_path))

    # Resample to 16kHz if needed (ASR expects 16kHz)
    if sr != 16000:
        import soxr
        audio = soxr.resample(audio, sr, 16000)
        sr = 16000

    model = get_asr_model()
    results = model.transcribe((audio, sr), language=asr_language)

    # Results is a list of transcription objects
    if results and results[0].text:
        return results[0].text.strip()
    return ""


def ensure_transcriptions(args) -> int:
    """Check for missing ref_text.txt and transcribe if needed. Returns count of transcribed files."""
    if args.no_transcribe:
        return 0

    # First: convert any MP3/MP4 files to ref_audio.wav
    for lang_dir in PROFILES_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        if args.language and lang_dir.name != args.language:
            continue

        for voice_dir in lang_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            if args.voice and voice_dir.name != args.voice:
                continue

            convert_audio_to_wav(voice_dir)

    transcribed = 0
    profiles_needing_transcription = []

    # Second pass: find profiles needing transcription
    for lang_dir in PROFILES_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        if args.language and lang_dir.name != args.language:
            continue

        for voice_dir in lang_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            if args.voice and voice_dir.name != args.voice:
                continue

            ref_audio = voice_dir / "ref_audio.wav"
            ref_text_file = voice_dir / "ref_text.txt"

            if ref_audio.exists() and not ref_text_file.exists():
                profiles_needing_transcription.append({
                    "language": lang_dir.name,
                    "voice": voice_dir.name,
                    "ref_audio": ref_audio,
                    "ref_text_file": ref_text_file,
                })

    if not profiles_needing_transcription:
        return 0

    print(f"{'='*60}")
    print(f"Auto-transcribing {len(profiles_needing_transcription)} profile(s) missing ref_text.txt")
    print(f"{'='*60}\n")

    for profile in profiles_needing_transcription:
        print(f"Transcribing {profile['language']}/{profile['voice']}...")
        try:
            text = transcribe_audio(profile["ref_audio"], profile["language"])
            profile["ref_text_file"].write_text(text, encoding="utf-8")
            print(f"    Transcription: {text}")
            print(f"    Saved: {profile['ref_text_file']}")
            transcribed += 1
        except Exception as e:
            print(f"    Error transcribing: {e}")

    # Unload ASR model to free VRAM for TTS
    global _asr_model
    if _asr_model is not None:
        print("\nUnloading ASR model to free VRAM...")
        import torch
        del _asr_model
        _asr_model = None
        torch.cuda.empty_cache()

    print()
    return transcribed


def get_model_size_from_name(model_name: str) -> str:
    """Extract model size from model name."""
    if "0.6B" in model_name or "0b6" in model_name.lower():
        return "0.6B"
    elif "1.7B" in model_name or "1b7" in model_name.lower():
        return "1.7B"
    return DEFAULT_MODEL_SIZE


def find_profiles(args, model_size: str) -> list:
    """Find all profiles that need processing for given model size."""
    profiles = []

    for lang_dir in PROFILES_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        if args.language and lang_dir.name != args.language:
            continue

        for voice_dir in lang_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            if args.voice and voice_dir.name != args.voice:
                continue

            ref_audio = voice_dir / "ref_audio.wav"
            ref_text_file = voice_dir / "ref_text.txt"
            cache_file = voice_dir / f"voice_clone_prompt_{model_size}.pt"

            if not ref_audio.exists():
                print(f"Skip {lang_dir.name}/{voice_dir.name}: missing ref_audio.wav")
                continue

            if not ref_text_file.exists():
                print(f"Skip {lang_dir.name}/{voice_dir.name}: missing ref_text.txt (transcription may have failed)")
                continue

            if cache_file.exists() and not args.force:
                print(f"Skip {lang_dir.name}/{voice_dir.name}: {cache_file.name} exists (use --force)")
                continue

            profiles.append({
                "language": lang_dir.name,
                "voice": voice_dir.name,
                "ref_audio": ref_audio,
                "ref_text_file": ref_text_file,
                "cache_file": cache_file,
            })

    return profiles


def process_profiles(profiles: list, model_name: str, model_size: str, device: str):
    """Load model and generate prompts for all profiles."""
    import torch

    if not profiles:
        print(f"No profiles to process for {model_size}.")
        return

    print(f"\n{'='*60}")
    print(f"Generating prompts for {model_size} model")
    print(f"{'='*60}")
    print(f"Found {len(profiles)} profile(s) to process:")
    for p in profiles:
        print(f"  - {p['language']}/{p['voice']}")
    print()

    # Load model
    print(f"Loading Qwen3-TTS model: {model_name}")
    print(f"Device: {device}")

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Verify model size matches expectation
    actual_size = getattr(model.model.config, 'tts_model_size', 'unknown')
    print(f"Model loaded (tts_model_size={actual_size}).\n")

    # Process each profile
    for i, profile in enumerate(profiles, 1):
        print(f"[{i}/{len(profiles)}] Processing {profile['language']}/{profile['voice']}...")

        ref_text = profile["ref_text_file"].read_text(encoding="utf-8").strip()

        prompt = model.create_voice_clone_prompt(
            ref_audio=str(profile["ref_audio"]),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        # create_voice_clone_prompt returns List[VoiceClonePromptItem];
        # normalize to single item for consistent storage
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]

        torch.save(prompt, profile["cache_file"])
        print(f"    Saved: {profile['cache_file'].name}")

    print(f"\nDone! Generated {len(profiles)} voice clone prompt(s) for {model_size}.")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice clone prompts for profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for default model (1.7B)
    python scripts/generate_voice_prompts.py

    # Generate for 0.6B model
    python scripts/generate_voice_prompts.py --model-size 0.6B

    # Generate for both model sizes
    python scripts/generate_voice_prompts.py --all-sizes

    # Force regenerate all prompts for a specific voice
    python scripts/generate_voice_prompts.py --voice martina --force --all-sizes
"""
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Regenerate even if prompt file exists")
    parser.add_argument("--language", "-l", default=None,
                        help="Only process specific language (e.g., es, en)")
    parser.add_argument("--voice", "-v", default=None,
                        help="Only process specific voice profile")
    parser.add_argument("--model-size", "-s", default=None, choices=["0.6B", "1.7B"],
                        help="Model size to generate prompts for (default: 1.7B)")
    parser.add_argument("--all-sizes", "-a", action="store_true",
                        help="Generate prompts for all model sizes (0.6B and 1.7B)")
    parser.add_argument("--model", "-m", default=None,
                        help="Custom Qwen3-TTS model path (overrides --model-size)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip auto-transcription of missing ref_text.txt files")
    args = parser.parse_args()

    # Auto-transcribe missing ref_text.txt files
    transcribed_count = ensure_transcriptions(args)
    if transcribed_count > 0:
        print(f"Transcribed {transcribed_count} audio file(s).\n")

    # Determine which model sizes to process
    if args.all_sizes:
        sizes_to_process = ["0.6B", "1.7B"]
    elif args.model:
        # Custom model specified - detect size from name
        size = get_model_size_from_name(args.model)
        sizes_to_process = [size]
        MODELS[size] = args.model  # Override with custom path
    elif args.model_size:
        sizes_to_process = [args.model_size]
    else:
        sizes_to_process = [DEFAULT_MODEL_SIZE]

    print(f"Model sizes to process: {sizes_to_process}")

    # Process each model size
    for model_size in sizes_to_process:
        model_name = MODELS[model_size]
        profiles = find_profiles(args, model_size)
        process_profiles(profiles, model_name, model_size, args.device)

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
