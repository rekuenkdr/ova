#!/usr/bin/env python3
"""Profile OVA pipeline latency: ASR → LLM → TTS"""

import argparse
import base64
import io
import os
import sys
import time
import wave
from pathlib import Path

# Add project root to sys.path so 'from ova...' imports work when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress verbose logging from OVA internals and HTTP libraries
import logging
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
for _name in ["ova.llm", "ova.utils", "sys", "llm", "tts", "asr", "api", "tools", "mcp"]:
    logging.getLogger(_name).setLevel(logging.WARNING)

from dotenv import load_dotenv
import requests
import numpy as np

# Load .env from project root (same as ova.sh)
load_dotenv(Path(__file__).parent.parent / ".env")

def record_audio(duration: float = 5.0, sample_rate: int = 48000) -> bytes:
    """Record audio from microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice: uv pip install sounddevice")
        sys.exit(1)

    print(f"\n🎤 Recording for {duration}s... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("✓ Recording complete\n")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# Default test audio - 3 seconds of silence with a beep
def generate_test_audio(duration: float = 3.0, sample_rate: int = 48000) -> bytes:
    """Generate a simple test WAV file."""
    samples = int(duration * sample_rate)
    # Simple sine wave beep
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    audio = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def profile_asr(wav_bytes: bytes, server_url: str = "http://localhost:8100", embedded: bool = False, language: str | None = None) -> tuple[str, float]:
    """Profile ASR transcription.

    Args:
        wav_bytes: WAV audio bytes
        server_url: ASR server URL (for HTTP mode) or backend URL (for embedded mode)
        embedded: If True, use embedded ASR via backend /v1/speech-to-text endpoint
        language: Language code for ASR (e.g., 'es', 'en')
    """
    if embedded:
        url = f"{server_url}/v1/speech-to-text"
        params = {"language": language} if language else {}
        start = time.perf_counter()
        resp = requests.post(
            url,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            params=params,
            timeout=60,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        text = resp.json()["text"]
        return text, elapsed
    else:
        # HTTP ASR server mode
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        audio_url = f"data:audio/wav;base64,{audio_b64}"

        url = f"{server_url}/v1/chat/completions"
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}}
                    ],
                }
            ]
        }

        start = time.perf_counter()
        resp = requests.post(url, json=data, timeout=60)
        elapsed = time.perf_counter() - start

        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text, elapsed


_llm_provider = None

def _get_llm_provider(model: str):
    global _llm_provider
    if _llm_provider is None or _llm_provider.model != model:
        from ova.llm import create_llm_provider
        _llm_provider = create_llm_provider(model)
    return _llm_provider

def profile_llm(text: str, model: str = "ministral-3:3b-instruct-2512-q4_K_M", system_prompt: str = "Respond briefly in one sentence.") -> tuple[str, float]:
    """Profile LLM chat response."""
    provider = _get_llm_provider(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    start = time.perf_counter()
    response = provider.chat(messages=messages)
    elapsed = time.perf_counter() - start

    return response.content or "", elapsed


def profile_tts(text: str, backend_url: str = "http://localhost:5173", language: str | None = None) -> tuple[int, float, float]:
    """Profile TTS generation. Returns (audio_bytes, total_time, time_to_first_byte)."""
    url = f"{backend_url}/v1/text-to-speech"
    if language:
        url += f"?language={language}"
    data = {"text": text}

    start = time.perf_counter()
    resp = requests.post(url, json=data, timeout=120, stream=True)

    first_byte_time = None
    chunks = []
    for chunk in resp.iter_content(chunk_size=4096):
        if first_byte_time is None:
            first_byte_time = time.perf_counter() - start
        chunks.append(chunk)

    total_time = time.perf_counter() - start
    audio_bytes = b"".join(chunks)

    return len(audio_bytes), total_time, first_byte_time or total_time


def profile_full_pipeline(wav_bytes: bytes, backend_url: str = "http://localhost:5173", language: str | None = None) -> tuple[int, float, float]:
    """Profile full voice-to-voice pipeline via /chat endpoint.

    Returns: (audio_size, total_time, ttfb)
    """
    url = f"{backend_url}/v1/chat/audio"
    if language:
        url += f"?language={language}"

    start = time.perf_counter()
    resp = requests.post(
        url,
        data=wav_bytes,
        headers={"Content-Type": "audio/wav"},
        timeout=120,
        stream=True,
    )

    first_byte_time = None
    chunks = []
    for chunk in resp.iter_content(chunk_size=4096):
        if first_byte_time is None:
            first_byte_time = time.perf_counter() - start
        chunks.append(chunk)

    total_time = time.perf_counter() - start
    audio_bytes = b"".join(chunks)

    return len(audio_bytes), total_time, first_byte_time or total_time


def calc_rtf(audio_bytes: int, elapsed: float, sample_rate: int = 24000) -> float:
    """Real-Time Factor. RTF < 1.0 = faster than real-time."""
    audio_duration = (audio_bytes - 44) / (sample_rate * 2)  # minus WAV header, int16
    return elapsed / audio_duration if audio_duration > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Profile OVA pipeline latency")
    parser.add_argument("--audio", type=Path, help="WAV file to use")
    parser.add_argument("--record", type=float, nargs="?", const=5.0, metavar="SECS", help="Record from mic (default: 5s)")
    parser.add_argument("--text", type=str, default="Hola, ¿cómo estás?", help="Text for LLM/TTS test")
    parser.add_argument("--asr-url", default=os.getenv("OVA_ASR_SERVER_URL", "http://localhost:8100"), help="ASR server URL")
    parser.add_argument("--backend-url", default="http://localhost:5173", help="OVA backend URL")
    parser.add_argument("--llm-model", default=os.getenv("OVA_CHAT_MODEL", "ministral-3:3b-instruct-2512-q4_K_M"), help="Ollama model")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for averaging")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (not counted)")
    parser.add_argument("--standalone-asr", action="store_true", help="Use standalone ASR HTTP server instead of embedded subprocess")
    parser.add_argument("--language", default=os.getenv("OVA_LANGUAGE", "es"), help="Language for ASR/TTS (default: es)")
    parser.add_argument("--voice", default=os.getenv("OVA_QWEN3_VOICE", "eva"), help="Voice profile for test audio and prompt (default: eva)")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    profiles_dir = project_root / "profiles"
    prompts_dir = project_root / "prompts"

    # Load, record, or generate test audio
    if args.record:
        wav_bytes = record_audio(duration=args.record)
    elif args.audio and args.audio.exists():
        wav_bytes = args.audio.read_bytes()
        print(f"Using audio: {args.audio}")
    else:
        # Try specific profile first, then language dir, then fallback
        voice_ref = profiles_dir / args.language / args.voice / "ref_audio.wav"
        if voice_ref.exists():
            wav_bytes = voice_ref.read_bytes()
            print(f"Using profile audio: {voice_ref.relative_to(project_root)}")
        else:
            lang_dir = profiles_dir / args.language
            ref_audios = sorted(lang_dir.glob("*/ref_audio.wav")) if lang_dir.exists() else []
            if ref_audios:
                wav_bytes = ref_audios[0].read_bytes()
                print(f"Using profile audio: {ref_audios[0].relative_to(project_root)}")
            else:
                wav_bytes = generate_test_audio()
                print("Using generated test tone (3s @ 440Hz)")

    # Load system prompt from profile or fallback to default
    prompt_file = profiles_dir / args.language / args.voice / "prompt.txt"
    if not prompt_file.exists():
        prompt_file = prompts_dir / args.language / "default.txt"
    if prompt_file.exists():
        system_prompt = prompt_file.read_text().strip()
        print(f"Using system prompt: {prompt_file.relative_to(project_root)}")
    else:
        system_prompt = "Respond briefly in one sentence."
        print("Using default system prompt (no profile prompt found)")

    print(f"Language: {args.language}, Voice: {args.voice}")
    print(f"Test text: {args.text}")
    print(f"Runs: {args.runs} (+ {args.warmup} warmup)")
    print("-" * 60)

    # Check services
    if args.standalone_asr:
        try:
            requests.get(f"{args.asr_url}/health", timeout=5)
            print(f"✓ ASR server ready at {args.asr_url}")
        except Exception as e:
            print(f"✗ ASR server not available: {e}")
            return
    else:
        print("✓ ASR: embedded mode (subprocess)")

    try:
        requests.get(f"{args.backend_url}/v1/info", timeout=30)
        print(f"✓ Backend ready at {args.backend_url}")
    except Exception as e:
        print(f"✗ Backend not available: {e}")
        return

    print("-" * 60)

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} runs)...")
        for _ in range(args.warmup):
            try:
                if args.standalone_asr:
                    profile_asr(wav_bytes, args.asr_url, language=args.language)
                else:
                    profile_asr(wav_bytes, args.backend_url, embedded=True, language=args.language)
                profile_llm(args.text, args.llm_model, system_prompt=system_prompt)
                profile_tts(args.text, args.backend_url, language=args.language)
            except Exception as e:
                print(f"  Warmup error: {e}")

    # Profile individual components
    print("\n" + "=" * 60)
    print("INDIVIDUAL COMPONENT PROFILING")
    print("=" * 60)

    asr_times = []
    llm_times = []
    tts_times = []
    tts_ttfb = []
    tts_rtf = []

    for i in range(args.runs):
        print(f"\nRun {i+1}/{args.runs}:")

        # ASR
        try:
            if args.standalone_asr:
                text, asr_t = profile_asr(wav_bytes, args.asr_url, language=args.language)
            else:
                text, asr_t = profile_asr(wav_bytes, args.backend_url, embedded=True, language=args.language)
            asr_times.append(asr_t)
            print(f"  ASR:  {asr_t*1000:7.1f}ms  →  \"{text[:50]}...\"" if len(text) > 50 else f"  ASR:  {asr_t*1000:7.1f}ms  →  \"{text}\"")
        except Exception as e:
            print(f"  ASR:  ERROR - {e}")

        # LLM
        try:
            response, llm_t = profile_llm(args.text, args.llm_model, system_prompt=system_prompt)
            llm_times.append(llm_t)
            print(f"  LLM:  {llm_t*1000:7.1f}ms  →  \"{response[:50]}...\"" if len(response) > 50 else f"  LLM:  {llm_t*1000:7.1f}ms  →  \"{response}\"")
        except Exception as e:
            print(f"  LLM:  ERROR - {e}")

        # TTS
        try:
            size, tts_t, ttfb = profile_tts(args.text, args.backend_url, language=args.language)
            tts_times.append(tts_t)
            tts_ttfb.append(ttfb)
            rtf = calc_rtf(size, tts_t)
            tts_rtf.append(rtf)
            print(f"  TTS:  {tts_t*1000:7.1f}ms  (TTFB: {ttfb*1000:.1f}ms, {size/1024:.1f}KB, RTF: {rtf:.2f})")
        except Exception as e:
            print(f"  TTS:  ERROR - {e}")

    # Full pipeline
    print("\n" + "=" * 60)
    print("FULL PIPELINE PROFILING (voice → voice)")
    print("=" * 60)

    full_times = []
    full_ttfb = []
    full_rtf = []

    for i in range(args.runs):
        try:
            size, total, ttfb = profile_full_pipeline(wav_bytes, args.backend_url, language=args.language)
            full_times.append(total)
            full_ttfb.append(ttfb)
            rtf = calc_rtf(size, total)
            full_rtf.append(rtf)
            print(f"  Run {i+1}: {total*1000:7.1f}ms total  (TTFB (Playback): {ttfb*1000:.1f}ms, {size/1024:.1f}KB, RTF: {rtf:.2f})")
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (averages)")
    print("=" * 60)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    asr_avg = avg(asr_times) * 1000
    llm_avg = avg(llm_times) * 1000
    tts_avg = avg(tts_times) * 1000
    tts_ttfb_avg = avg(tts_ttfb) * 1000
    tts_rtf_avg = avg(tts_rtf)
    full_avg = avg(full_times) * 1000
    full_ttfb_avg = avg(full_ttfb) * 1000
    full_rtf_avg = avg(full_rtf)

    total_components = asr_avg + llm_avg + tts_avg

    print(f"\n  Component Breakdown:")
    print(f"  ├─ ASR:     {asr_avg:7.1f}ms  ({asr_avg/total_components*100:5.1f}%)")
    print(f"  ├─ LLM:     {llm_avg:7.1f}ms  ({llm_avg/total_components*100:5.1f}%)")
    print(f"  └─ TTS:     {tts_avg:7.1f}ms  ({tts_avg/total_components*100:5.1f}%)  [TTFB: {tts_ttfb_avg:.1f}ms, RTF: {tts_rtf_avg:.2f}]")
    print(f"  ─────────────────────────")
    print(f"  Sum:        {total_components:7.1f}ms")
    print(f"\n  Full Pipeline:")
    print(f"  └─ Total:   {full_avg:7.1f}ms  [TTFB (Playback): {full_ttfb_avg:.1f}ms, RTF: {full_rtf_avg:.2f}]")

    # Bottleneck analysis
    print("\n  Bottleneck Analysis:")
    components = [("ASR", asr_avg), ("LLM", llm_avg), ("TTS", tts_avg)]
    components.sort(key=lambda x: x[1], reverse=True)
    print(f"  └─ Slowest: {components[0][0]} ({components[0][1]:.1f}ms)")


if __name__ == "__main__":
    main()
