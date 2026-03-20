"""FastAPI application for OVA voice assistant.

IMPORTANT: ASR runs as a standalone script subprocess to ensure vLLM gets
a completely clean CUDA context with proper `if __name__ == '__main__':` guard.
Uses Unix socket for IPC to avoid conflicts with vLLM's stdout usage.
"""
import secrets
import subprocess
import socket
import time
import tempfile
import glob
import uuid
from pathlib import Path
from typing import Optional
import sys
import os

# =============================================================================
# STEP 1: Start ASR as standalone script FIRST, before ANY torch/CUDA imports
# =============================================================================

_asr_process: Optional[subprocess.Popen] = None
_asr_socket: Optional[socket.socket] = None
_asr_socket_path: Optional[str] = None
_asr_lock = __import__('threading').Lock()  # Serializes IPC access to the single ASR subprocess


def _start_asr_subprocess():
    """Start ASR as standalone script with clean CUDA context."""
    global _asr_process, _asr_socket, _asr_socket_path

    # Clean up stale sockets from previous runs (verify socket type to avoid symlink attacks)
    import stat
    for stale in glob.glob(os.path.join(tempfile.gettempdir(), "ova_asr_*.sock")):
        try:
            if stat.S_ISSOCK(os.lstat(stale).st_mode):
                os.unlink(stale)
        except OSError:
            pass

    # Create socket path in temp directory (UUID for uniqueness, avoids PID race conditions)
    _asr_socket_path = os.path.join(tempfile.gettempdir(), f"ova_asr_{uuid.uuid4().hex}.sock")

    # Run asr_server.py as a script - this gives a fresh Python interpreter
    # with proper `if __name__ == '__main__':` guard for vLLM
    _asr_process = subprocess.Popen(
        [sys.executable, "-m", "ova.asr_server", _asr_socket_path],
        stdin=subprocess.DEVNULL,
        stdout=None,  # Let vLLM use stdout freely
        stderr=None,  # Let stderr go to parent's stderr (for logging)
    )

    # Connect to Unix socket (with retries for server startup)
    _asr_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(60):  # 60 second timeout
        try:
            _asr_socket.connect(_asr_socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(1)
    else:
        raise RuntimeError("ASR subprocess socket not available")

    # Wait for ready signal
    response = _asr_recv()
    if response.get("status") == "ready":
        return True
    elif response.get("status") == "error":
        raise RuntimeError(f"ASR subprocess failed to start: {response.get('message')}")
    raise RuntimeError("ASR subprocess failed to signal ready")


def _asr_send(data: dict):
    """Send pickle message to ASR subprocess (faster than JSON for numpy arrays)."""
    import pickle
    msg = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    _asr_socket.sendall(len(msg).to_bytes(4, 'big') + msg)


class _RestrictedUnpickler:
    """Restricted unpickler for ASR IPC - minimal allowlist.

    Only allows types actually used in the ASR protocol:
    - dict: message structure
    - str: action, status, text fields
    - int: sample rate
    - NoneType: optional fields
    - numpy float32 array: audio data
    """
    SAFE_MODULES = {
        'builtins': {'dict', 'str', 'int', 'NoneType'},
        'numpy': {'ndarray', 'dtype', 'float32'},
        'numpy.core.multiarray': {'_reconstruct'},
        'numpy._core.multiarray': {'_reconstruct'},  # numpy 2.0+
        'numpy.core.numeric': {'_frombuffer'},
        'numpy._core.numeric': {'_frombuffer'},  # numpy 2.0+
    }

    @classmethod
    def loads(cls, data: bytes):
        import pickle
        import io

        class Unpickler(pickle.Unpickler):
            def find_class(inner_self, module, name):
                if module in cls.SAFE_MODULES and name in cls.SAFE_MODULES[module]:
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(f"Blocked unsafe pickle class: {module}.{name}")

        return Unpickler(io.BytesIO(data)).load()


_MAX_IPC_MSG_SIZE = 50 * 1024 * 1024  # 50MB


def _asr_recv() -> dict:
    """Receive pickle message from ASR subprocess with restricted unpickling."""
    length_bytes = _asr_socket.recv(4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'big')
    if length > _MAX_IPC_MSG_SIZE:
        raise ValueError(f"IPC message too large: {length} bytes (max {_MAX_IPC_MSG_SIZE})")
    data = b''
    while len(data) < length:
        chunk = _asr_socket.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return _RestrictedUnpickler.loads(data)


def _asr_call(data: dict) -> dict:
    """Thread-safe ASR IPC: send command and receive response atomically.

    The ASR subprocess handles one request at a time on a single socket.
    This lock prevents concurrent callers from interleaving send/recv pairs,
    which would route responses to the wrong caller.
    """
    with _asr_lock:
        _asr_send(data)
        return _asr_recv()


# Start subprocess immediately at import time (before torch is imported anywhere)
_start_asr_subprocess()

# =============================================================================
# STEP 2: NOW safe to import torch/TTS (ASR subprocess already has clean CUDA)
# =============================================================================

import base64

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import json
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import re

from .audio import numpy_to_wav_bytes, create_streaming_wav_header
from .pipeline import (
    OVAPipeline, TTS_ENGINE, LANGUAGE,
    STREAM_FORMAT, KOKORO_VOICE, PCM_PREBUFFER_SAMPLES, DEFAULT_SR,
    KOKORO_VOICES, QWEN3_LANGUAGES,
)
from .prosody import parse_prosody, PauseSegment, TextSegment


# Security: Pattern for valid language/profile names (alphanumeric, underscore, hyphen)
_SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def _is_safe_name(name: str) -> bool:
    """Validate that a name is safe for use in file paths (no traversal)."""
    if not name or not isinstance(name, str):
        return False
    if len(name) > 64:  # Reasonable length limit
        return False
    return bool(_SAFE_NAME_PATTERN.match(name))


def _is_safe_env_value(value: str) -> bool:
    """Validate that a value is safe for .env files (no injection)."""
    if not isinstance(value, str):
        return False
    if len(value) > 256:  # Reasonable length limit
        return False
    # Block newlines, shell metacharacters, and null bytes
    dangerous = {'\n', '\r', '\0', '`', '$', '(', ')', ';', '|', '&', '<', '>', '\\'}
    return not any(c in value for c in dangerous)


from starlette.responses import JSONResponse

from .utils import get_logger, DEBUG

# Component-specific loggers for colored output
logger_sys = get_logger("sys")
logger_api = get_logger("api")
logger_tts = get_logger("tts")
logger_asr = get_logger("asr")

def _log_ttfa(gen, start):
    """Wrap a generator to log TTFA on the first yielded chunk (DEBUG only)."""
    first = True
    for chunk in gen:
        if first:
            logger_api.info(f"TTFA (Playback): {(time.perf_counter() - start)*1000:.0f}ms")
            first = False
        yield chunk

# Early TTS decode: start TTS before full LLM response
EARLY_TTS_DECODE = os.getenv("OVA_EARLY_TTS_DECODE", "true").lower() == "true"

# Barge-in: allow user to interrupt TTS by speaking
BARGE_IN_ENABLED = os.getenv("OVA_ENABLE_BARGE_IN", "true").lower() == "true"
VAD_THRESHOLD = float(os.getenv("OVA_VAD_THRESHOLD", "0.5"))

# Auto-send timing (barge-in recording phase)
AUTO_SEND_SILENCE_MS = int(os.getenv("OVA_AUTO_SEND_SILENCE_MS", "1200"))
AUTO_SEND_CONFIRM_MS = int(os.getenv("OVA_AUTO_SEND_CONFIRM_MS", "128"))
AUTO_SEND_TIMEOUT_MS = int(os.getenv("OVA_AUTO_SEND_TIMEOUT_MS", "3000"))

# Barge-in cooldown: suppress barge-in for N ms after auto-send fires
BARGE_IN_COOLDOWN_MS = int(os.getenv("OVA_BARGE_IN_COOLDOWN_MS", "1200"))

# VAD confirm frames: number of consecutive Silero frames (~32ms each) to confirm speech
VAD_CONFIRM_FRAMES = int(os.getenv("OVA_VAD_CONFIRM_FRAMES", "2"))

# Barge-in grace period: delay VAD start after playback begins (ms) to avoid echo false positives
BARGE_IN_GRACE_MS = int(os.getenv("OVA_BARGE_IN_GRACE_MS", "500"))

# Backchannel filter: discard filler words (yeah, ok, mmm) during barge-in
BACKCHANNEL_FILTER_ENABLED = os.getenv("OVA_BACKCHANNEL_FILTER", "false").lower() == "true"

# Wake word detection (client-side, frontend-only)
WAKE_WORD_ENABLED = os.getenv("OVA_ENABLE_WAKE_WORD", "false").lower() == "true"
WAKE_WORD = os.getenv("OVA_WAKE_WORD", "hey nova").strip().lower()

# Full-duplex mode
DUPLEX_ENABLED = os.getenv("OVA_ENABLE_DUPLEX", "false").lower() == "true"
DUPLEX_SILENCE_TIMEOUT_MS = int(os.getenv("OVA_DUPLEX_SILENCE_TIMEOUT_MS", "800"))
DUPLEX_BOT_STOP_DELAY_MS = int(os.getenv("OVA_DUPLEX_BOT_STOP_DELAY_MS", "500"))
DUPLEX_BACKCHANNEL_TIMEOUT_MS = int(os.getenv("OVA_DUPLEX_BACKCHANNEL_TIMEOUT_MS", "500"))
DUPLEX_VAD_THRESHOLD = float(os.getenv("OVA_DUPLEX_VAD_THRESHOLD", "0.5"))
DUPLEX_VAD_CONFIRM_MS = int(os.getenv("OVA_DUPLEX_VAD_CONFIRM_MS", "64"))
DUPLEX_VAD_SILENCE_MS = int(os.getenv("OVA_DUPLEX_VAD_SILENCE_MS", "320"))
DUPLEX_INACTIVITY_TIMEOUT_S = int(os.getenv("OVA_DUPLEX_INACTIVITY_TIMEOUT_S", "300"))
DUPLEX_INTERRUPT_COOLDOWN_MS = int(os.getenv("OVA_DUPLEX_INTERRUPT_COOLDOWN_MS", "0"))

# API key authentication (optional - empty means disabled)
API_KEY = os.getenv("OVA_API_KEY", "").strip()

# Security: Input size limits (0 = disabled)
MAX_TEXT_LENGTH = int(os.getenv("OVA_MAX_TEXT_LENGTH", "4096"))      # characters
MAX_IMAGE_SIZE = int(os.getenv("OVA_MAX_IMAGE_SIZE", "20971520"))    # 20MB
MAX_AUDIO_SIZE = int(os.getenv("OVA_MAX_AUDIO_SIZE", "20971520"))    # 20MB

# Security: Valid values for settings (configurable via env)
VALID_TTS_ENGINES = set(os.getenv("OVA_SUPPORTED_TTS_ENGINES", "qwen3,kokoro").split(","))
VALID_LANGUAGES = set(os.getenv("OVA_SUPPORTED_LANGUAGES", "es,en,fr,de,it,pt,ja,zh,ko,ru,hi").split(","))
VALID_STREAM_FORMATS = set(os.getenv("OVA_SUPPORTED_STREAM_FORMATS", "pcm,wav").split(","))

# Security: Disable restart endpoint in production
RESTART_ENDPOINT_DISABLED = os.getenv("OVA_DISABLE_RESTART_ENDPOINT", "false").lower() == "true"

# Security: Disable frontend settings panel
FRONTEND_SETTINGS_DISABLED = os.getenv("OVA_DISABLE_FRONTEND_SETTINGS", "false").lower() == "true"

# Security: Disable multimodal (image) input
MULTIMODAL_DISABLED = os.getenv("OVA_DISABLE_MULTIMODAL", "false").lower() == "true"


class TextChatRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH or 4096)
    image: Optional[str] = Field(default=None, max_length=MAX_IMAGE_SIZE or 20_971_520)


class TTSRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH or 4096)


class SettingsUpdate(BaseModel):
    language: Optional[str] = None
    tts_engine: Optional[str] = None
    voice: Optional[str] = None
    stream_format: Optional[str] = None
    conversation_mode: Optional[str] = None  # "half-duplex" or "full-duplex"


class BatchTTSItem(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH or 4096)
    voice: Optional[str] = None
    language: Optional[str] = None


class BatchTTSRequest(BaseModel):
    items: list[BatchTTSItem] = Field(min_length=1, max_length=64)


# Server configuration
BACKEND_HOST = os.getenv("OVA_BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("OVA_BACKEND_PORT", "5173")
FRONTEND_HOST = os.getenv("OVA_FRONTEND_HOST", "localhost")
FRONTEND_PORT = os.getenv("OVA_FRONTEND_PORT", "8080")

# Security: warn when binding non-loopback without API key
if BACKEND_HOST not in ("localhost", "127.0.0.1") and not API_KEY:
    logger_sys.warning(
        "SECURITY WARNING: OVA is binding to %s without OVA_API_KEY set. "
        "All endpoints are unauthenticated. Set OVA_API_KEY in .env for non-localhost deployments.",
        BACKEND_HOST,
    )


def _build_cors_origins():
    """Build CORS origins list, handling 0.0.0.0 (bind all interfaces).

    When binding to 0.0.0.0, browsers connect via localhost or real IP,
    so we add localhost and 127.0.0.1 as allowed origins.
    """
    origins = set()
    for host, port in [(BACKEND_HOST, BACKEND_PORT), (FRONTEND_HOST, FRONTEND_PORT)]:
        origins.add(f"http://{host}:{port}")
        # 0.0.0.0 binds all interfaces - browsers connect via localhost or real IP
        if host == "0.0.0.0":
            origins.add(f"http://localhost:{port}")
            origins.add(f"http://127.0.0.1:{port}")
    return list(origins)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Stream-Supported", "X-Early-TTS"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Optional API key authentication. When OVA_API_KEY is set, all requests must
    include Authorization: Bearer <key>. When unset, all requests pass through."""
    if not API_KEY:
        return await call_next(request)
    # Allow CORS preflight through — OPTIONS never carries Authorization
    if request.method == "OPTIONS":
        return await call_next(request)
    # Allow browser requests from trusted frontend origins (CORS restricts browser access).
    # Sec-Fetch-Site is set automatically by browsers and cannot be omitted by browser fetch/XHR.
    # Non-browser clients (curl, Python requests) don't send it by default, so spoofing
    # Origin alone is not sufficient to bypass auth.
    origin = request.headers.get("origin", "")
    sec_fetch_site = request.headers.get("sec-fetch-site", "")
    if origin and origin in _build_cors_origins() and sec_fetch_site:
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if secrets.compare_digest(auth, f"Bearer {API_KEY}"):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"error": "Unauthorized"})


# Pass ASR IPC functions to pipeline (no direct ASR init in main process)
pipeline = OVAPipeline(asr_send=_asr_send, asr_recv=_asr_recv, asr_call=_asr_call)

def _validate_voice_language(voice: str = None, language: str = None) -> Optional[Response]:
    """Validate voice/language params. Returns error Response or None."""
    if language and language not in VALID_LANGUAGES:
        return Response(status_code=400, content=f"Invalid language: {language}. Valid: {sorted(VALID_LANGUAGES)}")
    if voice and pipeline.tts_engine == "qwen3":
        found = any(voice in voices for voices in pipeline.all_voice_prompts.values())
        if not found:
            all_voices = sorted({v for voices in pipeline.all_voice_prompts.values() for v in voices})
            return Response(status_code=400, content=f"Voice '{voice}' not found. Available: {all_voices}")
    return None


def _tools_active() -> bool:
    """Check if tool/function calling is active on the pipeline."""
    return pipeline.tool_registry.enabled


# Server ready flag - set after all warmups complete
_server_ready = False


@app.on_event("startup")
async def startup_event():
    """Warmup ASR and TTS in worker thread pool (required for CUDA graph TLS)."""
    global _server_ready
    import asyncio
    import anyio
    # Store event loop for thread-safe EventBus publishing
    from .events import event_bus
    event_bus.set_loop(asyncio.get_running_loop())
    # Warmup ASR in subprocess via IPC
    await anyio.to_thread.run_sync(pipeline.warmup_asr)
    await anyio.to_thread.run_sync(pipeline.warmup_llm)
    # Run TTS warmup in the same thread pool that handles sync generators
    await anyio.to_thread.run_sync(pipeline.warmup_tts)
    _server_ready = True
    logger_api.info("Server ready - all warmups complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown ASR subprocess, TTS, and unload LLM from Ollama."""
    # Clean up TTS first (release GPU memory before ASR shutdown)
    try:
        pipeline.cleanup()
    except Exception as e:
        logger_sys.warning(f"TTS cleanup failed: {e}")

    # Check for restart marker - if present, keep LLM loaded for faster restart
    keep_llm_marker = Path(__file__).parent.parent / ".ova" / "keep_llm"
    if keep_llm_marker.exists():
        keep_llm_marker.unlink()
    else:
        # Unload LLM from VRAM (provider handles specifics)
        try:
            pipeline.llm_provider.unload()
        except Exception:
            pass

    global _asr_process, _asr_socket, _asr_socket_path
    if _asr_socket:
        try:
            _asr_send({"action": "shutdown"})
            _asr_socket.close()
        except (OSError, socket.error):
            pass
    if _asr_process and _asr_process.poll() is None:
        try:
            _asr_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _asr_process.terminate()
    # Clean up socket file
    if _asr_socket_path and os.path.exists(_asr_socket_path):
        try:
            os.unlink(_asr_socket_path)
        except OSError:
            pass


def _generate_pcm_for_segments(segments, first_chunk_state, voice=None, language=None, first_chunk_emit_every=None):
    """Generate PCM audio for a list of prosody segments.

    Args:
        segments: List of TextSegment and PauseSegment from parse_prosody().
        first_chunk_state: Dict with mutable state: {'first': bool, 'sr': int}.
            Updated in-place so callers track WAV header and sample rate across calls.
        voice: Optional voice name override for TTS.
        language: Optional language code override for TTS.
        first_chunk_emit_every: Optional override for Phase 1 emit interval (0 to disable).

    Yields:
        PCM bytes (int16) for each segment. Yields WAV header before first audio.
    """
    MIN_CHUNK_BYTES = 2048

    for segment in segments:
        if pipeline._interrupt.is_set():
            return

        if isinstance(segment, PauseSegment):
            sr = first_chunk_state['sr']
            if first_chunk_state['first']:
                yield create_streaming_wav_header(sr)
                first_chunk_state['first'] = False
            silence = np.zeros(int(sr * segment.duration_sec), dtype=np.float32)
            yield _audio_to_pcm(silence)

        elif isinstance(segment, TextSegment):
            pcm_buffer = b""
            for chunk, sr in pipeline.tts_streaming(segment.text, voice=voice, language=language, first_chunk_emit_every=first_chunk_emit_every):
                first_chunk_state['sr'] = sr
                if first_chunk_state['first']:
                    yield create_streaming_wav_header(sr)
                    first_chunk_state['first'] = False

                pcm = _audio_to_pcm(chunk)
                pcm_buffer += pcm

                if len(pcm_buffer) >= MIN_CHUNK_BYTES:
                    yield pcm_buffer
                    pcm_buffer = b""

            if pcm_buffer:
                yield pcm_buffer


def generate_audio_stream(text: str, voice=None, language=None):
    """
    Generator that yields audio.
    pcm mode: WAV header once + raw PCM chunks (true streaming)
    wav mode: Each chunk as complete WAV file (streaming)
    Fallback: Complete WAV file (non-streaming, if streaming not supported)
    """
    try:
        # Clear stale interrupt from previous request
        pipeline._interrupt.clear()

        if pipeline.supports_streaming and STREAM_FORMAT == "pcm":
            segments = parse_prosody(text)
            state = {'first': True, 'sr': DEFAULT_SR}

            for pcm_bytes in _generate_pcm_for_segments(segments, state, voice=voice, language=language):
                yield pcm_bytes

        elif pipeline.supports_streaming:
            # WAV streaming: each chunk as complete WAV file
            chunks_yielded = 0

            for chunk, sr in pipeline.tts_streaming(text, voice=voice, language=language):
                wav_bytes = numpy_to_wav_bytes(chunk, sr)
                yield wav_bytes
                chunks_yielded += 1

            logger_tts.info(f"WAV streaming completed: {chunks_yielded} chunks")

        else:
            # Non-streaming fallback: single complete WAV
            wav_bytes = pipeline.tts(text, voice=voice, language=language)
            yield wav_bytes
            logger_tts.info("WAV TTS completed (non-streaming)")

    except Exception as e:
        logger_tts.error(f"TTS failed: {e}")
        raise


def _clean_markdown(text: str) -> str:
    """Remove markdown artifacts and system tags for TTS."""
    import re
    text = text.replace("**", "").replace("_", "").replace("__", "").replace("#", "").replace("...", ",").replace("--", ",")
    text = text.replace("[interrupted]", "").replace("[interrupted before speaking]", "")
    text = re.sub(r'\[your full unsaid response was:.*?\]', '', text, flags=re.DOTALL)
    return text.strip()


def _audio_to_pcm(audio_chunk) -> bytes:
    """Convert audio chunk to int16 PCM bytes."""
    if audio_chunk.dtype == np.int16:
        audio_chunk = audio_chunk.astype(np.float32) / 32768.0
    else:
        audio_chunk = audio_chunk.astype(np.float32)
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    return (audio_chunk * 32767.0).astype(np.int16).tobytes()


def generate_interleaved_audio_stream(input_text: str, image: str = None, voice=None, language=None):
    """
    Single-kick early TTS with natural continuation and prosody support.

    Gating: Start TTS once when ANY condition met:
    - Sentence boundary (. ? ! \\n)
    - Pause tag detected ([pause:X] / [p:X])
    - OR buffer >= 40 chars
    - OR token count >= 12

    Then lock - early chunk is synthesized directly, remainder goes through
    prosody parsing to handle any [pause:X] tags.

    Args:
        input_text: User input text (transcribed audio or direct text)
        image: Optional base64-encoded image for vision models
        voice: Optional voice name override for TTS
        language: Optional language code override for TTS
    """
    from .prosody import _PAUSE_PATTERN, MAX_PAUSE_DURATION
    _PARTIAL_PAUSE = re.compile(r'\[\s*(?:pause|p)[\s:.\d]*$', re.IGNORECASE)

    try:
        # Clear stale interrupt from previous request
        pipeline._interrupt.clear()
        tts_sent_text = ""

        # When tools are active, use non-streaming LLM with tool loop,
        # then stream the final text response through TTS
        if _tools_active():
            chat_response = pipeline.chat_with_tools(input_text, image=image)
            segments = parse_prosody(_clean_markdown(chat_response))
            state = {'first': True, 'sr': DEFAULT_SR}
            for pcm_bytes in _generate_pcm_for_segments(segments, state, voice=voice, language=language):
                yield pcm_bytes
            return

        state = {'first': True, 'sr': DEFAULT_SR}
        start = time.perf_counter()

        # Gating state
        SENTENCE_END = {".", "?", "!", "\n"}
        MIN_CHARS = 40
        MIN_TOKENS = 12

        buffer = ""
        token_count = 0
        tts_started = False
        remainder = ""

        # Iterate raw tokens
        for token in pipeline.chat_streaming_tokens(input_text, image=image):
            # Check interrupt flag — break exits the for loop, which closes
            # chat_streaming_tokens (completed=False → [interrupted] in context)
            if pipeline._interrupt.is_set():
                logger_tts.info("Interleaved stream interrupted during token loop")
                break

            buffer += token
            token_count += 1

            if not tts_started:
                # Check if a pause tag just completed in the buffer
                pause_match = _PAUSE_PATTERN.search(buffer)

                # Don't trigger gating if buffer contains incomplete pause tag
                maybe_in_pause = bool(_PARTIAL_PAUSE.search(buffer))

                # Check gating conditions (pause tag acts as a trigger too)
                should_start = (
                    pause_match
                    or (any(ch in SENTENCE_END for ch in token) and not maybe_in_pause)
                    or (len(buffer) >= MIN_CHARS and not maybe_in_pause)
                    or (token_count >= MIN_TOKENS and not maybe_in_pause)
                )

                if should_start:
                    tts_started = True

                    if pause_match:
                        # Split at pause tag: text before → early TTS, pause → silence, rest → remainder prefix
                        early_text = _clean_markdown(buffer[:pause_match.start()])
                        pause_dur = min(float(pause_match.group(1)), MAX_PAUSE_DURATION)
                        after_pause = buffer[pause_match.end():]

                        logger_tts.debug(f"Early TTS kick (pause tag) @ {(time.perf_counter()-start)*1000:.0f}ms: '{early_text[:50]}...'")

                        # Synthesize text before the pause
                        if early_text:
                            for pcm_bytes in _generate_pcm_for_segments([TextSegment(text=early_text)], state, voice=voice, language=language):
                                yield pcm_bytes
                            tts_sent_text += early_text + " "

                        # Insert pause silence
                        if pause_dur > 0:
                            for pcm_bytes in _generate_pcm_for_segments([PauseSegment(duration_sec=pause_dur)], state, voice=voice, language=language):
                                yield pcm_bytes

                        # Any text after the pause tag goes to remainder
                        remainder = after_pause
                    else:
                        # Normal gating (sentence boundary / buffer size)
                        early_text = _clean_markdown(buffer)
                        logger_tts.debug(f"Early TTS kick @ {(time.perf_counter()-start)*1000:.0f}ms, {token_count} tokens: '{early_text[:50]}...'")

                        for pcm_bytes in _generate_pcm_for_segments([TextSegment(text=early_text)], state, voice=voice, language=language):
                            yield pcm_bytes
                        tts_sent_text += early_text + " "
            else:
                # TTS already kicked - accumulate remainder
                remainder += token

        # Handle case where LLM finished before trigger
        if not tts_started and buffer and not pipeline._interrupt.is_set():
            early_text = _clean_markdown(buffer)
            logger_tts.debug(f"Late TTS kick (LLM exhausted) @ {(time.perf_counter()-start)*1000:.0f}ms: '{early_text[:50]}...'")
            segments = parse_prosody(early_text)
            for pcm_bytes in _generate_pcm_for_segments(segments, state, voice=voice, language=language):
                yield pcm_bytes
            tts_sent_text += early_text + " "

        # Speak remainder through prosody parsing (handles any [pause:X] tags)
        remainder_clean = _clean_markdown(remainder)
        if remainder_clean and not pipeline._interrupt.is_set():
            logger_tts.debug(f"Remainder TTS @ {(time.perf_counter()-start)*1000:.0f}ms: '{remainder_clean[:50]}...'")
            segments = parse_prosody(remainder_clean)
            for pcm_bytes in _generate_pcm_for_segments(segments, state, voice=voice, language=language):
                yield pcm_bytes
            if not pipeline._interrupt.is_set():
                tts_sent_text += remainder_clean

        logger_tts.debug(f"Two-phase TTS complete")

    except Exception as e:
        logger_tts.error(f"Interleaved TTS failed: {e}")
        return
    finally:
        if pipeline._interrupt.is_set() and tts_sent_text:
            pipeline.correct_interrupted_context(tts_sent_text)
        if tts_sent_text:
            logger_api.debug(f"TTS sent text before exit: {tts_sent_text[:100]}...")


@app.post("/v1/chat/{voice_id}/audio", response_class=Response)
@app.post("/v1/chat/audio", response_class=Response)
async def chat_request_handler(request: Request, voice_id: str | None = None, language: str | None = None):
    """
    Audio chat endpoint. Always streams when Qwen3 TTS is active.

    Path params:
        voice_id: optional voice name (sticky — changes session)
    Query params:
        language: optional language code (sticky — changes session)

    Returns:
        - Streaming: chunked audio/wav responses
        - Non-streaming: single audio/wav response (fallback)
    """
    request_start = time.perf_counter()
    voice = voice_id
    err = _validate_voice_language(voice, language)
    if err:
        return err

    audio_in = await request.body()

    # Security: Validate audio input size
    if MAX_AUDIO_SIZE and len(audio_in) > MAX_AUDIO_SIZE:
        return Response(status_code=413, content="Audio input too large")

    # Sticky: switch session if language/voice changed
    if language and language != pipeline.language:
        if pipeline.tts_engine == "qwen3":
            pipeline.switch_language(language, voice)
        else:
            pipeline.switch_kokoro_language(language)
            if voice:
                pipeline.switch_kokoro_voice(voice)
    elif voice and voice != pipeline.current_voice:
        if pipeline.tts_engine == "qwen3":
            pipeline.switch_voice(voice)
        else:
            pipeline.switch_kokoro_voice(voice)

    transcribed_text = pipeline.transcribe(audio_in)

    if not transcribed_text:
        return Response(content=bytes(), media_type="audio/wav")

    if pipeline.supports_streaming:
        if EARLY_TTS_DECODE and STREAM_FORMAT == "pcm" and not _tools_active():
            logger_api.info(f"Using early TTS decode for: {transcribed_text[:50]}... (lang: {pipeline.language})")
            gen = generate_interleaved_audio_stream(transcribed_text, voice=voice, language=language)
            gen = _log_ttfa(gen, request_start)
            return StreamingResponse(
                gen,
                media_type="audio/wav",
                headers={"X-Stream-Supported": "true", "X-Early-TTS": "true"}
            )
        else:
            chat_response = pipeline.chat_with_tools(transcribed_text) if _tools_active() else pipeline.chat(transcribed_text)
            logger_api.info(f"Using streaming TTS for: {chat_response[:50]}... (lang: {pipeline.language})")
            gen = generate_audio_stream(chat_response, voice=voice, language=language)
            gen = _log_ttfa(gen, request_start)
            return StreamingResponse(
                gen,
                media_type="audio/wav",
                headers={"X-Stream-Supported": "true"}
            )
    else:
        # Non-streaming fallback
        import anyio
        def _chat_and_tts():
            resp = pipeline.chat_with_tools(transcribed_text) if _tools_active() else pipeline.chat(transcribed_text)
            return pipeline.tts(resp, voice=voice, language=language)
        audio_out = await anyio.to_thread.run_sync(_chat_and_tts)
        return Response(
            content=audio_out,
            media_type="audio/wav",
            headers={"X-Stream-Supported": "false"}
        )


@app.post("/v1/chat/{voice_id}", response_class=Response)
@app.post("/v1/chat", response_class=Response)
async def text_chat_handler(request: Request, body: TextChatRequest, voice_id: str | None = None, language: str | None = None):
    """
    Text/image chat endpoint. Accepts JSON with text and optional base64 image.
    Returns voice response (TTS). Always streams when Qwen3 TTS is active.

    Path params:
        voice_id: optional voice name (sticky — changes session)
    Query params:
        language: optional language code (sticky — changes session)
    Body:
        text: string - The message text
        image: string (optional) - Base64 encoded image (data URL format)
    """
    request_start = time.perf_counter()
    voice = voice_id
    err = _validate_voice_language(voice, language)
    if err:
        return err

    # Security: Validate input sizes
    if MAX_TEXT_LENGTH and len(body.text) > MAX_TEXT_LENGTH:
        return Response(status_code=413, content="Text input too large")
    if MAX_IMAGE_SIZE and body.image and len(body.image) > MAX_IMAGE_SIZE:
        return Response(status_code=413, content="Image input too large")

    # Security: Block image input if multimodal is disabled
    if MULTIMODAL_DISABLED and body.image:
        return Response(status_code=403, content="Image input is disabled")

    # Sticky: switch session if language/voice changed
    if language and language != pipeline.language:
        if pipeline.tts_engine == "qwen3":
            pipeline.switch_language(language, voice)
        else:
            pipeline.switch_kokoro_language(language)
            if voice:
                pipeline.switch_kokoro_voice(voice)
    elif voice and voice != pipeline.current_voice:
        if pipeline.tts_engine == "qwen3":
            pipeline.switch_voice(voice)
        else:
            pipeline.switch_kokoro_voice(voice)

    text = body.text.strip()
    image_data = body.image

    if not text and not image_data:
        return Response(content=bytes(), media_type="audio/wav")

    # Extract base64 data from data URL if present
    image_base64 = None
    if image_data:
        if image_data.startswith("data:"):
            # Format: data:image/jpeg;base64,/9j/4AAQ...
            try:
                image_base64 = image_data.split(",", 1)[1]
            except IndexError:
                image_base64 = None
        else:
            image_base64 = image_data

    if pipeline.supports_streaming:
        if EARLY_TTS_DECODE and STREAM_FORMAT == "pcm" and not _tools_active():
            logger_api.info(f"Using early TTS decode for: {text[:50]}... (image: {bool(image_base64)}, lang: {pipeline.language})")
            gen = generate_interleaved_audio_stream(text, image=image_base64, voice=voice, language=language)
            gen = _log_ttfa(gen, request_start)
            return StreamingResponse(
                gen,
                media_type="audio/wav",
                headers={"X-Stream-Supported": "true", "X-Early-TTS": "true"}
            )
        else:
            chat_response = pipeline.chat_with_tools(text, image=image_base64) if _tools_active() else pipeline.chat(text, image=image_base64)
            logger_api.info(f"Using streaming TTS for: {chat_response[:50]}... (lang: {pipeline.language})")
            gen = generate_audio_stream(chat_response, voice=voice, language=language)
            gen = _log_ttfa(gen, request_start)
            return StreamingResponse(
                gen,
                media_type="audio/wav",
                headers={"X-Stream-Supported": "true"}
            )
    else:
        # Non-streaming fallback
        import anyio
        logger_api.info(f"Text chat: '{text[:50]}...' (image: {bool(image_base64)}, lang: {pipeline.language})")
        def _chat_and_tts():
            resp = pipeline.chat_with_tools(text, image=image_base64) if _tools_active() else pipeline.chat(text, image=image_base64)
            return pipeline.tts(resp, voice=voice, language=language)
        audio_out = await anyio.to_thread.run_sync(_chat_and_tts)
        return Response(
            content=audio_out,
            media_type="audio/wav",
            headers={"X-Stream-Supported": "false"}
        )


@app.post("/v1/text-to-speech/batch")
async def batch_tts_handler(body: BatchTTSRequest):
    """Batch text-to-speech: synthesize multiple texts (potentially with different voices) in one call.

    Returns NDJSON where each line is a completed item:
        {"index": 0, "status": "ok", "voice": "samantha", "language": "en", "audio_b64": "UklGR..."}

    All items are processed in a single batched pass through the transformer.
    """
    import anyio

    if pipeline.tts_engine != "qwen3":
        return Response(status_code=400, content="Batch TTS requires Qwen3 engine")

    # Validate items
    for i, item in enumerate(body.items):
        text = item.text.strip()
        if not text:
            return Response(status_code=400, content=f"Item {i} has empty text")
        if item.voice and not _is_safe_name(item.voice):
            return Response(status_code=400, content=f"Item {i} has invalid voice name")
        if item.language and not _is_safe_name(item.language):
            return Response(status_code=400, content=f"Item {i} has invalid language name")

    def generate_batch_tts_ndjson():
        batch_items = [
            {"text": item.text.strip(), "voice": item.voice, "language": item.language}
            for item in body.items
        ]

        results = pipeline.tts_batch(batch_items)

        lines = []
        for i, (wav_bytes, voice, language) in enumerate(results):
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            line = json.dumps({
                "index": i,
                "status": "ok",
                "audio_b64": audio_b64,
                "voice": body.items[i].voice,
                "language": body.items[i].language,
            })
            lines.append(line.encode() + b"\n")
        return lines

    # Run the blocking TTS generation in a worker thread and collect results
    ndjson_lines = await anyio.to_thread.run_sync(generate_batch_tts_ndjson)
    return Response(
        content=b"".join(ndjson_lines),
        media_type="application/x-ndjson",
    )


@app.post("/v1/text-to-speech/{voice_id}", response_class=Response)
@app.post("/v1/text-to-speech", response_class=Response)
async def tts_handler(request: Request, body: TTSRequest, voice_id: str | None = None, language: str | None = None):
    """Pure text-to-speech. Synthesizes text exactly as given (no LLM).

    Path params:
        voice_id: optional voice name (non-sticky, per-request only)
    Query params:
        language: optional language code (non-sticky, per-request only)
    Body:
        text: string - The text to synthesize
    """
    request_start = time.perf_counter()
    voice = voice_id
    err = _validate_voice_language(voice, language)
    if err:
        return err

    if MAX_TEXT_LENGTH and len(body.text) > MAX_TEXT_LENGTH:
        return Response(status_code=413, content="Text input too large")

    text = body.text.strip()
    if not text:
        return Response(content=bytes(), media_type="audio/wav")

    if pipeline.supports_streaming:
        logger_api.info(f"TTS (pure): streaming '{text[:50]}...'")
        gen = generate_audio_stream(text, voice=voice, language=language)
        gen = _log_ttfa(gen, request_start)
        return StreamingResponse(
            gen,
            media_type="audio/wav",
            headers={"X-Stream-Supported": "true"}
        )
    else:
        # Non-streaming fallback
        import anyio
        logger_api.info(f"TTS (pure): non-streaming '{text[:50]}...'")
        audio_out = await anyio.to_thread.run_sync(
            lambda: pipeline.tts(text, voice=voice, language=language)
        )
        return Response(
            content=audio_out,
            media_type="audio/wav",
            headers={"X-Stream-Supported": "false"}
        )


@app.post("/v1/speech-to-text")
async def transcribe_audio(request: Request, language: str | None = None):
    """Standalone speech-to-text. Accepts raw WAV bytes, returns transcribed text.

    Query params:
        language: optional language code override for ASR (e.g., 'es', 'en')

    Returns:
        JSON: {"text": "transcribed text"}
    """
    import anyio

    audio_in = await request.body()

    if MAX_AUDIO_SIZE and len(audio_in) > MAX_AUDIO_SIZE:
        return Response(status_code=413, content="Audio input too large")

    text = await anyio.to_thread.run_sync(
        lambda: pipeline.transcribe(audio_in, language=language)
    )

    return {"text": text or ""}


@app.get("/v1/health")
async def ready():
    """Check if server is fully initialized (all warmups complete)."""
    if not _server_ready:
        return Response(status_code=503, content="Server warming up")
    return {"ready": True}


@app.post("/v1/interrupt")
async def interrupt_endpoint():
    """Signal the pipeline to stop current TTS generation."""
    pipeline.interrupt()
    return {"status": "interrupted"}


@app.get("/v1/context")
async def get_context():
    """Return current conversation context (debug/testing only)."""
    if not DEBUG:
        return JSONResponse(status_code=404, content={"error": "Debug mode not enabled"})
    return {"context": pipeline.context}


@app.get("/v1/info")
async def info():
    """Return info about the current pipeline configuration."""
    info = {
        "voice": getattr(pipeline, 'current_voice', None),
        "tts_engine": TTS_ENGINE,
        "llm_provider": os.getenv("OVA_LLM_PROVIDER", "ollama"),
        "language": LANGUAGE,
        "supports_streaming": pipeline.supports_streaming,
        "pcm_prebuffer_samples": PCM_PREBUFFER_SAMPLES,
        "early_tts_decode": EARLY_TTS_DECODE,
        "frontend_settings_disabled": FRONTEND_SETTINGS_DISABLED,
        "multimodal_disabled": MULTIMODAL_DISABLED,
        "barge_in_enabled": BARGE_IN_ENABLED,
        "vad_threshold": VAD_THRESHOLD,
        "auto_send_silence_ms": AUTO_SEND_SILENCE_MS,
        "auto_send_confirm_ms": AUTO_SEND_CONFIRM_MS,
        "auto_send_timeout_ms": AUTO_SEND_TIMEOUT_MS,
        "barge_in_cooldown_ms": BARGE_IN_COOLDOWN_MS,
        "vad_confirm_frames": VAD_CONFIRM_FRAMES,
        "barge_in_grace_ms": BARGE_IN_GRACE_MS,
        "backchannel_filter_enabled": BACKCHANNEL_FILTER_ENABLED,
        "wake_word_enabled": WAKE_WORD_ENABLED,
        "wake_word": WAKE_WORD,
        "duplex_enabled": DUPLEX_ENABLED,
        "debug": DEBUG,
    }

    # Only expose tool/MCP internals in debug mode
    if DEBUG:
        info["tools_enabled"] = pipeline.tool_registry.enabled
        info["tools_available"] = pipeline.tool_registry.get_tool_names()
        info["mcp_enabled"] = pipeline._mcp_manager is not None
        info["mcp_servers"] = pipeline._mcp_manager.get_status() if pipeline._mcp_manager else {}

    return info


@app.get("/v1/settings")
async def get_settings():
    """Return current settings and available profiles with their prompts."""
    from .pipeline import LANGUAGE_NAMES

    profiles_dir = Path(__file__).parent.parent / "profiles"
    prompts_dir = Path(__file__).parent.parent / "prompts"
    available_profiles = {}
    default_prompts = {}

    for lang_dir in profiles_dir.iterdir():
        if lang_dir.is_dir():
            lang = lang_dir.name
            profiles = {}
            for p in lang_dir.iterdir():
                if p.is_dir() and ((p / "ref_audio.wav").exists() or list(p.glob("voice_clone_prompt_*.pt"))):
                    # Load prompt for this profile
                    prompt_file = p / "prompt.txt"
                    if prompt_file.exists():
                        prompt = prompt_file.read_text(encoding="utf-8").strip()
                    else:
                        default = prompts_dir / lang / "default.txt"
                        if default.exists():
                            prompt = default.read_text(encoding="utf-8").strip()
                        else:
                            prompt = f"You are a helpful assistant. Always respond in {LANGUAGE_NAMES.get(lang, lang)}."
                    profiles[p.name] = {"prompt": prompt}
            if profiles:
                available_profiles[lang] = profiles

    # Load default prompts only for Kokoro (no voice profiles)
    if TTS_ENGINE == "kokoro":
        for lang_dir in prompts_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                default_file = lang_dir / "default.txt"
                if default_file.exists():
                    default_prompts[lang] = default_file.read_text(encoding="utf-8").strip()
                else:
                    default_prompts[lang] = f"You are a helpful assistant. Always respond in {LANGUAGE_NAMES.get(lang, lang)}."

    # Build normalized voice/language lists for discovery API
    if TTS_ENGINE == "qwen3":
        available_languages = sorted(available_profiles.keys())
        available_voices = {
            lang: [{"id": name, "name": name} for name in sorted(profiles.keys())]
            for lang, profiles in available_profiles.items()
        }
    else:
        available_languages = sorted(k for k, v in KOKORO_VOICES.items() if v)
        available_voices = KOKORO_VOICES

    qwen3_languages = [{"code": l, "name": LANGUAGE_NAMES[l]} for l in QWEN3_LANGUAGES]
    kokoro_languages = [{"code": l, "name": LANGUAGE_NAMES[l]} for l in sorted(k for k, v in KOKORO_VOICES.items() if v)]

    return {
        "current": {
            "language": pipeline.language,
            "tts_engine": TTS_ENGINE,
            "voice": pipeline.current_voice,
            "stream_format": STREAM_FORMAT,
            "system_prompt": pipeline.system_prompt,
            "conversation_mode": "full-duplex" if DUPLEX_ENABLED else "half-duplex",
        },
        "profiles": available_profiles,
        "default_prompts": default_prompts,
        "languages": available_languages,
        "voices": available_voices,
        "kokoro_voices": KOKORO_VOICES,
        "qwen3_languages": qwen3_languages,
        "kokoro_languages": kokoro_languages,
    }


@app.post("/v1/settings")
async def update_settings(settings: SettingsUpdate):
    """Update settings in .env file."""
    if FRONTEND_SETTINGS_DISABLED:
        return JSONResponse(status_code=403, content={"error": "Settings are disabled"})

    env_path = Path(__file__).parent.parent / ".env"

    env_mapping = {
        "language": "OVA_LANGUAGE",
        "tts_engine": "OVA_TTS_ENGINE",
        "voice": "OVA_QWEN3_VOICE" if TTS_ENGINE == "qwen3" else "OVA_KOKORO_VOICE",
        "stream_format": "OVA_STREAM_FORMAT",
        "conversation_mode": "OVA_ENABLE_DUPLEX",
    }

    current = {
        "language": pipeline.language,
        "tts_engine": TTS_ENGINE,
        "voice": pipeline.current_voice,
        "stream_format": STREAM_FORMAT,
        "conversation_mode": "full-duplex" if DUPLEX_ENABLED else "half-duplex",
    }

    changes = {}
    for key, env_var in env_mapping.items():
        new_val = getattr(settings, key, None)
        if new_val is not None and new_val != current.get(key):
            changes[env_var] = new_val

    if not changes:
        return {"restart_required": False, "message": "No changes"}

    # Translate conversation_mode to boolean env value
    if "OVA_ENABLE_DUPLEX" in changes:
        changes["OVA_ENABLE_DUPLEX"] = "true" if changes["OVA_ENABLE_DUPLEX"] == "full-duplex" else "false"

    # Security: Validate all values before writing to .env
    for env_var, value in changes.items():
        if not _is_safe_env_value(value):
            return {"restart_required": False, "error": f"Invalid value for {env_var}"}
        # Extra validation for path-related values
        if env_var in {"OVA_LANGUAGE", "OVA_QWEN3_VOICE", "OVA_KOKORO_VOICE"}:
            if not _is_safe_name(value):
                return {"restart_required": False, "error": f"Invalid name for {env_var}"}

    # Security: Semantic validation for known settings
    if "OVA_TTS_ENGINE" in changes and changes["OVA_TTS_ENGINE"] not in VALID_TTS_ENGINES:
        return {"restart_required": False, "error": f"Invalid TTS engine: {changes['OVA_TTS_ENGINE']}"}
    if "OVA_LANGUAGE" in changes and changes["OVA_LANGUAGE"] not in VALID_LANGUAGES:
        return {"restart_required": False, "error": f"Invalid language: {changes['OVA_LANGUAGE']}"}
    if "OVA_STREAM_FORMAT" in changes and changes["OVA_STREAM_FORMAT"] not in VALID_STREAM_FORMATS:
        return {"restart_required": False, "error": f"Invalid stream format: {changes['OVA_STREAM_FORMAT']}"}
    if "OVA_ENABLE_DUPLEX" in changes and changes["OVA_ENABLE_DUPLEX"] not in ("true", "false"):
        return {"restart_required": False, "error": "Invalid conversation mode"}

    # Helper to persist changes to .env
    def save_env(env_changes):
        env_content = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_content[k.strip()] = v.strip()
        env_content.update(env_changes)
        lines = [f"{k}={v}" for k, v in sorted(env_content.items())]
        content = "\n".join(lines) + "\n"
        # Security: Atomic write to prevent corruption on crash
        temp_fd, temp_path = tempfile.mkstemp(dir=env_path.parent, suffix=".tmp")
        try:
            with os.fdopen(temp_fd, 'w') as f:
                f.write(content)
            os.replace(temp_path, env_path)  # Atomic on POSIX
            os.chmod(env_path, 0o600)  # Owner-only read/write (may contain API keys)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    change_keys = set(changes.keys())

    # Determine the voice env var for the active engine
    voice_env = "OVA_QWEN3_VOICE" if pipeline.tts_engine == "qwen3" else "OVA_KOKORO_VOICE"

    # Check if only voice changed (hot-switch)
    if change_keys == {voice_env}:
        new_voice = changes[voice_env]
        if pipeline.tts_engine == "qwen3":
            if pipeline.switch_voice(new_voice):
                save_env(changes)
                return {"restart_required": False, "message": f"Switched to voice: {new_voice}"}
            else:
                return {"restart_required": True, "message": "Voice not preloaded, restart required"}
        else:
            if pipeline.switch_kokoro_voice(new_voice):
                save_env(changes)
                return {"restart_required": False, "message": f"Switched to voice: {new_voice}"}

    # Check if language changed, with optional voice change (hot-switch)
    if (change_keys <= {"OVA_LANGUAGE", voice_env} and
        "OVA_LANGUAGE" in change_keys):
        new_language = changes["OVA_LANGUAGE"]
        new_voice = changes.get(voice_env)
        if pipeline.tts_engine == "qwen3":
            if pipeline.switch_language(new_language, new_voice):
                save_env(changes)
                return {
                    "restart_required": False,
                    "message": f"Switched to language: {new_language}, voice: {pipeline.current_voice}"
                }
            else:
                return {"restart_required": True, "message": "Language switch failed, restart required"}
        else:
            if pipeline.switch_kokoro_language(new_language):
                if new_voice:
                    pipeline.switch_kokoro_voice(new_voice)
                save_env(changes)
                return {
                    "restart_required": False,
                    "message": f"Switched to language: {new_language}, voice: {pipeline.current_voice}"
                }
            else:
                return {"restart_required": True, "message": "Language switch failed, restart required"}

    # Other changes require restart
    save_env(changes)

    # Trigger uvicorn reload
    Path(__file__).touch()

    return {"restart_required": True}


class ReloadPromptRequest(BaseModel):
    language: Optional[str] = None
    profile: Optional[str] = None
    prompt: Optional[str] = None  # Direct prompt content
    clear_history: Optional[bool] = False  # Clear conversation context


@app.post("/v1/settings/prompt")
async def reload_prompt(request: ReloadPromptRequest = None):
    """Update system prompt (session-only, no restart required).

    If prompt provided, uses that directly.
    If language/profile provided, loads that profile's prompt from file.
    Otherwise reloads current profile's prompt from file.
    If clear_history is True, clears conversation context.
    """
    if FRONTEND_SETTINGS_DISABLED:
        return JSONResponse(status_code=403, content={"error": "Settings are disabled"})
    if request and request.prompt is not None:
        # Use provided prompt directly
        pipeline.system_prompt = request.prompt.strip()
        with pipeline._context_lock:
            pipeline.context[0]["content"] = pipeline._build_system_content()
        logger_api.info("Updated prompt from UI")
    elif request and request.language and request.profile:
        # Security: Validate language and profile names to prevent path traversal
        if not _is_safe_name(request.language) or not _is_safe_name(request.profile):
            return {"success": False, "error": "Invalid language or profile name"}

        # Load specific profile's prompt from file
        from .pipeline import LANGUAGE_NAMES
        profiles_dir = Path(__file__).parent.parent / "profiles"
        prompt_file = (profiles_dir / request.language / request.profile / "prompt.txt").resolve()

        # Security: Verify path is contained within profiles directory (prevent symlink attacks)
        if not prompt_file.is_relative_to(profiles_dir.resolve()):
            return {"success": False, "error": "Invalid path"}

        if prompt_file.exists():
            new_prompt = prompt_file.read_text(encoding="utf-8").strip()
        else:
            default = profiles_dir.parent / "prompts" / request.language / "default.txt"
            if default.exists():
                new_prompt = default.read_text(encoding="utf-8").strip()
            else:
                new_prompt = f"You are a helpful assistant. Always respond in {LANGUAGE_NAMES.get(request.language, request.language)}."

        pipeline.system_prompt = new_prompt
        with pipeline._context_lock:
            pipeline.context[0]["content"] = pipeline._build_system_content()
        logger_api.info(f"Loaded prompt for {request.language}/{request.profile}")
    else:
        pipeline.reload_prompt()

    # Clear conversation history if requested (independent of prompt handling)
    if request and request.clear_history:
        with pipeline._context_lock:
            pipeline.context = [{"role": "system", "content": pipeline._build_system_content()}]
        logger_api.info("Cleared conversation context")

    return {"success": True, "prompt": pipeline.system_prompt}


@app.post("/v1/restart")
async def restart_server():
    """Trigger server restart via ova.sh.

    Spawns the restart command in a detached subprocess and returns immediately.
    The subprocess will stop and restart the backend after a brief delay.

    Can be disabled with OVA_DISABLE_RESTART_ENDPOINT=true for production.
    """
    if RESTART_ENDPOINT_DISABLED:
        return {"success": False, "error": "Restart endpoint is disabled"}

    import asyncio

    ova_script = Path(__file__).parent.parent / "ova.sh"
    if not ova_script.exists():
        return {"success": False, "error": "ova.sh not found"}

    logger_api.info("Restart requested - spawning ova.sh restart")

    # Use asyncio to delay the restart slightly so response can be sent
    async def delayed_restart():
        await asyncio.sleep(0.5)  # Give time for HTTP response to complete
        # Spawn detached subprocess - it will outlive this process
        subprocess.Popen(
            [str(ova_script), "restart"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    asyncio.create_task(delayed_restart())
    return {"success": True, "message": "Restart initiated"}


@app.get("/v1/events")
async def event_stream(request: Request):
    """Server-Sent Events endpoint for real-time push notifications.

    Tools (e.g. timer) publish events via EventBus; connected clients
    receive them as SSE. Auto-reconnect is handled by the browser EventSource API.
    """
    import asyncio
    from .events import event_bus

    queue = event_bus.subscribe()
    if queue is None:
        return JSONResponse(status_code=503, content={"error": "Too many event subscribers"})

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            event_bus.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/v1/speech-to-text/stream")
async def websocket_asr(websocket: WebSocket):
    """Stream audio chunks, receive partial transcripts."""
    # Security: API key check for WebSocket (via query param, since browsers can't set WS headers)
    if API_KEY:
        origin = websocket.headers.get("origin", "")
        sec_fetch_site = websocket.headers.get("sec-fetch-site", "")
        is_trusted_origin = origin and origin in _build_cors_origins() and sec_fetch_site
        if not is_trusted_origin:
            ws_key = websocket.query_params.get("api_key", "")
            if not secrets.compare_digest(ws_key, API_KEY):
                await websocket.close(code=4001)
                logger_asr.warning("WebSocket rejected: invalid or missing API key")
                return

    # Security: Validate origin header (CORS middleware doesn't cover WebSocket)
    origin = websocket.headers.get("origin", "")
    allowed_origins = _build_cors_origins()
    if origin and origin not in allowed_origins:
        await websocket.close(code=4003)
        logger_asr.warning(f"WebSocket rejected: invalid origin {origin}")
        return

    await websocket.accept()

    import anyio

    # Extract optional language override from query params (e.g., ?language=en for wake word)
    language = websocket.query_params.get("language")

    try:
        # Initialize streaming state via subprocess IPC
        await anyio.to_thread.run_sync(
            lambda: pipeline.transcribe_streaming_init(language=language)
        )

        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                break

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Security: Cap frame size (1MB ≈ 16s of 16kHz float32 audio)
                    raw = data["bytes"]
                    if len(raw) > 1_048_576:
                        await websocket.send_json({"error": "Audio frame too large"})
                        continue
                    # Audio chunk (float32 PCM at 16kHz)
                    pcm = np.frombuffer(raw, dtype=np.float32)

                    text = await anyio.to_thread.run_sync(
                        pipeline.transcribe_streaming_chunk, pcm
                    )
                    if text:
                        logger_asr.debug(f"ASR streaming partial: {text}")

                    await websocket.send_json({"partial": text})

                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("action") == "end":
                        text = await anyio.to_thread.run_sync(
                            pipeline.transcribe_streaming_finish
                        )
                        logger_asr.info(f"ASR streaming final: {text}")

                        await websocket.send_json({"final": text})
                        break

    except WebSocketDisconnect:
        logger_asr.info("ASR WebSocket disconnected")
    except Exception as e:
        logger_asr.error(f"ASR WebSocket error: {e}")
        try:
            error_msg = str(e) if DEBUG else "Internal ASR error"
            await websocket.send_json({"error": error_msg})
        except (WebSocketDisconnect, RuntimeError):
            pass


@app.websocket("/v1/duplex")
async def websocket_duplex(websocket: WebSocket):
    """Full-duplex voice assistant WebSocket.

    Carries bidirectional audio (PCM int16) and JSON control/status messages
    on a single persistent connection.
    """
    if not DUPLEX_ENABLED:
        await websocket.close(code=4000, reason="Duplex mode not enabled")
        return

    # Security: API key check (same as ASR WebSocket)
    if API_KEY:
        origin = websocket.headers.get("origin", "")
        sec_fetch_site = websocket.headers.get("sec-fetch-site", "")
        is_trusted_origin = origin and origin in _build_cors_origins() and sec_fetch_site
        if not is_trusted_origin:
            ws_key = websocket.query_params.get("api_key", "")
            if not secrets.compare_digest(ws_key, API_KEY):
                await websocket.close(code=4001)
                logger_api.warning("Duplex WebSocket rejected: invalid or missing API key")
                return

    # Security: Validate origin
    origin = websocket.headers.get("origin", "")
    allowed_origins = _build_cors_origins()
    if origin and origin not in allowed_origins:
        await websocket.close(code=4003)
        logger_api.warning(f"Duplex WebSocket rejected: invalid origin {origin}")
        return

    await websocket.accept()

    # Parse optional config from query params
    language = websocket.query_params.get("language", pipeline.language)
    voice = websocket.query_params.get("voice", pipeline.current_voice)

    from .duplex import DuplexSession

    session = DuplexSession(
        websocket=websocket,
        pipeline=pipeline,
        generate_interleaved_fn=generate_interleaved_audio_stream,
        generate_audio_fn=generate_audio_stream,
        tools_active_fn=_tools_active,
        clean_markdown_fn=_clean_markdown,
        language=language,
        voice=voice,
        vad_threshold=DUPLEX_VAD_THRESHOLD,
        vad_confirm_ms=DUPLEX_VAD_CONFIRM_MS,
        vad_silence_ms=DUPLEX_VAD_SILENCE_MS,
        silence_timeout_ms=DUPLEX_SILENCE_TIMEOUT_MS,
        bot_stop_delay_ms=DUPLEX_BOT_STOP_DELAY_MS,
        backchannel_timeout_ms=DUPLEX_BACKCHANNEL_TIMEOUT_MS,
        interrupt_cooldown_ms=DUPLEX_INTERRUPT_COOLDOWN_MS,
        inactivity_timeout_s=DUPLEX_INACTIVITY_TIMEOUT_S,
        sample_rate=DEFAULT_SR,
    )

    logger_api.info(f"Duplex session started (lang={language}, voice={voice})")
    await session.run()
    logger_api.info("Duplex session ended")
