#!/usr/bin/env python3
"""ASR standalone server - runs as separate process with clean CUDA context.

Uses Unix socket for IPC to avoid conflicts with vLLM's stdout usage.
This script must be run with `if __name__ == '__main__':` guard for vLLM.
"""
import os
from pathlib import Path

# Load .env FIRST to get HF_HUB_OFFLINE and other settings
# Must happen before any HuggingFace/vLLM imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Check DEBUG mode - when true, keep all verbose logging
DEBUG = os.getenv("OVA_DEBUG", "").lower() == "true"

# CRITICAL: Always needed for vLLM (regardless of DEBUG mode)
# Disable vLLM's V1 multiprocessing to avoid spawn warning
# qwen_asr imports vLLM at module level, so the if __name__ guard doesn't help
# This runs vLLM in single-process mode (fine for ASR with one request at a time)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

# Suppress noise only when not in DEBUG mode (must be set BEFORE vLLM import)
if not DEBUG:
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")  # Disable vLLM's log config
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")  # Reduce vLLM noise
    os.environ.setdefault("TQDM_DISABLE", "1")  # Disable progress bars
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # Reduce transformers noise

import socket
import sys


def _safe_error_message(e: Exception, context: str = "Operation") -> str:
    """Return error message safe for IPC response.

    In DEBUG mode: include full exception details for debugging.
    In production: return generic message to avoid leaking internals.
    """
    if DEBUG:
        return f"{context} failed: {e}"
    return f"{context} failed"


def main():
    """Main entry point - vLLM imports happen here, after __main__ guard."""
    import numpy as np
    import logging

    # ANSI color codes
    CYAN = "\033[36m"
    RESET = "\033[0m"

    # Configure colored logging for ASR subprocess
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # Color the [ASR] tag cyan
            return f"{self.formatTime(record)} {record.levelname} {CYAN}[ASR]{RESET} {record.getMessage()}"

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger = logging.getLogger("asr_server")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Suppress noisy third-party loggers when not in DEBUG mode
    if not DEBUG:
        for name in ["vllm", "flashinfer", "transformers", "filelock"]:
            logging.getLogger(name).setLevel(logging.ERROR)

    # Get socket path from command line
    if len(sys.argv) < 2:
        logger.error("Usage: python -m ova.asr_server <socket_path>")
        sys.exit(1)

    socket_path = sys.argv[1]

    # Create Unix socket server
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)  # Owner-only access
    server.listen(1)
    logger.info(f"ASR server listening on {socket_path}")

    # Accept connection from parent
    conn, _ = server.accept()
    logger.info("Parent connected")

    # NOW import qwen_asr - vLLM will be first to touch CUDA
    from qwen_asr import Qwen3ASRModel

    ASR_MODEL = os.getenv("OVA_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")

    # Language configuration - map ISO codes to Qwen3-ASR language names
    LANGUAGE_MAP = {
        "es": "Spanish", "en": "English", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
        "ko": "Korean", "ru": "Russian", "ar": "Arabic", "nl": "Dutch",
        "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
    }

    lang_code = os.getenv("OVA_LANGUAGE", "es")
    ASR_LANGUAGE = LANGUAGE_MAP.get(lang_code, None)  # Default for warmup, overridden per-call

    logger.info(f"Loading ASR model: {ASR_MODEL} (language: {ASR_LANGUAGE or 'auto-detect'})")

    try:
        model = Qwen3ASRModel.LLM(
            model=ASR_MODEL,
            dtype="bfloat16",  # Skip auto dtype detection (avoids safetensors metadata fetch bug)
            gpu_memory_utilization=float(os.getenv("OVA_ASR_GPU_MEMORY_UTILIZATION", "0.4")),
            max_new_tokens=512,
            max_model_len=int(os.getenv("OVA_ASR_MAX_MODEL_LEN", "2048")),
            enforce_eager=False,  # CUDA graphs enabled for performance
            seed=42,  # Explicit seed for reproducible ASR (suppresses random state warning)
            compilation_config={"cudagraph_num_of_warmups": 3},
        )
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        _send(conn, {"status": "error", "message": _safe_error_message(e, "ASR model load")})
        conn.close()
        server.close()
        os.unlink(socket_path)
        sys.exit(1)

    # Signal ready
    _send(conn, {"status": "ready"})

    # Streaming state
    state = None

    # Main loop
    while True:
        try:
            cmd = _recv(conn)
            if cmd is None:
                logger.info("Connection closed")
                break

            action = cmd.get("action")

            if action == "transcribe":
                audio = cmd["audio"]  # Already numpy array via pickle
                sr = cmd["sr"]
                language = cmd.get("language", ASR_LANGUAGE)  # Allow per-call override
                try:
                    results = model.transcribe((audio, sr), language=language)
                    text = results[0].text.strip() if results and results[0].text else ""
                    _send(conn, {"status": "result", "text": text})
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    _send(conn, {"status": "error", "message": _safe_error_message(e, "Transcription")})

            elif action == "stream_init":
                language = cmd.get("language", ASR_LANGUAGE)  # Allow per-call override
                try:
                    state = model.init_streaming_state(
                        chunk_size_sec=float(os.getenv("OVA_ASR_CHUNK_SIZE_SEC", "0.5")),
                        unfixed_chunk_num=int(os.getenv("OVA_ASR_UNFIXED_CHUNK_NUM", "4")),
                        unfixed_token_num=int(os.getenv("OVA_ASR_UNFIXED_TOKEN_NUM", "5")),
                        language=language,
                    )
                    _send(conn, {"status": "ok"})
                except Exception as e:
                    logger.error(f"Stream init error: {e}")
                    _send(conn, {"status": "error", "message": _safe_error_message(e, "Stream init")})

            elif action == "stream_chunk":
                if state is None:
                    _send(conn, {"status": "error", "message": "Not initialized"})
                    continue
                try:
                    audio = cmd["audio"]  # Already numpy array via pickle
                    model.streaming_transcribe(audio, state)
                    _send(conn, {"status": "partial", "text": state.text})
                except Exception as e:
                    logger.error(f"Stream chunk error: {e}")
                    _send(conn, {"status": "error", "message": _safe_error_message(e, "Stream chunk")})

            elif action == "stream_finish":
                if state is None:
                    _send(conn, {"status": "error", "message": "Not initialized"})
                    continue
                try:
                    model.finish_streaming_transcribe(state)
                    text = state.text
                    state = None
                    _send(conn, {"status": "final", "text": text})
                except Exception as e:
                    logger.error(f"Stream finish error: {e}")
                    _send(conn, {"status": "error", "message": _safe_error_message(e, "Stream finish")})

            elif action == "warmup":
                try:
                    import soundfile as sf
                    from pathlib import Path

                    # Load real audio from profile for proper warmup (generates actual tokens)
                    # This warms CUDA graphs with realistic workload
                    ref_audio_path = cmd.get("audio_path")
                    if ref_audio_path and Path(ref_audio_path).exists():
                        audio, sr = sf.read(ref_audio_path, dtype='float32')
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)  # Mono
                        # Resample to 16kHz if needed
                        if sr != 16000:
                            import soxr
                            audio = soxr.resample(audio, sr, 16000)
                        logger.info(f"Loaded warmup audio: {len(audio)/16000:.1f}s")
                    else:
                        # Fallback to silence if no audio provided
                        audio = np.zeros(48000, dtype=np.float32)
                        logger.info("Using silent audio for warmup (no ref_audio provided)")

                    # Pass 1: Full audio - triggers token generation
                    model.transcribe((audio, 16000), language=ASR_LANGUAGE)
                    logger.info("ASR warmup pass 1/3")

                    # Pass 2: Same audio again - should be faster (graphs warmed)
                    model.transcribe((audio, 16000), language=ASR_LANGUAGE)
                    logger.info("ASR warmup pass 2/3")

                    # Pass 3: Final pass - verify stability
                    model.transcribe((audio, 16000), language=ASR_LANGUAGE)
                    logger.info("ASR warmup pass 3/3")

                    _send(conn, {"status": "ok"})
                except Exception as e:
                    logger.warning(f"Warmup error: {e}")
                    _send(conn, {"status": "ok"})

            elif action == "shutdown":
                logger.info("Shutting down")
                break

            else:
                _send(conn, {"status": "error", "message": f"Unknown action: {action}"})

        except Exception as e:
            logger.error(f"Error: {e}")
            try:
                _send(conn, {"status": "error", "message": _safe_error_message(e, "IPC")})
            except:
                break

    conn.close()
    server.close()
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # Cleanup PyTorch distributed resources (vLLM initializes NCCL internally)
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass  # Best effort cleanup


def _send(conn: socket.socket, data: dict):
    """Send pickle message with length prefix (faster than JSON for numpy arrays)."""
    import pickle
    msg = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(len(msg).to_bytes(4, 'big') + msg)


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


def _recv(conn: socket.socket) -> dict:
    """Receive pickle message with length prefix and restricted unpickling."""
    length_bytes = conn.recv(4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'big')
    if length > _MAX_IPC_MSG_SIZE:
        raise ValueError(f"IPC message too large: {length} bytes (max {_MAX_IPC_MSG_SIZE})")
    data = b''
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return _RestrictedUnpickler.loads(data)


if __name__ == "__main__":
    main()
