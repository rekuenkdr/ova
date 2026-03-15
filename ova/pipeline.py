import io
import os
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
from .audio import numpy_to_wav_bytes
from .prosody import strip_prosody_tags
from .utils import DEBUG, get_device, get_logger

# Component-specific loggers with colored output
logger_sys = get_logger("sys")
logger_asr = get_logger("asr")
logger_llm = get_logger("llm")
logger_tts = get_logger("tts")

# NOTE: Do NOT import torch, qwen_tts, or kokoro at module level
# ASR subprocess must initialize BEFORE any CUDA context is created in main process
# Otherwise vLLM is forced to use 'spawn' multiprocessing (slower, more memory)

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration from environment
DEFAULT_SR = 24000

# Language settings
LANGUAGE = os.getenv("OVA_LANGUAGE", "es")
LANGUAGE_NAMES = {
    "es": "Spanish", "en": "English", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "ru": "Russian", "hi": "Hindi",
}

# Qwen3-TTS supported languages
QWEN3_LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

# ASR language names (Qwen3-ASR supports more languages than TTS)
ASR_LANGUAGE_MAP = {
    "es": "Spanish", "en": "English", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "ru": "Russian", "ar": "Arabic", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
}

KOKORO_LANG_CODES = {"es": "e", "en": "a", "en-gb": "b", "fr": "f", "it": "i", "pt": "p", "ja": "j", "zh": "z"}

# Voice profile
QWEN3_VOICE = os.getenv("OVA_QWEN3_VOICE", "myvoice")

# TTS engine: "qwen3" or "kokoro"
TTS_ENGINE = os.getenv("OVA_TTS_ENGINE", "qwen3")

# Models
CHAT_MODEL = os.getenv("OVA_CHAT_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
QWEN3_TTS_MODEL = os.getenv("OVA_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
KOKORO_MODEL = os.getenv("OVA_KOKORO_MODEL", "hexgrad/Kokoro-82M")
KOKORO_VOICE = os.getenv("OVA_KOKORO_VOICE", "af_heart")

# Kokoro voices by language (authoritative list — frontend loads from backend)
KOKORO_VOICES = {
    "en": [
        {"id": "af_heart", "name": "Heart (Female)"},
        {"id": "af_alloy", "name": "Alloy (Female)"},
        {"id": "af_bella", "name": "Bella (Female)"},
        {"id": "af_jessica", "name": "Jessica (Female)"},
        {"id": "af_nicole", "name": "Nicole (Female)"},
        {"id": "af_nova", "name": "Nova (Female)"},
        {"id": "af_river", "name": "River (Female)"},
        {"id": "af_sarah", "name": "Sarah (Female)"},
        {"id": "af_sky", "name": "Sky (Female)"},
        {"id": "am_adam", "name": "Adam (Male)"},
        {"id": "am_echo", "name": "Echo (Male)"},
        {"id": "am_eric", "name": "Eric (Male)"},
        {"id": "am_liam", "name": "Liam (Male)"},
        {"id": "am_michael", "name": "Michael (Male)"},
        {"id": "bf_alice", "name": "Alice (British Female)"},
        {"id": "bf_emma", "name": "Emma (British Female)"},
        {"id": "bf_lily", "name": "Lily (British Female)"},
        {"id": "bm_daniel", "name": "Daniel (British Male)"},
        {"id": "bm_george", "name": "George (British Male)"},
    ],
    "es": [
        {"id": "ef_dora", "name": "Dora (Female)"},
        {"id": "em_alex", "name": "Alex (Male)"},
        {"id": "em_santa", "name": "Santa (Male)"},
    ],
    "fr": [
        {"id": "ff_siwis", "name": "Siwis (Female)"},
    ],
    "de": [],
    "it": [
        {"id": "if_sara", "name": "Sara (Female)"},
        {"id": "im_nicola", "name": "Nicola (Male)"},
    ],
    "pt": [
        {"id": "pf_dora", "name": "Dora (Female)"},
        {"id": "pm_alex", "name": "Alex (Male)"},
        {"id": "pm_santa", "name": "Santa (Male)"},
    ],
    "ja": [
        {"id": "jf_alpha", "name": "Alpha (Female)"},
        {"id": "jf_gongitsune", "name": "Gongitsune (Female)"},
        {"id": "jf_nezumi", "name": "Nezumi (Female)"},
        {"id": "jm_kumo", "name": "Kumo (Male)"},
    ],
    "zh": [
        {"id": "zf_xiaobei", "name": "Xiaobei (Female)"},
        {"id": "zf_xiaoni", "name": "Xiaoni (Female)"},
        {"id": "zf_xiaoxiao", "name": "Xiaoxiao (Female)"},
        {"id": "zf_xiaoyi", "name": "Xiaoyi (Female)"},
        {"id": "zm_yunjian", "name": "Yunjian (Male)"},
        {"id": "zm_yunxi", "name": "Yunxi (Male)"},
    ],
    "hi": [
        {"id": "hf_alpha", "name": "Alpha (Female)"},
        {"id": "hf_beta", "name": "Beta (Female)"},
        {"id": "hm_omega", "name": "Omega (Male)"},
        {"id": "hm_psi", "name": "Psi (Male)"},
    ],
}

# Streaming format for Qwen3: "pcm" or "wav"
QWEN3_STREAM_FORMAT = os.getenv("OVA_QWEN3_STREAM_FORMAT", "pcm")

# PCM streaming tuning
PCM_EMIT_EVERY_FRAMES = int(os.getenv("OVA_PCM_EMIT_EVERY_FRAMES", "8"))
PCM_PREBUFFER_SAMPLES = int(os.getenv("OVA_PCM_PREBUFFER_SAMPLES", "9600"))

# Streaming optimizations (torch.compile)
# Auto-selects optimal settings based on QWEN3_STREAM_FORMAT (pcm vs wav)
ENABLE_STREAMING_OPTIMIZATIONS = os.getenv("OVA_ENABLE_STREAMING_OPTIMIZATIONS", "true").lower() == "true"

# Two-phase TTS streaming (aggressive first chunk for lower latency)
FIRST_CHUNK_EMIT_EVERY = int(os.getenv("OVA_FIRST_CHUNK_EMIT_EVERY", "5"))
FIRST_CHUNK_DECODE_WINDOW = int(os.getenv("OVA_FIRST_CHUNK_DECODE_WINDOW", "48"))
FIRST_CHUNK_FRAMES = int(os.getenv("OVA_FIRST_CHUNK_FRAMES", "48"))

# Decode window size for steady-state streaming (phase 2)
PCM_DECODE_WINDOW = int(os.getenv("OVA_PCM_DECODE_WINDOW", "80"))

# Codebook CUDA graph acceleration (~1.13x faster codebook generation, ~500MB extra VRAM)
CODEBOOK_CUDA_GRAPH = os.getenv("OVA_CODEBOOK_CUDA_GRAPH", "false").lower() == "true"

# Paged attention engine (alternative to torch.compile, requires flash-attn, triton, xxhash)
USE_PAGED_ENGINE = os.getenv("OVA_USE_PAGED_ENGINE", "false").lower() == "true"
PAGED_GPU_MEMORY_UTILIZATION = float(os.getenv("OVA_PAGED_GPU_MEMORY_UTILIZATION", "0.9"))

# Maximum TTS frames (prevents runaway generation)
# Model-specific values: 0.6B=1500, 1.7B=8000 (library limit: 8000)
MAX_TTS_FRAMES = int(os.getenv("OVA_MAX_TTS_FRAMES", "8000"))

# Maximum LLM output tokens (0 = unlimited)
# Model-specific values: 0.6B=300, 1.7B=0 (unlimited)
LLM_MAX_TOKENS = int(os.getenv("OVA_LLM_MAX_TOKENS", "0"))

# Maximum conversation context messages (0 = unlimited)
# Prevents unbounded memory growth in long sessions
MAX_CONTEXT_MESSAGES = int(os.getenv("OVA_MAX_CONTEXT_MESSAGES", "50"))

class OVAPipeline:
    def __init__(self, asr_send: Callable, asr_recv: Callable, asr_call: Callable = None):
        """Initialize OVA pipeline.

        Args:
            asr_send: Function to send commands to ASR subprocess
            asr_recv: Function to receive responses from ASR subprocess
            asr_call: Thread-safe send+recv in one atomic call (preferred)
        """
        self._asr_send = asr_send
        self._asr_recv = asr_recv
        self._asr_call = asr_call
        self._interrupt = threading.Event()
        self._tts_idle = threading.Event()
        self._tts_idle.set()  # Initially idle
        self._context_lock = threading.Lock()
        self.language = LANGUAGE
        self.tts_engine = TTS_ENGINE

        # Load system prompt
        # For Qwen3: use profile-specific prompt if available
        # For Kokoro: use language default prompt (no voice profiles)
        if self.tts_engine == "qwen3":
            profile_dir = Path(__file__).parent.parent / "profiles" / LANGUAGE / QWEN3_VOICE
            if not profile_dir.exists():
                raise ValueError(f"Profile directory not found: {profile_dir}")
            self.profile_dir = profile_dir
            prompt_file = profile_dir / "prompt.txt"
            if prompt_file.exists():
                self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            else:
                self.system_prompt = self._load_default_prompt()
        else:
            # Kokoro: no voice profiles, use default prompt
            self.profile_dir = None
            self.system_prompt = self._load_default_prompt()

        self.context = [{"role": "system", "content": self.system_prompt}]

        # NOW safe to call get_device() (ASR subprocess already started with clean CUDA)
        self.device = get_device()

        # Initialize TTS
        if self.tts_engine == "qwen3":
            self._init_qwen3_tts()
        else:
            self._init_kokoro_tts()

        # Chat model + LLM provider
        self.chat_model = CHAT_MODEL
        from .llm import create_llm_provider
        self.llm_provider = create_llm_provider(self.chat_model)

        # Tool/function calling registry
        from .tools import registry as tool_registry
        tool_registry.discover()
        self.tool_registry = tool_registry

        # MCP client (external tool servers)
        self._mcp_manager = None
        if os.getenv("OVA_ENABLE_MCP", "false").lower() == "true":
            try:
                from .mcp_client import MCPClientManager, load_mcp_config
                config = load_mcp_config()
                if config.get("mcpServers"):
                    self._mcp_manager = MCPClientManager()
                    self._mcp_manager.start()
                    self._mcp_manager.connect_all(config)
                    self.tool_registry.register_mcp(self._mcp_manager)
                else:
                    logger_sys.info("MCP enabled but no servers configured")
            except Exception as e:
                logger_sys.warning(f"MCP init failed (non-fatal): {e}")
                self._mcp_manager = None

        # Inject tool instructions into system prompt if tools are enabled
        self.context[0]["content"] = self._build_system_content()

        logger_sys.info(f"OVA initialized: lang={LANGUAGE}, voice={self.current_voice}, tts={TTS_ENGINE}, tools={self.tool_registry.enabled}")

    def _trim_context(self):
        """Trim conversation context to MAX_CONTEXT_MESSAGES.

        Preserves the system message at index 0 and keeps the most recent messages.
        Caller must hold _context_lock.
        """
        if MAX_CONTEXT_MESSAGES > 0 and len(self.context) > MAX_CONTEXT_MESSAGES:
            self.context = [self.context[0]] + self.context[-(MAX_CONTEXT_MESSAGES - 1):]

    def _load_default_prompt(self) -> str:
        """Load default system prompt for current language."""
        default_prompt = Path(__file__).parent.parent / "prompts" / self.language / "default.txt"
        if default_prompt.exists():
            return default_prompt.read_text(encoding="utf-8").strip()
        return f"You are a helpful assistant. Always respond in {LANGUAGE_NAMES.get(self.language, self.language)}."

    def _build_system_content(self) -> str:
        """Build full system message content with tool instructions before personality."""
        content = self.system_prompt
        if self.tool_registry.enabled:
            tool_names_list = self.tool_registry.get_tool_names()
            tool_names = ", ".join(tool_names_list)
            has_web_search = "web_search" in tool_names_list

            parts = []
            parts.append(f"You have these tools: {tool_names}. ALWAYS call a tool when the question matches its purpose. NEVER guess. After receiving a tool result, use the EXACT data from it in your response — never approximate or make up values.")
            parts.append("For time or date questions, call get_current_datetime. For timer requests, call set_timer or check_timers.")
            if has_web_search:
                parts.append(
                    "For web_search, only use it when your own knowledge is insufficient: "
                    "breaking news, live scores, current prices, today's weather, or recent events. "
                    "Do NOT search for general knowledge, historical facts, definitions, or anything you already know well — just answer directly."
                )

            # Tool instructions FIRST, then personality
            content = " ".join(parts) + "\n\n" + self.system_prompt
        return content

    def _get_model_size_suffix(self) -> str:
        """Get model size suffix for voice clone prompt files.

        Returns suffix like '0.6B' or '1.7B' based on loaded TTS model config.
        Voice clone prompts are model-specific due to different speaker encoder dimensions.
        """
        model_size = getattr(self.tts_model.model.config, 'tts_model_size', '1b7')
        # Map config values to human-readable suffixes
        size_map = {"0b6": "0.6B", "1b7": "1.7B"}
        return size_map.get(model_size, model_size)

    def _get_llm_options(self) -> dict:
        """Get LLM options, applying token limit if set."""
        if LLM_MAX_TOKENS > 0:
            return {"num_predict": LLM_MAX_TOKENS}
        return {}

    def _load_voice_prompt(self, voice_dir: Path):
        """Load voice clone prompt for current model size.

        Tries model-specific file first (voice_clone_prompt_{size}.pt),
        falls back to generating from ref_audio.wav + ref_text.txt if needed.

        Args:
            voice_dir: Path to voice profile directory

        Returns:
            Voice clone prompt or None if not available
        """
        import torch
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        # Allowlist the trusted dataclass for safe deserialization
        # This blocks arbitrary code execution while allowing VoiceClonePromptItem
        torch.serialization.add_safe_globals([VoiceClonePromptItem])

        suffix = self._get_model_size_suffix()
        voice_name = voice_dir.name

        # Expected speaker embedding dimensions per model size
        expected_dims = {"0.6B": 1024, "1.7B": 2048}
        expected_dim = expected_dims.get(suffix)

        # Try model-specific file first
        specific_file = voice_dir / f"voice_clone_prompt_{suffix}.pt"
        if specific_file.exists():
            prompt = torch.load(specific_file, weights_only=True)
            # create_voice_clone_prompt returns List[VoiceClonePromptItem];
            # normalize to single item for consistent storage
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            # Validate speaker embedding dimension
            if expected_dim and hasattr(prompt, 'ref_spk_embedding'):
                actual_dim = prompt.ref_spk_embedding.shape[-1]
                if actual_dim != expected_dim:
                    logger_tts.error(
                        f"Voice prompt dimension mismatch for {voice_name}: "
                        f"expected {expected_dim}, got {actual_dim}. "
                        f"Regenerate with: python scripts/generate_voice_prompts.py --voice {voice_name} --model-size {suffix}"
                    )
                    return None
            logger_tts.info(f"Loaded voice: {voice_name} ({suffix})")
            return prompt

        # Fallback: legacy file (warn about potential incompatibility)
        legacy_file = voice_dir / "voice_clone_prompt.pt"
        if legacy_file.exists():
            logger_tts.warning(
                f"Using legacy voice_clone_prompt.pt for {voice_name}. "
                f"If you see dimension errors, regenerate with: "
                f"python scripts/generate_voice_prompts.py --voice {voice_name} --model-size {suffix}"
            )
            prompt = torch.load(legacy_file, weights_only=True)
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            # Validate speaker embedding dimension for legacy files too
            if expected_dim and hasattr(prompt, 'ref_spk_embedding'):
                actual_dim = prompt.ref_spk_embedding.shape[-1]
                if actual_dim != expected_dim:
                    logger_tts.error(
                        f"Legacy prompt dimension mismatch for {voice_name}: "
                        f"expected {expected_dim}, got {actual_dim}. "
                        f"Regenerate with: python scripts/generate_voice_prompts.py --voice {voice_name} --model-size {suffix}"
                    )
                    return None
            return prompt

        # Generate from reference files
        ref_audio = voice_dir / "ref_audio.wav"
        ref_text_file = voice_dir / "ref_text.txt"
        if ref_audio.exists() and ref_text_file.exists():
            ref_text = ref_text_file.read_text(encoding="utf-8").strip()
            logger_tts.info(f"Generating voice clone prompt for {voice_name} ({suffix})...")
            prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            # Save with model-specific name
            torch.save(prompt, specific_file)
            logger_tts.info(f"Created and cached: {specific_file.name}")
            return prompt

        return None

    def _init_qwen3_tts(self):
        """Initialize Qwen3-TTS with voice cloning."""
        # Lazy imports - safe now since ASR subprocess already has clean CUDA
        import torch
        from qwen_tts import Qwen3TTSModel

        # Enable TensorFloat32 for better performance on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')

        # torch 2.10 lowered cudagraph_dynamic_shape_warn_limit from 50 to 8.
        # Streaming TTS generates ~9 distinct shapes (same as torch 2.9), restore previous threshold.
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 50

        logger_tts.info(f"Loading Qwen3-TTS: {QWEN3_TTS_MODEL}")
        self.tts_model = Qwen3TTSModel.from_pretrained(
            QWEN3_TTS_MODEL,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Log model size for debugging
        model_size = self._get_model_size_suffix()
        logger_tts.info(f"TTS model size: {model_size}")
        logger_tts.info("EOS tokens: [2150, 2157, 151670, 151673, 151645, 151643]")

        # Preload ALL voice prompts for ALL languages (enables hot-switching)
        # CUDA-safe: load once at startup, switch by reference only
        # NOTE: Voice prompts are model-specific (different speaker encoder dimensions)
        self.all_voice_prompts = {}  # {lang_code: {voice_name: prompt_tensor}}
        self.current_voice = QWEN3_VOICE
        profiles_root = Path(__file__).parent.parent / "profiles"

        for lang_dir in profiles_root.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            self.all_voice_prompts[lang] = {}
            for voice_dir in lang_dir.iterdir():
                if not voice_dir.is_dir():
                    continue
                prompt = self._load_voice_prompt(voice_dir)
                if prompt is not None:
                    self.all_voice_prompts[lang][voice_dir.name] = prompt

        # Remove empty languages
        self.all_voice_prompts = {k: v for k, v in self.all_voice_prompts.items() if v}

        # Shortcut to current language (updated on switch_language)
        self.voice_prompts = self.all_voice_prompts.get(self.language, {})

        # Validate at least one voice prompt was loaded
        total_voices = sum(len(v) for v in self.all_voice_prompts.values())
        if not self.voice_prompts:
            logger_tts.error(
                f"No voice prompts found for language {self.language} (model size {model_size}). "
                f"Run: python scripts/generate_voice_prompts.py --model-size {model_size}"
            )
            raise RuntimeError(f"No valid voice prompts for {model_size} model")

        # Set current voice prompt
        self.voice_clone_prompt = self.voice_prompts.get(QWEN3_VOICE)
        if self.voice_clone_prompt:
            logger_tts.info(f"Active voice: {QWEN3_VOICE} ({total_voices} voices preloaded across {len(self.all_voice_prompts)} languages)")
        else:
            logger_tts.warning(f"Voice {QWEN3_VOICE} not found, using default")

        self._apply_qwen3_optimizations()
        # NOTE: Warmup moved to api.py startup event to run in async context
        # (CUDA graph TLS requires same thread context as inference)

    def warmup_tts(self):
        """Warmup TTS to trigger torch.compile. Call from FastAPI startup event."""
        import sys

        if self.tts_engine != "qwen3" or not self.voice_clone_prompt:
            return

        # Paged engine requires explicit warmup (KV cache allocation + CUDA graph capture)
        # before any stream_generate_voice_clone() calls
        if USE_PAGED_ENGINE:
            import torch
            logger_tts.info("Warming up paged attention engine (KV cache + CUDA graphs)...")
            with torch.inference_mode():
                self.tts_model.model.warmup_paged_engine()

        logger_tts.info("Warming up TTS (triggering torch.compile)...")
        try:
            # Two passes: first triggers torch.compile, second ensures CUDA graphs captured
            warmup_texts = [
                # Pass 1: Medium text - triggers initial torch.compile
                "Hola, esto es una prueba de calentamiento para activar la compilación completa del modelo.",
                # Pass 2: Slightly different - ensures reduce-overhead CUDA graphs are captured
                "Segunda prueba para asegurar que los grafos de CUDA están completamente capturados.",
            ]

            for i, text in enumerate(warmup_texts, 1):
                chunk_count = 0
                for chunk, sr in self.tts_model.stream_generate_voice_clone(
                    text=text,
                    language=LANGUAGE_NAMES.get(self.language, "Spanish"),
                    voice_clone_prompt=self.voice_clone_prompt,
                    emit_every_frames=PCM_EMIT_EVERY_FRAMES,
                    decode_window_frames=PCM_DECODE_WINDOW,
                    overlap_samples=0,
                    first_chunk_emit_every=FIRST_CHUNK_EMIT_EVERY,
                    first_chunk_decode_window=FIRST_CHUNK_DECODE_WINDOW,
                    first_chunk_frames=FIRST_CHUNK_FRAMES,
                    max_frames=MAX_TTS_FRAMES,
                    use_paged_engine=USE_PAGED_ENGINE,
                ):
                    chunk_count += 1
                logger_tts.info(f"TTS warmup pass {i}/{len(warmup_texts)}: {chunk_count} chunks")
                sys.stdout.flush()
                sys.stderr.flush()

            # Capture codebook CUDA graph after warmup (same thread context required)
            # Must run under inference_mode — warmup passes mark model tensors as
            # inference tensors, and CUDA graph capture does inplace updates on them
            if CODEBOOK_CUDA_GRAPH and QWEN3_STREAM_FORMAT == "pcm" and not USE_PAGED_ENGINE:
                import torch
                logger_tts.info("Capturing codebook CUDA graph (private pool)...")
                with torch.inference_mode():
                    self.tts_model.capture_codebook_cuda_graph(warmup_runs=3)
                logger_tts.info("Codebook CUDA graph captured")

            logger_tts.info("TTS warmup complete")
        except Exception as e:
            logger_tts.warning(f"TTS warmup failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()

    def warmup_asr(self):
        """Warmup ASR in subprocess. Call from FastAPI startup event."""
        logger_asr.info("Warming up ASR via subprocess...")
        try:
            # Send ref_audio path for realistic warmup (generates actual tokens)
            # For Kokoro, profile_dir is None so we skip ref_audio
            audio_path = None
            if self.profile_dir:
                ref_audio_path = self.profile_dir / "ref_audio.wav"
                if ref_audio_path.exists():
                    audio_path = str(ref_audio_path)
            response = self._asr_call({
                "action": "warmup",
                "audio_path": audio_path
            })
            if response.get("status") == "ok":
                logger_asr.info("ASR warmup complete")
            else:
                logger_asr.warning(f"ASR warmup response: {response}")
        except Exception as e:
            logger_asr.warning(f"ASR warmup failed (non-fatal): {e}")

    def warmup_llm(self):
        """Warmup LLM by sending a test request.

        Ensures the model is loaded into VRAM before first user request.
        """
        self.llm_provider.warmup()

    def _apply_qwen3_optimizations(self):
        """Apply torch.compile optimizations based on stream format."""
        if ENABLE_STREAMING_OPTIMIZATIONS and self.device != "cpu":
            if USE_PAGED_ENGINE:
                # Paged attention engine: explicit paged KV cache + CUDA graphs
                # Mutually exclusive with torch.compile and codebook optimizations
                logger_tts.info("Enabling paged attention engine")
                self.tts_model.enable_streaming_optimizations(
                    decode_window_frames=PCM_DECODE_WINDOW,
                    use_compile=False,
                    use_cuda_graphs=False,
                    use_paged_engine=True,
                    paged_gpu_memory_utilization=PAGED_GPU_MEMORY_UTILIZATION,
                )
            elif QWEN3_STREAM_FORMAT == "pcm":
                # Streaming mode: optimize for low latency with reduce-overhead
                if CODEBOOK_CUDA_GRAPH:
                    # Fast codebook + manual CUDA graph (requires wip/experimental fork)
                    # compile_codebook_predictor=False avoids conflict with reduce-overhead CUDA graphs
                    logger_tts.info("Enabling PCM streaming optimizations: compile_mode=reduce-overhead, fast_codebook=True, codebook_cuda_graph=True")
                    self.tts_model.enable_streaming_optimizations(
                        decode_window_frames=PCM_DECODE_WINDOW,
                        use_compile=True,
                        use_cuda_graphs=True,
                        compile_mode="reduce-overhead",
                        use_fast_codebook=True,
                        use_codebook_cuda_graph=True,
                        compile_codebook_predictor=False,
                    )
                else:
                    # Default: no codebook params, let fork defaults handle it
                    # Compatible with both fork main and wip/experimental branches
                    logger_tts.info("Enabling PCM streaming optimizations: compile_mode=reduce-overhead")
                    self.tts_model.enable_streaming_optimizations(
                        decode_window_frames=PCM_DECODE_WINDOW,
                        use_compile=True,
                        use_cuda_graphs=True,
                        compile_mode="reduce-overhead",
                    )
            else:
                # WAV mode: optimize for throughput with max-autotune
                logger_tts.info("Enabling WAV batch optimizations: compile_mode=max-autotune, fast_codebook=True")
                self.tts_model.enable_streaming_optimizations(
                    decode_window_frames=300,  # Larger window for batch processing
                    use_compile=True,
                    use_cuda_graphs=False,
                    compile_mode="max-autotune",
                    use_fast_codebook=True,  # 2-3x speedup bypassing HF generate()
                    compile_codebook_predictor=True,
                )
            logger_tts.info("Optimizations enabled")

    def _resolve_voice(self, voice: str = None, language: str = None):
        """Resolve (voice_clone_prompt, language_name) from optional overrides.

        Voice lookup: target language -> current language -> all languages -> default.
        Language: override -> session default -> LANGUAGE_NAMES map.
        """
        tts_lang = language or self.language
        lang_name = LANGUAGE_NAMES.get(tts_lang, LANGUAGE_NAMES.get(self.language, "English"))

        if voice is None:
            return self.voice_clone_prompt, lang_name

        # 1. Target language
        if tts_lang in self.all_voice_prompts and voice in self.all_voice_prompts[tts_lang]:
            return self.all_voice_prompts[tts_lang][voice], lang_name

        # 2. Current session language
        if voice in self.voice_prompts:
            return self.voice_prompts[voice], lang_name

        # 3. Search all languages
        for voices in self.all_voice_prompts.values():
            if voice in voices:
                return voices[voice], lang_name

        # 4. Not found
        logger_tts.warning(f"Voice '{voice}' not found, using session default")
        return self.voice_clone_prompt, lang_name

    def _init_kokoro_tts(self):
        """Initialize Kokoro TTS."""
        import warnings

        # Suppress Kokoro model loading warnings (from PyTorch internals)
        warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

        # Lazy import - safe now since ASR subprocess already has clean CUDA
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "Kokoro TTS requires the 'kokoro' package. "
                "Install with: pip install kokoro"
            )

        lang_code = KOKORO_LANG_CODES.get(LANGUAGE, "a")
        if LANGUAGE not in KOKORO_LANG_CODES:
            logger_tts.warning(f"Language '{LANGUAGE}' not supported by Kokoro, defaulting to English")

        logger_tts.info(f"Loading Kokoro TTS: {KOKORO_MODEL} (lang={lang_code})")
        self.tts_model = KPipeline(lang_code=lang_code, repo_id=KOKORO_MODEL)
        self.voice_clone_prompt = None  # Kokoro doesn't use voice cloning
        self.current_voice = KOKORO_VOICE  # Instance variable for hot-switching
        self.voice_prompts = {}  # Empty for Kokoro (no voice cloning)
        self.all_voice_prompts = {}  # Empty for Kokoro (no voice cloning)

        # Warm up
        self.tts_model("Test", voice=self.current_voice)

    def tts(self, text: str, *, voice: str = None, language: str = None) -> bytes:
        """Generate speech from text."""
        if self.tts_engine == "qwen3":
            prompt, lang_name = self._resolve_voice(voice, language)
            return self._tts_qwen3(text, voice_clone_prompt=prompt, language_name=lang_name)
        else:
            return self._tts_kokoro(text, voice=voice)

    def _tts_qwen3(self, text: str, *, voice_clone_prompt=None, language_name=None) -> bytes:
        """Generate speech using Qwen3-TTS."""
        import torch
        language = language_name or LANGUAGE_NAMES.get(self.language, "Spanish")
        prompt = voice_clone_prompt or self.voice_clone_prompt

        with torch.inference_mode():
            if prompt:
                wavs, sr = self.tts_model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                )
            else:
                wavs, sr = self.tts_model.generate(text=text, language=language)

        return numpy_to_wav_bytes(wavs[0], sr)

    def _tts_kokoro(self, text: str, *, voice: str = None) -> bytes:
        """Generate speech using Kokoro TTS."""
        voice_name = voice or self.current_voice
        generator = self.tts_model(text, voice=voice_name)

        chunks = []
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size > 0:
                chunks.append(audio)

        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return numpy_to_wav_bytes(arr, sr=DEFAULT_SR)

    def tts_batch(self, items: list[dict]) -> list[tuple[bytes, str | None, str | None]]:
        """Non-streaming batch TTS. Returns list of (wav_bytes, voice, language) per item.

        Uses generate_voice_clone(text=List[str]) which processes all items in a
        single forward pass — 2.6x faster than streaming batch for buffered responses.

        Args:
            items: List of dicts with keys: text (str), voice (str|None), language (str|None)

        Returns:
            List of (wav_bytes, resolved_voice, resolved_language) tuples, one per item.
        """
        import torch

        if self.tts_engine != "qwen3":
            raise ValueError("Batch TTS only supported for Qwen3 TTS")

        if not items:
            return []

        # Resolve per-item voice prompts and language names
        texts = []
        prompts = []
        lang_names = []

        for item in items:
            text = item.get("text", "").strip()
            if not text:
                raise ValueError("Each batch item must have non-empty text")
            texts.append(text)

            prompt, lang_name = self._resolve_voice(
                voice=item.get("voice"),
                language=item.get("language"),
            )
            if prompt is None:
                raise ValueError(f"No voice prompt available for item: {item}")
            prompts.append(prompt)
            lang_names.append(lang_name)

        start = time.perf_counter()

        # Wait for any streaming TTS to finish (shares the same CUDA-compiled model)
        if not self._tts_idle.wait(timeout=30):
            logger_tts.warning("Streaming TTS did not finish in time for batch, proceeding")
        self._tts_idle.clear()

        try:
            with torch.inference_mode():
                wavs, sr = self.tts_model.generate_voice_clone(
                    text=texts,
                    language=lang_names,
                    voice_clone_prompt=prompts,
                )

            results = []
            for i in range(len(items)):
                wav_bytes = numpy_to_wav_bytes(wavs[i], sr)
                results.append((wav_bytes, items[i].get("voice"), items[i].get("language")))

            elapsed = time.perf_counter() - start
            logger_tts.info(f"Batch TTS: {len(items)} items in {elapsed:.2f}s")

            return results
        finally:
            self._tts_idle.set()

    def tts_streaming_frames(self, text: str, **kwargs) -> Generator:
        """Frame-wrapped streaming TTS. Yields AudioFrame objects.

        This is a thin wrapper over tts_streaming() that adds frame metadata.
        Existing callers continue to use tts_streaming() directly.
        """
        from .frames import AudioFrame
        idx = 0
        for chunk, sr in self.tts_streaming(text, **kwargs):
            yield AudioFrame(audio=chunk, sample_rate=sr, chunk_index=idx, is_first=(idx == 0))
            idx += 1

    def tts_streaming(self, text: str, *, voice: str = None, language: str = None, first_chunk_emit_every: int = None) -> Generator[Tuple[bytes, int], None, None]:
        """Streaming TTS - yields (pcm_chunk, sample_rate) tuples."""
        if self.tts_engine == "qwen3":
            prompt, lang_name = self._resolve_voice(voice, language)
            if prompt:
                yield from self._tts_qwen3_streaming(text, voice_clone_prompt=prompt, language_name=lang_name, first_chunk_emit_every=first_chunk_emit_every)
                return
        # Non-streaming fallback
        wav_bytes = self.tts(text, voice=voice, language=language)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        yield audio, sr

    def _assert_no_overlap_duplication(self, prev: np.ndarray, curr: np.ndarray, n: int = 512, tol: float = 0.85) -> None:
        """Assert that chunk boundaries don't have duplicate overlap regions.

        Compares the last n samples of prev with the first n samples of curr.
        High correlation indicates overlap was not properly removed.

        Args:
            prev: Previous audio chunk
            curr: Current audio chunk
            n: Number of samples to compare at boundary
            tol: Correlation threshold (0.85 = 85% similarity triggers assertion)

        Raises:
            AssertionError: If overlap duplication detected
        """
        if prev.size < n or curr.size < n:
            return

        prev_tail = prev[-n:]
        curr_head = curr[:n]

        # Normalize for correlation
        prev_std = np.std(prev_tail)
        curr_std = np.std(curr_head)

        if prev_std < 1e-6 or curr_std < 1e-6:
            # Near-silence, skip check
            return

        prev_norm = (prev_tail - np.mean(prev_tail)) / prev_std
        curr_norm = (curr_head - np.mean(curr_head)) / curr_std

        correlation = np.abs(np.dot(prev_norm, curr_norm) / n)

        assert correlation < tol, (
            f"Overlap duplication detected: correlation={correlation:.3f} >= {tol} "
            f"(prev_tail vs curr_head, n={n})"
        )

    def _assert_no_rms_jump(self, prev: np.ndarray, curr: np.ndarray, max_ratio: float = 3.0) -> None:
        """Assert no sudden RMS volume jump between chunks.

        Large RMS jumps indicate discontinuities that cause audible clicks/pops.

        Args:
            prev: Previous audio chunk
            curr: Current audio chunk
            max_ratio: Maximum allowed RMS ratio between chunks

        Raises:
            AssertionError: If RMS jump exceeds threshold
        """
        # Use last/first 512 samples for boundary RMS
        n = min(512, prev.size, curr.size)
        if n < 64:
            return

        prev_rms = np.sqrt(np.mean(prev[-n:] ** 2))
        curr_rms = np.sqrt(np.mean(curr[:n] ** 2))

        # Skip if either is near silence (0.01 = ~-40dB, natural pauses)
        if prev_rms < 0.01 or curr_rms < 0.01:
            return

        ratio = max(prev_rms, curr_rms) / min(prev_rms, curr_rms)

        assert ratio <= max_ratio, (
            f"RMS jump detected: ratio={ratio:.2f} > {max_ratio} "
            f"(prev_rms={prev_rms:.4f}, curr_rms={curr_rms:.4f})"
        )

    def _tts_qwen3_streaming(self, text: str, *, voice_clone_prompt=None, language_name=None, first_chunk_emit_every: int = None) -> Generator[Tuple[bytes, int], None, None]:
        """Streaming TTS with Qwen3."""
        language = language_name or LANGUAGE_NAMES.get(self.language, "Spanish")
        prompt = voice_clone_prompt or self.voice_clone_prompt
        start = time.perf_counter()
        chunk_idx = 0
        prev_chunk = None

        # Wait for previous generation to finish (interrupt makes this fast)
        if not self._tts_idle.wait(timeout=5):
            logger_tts.warning("Previous TTS generation did not finish in time, proceeding anyway")

        self._tts_idle.clear()  # Mark busy
        self._interrupt.clear()

        try:
            for chunk, sr in self.tts_model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                emit_every_frames=PCM_EMIT_EVERY_FRAMES,
                decode_window_frames=PCM_DECODE_WINDOW,
                overlap_samples=0,
                first_chunk_emit_every=first_chunk_emit_every if first_chunk_emit_every is not None else FIRST_CHUNK_EMIT_EVERY,
                first_chunk_decode_window=FIRST_CHUNK_DECODE_WINDOW,
                first_chunk_frames=FIRST_CHUNK_FRAMES,
                max_frames=MAX_TTS_FRAMES,
                use_paged_engine=USE_PAGED_ENGINE,
            ):
                if self._interrupt.is_set():
                    logger_tts.info("TTS interrupted by barge-in")
                    return
                if chunk.size > 0:
                    # Runtime assertions to detect overlap/RMS issues
                    if prev_chunk is not None:
                        try:
                            self._assert_no_overlap_duplication(prev_chunk, chunk)
                            self._assert_no_rms_jump(prev_chunk, chunk)
                        except AssertionError as e:
                            if DEBUG:
                                logger_tts.warning(f"TTS chunk {chunk_idx} assertion: {e}")

                    logger_tts.debug(f"TTS chunk {chunk_idx}: {chunk.size} samples @ {(time.perf_counter()-start)*1000:.0f}ms")
                    prev_chunk = chunk
                    chunk_idx += 1
                    yield chunk, sr
        except Exception as e:
            logger_tts.error(f"Streaming TTS failed: {e}")
            raise
        finally:
            self._tts_idle.set()  # Mark idle

    def cleanup(self):
        """Release GPU memory and MCP connections before shutdown.

        Explicitly deletes TTS model and clears CUDA cache to prevent
        memory leaks when restarting or switching TTS engines.
        """
        import gc
        import torch

        # Shut down MCP client if active
        if self._mcp_manager is not None:
            try:
                self._mcp_manager.shutdown()
            except Exception as e:
                logger_sys.warning(f"MCP shutdown error: {e}")
            self._mcp_manager = None

        logger_sys.info("Releasing TTS resources...")

        # Clear TTS model
        if hasattr(self, 'tts_model') and self.tts_model is not None:
            del self.tts_model
            self.tts_model = None

        # Clear voice prompts (GPU tensors for Qwen3)
        if hasattr(self, 'all_voice_prompts') and self.all_voice_prompts:
            self.all_voice_prompts.clear()
        if hasattr(self, 'voice_prompts') and self.voice_prompts:
            self.voice_prompts.clear()
        if hasattr(self, 'voice_clone_prompt'):
            self.voice_clone_prompt = None

        # Force garbage collection and CUDA cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger_sys.info("TTS resources released")

    def interrupt(self):
        """Signal current TTS generation to stop."""
        self._interrupt.set()
        logger_tts.info("Barge-in interrupt signal received")

    @property
    def supports_streaming(self) -> bool:
        """Check if current config supports streaming."""
        return self.tts_engine == "qwen3" and self.voice_clone_prompt is not None

    def reload_prompt(self):
        """Reload system prompt from file (session-only, no restart required)."""
        # For Qwen3: try profile-specific prompt first
        # For Kokoro: profile_dir is None, use default prompt
        if self.profile_dir:
            prompt_file = self.profile_dir / "prompt.txt"
            if prompt_file.exists():
                self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()
                with self._context_lock:
                    self.context[0]["content"] = self._build_system_content()
                logger_sys.info(f"Reloaded prompt for profile {self.profile_dir.name}")
                return

        # Fallback to language default prompt
        self.system_prompt = self._load_default_prompt()
        with self._context_lock:
            self.context[0]["content"] = self._build_system_content()
        logger_sys.info(f"Reloaded default prompt for language {self.language}")

    def switch_voice(self, voice_name: str) -> bool:
        """Switch to a different voice profile (CUDA-safe, no restart required).

        Only switches the voice_clone_prompt pointer - no new allocations.
        All voices must be preloaded at startup.

        Returns True if switch successful, False if voice not found.
        """
        if self.tts_engine != "qwen3":
            logger_tts.warning("Voice switching only supported for Qwen3 TTS")
            return False

        if voice_name not in self.voice_prompts:
            logger_tts.error(f"Voice '{voice_name}' not preloaded. Available: {list(self.voice_prompts.keys())}")
            return False

        if voice_name == self.current_voice:
            logger_tts.info(f"Voice already set to {voice_name}")
            return True

        # Switch by reference only - CUDA-safe
        self.voice_clone_prompt = self.voice_prompts[voice_name]
        self.current_voice = voice_name
        self.profile_dir = Path(__file__).parent.parent / "profiles" / self.language / voice_name
        logger_tts.info(f"Switched to voice: {voice_name}")
        return True

    def switch_kokoro_voice(self, voice_name: str) -> bool:
        """Switch Kokoro voice (no restart required).

        Kokoro voices are just string parameters passed at inference time,
        so switching is trivial - no CUDA concerns.

        Returns True if voice is in the known voices list, False otherwise.
        """
        if self.tts_engine != "kokoro":
            logger_tts.warning("Kokoro voice switching only supported for Kokoro TTS")
            return False

        # Validate against known Kokoro voices
        all_ids = {v["id"] for voices in KOKORO_VOICES.values() for v in voices}
        if voice_name not in all_ids:
            logger_tts.error(f"Kokoro voice '{voice_name}' not in known voices: {sorted(all_ids)}")
            return False

        if voice_name == self.current_voice:
            logger_tts.info(f"Kokoro voice already set to {voice_name}")
            return True

        self.current_voice = voice_name
        logger_tts.info(f"Switched Kokoro voice to: {voice_name}")
        return True

    def switch_kokoro_language(self, new_language: str) -> bool:
        """Switch Kokoro TTS language (hot-reload, no restart required).

        KPipeline initializes language-specific G2P (grapheme-to-phoneme) processors
        during __init__. Simply changing lang_code doesn't reinitialize these processors.
        We must create a new KPipeline instance, but pass the existing KModel to avoid
        GPU reallocation (KModel is language-agnostic).

        Args:
            new_language: Language code (e.g., 'es', 'en', 'fr')

        Returns:
            True if switch successful, False if language not supported.
        """
        if self.tts_engine != "kokoro":
            logger_tts.warning("Kokoro language switching only supported for Kokoro TTS")
            return False

        if new_language == self.language:
            logger_tts.info(f"Language already set to {new_language}")
            return True

        # Check if language is supported by Kokoro
        if new_language not in KOKORO_LANG_CODES:
            logger_tts.error(f"Language '{new_language}' not supported by Kokoro. Supported: {list(KOKORO_LANG_CODES.keys())}")
            return False

        # Get existing KModel to reuse (language-agnostic, no VRAM increase)
        existing_model = self.tts_model.model

        # Create new KPipeline with new language, passing existing model
        # This reinitializes the G2P processors while reusing GPU weights
        lang_code = KOKORO_LANG_CODES[new_language]
        from kokoro import KPipeline
        self.tts_model = KPipeline(lang_code=lang_code, model=existing_model)

        self.language = new_language

        # Reload system prompt for the new language
        self.reload_prompt()

        # Clear conversation context (old messages are in the previous language)
        with self._context_lock:
            self.context = [{"role": "system", "content": self._build_system_content()}]

        logger_tts.info(f"Switched Kokoro language to: {new_language} (lang_code={lang_code})")
        return True

    def switch_language(self, new_language: str, voice_name: str = None) -> bool:
        """Switch to a different language (hot-reload, no restart required).

        Uses preloaded voice prompts — no disk I/O at runtime.

        Args:
            new_language: Language code (e.g., 'es', 'en', 'fr')
            voice_name: Optional voice to switch to. If None, uses first available.

        Returns True if switch successful, False if language has no voices.
        """
        if self.tts_engine != "qwen3":
            logger_tts.warning("Language switching only supported for Qwen3 TTS")
            return False

        if new_language == self.language and voice_name is None:
            logger_tts.info(f"Language already set to {new_language}")
            return True

        lang_voices = self.all_voice_prompts.get(new_language, {})
        if not lang_voices:
            logger_tts.error(f"No voices for language: {new_language}")
            return False

        # Determine target voice
        if voice_name and voice_name in lang_voices:
            target_voice = voice_name
        elif voice_name:
            logger_tts.warning(f"Voice '{voice_name}' not available in {new_language}, using first available")
            target_voice = next(iter(lang_voices))
        else:
            target_voice = next(iter(lang_voices))

        # Update state (all reference swaps, no allocation)
        self.language = new_language
        self.voice_prompts = lang_voices
        self.current_voice = target_voice
        self.voice_clone_prompt = lang_voices[target_voice]
        self.profile_dir = Path(__file__).parent.parent / "profiles" / new_language / target_voice

        # Reload system prompt for the new profile
        self.reload_prompt()

        # Clear conversation context (old messages are in the previous language)
        with self._context_lock:
            self.context = [{"role": "system", "content": self._build_system_content()}]

        logger_tts.info(f"Switched to {new_language}/{target_voice} ({len(lang_voices)} voices)")
        return True

    def _get_asr_language(self) -> Optional[str]:
        """Get ASR language name for current pipeline language, or None for auto-detect."""
        return ASR_LANGUAGE_MAP.get(self.language)

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """Transcribe audio to text using ASR subprocess.

        Args:
            wav_bytes: Raw WAV audio bytes.
            language: Optional language code override (e.g., 'es', 'en').
                      If None, uses the pipeline's current language.
        """
        try:
            # Convert WAV bytes to numpy array
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Resolve ASR language: explicit override or pipeline default
            asr_language = ASR_LANGUAGE_MAP.get(language) if language else self._get_asr_language()

            # Send to ASR subprocess via pickle IPC (numpy arrays serialized directly)
            response = self._asr_call({"action": "transcribe", "audio": audio, "sr": sr, "language": asr_language})

            if response.get("status") == "result":
                text = response.get("text", "")
                if text:
                    logger_asr.debug(f"Transcribed: {text[:50]}...")
                return text
            elif response.get("status") == "error":
                logger_asr.error(f"ASR transcription error: {response.get('message')}")
                return ""
            else:
                logger_asr.warning(f"Unexpected ASR response: {response}")
                return ""
        except Exception as e:
            logger_asr.error(f"ASR transcription failed: {e}")
            return ""

    def transcribe_streaming_init(self, language: str | None = None):
        """Initialize streaming ASR session via subprocess.

        Args:
            language: Optional language code override (e.g., 'en').
                      If None, uses the pipeline's current language.
        """
        asr_language = ASR_LANGUAGE_MAP.get(language) if language else self._get_asr_language()
        response = self._asr_call({"action": "stream_init", "language": asr_language})
        if response.get("status") == "error":
            raise RuntimeError(f"ASR stream init failed: {response.get('message')}")

    def transcribe_streaming_chunk(self, pcm16k: np.ndarray) -> str:
        """Process one audio chunk via subprocess, return current transcript."""
        response = self._asr_call({"action": "stream_chunk", "audio": pcm16k})
        if response.get("status") == "error":
            logger_asr.error(f"ASR stream chunk error: {response.get('message')}")
            return ""
        return response.get("text", "")

    def transcribe_streaming_finish(self) -> str:
        """Finalize streaming transcription via subprocess."""
        response = self._asr_call({"action": "stream_finish"})
        if response.get("status") == "error":
            logger_asr.error(f"ASR stream finish error: {response.get('message')}")
            return ""
        return response.get("text", "")

    def chat(self, text: str, image: Optional[str] = None) -> str:
        """Send message to chat model and get response.

        Args:
            text: The user's message text
            image: Optional base64-encoded image data (for vision models)

        Returns:
            The assistant's response text
        """
        with self._context_lock:
            user_message = self.llm_provider.build_user_message(text, image)
            self.context.append(user_message)

            response = self.llm_provider.chat(
                messages=self.context,
                options=self._get_llm_options(),
            )

            response_text = (response.content or "").replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()
            logger_llm.debug(f"LLM response: {response_text}")

            # Strip prosody tags from context to prevent [pause:X] polluting conversation history
            context_text = strip_prosody_tags(response_text)
            self.context.append({"role": "assistant", "content": context_text})
            self._trim_context()

        return response_text

    def chat_with_tools(self, text: str, image: Optional[str] = None) -> str:
        """Send message to chat model with tool calling support.

        Uses non-streaming chat in a loop: if the LLM returns tool_calls,
        execute them and feed results back until the LLM produces a final
        text response. Capped at MAX_TOOL_ITERATIONS to prevent runaway loops.

        Args:
            text: The user's message text
            image: Optional base64-encoded image data (for vision models)

        Returns:
            The assistant's final text response (after all tool calls resolved)
        """
        from .tools import MAX_TOOL_ITERATIONS

        with self._context_lock:
            user_message = self.llm_provider.build_user_message(text, image)
            self.context.append(user_message)

            raw_tools = self.tool_registry.get_enabled_functions()
            tools = self.llm_provider.format_tools(raw_tools)

            for iteration in range(MAX_TOOL_ITERATIONS):
                response = self.llm_provider.chat(
                    messages=self.context,
                    tools=tools,
                    options=self._get_llm_options(),
                )

                if not response.tool_calls:
                    # Final text response — no more tool calls
                    logger_llm.info(f"No tool calls on iteration {iteration + 1} — LLM produced final text response")
                    break

                # Process tool calls — append raw_message for provider-specific context
                self.context.append(response.raw_message)
                for tc in response.tool_calls:
                    logger_llm.info(f"Tool call: {tc.name}({tc.arguments})")
                    result = self.tool_registry.execute(tc.name, tc.arguments)
                    logger_llm.info(f"Tool result: {result[:100]}")
                    self.context.append(
                        self.llm_provider.build_tool_result_message(tc.id, tc.name, result)
                    )
            else:
                # Exhausted iterations — force a no-tools response
                logger_llm.warning(f"Tool loop hit max iterations ({MAX_TOOL_ITERATIONS}), forcing final response")
                response = self.llm_provider.chat(
                    messages=self.context,
                    options=self._get_llm_options(),
                )

            response_text = (response.content or "").replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()
            logger_llm.debug(f"LLM response (with tools): {response_text}")

            context_text = strip_prosody_tags(response_text)
            self.context.append({"role": "assistant", "content": context_text})
            self._trim_context()

        return response_text

    def chat_streaming_tokens(self, text: str, image: Optional[str] = None) -> Iterator[str]:
        """Stream raw LLM tokens without chunking.

        Returns individual tokens as they arrive from the LLM provider.
        Caller handles gating/buffering for early TTS.

        Args:
            text: The user's message text
            image: Optional base64-encoded image data (for vision models)

        Yields:
            Individual tokens from the LLM
        """
        user_message = self.llm_provider.build_user_message(text, image)
        with self._context_lock:
            self.context.append(user_message)

        full_response = ""
        completed = False
        try:
            for token in self.llm_provider.chat(
                messages=self.context,
                stream=True,
                options=self._get_llm_options(),
            ):
                full_response += token
                yield token
            completed = True
        finally:
            # Always save response to context, even if interrupted (barge-in)
            if full_response.strip():
                clean_full = full_response.replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()
                clean_full = strip_prosody_tags(clean_full)
                with self._context_lock:
                    if not completed:
                        self.context.append({"role": "assistant", "content": f"{clean_full} [interrupted]"})
                        logger_llm.info(f"Context saved (interrupted): {clean_full[:80]}...")
                    else:
                        logger_llm.debug(f"LLM response: {clean_full}")
                        self.context.append({"role": "assistant", "content": clean_full})
                    self._trim_context()

    def correct_interrupted_context(self, spoken_text: str) -> None:
        """Annotate last assistant context entry with what the user heard vs the full response.

        Called by api.py when barge-in happens after LLM completed but during TTS.
        The full response was already saved by chat_streaming_tokens(), so this
        retroactively annotates it so the LLM knows what was heard and won't repeat itself.

        Format preserves [interrupted] as the marker (so existing system prompt rules fire)
        while appending the full response so the LLM has memory and won't repeat content.
        """
        spoken_clean = strip_prosody_tags(
            spoken_text.replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()
        )
        with self._context_lock:
            for i in range(len(self.context) - 1, -1, -1):
                if self.context[i].get("role") == "assistant":
                    full = self.context[i]["content"]
                    if not spoken_clean:
                        self.context[i]["content"] = "[interrupted before speaking]"
                    else:
                        # Extract only the unsaid portion (full = spoken + unsaid)
                        if full.startswith(spoken_clean):
                            unsaid = full[len(spoken_clean):].strip()
                        else:
                            unsaid = full

                        if unsaid:
                            self.context[i]["content"] = (
                                f"{spoken_clean} [interrupted]\n"
                                f"[your full unsaid response was: {unsaid}]"
                            )
                        else:
                            # User heard everything — just mark as interrupted
                            self.context[i]["content"] = f"{spoken_clean} [interrupted]"
                    logger_llm.info(
                        f"Context corrected for barge-in: '{full[:60]}...' → '{self.context[i]['content'][:60]}...'"
                    )
                    return
