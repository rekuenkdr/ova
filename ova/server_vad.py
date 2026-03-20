"""Server-side Silero VAD using ONNX Runtime.

Mirrors the sliding-window detection logic from client vad.js but runs
server-side so duplex sessions can perform turn-taking without round-tripping
to the browser.

Uses the same /models/silero_vad_16k_op15.onnx model already shipped with OVA.
"""

import numpy as np
from pathlib import Path

from .utils import get_logger

logger = get_logger("vad")

# Silero constants (must match model expectations)
_SILERO_SR = 16000
_SILERO_WINDOW = 512  # 512 samples at 16kHz = 32ms per frame
_FRAME_MS = (_SILERO_WINDOW / _SILERO_SR) * 1000  # 32ms


class ServerVAD:
    """Silero VAD wrapper for server-side voice activity detection.

    Accepts raw PCM int16 at 16kHz.  Produces speech_start / speech_end
    callbacks using the same N-of-M onset / consecutive-silence offset
    state machine as the client ``vad.js``.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        confirm_ms: float = 64,
        silence_ms: float = 320,
        on_speech_start=None,
        on_speech_end=None,
    ):
        self.threshold = threshold

        # Frame counts derived from ms
        self.confirm_frames = max(1, int(np.ceil(confirm_ms / _FRAME_MS)))
        self.silence_frames = max(1, int(np.ceil(silence_ms / _FRAME_MS)))

        # Sliding window for onset (N-of-M)
        self._confirm_required = self.confirm_frames
        self._confirm_window = max(self._confirm_required, int(np.ceil(self._confirm_required * 1.5)))

        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        # Event queue for polling-based consumption (used by duplex mode)
        self._pending_events: list[str] = []

        # ONNX session (lazy-loaded, shared across instances via class var)
        self._session = None

        # Silero LSTM hidden state
        self._state = np.zeros(2 * 1 * 128, dtype=np.float32)
        self._sr_tensor = None  # cached

        # Accumulation buffer for incoming PCM
        self._buf = np.array([], dtype=np.float32)

        # Detection state machine
        self._speech_active = False
        self._frames_above = 0
        self._frames_below = 0
        self._frame_history = [False] * self._confirm_window
        self._frame_history_idx = 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    _shared_session = None  # class-level: one ONNX session for all instances

    @classmethod
    def _load_model(cls):
        if cls._shared_session is not None:
            return cls._shared_session

        model_path = Path(__file__).parent.parent / "models" / "silero_vad_16k_op15.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {model_path}")

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # suppress warnings

        cls._shared_session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Silero VAD ONNX model loaded (server-side)")
        return cls._shared_session

    def _ensure_session(self):
        if self._session is None:
            self._session = self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def feed(self, pcm_int16: bytes | np.ndarray):
        """Feed raw PCM int16 samples at 16kHz.  Fires callbacks as needed."""
        self._ensure_session()

        if isinstance(pcm_int16, (bytes, bytearray)):
            samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        elif pcm_int16.dtype == np.int16:
            samples = pcm_int16.astype(np.float32) / 32768.0
        else:
            samples = pcm_int16.astype(np.float32)

        self._buf = np.concatenate([self._buf, samples])

        while len(self._buf) >= _SILERO_WINDOW:
            window = self._buf[:_SILERO_WINDOW]
            self._buf = self._buf[_SILERO_WINDOW:]
            prob = self._infer(window)
            self._handle_detection(prob)

    def drain_events(self) -> list[str]:
        """Return and clear pending VAD events (``"start"`` / ``"end"``)."""
        events = self._pending_events
        self._pending_events = []
        return events

    def reset(self):
        """Reset all state (for new session)."""
        self._state = np.zeros(2 * 1 * 128, dtype=np.float32)
        self._buf = np.array([], dtype=np.float32)
        self._speech_active = False
        self._frames_above = 0
        self._frames_below = 0
        self._frame_history = [False] * self._confirm_window
        self._frame_history_idx = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _infer(self, window: np.ndarray) -> float:
        """Run Silero inference on a single 512-sample window, return probability."""
        input_tensor = window.reshape(1, _SILERO_WINDOW)
        state_tensor = self._state.reshape(2, 1, 128)

        if self._sr_tensor is None:
            self._sr_tensor = np.array([_SILERO_SR], dtype=np.int64)

        results = self._session.run(
            None,
            {"input": input_tensor, "state": state_tensor, "sr": self._sr_tensor},
        )

        prob = float(results[0][0])
        self._state = results[1].flatten().copy()
        return prob

    def _handle_detection(self, prob: float):
        """State machine: sliding window onset, consecutive silence offset."""
        is_above = prob >= self.threshold

        if not self._speech_active:
            # --- Speech onset: N out of M ---
            old_val = 1 if self._frame_history[self._frame_history_idx] else 0
            self._frame_history[self._frame_history_idx] = is_above
            self._frame_history_idx = (self._frame_history_idx + 1) % self._confirm_window
            self._frames_above += (1 if is_above else 0) - old_val

            if is_above:
                self._frames_below = 0
            else:
                self._frames_below += 1

            if self._frames_above >= self._confirm_required:
                self._speech_active = True
                self._frames_below = 0
                self._pending_events.append("start")
                if self.on_speech_start:
                    self.on_speech_start()
        else:
            # --- Silence offset: consecutive frames ---
            if is_above:
                self._frames_below = 0
            else:
                self._frames_below += 1
                if self._frames_below >= self.silence_frames:
                    self._speech_active = False
                    self._frames_above = 0
                    self._frames_below = 0
                    self._frame_history = [False] * self._confirm_window
                    self._frame_history_idx = 0
                    self._pending_events.append("end")
                    if self.on_speech_end:
                        self.on_speech_end()
