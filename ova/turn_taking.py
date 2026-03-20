"""Turn-taking state machine for full-duplex voice assistant.

Event-driven: receives VAD, ASR, and TTS events, produces decisions that
the DuplexSession acts on.
"""

import time
from enum import Enum, auto
from typing import Optional

from .utils import get_logger

logger = get_logger("turn")


class State(Enum):
    IDLE = auto()
    USER_SPEAKING = auto()
    BOT_THINKING = auto()
    BOT_SPEAKING = auto()


class Decision(Enum):
    START_ASR = auto()
    END_TURN = auto()
    INTERRUPT = auto()
    BACKCHANNEL_IGNORE = auto()


# Per-language backchannel phrase sets (server-side copy of config.js BACKCHANNELS)
BACKCHANNELS = {
    "es": {"si", "sí", "vale", "ajá", "aja", "mmm", "ok", "okay", "claro", "ya", "bien", "ah", "oh"},
    "en": {"yeah", "yes", "ok", "okay", "uh huh", "hmm", "right", "sure", "mhm", "mm", "ah", "oh", "got it", "i see"},
    "fr": {"oui", "ouais", "mmm", "d'accord", "ok", "okay", "ah", "oh", "bien", "bon"},
    "de": {"ja", "ok", "okay", "mmm", "ah", "oh", "genau", "richtig", "gut", "stimmt"},
    "it": {"sì", "si", "ok", "okay", "mmm", "ah", "oh", "bene", "certo", "giusto"},
    "pt": {"sim", "ok", "okay", "mmm", "ah", "oh", "bem", "certo", "tá"},
    "ja": {"はい", "うん", "ああ", "ええ", "そう", "ok", "okay", "mmm"},
    "zh": {"嗯", "对", "好", "是", "哦", "ok", "okay", "mmm"},
    "ko": {"네", "응", "어", "그래", "ok", "okay", "mmm"},
    "ru": {"да", "ага", "угу", "ок", "хорошо", "ладно", "mmm"},
    "hi": {"हाँ", "हां", "ठीक", "अच्छा", "ok", "okay", "mmm"},
}


class TurnTakingStateMachine:
    """Manages conversational turn-taking for a single duplex session.

    Thread-safe: all mutations go through event methods which are called
    from the DuplexSession's single asyncio task.
    """

    def __init__(
        self,
        language: str = "en",
        silence_timeout_ms: float = 800,
        bot_stop_delay_ms: float = 500,
        backchannel_timeout_ms: float = 500,
        interrupt_cooldown_ms: float = 1200,
    ):
        self.language = language
        self.silence_timeout_ms = silence_timeout_ms
        self.bot_stop_delay_ms = bot_stop_delay_ms
        self.backchannel_timeout_ms = backchannel_timeout_ms
        self.interrupt_cooldown_ms = interrupt_cooldown_ms

        self._state = State.IDLE
        self._last_interrupt_time: float = 0.0

        # Timestamps for timeouts
        self._speech_end_time: Optional[float] = None  # when VAD last went silent
        self._tts_complete_time: Optional[float] = None  # when TTS generation finished
        self._bot_speech_vad_time: Optional[float] = None  # VAD onset during bot speaking

        # Dynamic bot_stop_delay: accounts for client playback buffer
        self._dynamic_bot_stop_delay_ms: Optional[float] = None

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, new_state: State):
        if new_state != self._state:
            logger.info(f"Turn state: {self._state.name} → {new_state.name}")
            self._state = new_state

    # ------------------------------------------------------------------
    # VAD events
    # ------------------------------------------------------------------

    def on_vad_speech_start(self) -> Optional[Decision]:
        """Called when server VAD detects speech onset."""
        now = time.monotonic()

        if self._state == State.IDLE:
            self.state = State.USER_SPEAKING
            self._speech_end_time = None
            return Decision.START_ASR

        if self._state == State.USER_SPEAKING:
            # Already speaking — cancel any pending silence timeout
            self._speech_end_time = None
            return None

        if self._state == State.BOT_THINKING:
            # User spoke while bot is thinking — interrupt and take the floor
            logger.info("User speech during BOT_THINKING — interrupting")
            self._last_interrupt_time = now
            self.state = State.USER_SPEAKING
            self._speech_end_time = None
            return Decision.INTERRUPT

        if self._state == State.BOT_SPEAKING:
            # Potential interruption — wait for ASR partial to decide
            # Check cooldown
            if now - self._last_interrupt_time < self.interrupt_cooldown_ms / 1000:
                logger.debug("Interrupt cooldown active, ignoring VAD onset")
                return None
            self._bot_speech_vad_time = now
            return Decision.START_ASR  # Start ASR to classify backchannel vs real speech

        return None

    def on_vad_speech_end(self) -> Optional[Decision]:
        """Called when server VAD detects speech offset (silence)."""
        if self._state == State.USER_SPEAKING:
            self._speech_end_time = time.monotonic()
            # Don't transition yet — check_timeouts() will fire END_TURN
            return None

        if self._state == State.BOT_SPEAKING:
            # Speech ended during bot speaking — don't clear _bot_speech_vad_time
            # so that on_asr_partial() and backchannel timeout can still classify
            return None

        return None

    # ------------------------------------------------------------------
    # ASR events
    # ------------------------------------------------------------------

    def on_asr_partial(self, text: str) -> Optional[Decision]:
        """Called when ASR returns a partial transcript."""
        if self._state == State.BOT_SPEAKING and self._bot_speech_vad_time is not None:
            # We're in the "is this a backchannel?" evaluation window
            if self._is_backchannel(text):
                logger.info(f"Backchannel detected: '{text}' — ignoring")
                self._bot_speech_vad_time = None
                return Decision.BACKCHANNEL_IGNORE
            else:
                # Real speech — interrupt
                logger.info(f"Real speech during bot: '{text}' — interrupting")
                self._last_interrupt_time = time.monotonic()
                self._bot_speech_vad_time = None
                self.state = State.USER_SPEAKING
                self._speech_end_time = None
                return Decision.INTERRUPT

        return None

    def on_asr_final(self, text: str) -> Optional[Decision]:
        """Called when ASR returns a final transcript (end of streaming session)."""
        if self._state == State.USER_SPEAKING:
            self.state = State.BOT_THINKING
            self._speech_end_time = None
            return Decision.END_TURN

        return None

    # ------------------------------------------------------------------
    # TTS events
    # ------------------------------------------------------------------

    def on_tts_chunk_sent(self) -> None:
        """Called when a TTS audio chunk is sent to the client."""
        if self._state == State.BOT_THINKING:
            # Clear stale timestamps from any previous response (e.g. one
            # that was interrupted after its TTS had already completed).
            self._tts_complete_time = None
            self._dynamic_bot_stop_delay_ms = None
            self.state = State.BOT_SPEAKING

    def on_tts_complete(self) -> None:
        """Called when TTS generation is fully complete."""
        self._tts_complete_time = time.monotonic()

    def set_dynamic_bot_stop_delay(self, remaining_playback_ms: float) -> None:
        """Set bot_stop_delay to estimated remaining client playback time + base.

        The caller should pass the *remaining* playback time (total minus
        elapsed streaming time), not the total duration.
        Resets to base delay after transitioning to IDLE.
        """
        self._dynamic_bot_stop_delay_ms = max(
            self.bot_stop_delay_ms, remaining_playback_ms + self.bot_stop_delay_ms
        )
        logger.debug(
            f"Dynamic bot_stop_delay set to {self._dynamic_bot_stop_delay_ms:.0f}ms "
            f"(remaining {remaining_playback_ms:.0f}ms + base {self.bot_stop_delay_ms:.0f}ms)"
        )

    def on_bot_stop_expired(self) -> None:
        """Called when bot_stop_delay has elapsed after TTS complete."""
        if self._state == State.BOT_SPEAKING:
            self.state = State.IDLE
            self._tts_complete_time = None
            self._dynamic_bot_stop_delay_ms = None  # reset to base

    # ------------------------------------------------------------------
    # Timeout checks (called periodically by DuplexSession)
    # ------------------------------------------------------------------

    def check_timeouts(self) -> Optional[Decision]:
        """Check for pending timeouts. Returns a decision if one fires."""
        now = time.monotonic()

        # Silence timeout → end of user turn
        if (
            self._state == State.USER_SPEAKING
            and self._speech_end_time is not None
            and (now - self._speech_end_time) * 1000 >= self.silence_timeout_ms
        ):
            self.state = State.BOT_THINKING
            self._speech_end_time = None
            return Decision.END_TURN

        # Bot stop delay → transition to IDLE (use dynamic delay if set)
        effective_delay = self._dynamic_bot_stop_delay_ms or self.bot_stop_delay_ms
        if (
            self._state == State.BOT_SPEAKING
            and self._tts_complete_time is not None
            and (now - self._tts_complete_time) * 1000 >= effective_delay
        ):
            self.state = State.IDLE
            self._tts_complete_time = None
            self._dynamic_bot_stop_delay_ms = None  # reset to base
            return None

        # Backchannel timeout: if VAD triggered during bot speaking but no ASR partial
        # came within the timeout, treat as interruption
        if (
            self._state == State.BOT_SPEAKING
            and self._bot_speech_vad_time is not None
            and (now - self._bot_speech_vad_time) * 1000 >= self.backchannel_timeout_ms
        ):
            logger.info("Backchannel timeout — treating as interruption")
            self._last_interrupt_time = now
            self._bot_speech_vad_time = None
            self.state = State.USER_SPEAKING
            self._speech_end_time = None
            return Decision.INTERRUPT

        return None

    # ------------------------------------------------------------------
    # External state changes
    # ------------------------------------------------------------------

    def force_idle(self):
        """Force-reset to IDLE (e.g. on disconnect or session end)."""
        self.state = State.IDLE
        self._speech_end_time = None
        self._tts_complete_time = None
        self._bot_speech_vad_time = None
        self._dynamic_bot_stop_delay_ms = None

    def set_language(self, language: str):
        """Update language for backchannel detection."""
        self.language = language

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_backchannel(self, text: str) -> bool:
        """Check if ASR text matches a backchannel phrase for the current language."""
        if not text:
            return False
        normalized = text.strip().lower()
        phrases = BACKCHANNELS.get(self.language, BACKCHANNELS.get("en", set()))
        return normalized in phrases
