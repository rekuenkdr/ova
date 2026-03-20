"""Full-duplex WebSocket session for OVA voice assistant.

Manages a single persistent WebSocket connection that carries bidirectional
audio (PCM) and JSON control/status messages.  Three concurrent async tasks
handle ingest (mic → VAD → ASR), output (TTS PCM + JSON → client), and
processing (turn-taking decisions → LLM + TTS orchestration).
"""

import asyncio
import json
import time
from typing import Optional

import anyio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .server_vad import ServerVAD
from .turn_taking import TurnTakingStateMachine, State, Decision
from .utils import get_logger

logger = get_logger("duplex")


class DuplexSession:
    """Manages one full-duplex WebSocket connection lifecycle.

    Args:
        websocket: The accepted FastAPI WebSocket.
        pipeline: The OVAPipeline instance.
        generate_interleaved_fn: Reference to ``generate_interleaved_audio_stream``.
        generate_audio_fn: Reference to ``generate_audio_stream``.
        tools_active_fn: Callable returning bool (are tools enabled?).
        clean_markdown_fn: Reference to ``_clean_markdown``.
        language: Initial language code.
        voice: Initial voice name.
        vad_threshold: Server-side VAD threshold.
        vad_confirm_ms: VAD speech onset confirmation (ms).
        vad_silence_ms: VAD silence for speech offset (ms).
        silence_timeout_ms: Silence to consider end-of-turn.
        bot_stop_delay_ms: Grace after TTS finishes before IDLE.
        backchannel_timeout_ms: Max wait for ASR to classify backchannel.
        interrupt_cooldown_ms: Cooldown between interruptions.
        inactivity_timeout_s: Close session after N seconds of no speech.
        sample_rate: TTS output sample rate (for session.started message).
    """

    def __init__(
        self,
        websocket: WebSocket,
        pipeline,
        generate_interleaved_fn,
        generate_audio_fn,
        tools_active_fn,
        clean_markdown_fn,
        language: str = "en",
        voice: Optional[str] = None,
        vad_threshold: float = 0.5,
        vad_confirm_ms: float = 64,
        vad_silence_ms: float = 320,
        silence_timeout_ms: float = 800,
        bot_stop_delay_ms: float = 500,
        backchannel_timeout_ms: float = 500,
        interrupt_cooldown_ms: float = 0,
        inactivity_timeout_s: float = 300,
        sample_rate: int = 24000,
    ):
        self.ws = websocket
        self.pipeline = pipeline
        self._generate_interleaved = generate_interleaved_fn
        self._generate_audio = generate_audio_fn
        self._tools_active = tools_active_fn
        self._clean_markdown = clean_markdown_fn
        self.language = language
        self.voice = voice
        self.sample_rate = sample_rate
        self.inactivity_timeout_s = inactivity_timeout_s

        # Output queue: items are (type, data) tuples
        # type="binary" → data is bytes, type="json" → data is dict
        self._output_queue: asyncio.Queue = asyncio.Queue()

        # Server-side VAD (no callbacks — we poll via drain_events)
        self._vad = ServerVAD(
            threshold=vad_threshold,
            confirm_ms=vad_confirm_ms,
            silence_ms=vad_silence_ms,
        )

        # Turn-taking state machine
        self._turn = TurnTakingStateMachine(
            language=language,
            silence_timeout_ms=silence_timeout_ms,
            bot_stop_delay_ms=bot_stop_delay_ms,
            backchannel_timeout_ms=backchannel_timeout_ms,
            interrupt_cooldown_ms=interrupt_cooldown_ms,
        )

        # ASR state
        self._asr_active = False
        self._asr_text = ""  # latest partial/final
        # Buffer PCM at 16kHz to send ~500ms chunks to ASR (matches half-duplex)
        # Tiny per-frame calls (~85ms) cause excessive IPC + model forward passes
        self._asr_chunk_target = 8000  # 500ms at 16kHz
        self._asr_buffer = np.empty(0, dtype=np.float32)

        # Pre-speech ring buffer: keeps last ~500ms of audio so ASR gets the
        # speech onset that triggered VAD (otherwise first words are lost)
        self._pre_speech_capacity = 8000  # 500ms at 16kHz
        self._pre_speech_buf = np.empty(0, dtype=np.float32)

        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_activity = time.monotonic()
        # Generation counter: output loop only drops audio from stale generations
        self._gen_id = 0

        # Playback duration tracking (Fix 4)
        self._audio_bytes_sent = 0

        # Pending image for next voice turn (sent via session.image)
        self._pending_image: Optional[str] = None

        # Backchannel ASR during bot speaking
        self._backchannel_asr_active = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self):
        """Run the duplex session until disconnect or inactivity timeout."""
        self._running = True
        self._last_activity = time.monotonic()

        # Send session.started
        await self._send_json({"type": "session.started", "sample_rate": self.sample_rate})

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._ingest_loop)
                tg.start_soon(self._output_loop)
                tg.start_soon(self._timeout_loop)
        except (WebSocketDisconnect, asyncio.CancelledError):
            logger.info("Duplex session ended")
        except Exception as e:
            logger.error(f"Duplex session error: {e}")
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Ingest loop: reads WebSocket frames → VAD → ASR
    # ------------------------------------------------------------------

    async def _ingest_loop(self):
        """Read binary (audio) and text (JSON) frames from the WebSocket."""
        try:
            while self._running:
                data = await self.ws.receive()

                if data["type"] == "websocket.disconnect":
                    self._running = False
                    return

                if data["type"] != "websocket.receive":
                    continue

                if "bytes" in data and data["bytes"]:
                    raw = data["bytes"]
                    # Cap frame size (1MB)
                    if len(raw) > 1_048_576:
                        continue
                    self._last_activity = time.monotonic()
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    # Feed PCM int16 16kHz to VAD (runs synchronously, fast)
                    self._vad.feed(raw)

                    # Maintain pre-speech ring buffer (keeps last ~500ms)
                    # so ASR gets the speech onset that triggered VAD
                    if not self._asr_active and not self._backchannel_asr_active:
                        self._pre_speech_buf = np.concatenate([self._pre_speech_buf, pcm])
                        if len(self._pre_speech_buf) > self._pre_speech_capacity:
                            self._pre_speech_buf = self._pre_speech_buf[-self._pre_speech_capacity:]

                    # Poll VAD events inline (no scheduling delay)
                    for vad_event in self._vad.drain_events():
                        await self._handle_vad_event(vad_event)

                    # If ASR is active, buffer audio and send ~500ms chunks
                    if self._asr_active or self._backchannel_asr_active:
                        self._asr_buffer = np.concatenate([self._asr_buffer, pcm])

                        # Only call ASR when we have enough audio (~500ms)
                        while len(self._asr_buffer) >= self._asr_chunk_target:
                            chunk = self._asr_buffer[:self._asr_chunk_target]
                            self._asr_buffer = self._asr_buffer[self._asr_chunk_target:]
                            try:
                                text = await anyio.to_thread.run_sync(
                                    self.pipeline.transcribe_streaming_chunk, chunk
                                )
                            except Exception:
                                # ASR session not initialized — clear flags to stop feeding
                                self._asr_active = False
                                self._backchannel_asr_active = False
                                self._asr_buffer = np.empty(0, dtype=np.float32)
                                break
                            if text:
                                self._asr_text = text
                                await self._send_json({"type": "transcript", "text": text, "is_final": False})
                                # Check if this is backchannel during bot speaking
                                if self._backchannel_asr_active:
                                    decision = self._turn.on_asr_partial(text)
                                    if decision:
                                        await self._handle_decision(decision, text)

                elif "text" in data and data["text"]:
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "session.end":
                        self._running = False
                        await self._send_json({"type": "session.ended"})
                        return

                    elif msg_type == "session.interrupt":
                        # Manual interrupt from talkBtn tap
                        if self._turn.state == State.BOT_SPEAKING:
                            await self._handle_interrupt()

                    elif msg_type == "session.image":
                        # Queue image for next voice turn
                        self._pending_image = msg.get("image")
                        logger.debug("Image queued for next turn")

                    elif msg_type == "session.text":
                        # Text input from UI — process as if user spoke it
                        text = msg.get("text", "").strip()
                        image = msg.get("image")
                        if image:
                            self._pending_image = image
                        if text:
                            self._last_activity = time.monotonic()
                            # Interrupt any in-progress response
                            if self._processing_task and not self._processing_task.done():
                                self.pipeline.interrupt()
                                try:
                                    await asyncio.wait_for(self._processing_task, timeout=5.0)
                                except Exception:
                                    pass
                            # Stop any active ASR
                            if self._asr_active:
                                try:
                                    await anyio.to_thread.run_sync(self.pipeline.transcribe_streaming_finish)
                                except Exception:
                                    pass
                                self._asr_active = False
                            if self._backchannel_asr_active:
                                await self._stop_backchannel_asr()
                            self._turn.force_idle()
                            self._turn.state = State.BOT_THINKING
                            await self._send_json({"type": "bot.thinking"})
                            logger.info(f"User typed: {text}")
                            img = self._pending_image
                            self._pending_image = None
                            self._processing_task = asyncio.ensure_future(
                                self._process_response(text, image=img)
                            )

                    elif msg_type == "session.config":
                        # Runtime config update — sync pipeline state like /v1/settings does
                        if "language" in msg:
                            new_lang = msg["language"]
                            self.language = new_lang
                            self._turn.set_language(new_lang)
                            if self.pipeline.tts_engine == "qwen3":
                                self.pipeline.switch_language(new_lang)
                            else:
                                self.pipeline.switch_kokoro_language(new_lang)
                        if "voice" in msg:
                            new_voice = msg["voice"]
                            self.voice = new_voice
                            if self.pipeline.tts_engine == "qwen3":
                                self.pipeline.switch_voice(new_voice)
                            else:
                                self.pipeline.switch_kokoro_voice(new_voice)

        except WebSocketDisconnect:
            self._running = False
        except Exception as e:
            logger.error(f"Ingest error: {e}")
            self._running = False

    # ------------------------------------------------------------------
    # Output loop: drains queue → sends to WebSocket
    # ------------------------------------------------------------------

    async def _output_loop(self):
        """Send queued binary audio and JSON messages to the WebSocket."""
        try:
            while self._running:
                try:
                    item = await asyncio.wait_for(
                        self._output_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if not self._running:
                    return

                try:
                    if item[0] == "binary":
                        _, data, gen = item
                        # Drop audio from a stale generation (interrupted)
                        if gen != self._gen_id:
                            continue
                        await self.ws.send_bytes(data)
                    elif item[0] == "json":
                        await self.ws.send_json(item[1])
                except (WebSocketDisconnect, RuntimeError):
                    self._running = False
                    return
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Timeout loop: checks turn-taking timeouts + inactivity
    # ------------------------------------------------------------------

    async def _timeout_loop(self):
        """Periodically check turn-taking timeouts and session inactivity."""
        try:
            prev_state = self._turn.state
            while self._running:
                await asyncio.sleep(0.05)  # 50ms tick

                # Turn-taking timeouts
                decision = self._turn.check_timeouts()
                if decision:
                    await self._handle_decision(decision)

                # Detect BOT_SPEAKING → IDLE transition (bot_stop_delay expired)
                cur_state = self._turn.state
                if prev_state == State.BOT_SPEAKING and cur_state == State.IDLE:
                    # Clean up any lingering backchannel ASR
                    if self._backchannel_asr_active:
                        await self._stop_backchannel_asr()
                    await self._send_json({"type": "bot.idle"})
                prev_state = cur_state

                # Inactivity timeout
                if time.monotonic() - self._last_activity > self.inactivity_timeout_s:
                    logger.info("Duplex session inactivity timeout")
                    await self._send_json({"type": "session.ended"})
                    self._running = False
                    return
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # VAD event handling (polled inline from ingest loop)
    # ------------------------------------------------------------------

    async def _handle_vad_event(self, event_type: str):
        """Handle VAD event — called inline from ingest loop, no scheduling delay."""
        # Notify client
        await self._send_json({"type": "vad", "speech": event_type == "start"})

        if event_type == "start":
            decision = self._turn.on_vad_speech_start()
        else:
            decision = self._turn.on_vad_speech_end()

        if decision:
            await self._handle_decision(decision)

    # ------------------------------------------------------------------
    # Decision handling
    # ------------------------------------------------------------------

    async def _handle_decision(self, decision: Decision, asr_text: str = ""):
        """Act on a turn-taking decision."""
        if decision == Decision.START_ASR:
            if self._turn.state == State.BOT_SPEAKING:
                # Backchannel evaluation mode — start ASR but don't stop bot
                await self._start_backchannel_asr()
            else:
                await self._start_asr()

        elif decision == Decision.END_TURN:
            await self._end_turn()

        elif decision == Decision.INTERRUPT:
            await self._handle_interrupt()

        elif decision == Decision.BACKCHANNEL_IGNORE:
            await self._stop_backchannel_asr()

    async def _start_asr(self):
        """Initialize streaming ASR session."""
        if self._asr_active:
            return
        self._asr_text = ""
        # Seed ASR buffer with pre-speech audio so first words aren't lost
        self._asr_buffer = self._pre_speech_buf.copy()
        self._pre_speech_buf = np.empty(0, dtype=np.float32)
        try:
            await anyio.to_thread.run_sync(
                lambda: self.pipeline.transcribe_streaming_init(language=self.language)
            )
            self._asr_active = True
            logger.debug(f"ASR streaming started (pre-speech: {len(self._asr_buffer)} samples)")
        except Exception as e:
            logger.error(f"ASR init failed: {e}")

    async def _start_backchannel_asr(self):
        """Start ASR during bot speaking for backchannel classification."""
        if self._backchannel_asr_active:
            return
        self._asr_text = ""
        self._asr_buffer = self._pre_speech_buf.copy()
        self._pre_speech_buf = np.empty(0, dtype=np.float32)
        try:
            await anyio.to_thread.run_sync(
                lambda: self.pipeline.transcribe_streaming_init(language=self.language)
            )
            self._backchannel_asr_active = True
            logger.debug("Backchannel ASR started")
        except Exception as e:
            logger.error(f"Backchannel ASR init failed: {e}")

    async def _stop_backchannel_asr(self):
        """Stop backchannel ASR (backchannel was detected, discard)."""
        if not self._backchannel_asr_active:
            return
        self._asr_buffer = np.empty(0, dtype=np.float32)
        try:
            await anyio.to_thread.run_sync(self.pipeline.transcribe_streaming_finish)
        except Exception:
            pass
        self._backchannel_asr_active = False
        self._asr_text = ""
        logger.debug("Backchannel ASR stopped (discarded)")

    async def _end_turn(self):
        """User turn ended — finalize ASR, start LLM + TTS."""
        # Finalize ASR
        final_text = ""
        if self._asr_active:
            # Flush any remaining buffered audio before finalizing
            if len(self._asr_buffer) > 0:
                try:
                    text = await anyio.to_thread.run_sync(
                        self.pipeline.transcribe_streaming_chunk, self._asr_buffer
                    )
                    if text:
                        self._asr_text = text
                except Exception:
                    pass
                self._asr_buffer = np.empty(0, dtype=np.float32)
            try:
                final_text = await anyio.to_thread.run_sync(
                    self.pipeline.transcribe_streaming_finish
                )
            except Exception as e:
                logger.error(f"ASR finalize failed: {e}")
            self._asr_active = False

        if not final_text:
            final_text = self._asr_text  # fallback to last partial

        if not final_text or not final_text.strip():
            logger.debug("No speech detected, returning to IDLE")
            self._turn.force_idle()
            await self._send_json({"type": "bot.idle"})
            return

        # Send final transcript
        await self._send_json({"type": "transcript", "text": final_text, "is_final": True})
        logger.info(f"User said: {final_text}")

        # Notify bot thinking
        await self._send_json({"type": "bot.thinking"})

        # Wait for any previous processing to finish (interrupt flag will
        # make it exit quickly).  We must NOT clear _interrupt or start new
        # TTS while the old CUDA thread is still running.
        if self._processing_task and not self._processing_task.done():
            self.pipeline.interrupt()
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                logger.warning("Previous processing task did not finish in time")

        # Consume pending image (one-shot)
        image = self._pending_image
        self._pending_image = None

        # Start LLM + TTS in a new task
        self._processing_task = asyncio.ensure_future(
            self._process_response(final_text, image=image)
        )

    async def _handle_interrupt(self):
        """Interrupt bot speech — stop TTS, correct context, start new user turn."""
        logger.info("Interrupting bot speech")

        # Set interrupt flag — the TTS generator thread checks this on each
        # chunk and will exit.  Do NOT cancel() the processing task: that
        # leaves the CUDA worker thread orphaned with dangling tensors.
        self.pipeline.interrupt()

        # Notify client
        await self._send_json({"type": "bot.interrupted"})

        # Finalize backchannel ASR (it's now a real user turn)
        if self._backchannel_asr_active:
            self._backchannel_asr_active = False
            self._asr_active = True  # Transition to regular ASR

        # If ASR wasn't already active, start it
        if not self._asr_active:
            await self._start_asr()

        # If VAD is not currently detecting speech, seed the silence timeout
        # so the turn can end.  Without this, the VAD speech_end that already
        # fired (before the interrupt) is lost and ASR runs forever.
        if not self._vad.speech_active:
            self._turn._speech_end_time = time.monotonic()

    # ------------------------------------------------------------------
    # LLM + TTS processing
    # ------------------------------------------------------------------

    async def _process_response(self, user_text: str, image: Optional[str] = None):
        """Run LLM + TTS and send audio chunks to output queue."""
        try:
            pipeline = self.pipeline
            pipeline._interrupt.clear()
            self._audio_bytes_sent = 0
            self._gen_id += 1
            my_gen = self._gen_id
            first_chunk_time = None  # track when client started receiving audio

            # Generate interleaved audio stream (LLM → TTS)
            tts_sent_text = ""
            first_chunk = True

            def _generate():
                # generate_interleaved_audio_stream handles both tools and
                # non-tools paths internally, yielding PCM bytes
                for pcm_bytes in self._generate_interleaved(
                    user_text, image=image, voice=self.voice, language=self.language
                ):
                    yield pcm_bytes

            # Stream from sync generator across threads via queue
            chunk_queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            async def _producer():
                """Run sync generator in thread, push chunks to async queue."""
                def _run_gen():
                    try:
                        for chunk in _generate():
                            chunk_queue.put_nowait(chunk)
                    except Exception as e:
                        logger.error(f"TTS generation error: {e}")
                    finally:
                        chunk_queue.put_nowait(sentinel)

                await anyio.to_thread.run_sync(_run_gen)

            # Start producer
            producer_task = asyncio.ensure_future(_producer())

            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        if pipeline._interrupt.is_set():
                            break
                        continue

                    if chunk is sentinel:
                        break

                    if pipeline._interrupt.is_set():
                        break

                    # Skip WAV header for duplex (client uses raw PCM)
                    # WAV header is 44 bytes starting with "RIFF"
                    if isinstance(chunk, bytes) and first_chunk and len(chunk) >= 44:
                        if chunk[:4] == b'RIFF':
                            # Strip WAV header, send only PCM data
                            chunk = chunk[44:]
                            if not chunk:
                                continue

                    if chunk:
                        if first_chunk:
                            first_chunk = False
                            first_chunk_time = time.monotonic()
                            await self._send_json({"type": "bot.speaking"})
                        self._output_queue.put_nowait(("binary", chunk, my_gen))
                        self._audio_bytes_sent += len(chunk)
                        self._turn.on_tts_chunk_sent()
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass

            # Estimate remaining client playback time.
            # The client started playing when the first chunk arrived, so
            # subtract the time spent streaming from total playback duration.
            playback_s = (self._audio_bytes_sent / 2) / self.sample_rate
            playback_ms = playback_s * 1000
            if first_chunk_time:
                elapsed_ms = (time.monotonic() - first_chunk_time) * 1000
                remaining_ms = max(0, playback_ms - elapsed_ms)
            else:
                remaining_ms = playback_ms
            self._turn.set_dynamic_bot_stop_delay(remaining_ms)

            # TTS generation complete — stay in BOT_SPEAKING for the dynamic
            # bot_stop_delay so user speech during client playback buffer is
            # still treated as interruption.
            self._turn.on_tts_complete()

        except asyncio.CancelledError:
            logger.debug("Processing cancelled (interrupt)")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            await self._send_json({"type": "error", "message": str(e)})
            self._turn.force_idle()
            await self._send_json({"type": "bot.idle"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_json(self, data: dict):
        """Queue a JSON message for sending."""
        self._output_queue.put_nowait(("json", data))

    async def _cleanup(self):
        """Clean up session resources."""
        self._running = False

        # Interrupt any in-progress TTS
        self.pipeline.interrupt()

        # Wait for processing task to finish (interrupt flag stops it)
        if self._processing_task and not self._processing_task.done():
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._processing_task.cancel()

        # Finalize ASR if active
        if self._asr_active:
            try:
                await anyio.to_thread.run_sync(self.pipeline.transcribe_streaming_finish)
            except Exception:
                pass
            self._asr_active = False

        if self._backchannel_asr_active:
            try:
                await anyio.to_thread.run_sync(self.pipeline.transcribe_streaming_finish)
            except Exception:
                pass
            self._backchannel_asr_active = False

        self._turn.force_idle()
        logger.info("Duplex session cleaned up")
