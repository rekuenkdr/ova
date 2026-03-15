# OVA Frontend - Static Assets

This directory contains all frontend assets for the QWOVA application, including JavaScript modules, CSS stylesheets, theming system, and static images.

## Directory Structure

```
static/
├── css/
│   ├── animations.css      # Keyframe animations and transitions
│   ├── base.css            # Global styles, typography, layout
│   └── components.css      # UI component styles
├── js/
│   ├── app.js              # Application entry point
│   ├── audio.js            # Audio recording, playback, streaming
│   ├── config.js           # Shared configuration constants
│   ├── events.js           # SSE client for server→client push events
│   ├── notifications.js    # Toast/system notifications + alarm chime
│   ├── pcm-processor.js    # AudioWorklet for PCM streaming
│   ├── settings.js         # Settings panel management
│   ├── theme.js            # Theme switching system
│   ├── ui.js               # DOM management and UI state
│   ├── vad.js              # Silero VAD + RMS fallback for barge-in
│   └── wakeword.js         # VAD-gated wake word detection
├── themes/
│   ├── dark/
│   │   └── theme.css       # Dark theme
│   ├── light/
│   │   └── theme.css       # Light theme
│   ├── her/
│   │   └── theme.css       # Her (Samantha) theme
│   └── hal-9000/
│       └── theme.css       # HAL-9000 theme
└── img/
    └── *.png               # UI state screenshots
```

---

## JavaScript Modules

### Module Dependency Graph

```
app.js (entry point)
├── config.js          (constants)
├── theme.js           (theming)
├── ui.js              (DOM management)
├── audio.js           (audio handling)
│   └── pcm-processor.js (AudioWorklet)
├── settings.js        (settings panel)
├── events.js          (SSE push events)
├── notifications.js   (toast/system notifications)
├── vad.js             (voice activity detection)
└── wakeword.js        (wake word detection)
    └── vad.js         (imports VAD functions)
```

---

### config.js

Shared configuration constants for the frontend application.

| Constant | Default | Description |
|----------|---------|-------------|
| `API_BASE` | `http://localhost:5173` | Backend server URL |
| `API_CHAT_AUDIO` | `/v1/chat/audio` | Voice input endpoint (WAV audio) |
| `API_CHAT_TEXT` | `/v1/chat` | Text/image input endpoint |
| `API_TTS` | `/v1/text-to-speech` | Pure TTS endpoint |
| `API_INFO` | `/v1/info` | Pipeline info endpoint |
| `API_SETTINGS` | `/v1/settings` | Settings endpoint |
| `API_SETTINGS_PROMPT` | `/v1/settings/prompt` | Prompt reload endpoint |
| `API_HEALTH` | `/v1/health` | Health check endpoint |
| `API_RESTART` | `/v1/restart` | Restart endpoint |
| `API_INTERRUPT` | `/v1/interrupt` | Interrupt current TTS playback |
| `API_EVENTS` | `/v1/events` | SSE event stream endpoint |
| `WS_ASR` | `ws://localhost:5173/v1/speech-to-text/stream` | WebSocket for streaming ASR |
| `TARGET_SR` | `16000` | Target sample rate for ASR (Hz) |
| `CHUNK_MS` | `500` | Audio chunk duration for streaming ASR (ms) |
| `SILERO_MODEL_URL` | `/models/silero_vad_16k_op15.onnx` | Silero VAD ONNX model path |
| `VAD_CONFIRM_FRAMES` | `4` | Frames (~128ms) to confirm speech |
| `VAD_SILENCE_FRAMES` | `10` | Frames (~320ms) to confirm silence |
| `AUTO_SEND_SILENCE_MS` | `1000` | Silence duration to trigger auto-send (ms) |
| `AUTO_SEND_CONFIRM_MS` | `64` | Speech duration to arm auto-send (ms) |
| `AUTO_SEND_TIMEOUT_MS` | `3000` | Cancel recording if no speech (ms) |
| `BARGE_IN_COOLDOWN_MS` | `2000` | Suppress barge-in after auto-send (ms) |
| `BARGE_IN_GRACE_MS` | `500` | Delay VAD start after playback begins (ms) |
| `BACKCHANNELS` | `{es:[...], en:[...], ...}` | Per-language filler words filtered during barge-in |
| `WAKE_WORD_SILENCE_MS` | `600` | Silence timeout for wake word ASR (ms) |
| `DEBUG` | `false` | Debug mode (set from backend `/v1/info`) |

---

### theme.js

Theme switching system with persistence and system preference detection. Cycles through 4 themes: Dark, Light, Her (Samantha), and HAL-9000.

#### Features

- **localStorage Persistence**: Saves user theme preference with key `'ova-theme'`
- **System Preference Detection**: Respects `prefers-color-scheme` media query when no saved preference exists
- **Real-time System Updates**: Listens for OS theme changes and auto-switches if user hasn't set manual preference
- **Dynamic CSS Loading**: Swaps theme stylesheet by updating `<link>` href

#### How It Works

1. On initialization, checks localStorage for saved preference
2. Falls back to system preference if no saved preference
3. Sets `data-theme` attribute on `<html>` element
4. Updates theme CSS link to load `/static/themes/{theme}/theme.css`

#### Exported Functions

| Function | Description |
|----------|-------------|
| `initTheme()` | Initialize theme on page load |
| `applyTheme(theme)` | Apply theme and save to localStorage |
| `toggleTheme()` | Cycle to next theme |
| `getTheme()` | Get current theme name |
| `THEMES` | Array of available themes: `['dark', 'light', 'her', 'hal-9000']` |

---

### ui.js

DOM element management and UI state updates for the voice assistant interface.

#### Voice Button States

| State | Description | Visual |
|-------|-------------|--------|
| `idle` | Ready for input | Blue glow, mic icon |
| `recording` | Capturing audio | Red glow, pulsing mic |
| `waiting` | Processing request | Purple glow, spinner |
| `playing` | Playing response | Green glow, wave bars |
| `restart` | Server restarting | Red, reload icon |

#### Features

- **Image Handling**: Drag-drop, paste, and file input with preview
- **Text Input**: Auto-resizing textarea (min 48px, max 120px)
- **Smart Send Button**: Enables only when text or image is present
- **Progress Bar**: Animated restart progress indicator

#### Exported Functions

| Function | Description |
|----------|-------------|
| `initUI()` | Cache DOM element references |
| `getElements()` | Return cached element references |
| `setState(state, {label, sub})` | Update button state and text |
| `getState()` | Get current button state |
| `setInputsEnabled(enabled)` | Enable/disable all inputs |
| `updateSendButton()` | Update send button enabled state |
| `showImagePreview(dataUrl)` | Display image preview thumbnail |
| `clearImage()` | Remove image preview |
| `handleImageSelect(file)` | Process image file to data URL |
| `autoResizeTextInput()` | Resize textarea based on content |
| `getTextInputValue()` | Get trimmed textarea content |
| `clearTextInput()` | Clear textarea |
| `disableMultimodal()` | Remove text input section from DOM |
| `showRestartProgress()` | Show restart progress bar |
| `hideRestartProgress()` | Hide restart progress bar |
| `completeRestartProgress()` | Complete and hide progress bar |
| `showSuccessFlash()` | Flash success animation on button |

---

### audio.js

Comprehensive audio handling including recording, playback, streaming, and WebSocket ASR.

#### Recording

- Uses Web Audio API `ScriptProcessorNode` for audio capture
- Records at device sample rate, stores chunks as Float32Array
- Streams to ASR WebSocket in real-time (resampled to 16kHz)
- Falls back to batch mode if WebSocket unavailable

#### WebSocket ASR (Streaming Transcription)

```
┌──────────────┐    binary audio chunks     ┌──────────────┐
│   Browser    │ ─────────────────────────▶ │   Backend    │
│              │ ◀───────────────────────── │  /v1/stt/s   │
│              │    JSON partial/final      │              │
└──────────────┘                            └──────────────┘
```

- Opens persistent connection to `WS_ASR_URL`
- Resamples audio from device sample rate to 16kHz (linear interpolation)
- Sends ~500ms chunks as binary data
- Receives JSON messages with partial and final transcripts
- 5-second timeout if final transcript not received

#### Playback Modes

**Non-Streaming (WAV)**
- Uses HTML5 `<audio>` element
- Creates blob URL with `URL.createObjectURL()`

**Streaming PCM**
- WAV header with `dataSize=0x7FFFFFFF` followed by raw PCM chunks
- Uses AudioWorklet (`PCMStreamProcessor`) for buffer management
- Configurable pre-buffering before playback starts
- Handles buffer underruns gracefully

**Streaming WAV**
- Each chunk is a complete WAV file
- Parses headers, extracts sample rate
- Schedules buffer sources for sequential playback

#### Audio Utilities

| Function | Description |
|----------|-------------|
| `encodeWavPcm16(pcm, sr)` | Create WAV bytes from Float32 PCM |
| `parseWavToFloat32(arrayBuffer)` | Extract PCM and sample rate from WAV |
| `resampleLinear(input, srcSr, dstSr)` | Fast linear interpolation resampling |
| `flattenFloat32(chunks)` | Flatten array of Float32Arrays |
| `concatFloat32(a, b)` | Concatenate two Float32Arrays |

#### Exported Functions

| Function | Description |
|----------|-------------|
| `startRecording()` | Begin audio capture, try ASR WebSocket |
| `stopRecording()` | Stop recording, finalize ASR, return transcript or WAV blob |
| `isRecording()` | Check if currently recording |
| `sendTextMessage(text, imageData)` | POST text/image to `/v1/chat` |
| `fetchAndPlayTts(wavBlob)` | POST WAV to `/v1/chat/audio` with streaming |
| `sendAndPlayResponse(requestFn)` | Generic request + playback handler |
| `stopCurrentAudio()` | Stop playback and abort request |
| `stopStreamingPlayback()` | Close AudioWorklet and audio context |
| `abortCurrentRequest()` | Abort fetch request |
| `setPcmPrebufferSamples(samples)` | Update prebuffer threshold |

---

### pcm-processor.js

AudioWorklet processor for real-time PCM streaming playback with pre-buffering.

#### Features

- **Pre-buffering**: Waits for threshold samples before starting playback
- **Buffer Management**: Maintains internal Float32Array buffer
- **Underrun Handling**: Plays silence if buffer exhausted
- **Status Reporting**: Reports buffer level every ~0.27s (100 frames at 24kHz)

#### Message Protocol

**Incoming Messages**

| Message | Description |
|---------|-------------|
| `{config: {prebufferThreshold: N}}` | Update prebuffer threshold |
| `{samples: Float32Array}` | Add samples to buffer |
| `{end: true}` | Signal end of stream |
| `{stop: true}` | Stop playback and reset |

**Outgoing Events**

| Event | Description |
|-------|-------------|
| `{event: 'playing', buffered: N}` | Playback started |
| `{event: 'underrun', frame, had, needed, total}` | Buffer underrun |
| `{event: 'status', buffer: N, frame: X}` | Periodic status |
| `{event: 'finished', frame: N}` | Stream ended |

---

### settings.js

Settings panel management with backend API integration.

#### Settings Managed

| Setting | Type | Description |
|---------|------|-------------|
| Language | Select | UI and TTS language (es, en, fr, de, it, pt, ja, zh) |
| TTS Engine | Select | `qwen3` (voice cloning) or `kokoro` (predefined voices) |
| Voice Profile | Select | Qwen3 voice name (per language) |
| Voice ID | Select | Kokoro voice preset (per language) |
| Stream Format | Select | `pcm` (low latency) or `wav` (compatible) |
| System Prompt | Textarea | Editable system prompt |
| Clear History | Checkbox | Clear conversation on apply |

#### Hot-reload vs Restart

| Change Type | Requires Restart |
|-------------|-----------------|
| Language | No |
| Voice profile | No |
| System prompt | No |
| TTS engine | Yes |
| Stream format | Yes |

#### Features

- **Dynamic UI**: Shows Qwen3 or Kokoro settings based on selected engine
- **Language Availability**: Disables languages without Qwen3 profiles when Qwen3 selected
- **Prompt Management**: Auto-loads profile prompts, saves on blur or apply
- **Restart Flow**: Shows alert, triggers restart, polls `/v1/health` for completion

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/settings` | GET | Load current settings |
| `/v1/settings` | POST | Apply settings (returns `restart_required`) |
| `/v1/settings/prompt` | POST | Update system prompt in-session |
| `/v1/restart` | POST | Trigger server restart |
| `/v1/health` | GET | Poll for restart completion (503 while warming) |

#### Exported Functions

| Function | Description |
|----------|-------------|
| `initSettings()` | Initialize settings panel |
| `loadSettings()` | Fetch and display current settings |
| `disableSettings()` | Remove settings from DOM |
| `openSettings()` | Show settings panel |
| `closeSettings()` | Hide settings panel |

---

### events.js

SSE client for server-to-client push events. Connects to the `/v1/events` endpoint and dispatches events to registered handlers by type.

#### How It Works

1. Handlers register per event type with `onEvent("timer_expired", fn)`
2. `connectEventStream()` opens an `EventSource` to `/v1/events`
3. Named SSE events are parsed as JSON and dispatched to matching handlers
4. `EventSource` auto-reconnects on disconnect

#### Exported Functions

| Function | Description |
|----------|-------------|
| `onEvent(type, fn)` | Register a handler for an event type |
| `offEvent(type, fn)` | Unregister a handler |
| `connectEventStream()` | Connect to SSE endpoint |
| `disconnectEventStream()` | Close SSE connection |

---

### notifications.js

Hybrid notification system: OS-level system notifications when the tab is hidden, in-page toasts when visible. Alarm tone uses Web Audio API (no external files).

#### Exported Functions

| Function | Description |
|----------|-------------|
| `requestPermission()` | Request OS notification permission (safe to call multiple times) |
| `showNotification({title, message})` | Show system notification (hidden tab) or toast (visible tab) |
| `playAlarmTone()` | Play a two-tone sine wave chime (660Hz + 880Hz) |

---

### vad.js

Silero Neural Voice Activity Detection with RMS energy fallback. Detects speech during TTS playback for barge-in support.

#### How It Works

1. Loads Silero VAD v6 ONNX model via ONNX Runtime Web (lazy, once)
2. Monitors microphone via `ScriptProcessorNode` (512-sample buffer)
3. Downsamples to 16kHz, runs inference on 32ms windows
4. Speech onset: sliding window confirmation ("N out of M frames above threshold")
5. Silence detection: consecutive frames below threshold
6. Falls back to RMS energy detection if ONNX model fails to load

#### Exported Functions

| Function | Description |
|----------|-------------|
| `loadSileroModel()` | Pre-load Silero ONNX model (call during init) |
| `startVADMonitor(stream, opts)` | Start monitoring a mic stream for speech |
| `stopVADMonitor()` | Stop monitoring and release audio nodes |
| `isVADActive()` | Check if VAD is currently active |
| `isSileroReady()` | Check if Silero model is loaded |

#### startVADMonitor Options

| Option | Default | Description |
|--------|---------|-------------|
| `onSpeechStart` | — | Callback when speech is confirmed |
| `onSpeechEnd` | — | Callback when silence is confirmed after speech |
| `threshold` | `0.5` | Detection threshold (0.0-1.0) |
| `confirmFrames` | `2` | Frames above threshold to confirm speech |
| `silenceFrames` | `10` | Consecutive frames below threshold to confirm silence |
| `confirmMs` | — | Overrides `confirmFrames` (computed from 32ms frame) |
| `silenceMs` | — | Overrides `silenceFrames` (computed from 32ms frame) |

---

### wakeword.js

VAD-gated streaming ASR for configurable wake word detection. Coordinates Silero VAD with a dedicated ASR WebSocket to detect a wake phrase.

#### How It Works

1. Mic is captured and audio flows into a pre-speech ring buffer (~500ms)
2. VAD monitors for speech onset (low CPU — no ASR until speech detected)
3. Speech detected → opens ASR WebSocket (forced to English for reliable matching)
4. Pre-speech buffer + live audio streamed to ASR
5. Partial transcripts checked against wake phrase
6. Match found → callback fires, detection stops

#### Exported Functions

| Function | Description |
|----------|-------------|
| `startWakeWordDetection(stream, opts)` | Start wake word listening |
| `stopWakeWordDetection()` | Stop and release all resources |
| `isWakeWordActive()` | Check if detection is active |

#### startWakeWordDetection Options

| Option | Default | Description |
|--------|---------|-------------|
| `onWakeWord` | — | Callback when wake phrase is detected |
| `threshold` | `0.5` | VAD threshold |
| `silenceMs` | `600` | Silence duration to end utterance (ms) |
| `wakeWord` | `'hey nova'` | Wake word phrase to match |

---

### app.js

Application entry point and event binding.

#### Initialization Sequence

1. `initTheme()` - Load and apply saved/system theme
2. `initUI()` - Cache DOM element references
3. `setupVoiceButton()` - Voice input click handlers
4. `setupTextInput()` - Text input, image, send button handlers
5. `setupDragDrop()` - Drag-drop and paste image handlers
6. `setupThemeToggle()` - Theme button click handler
7. `fetchBackendConfig()` - Get PCM prebuffer, feature flags
8. `initSettings()` - Initialize settings panel (if enabled)

#### Voice Button Flow

```
┌───────┐  click   ┌───────────┐  click   ┌─────────┐
│ Idle  │ ───────▶ │ Recording │ ───────▶ │ Waiting │
└───────┘          └───────────┘          └────┬────┘
    ▲                                          │
    │              ┌─────────┐                 │
    └───────────── │ Playing │ ◀───────────────┘
                   └─────────┘  audio finished
```

#### Backend Config

Fetches `/v1/info` endpoint for:
- `pcm_prebuffer_samples` - PCM prebuffer threshold
- `frontend_settings_disabled` - Disables settings panel
- `multimodal_disabled` - Removes text input section

---

## CSS Architecture

### File Organization

| File | Purpose | Lines |
|------|---------|-------|
| `base.css` | Global styles, typography, layout | ~160 |
| `components.css` | All UI component styles | ~1,030 |
| `animations.css` | Keyframes and transitions | ~280 |

### CSS Variable System

All colors and effects use CSS custom properties (variables) defined in theme files. This enables seamless theme switching without duplicating component styles.

```css
/* Component uses variables */
.voice-btn {
  background: var(--bg-glass);
  color: var(--text-primary);
  box-shadow: var(--shadow-button);
}

/* Theme defines variables */
[data-theme="dark"] {
  --bg-glass: rgba(17, 24, 39, 0.6);
  --text-primary: #f1f5f9;
}
```

---

## base.css

Global styles and layout foundation.

#### Features

- **Reset**: Border-box sizing, zeroed margins/padding
- **Typography**: Inter font family, antialiased rendering
- **Layout**: CSS Grid with centered stage container (max 580px)
- **Grid Background**: Animated 50x50px grid pattern
- **Accessibility**: `:focus-visible` outlines, `.sr-only` class
- **Restart Progress**: Top-left animated progress bar

#### Key Classes

| Class | Description |
|-------|-------------|
| `.stage` | Centered content container |
| `.divider` | Horizontal line with text |
| `.mono` | Monospace font |
| `.hidden` | Display none |
| `.sr-only` | Screen reader only |
| `.restart-progress` | Top progress bar container |

---

## components.css

All UI component styles.

### Voice Button

```
┌─────────────────────────────────────┐
│         .voice-btn-container        │
│  ┌─────────────────────────────┐    │
│  │       .holo-ring            │    │  ← Rotating holographic ring
│  │  ┌─────────────────────┐    │    │
│  │  │    .pulse-ring      │    │    │  ← Expanding pulse animation
│  │  │  ┌─────────────┐    │    │    │
│  │  │  │   .talk     │    │    │    │  ← Main button (280px circle)
│  │  │  │  icon/label │    │    │    │
│  │  │  └─────────────┘    │    │    │
│  │  └─────────────────────┘    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

#### Button States

| State | Ring Color | Background | Animation |
|-------|------------|------------|-----------|
| Idle | Cyan/Blue gradient | Blue glow | None |
| Recording | Red gradient | Red glow | Icon pulse |
| Waiting | Purple gradient | Purple glow | Spinner |
| Playing | Green gradient | Green glow | Wave bars |
| Restart | Red | Red | Reload icon |

### Action Buttons

48x48 circular buttons with hover lift effect:
- Send button (glow when enabled)
- Image button
- Settings button
- Theme button

### Text Input

- Rounded pill input with auto-resize (48-120px height)
- Focus glow with cyan/teal border
- Mobile-responsive expansion

### Image Preview

- 64x64 thumbnail with remove button overlay
- Dark overlay on hover

### Settings Panel

- Full-screen semi-transparent backdrop with blur
- Slide-in panel from left (400px or 90vw on mobile)
- Form groups with labels, selects, textarea
- Custom toggle switches
- Apply/Cancel buttons

### Top Buttons

- Settings: Fixed top-left, gear icon
- Theme: Fixed top-right, sun/moon icon
- Rotating icon animation on hover

### Responsive Design

```css
@media (max-width: 480px) {
  /* Text input expands to full width */
  /* Image/send buttons slide in from sides */
  /* Spring animation for expansion */
}
```

---

## animations.css

All keyframe animations and transition utilities.

### Button Animations

| Animation | Duration | Description |
|-----------|----------|-------------|
| `holo-spin` | 8s | Holographic ring rotation |
| `pulse-expand` | 1.5s | Scale 1→1.3 with fade |
| `icon-pulse` | 1s | Breathing scale 1→1.1 |
| `wave-bar` | 0.8s | Vertical bar wave (staggered) |

### UI Animations

| Animation | Duration | Description |
|-----------|----------|-------------|
| `fade-in` | 0.3s | Opacity 0→1, translate up |
| `slide-in-left` | 0.3s | From -100% X to 0 |
| `scale-pop` | 0.2s | Elastic scale in |

### Loading

| Animation | Description |
|-----------|-------------|
| `spin` | 360° rotation |
| `progress-fill` | Width 0% → 85% → 95% |
| `dots-pulse` | Scale breathing with delay |

### Background

| Animation | Description |
|-----------|-------------|
| `grid-shift` | Background position offset |
| `glow-pulse` | Box-shadow intensity |
| `ambient-shift` | Hue rotation |

### Micro-interactions

| Animation | Description |
|-----------|-------------|
| `button-press` | Scale 1 → 0.97 → 1 |
| `success-flash` | Background color pulse |
| `error-shake` | Horizontal shake ±4px |

### Transition Classes

| Class | Timing |
|-------|--------|
| `.transition-all` | 0.2s ease |
| `.transition-fast` | 0.1s ease |
| `.transition-slow` | 0.4s ease |
| `.transition-spring` | 0.3s cubic-bezier (bouncy) |
| `.hover-lift` | translateY(-3px) on hover |
| `.hover-glow` | Box-shadow on hover |

### State Classes

| Class | Description |
|-------|-------------|
| `.appear` | Fade in animation |
| `.disappear` | Fade out animation |
| `.panel-enter` | Slide in from left |
| `.panel-exit` | Slide out to left |
| `.pop-in` | Scale pop animation |

### Accessibility

```css
@media (prefers-reduced-motion: reduce) {
  /* Disables all animations */
  /* Hides decorative rings */
}
```

---

## Theming System

### Architecture

The theming system uses CSS custom properties (variables) to define all colors and effects. Theme files only contain variable definitions, while component styles reference these variables.

```
┌─────────────────────────────────────────────────────────┐
│                    theme.js                             │
│  - Detects system preference                            │
│  - Loads saved preference from localStorage             │
│  - Sets data-theme attribute on <html>                  │
│  - Updates <link> href to load theme CSS                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│      themes/{dark,light,her,hal-9000}/theme.css         │
│  - Defines CSS variables at :root level                 │
│  - Uses [data-theme="..."] selector                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│     base.css / components.css / animations.css          │
│  - References variables: var(--bg-primary), etc.        │
│  - No color values hardcoded                            │
└─────────────────────────────────────────────────────────┘
```

### CSS Variable Categories

#### Backgrounds

| Variable | Dark | Light |
|----------|------|-------|
| `--bg-primary` | `#0a0e17` | `#f8fafc` |
| `--bg-secondary` | `#111827` | `#f1f5f9` |
| `--bg-tertiary` | `#1a2234` | `#e2e8f0` |
| `--bg-surface` | `rgba(17,24,39,0.8)` | `rgba(255,255,255,0.9)` |
| `--bg-glass` | `rgba(17,24,39,0.6)` | `rgba(255,255,255,0.7)` |

#### Text

| Variable | Dark | Light |
|----------|------|-------|
| `--text-primary` | `#f1f5f9` | `#0f172a` |
| `--text-secondary` | `#94a3b8` | `#475569` |
| `--text-muted` | `rgba(148,163,184,0.7)` | `rgba(71,85,105,0.7)` |

#### Accents

| Variable | Dark | Light |
|----------|------|-------|
| `--accent-primary` | `#06b6d4` (cyan) | `#0891b2` (teal) |
| `--accent-secondary` | `#3b82f6` (blue) | `#2563eb` (blue) |
| `--accent-glow` | Semi-transparent cyan | Semi-transparent teal |

#### Status Colors

| Variable | Dark | Light |
|----------|------|-------|
| `--status-success` | `#10b981` | `#059669` |
| `--status-danger` | `#ef4444` | `#dc2626` |
| `--status-warning` | `#f59e0b` | `#d97706` |
| `--status-processing` | `#a855f7` | `#9333ea` |

#### Special Effects

- **Holographic Rings**: Conic gradients combining accent colors
- **Button Shadows**: Deep shadows with color tints
- **Glass Borders**: Subtle white (dark) or dark (light) borders
- **Grid Lines**: Accent-tinted at low opacity

### Theme Aesthetics

**Dark Theme**
- Deep space aesthetic with cyan/blue accents
- High contrast text on dark backgrounds
- Glowing effects and holographic rings
- Sci-fi inspired visuals

**Light Theme**
- Clean, minimal aesthetic with teal/blue accents
- Soft shadows and subtle gradients
- Professional, accessible appearance
- Modern UI design language

**Her (Samantha) Theme**
- Warm, intimate palette inspired by the film *Her*
- Soft coral/salmon accents on muted backgrounds

**HAL-9000 Theme**
- Red-on-black aesthetic inspired by HAL 9000
- Monochrome with red accent highlights

### Adding a New Theme

1. Create directory: `themes/{theme-name}/`
2. Create `theme.css` with all required CSS variables
3. Add theme name to `THEMES` array in `theme.js`
4. Theme will be available via `applyTheme('{theme-name}')`

---

## Images

The `img/` directory contains UI state screenshots for documentation.
