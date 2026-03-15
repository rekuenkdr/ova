/**
 * UI Module - DOM management and state updates
 */

// DOM element references
let elements = {};

/**
 * Initialize UI module with DOM elements
 */
export function initUI() {
  elements = {
    // Voice button
    talkBtn: document.getElementById('talkBtn'),
    labelEl: document.getElementById('label'),
    subEl: document.getElementById('sub'),

    // Text input
    textInput: document.getElementById('textInput'),
    sendBtn: document.getElementById('sendBtn'),

    // Image
    imageBtn: document.getElementById('imageBtn'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    removeImageBtn: document.getElementById('removeImageBtn'),

    // Progress
    restartProgress: document.getElementById('restartProgress'),
  };

  return elements;
}

/**
 * Disable multimodal input - removes text input section and divider from DOM
 */
export function disableMultimodal() {
  // Remove "or type" divider
  const divider = document.querySelector('.divider');
  if (divider) {
    divider.remove();
  }

  // Remove entire input section (text input, send button, image button, image preview)
  const inputSection = document.querySelector('.input-section');
  if (inputSection) {
    inputSection.remove();
  }
}

/**
 * Get UI elements
 */
export function getElements() {
  return elements;
}

/**
 * Set voice button state
 */
export function setState(state, { label, sub } = {}) {
  const btn = elements.talkBtn;
  if (!btn) return;

  btn.classList.remove('state-idle', 'state-recording', 'state-waiting', 'state-playing', 'state-restart');
  btn.classList.add(`state-${state}`);

  if (label != null && elements.labelEl) {
    elements.labelEl.textContent = label;
  }
  if (sub != null && elements.subEl) {
    elements.subEl.textContent = sub;
  }
}

/**
 * Get current state
 */
export function getState() {
  const btn = elements.talkBtn;
  if (!btn) return 'idle';

  if (btn.classList.contains('state-recording')) return 'recording';
  if (btn.classList.contains('state-waiting')) return 'waiting';
  if (btn.classList.contains('state-playing')) return 'playing';
  if (btn.classList.contains('state-restart')) return 'restart';
  return 'idle';
}

/**
 * Enable/disable input elements
 */
export function setInputsEnabled(enabled) {
  const { textInput, sendBtn, imageBtn } = elements;

  if (textInput) textInput.disabled = !enabled;
  if (sendBtn) {
    sendBtn.disabled = !enabled || (!textInput?.value.trim() && !hasAttachedImage());
  }
  if (imageBtn) imageBtn.disabled = !enabled;
}

/**
 * Update send button state
 */
export function updateSendButton() {
  const { textInput, sendBtn } = elements;
  if (sendBtn) {
    sendBtn.disabled = !textInput?.value.trim() && !hasAttachedImage();
  }
}

/**
 * Check if image is attached
 */
export function hasAttachedImage() {
  return elements.imagePreview?.classList.contains('has-image') || false;
}

/**
 * Get attached image data
 */
export function getAttachedImage() {
  if (!hasAttachedImage()) return null;
  return elements.previewImg?.src || null;
}

/**
 * Show image preview
 */
export function showImagePreview(dataUrl) {
  const { imagePreview, previewImg } = elements;
  if (previewImg) previewImg.src = dataUrl;
  if (imagePreview) imagePreview.classList.add('has-image');
  updateSendButton();
}

/**
 * Clear image preview
 */
export function clearImage() {
  const { imagePreview, previewImg, imageInput } = elements;
  if (previewImg) previewImg.src = '';
  if (imagePreview) imagePreview.classList.remove('has-image');
  if (imageInput) imageInput.value = '';
  updateSendButton();
}

/**
 * Handle image file selection
 */
export function handleImageSelect(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    showImagePreview(e.target.result);
  };
  reader.readAsDataURL(file);
}

/**
 * Auto-resize text input
 */
export function autoResizeTextInput() {
  const { textInput } = elements;
  if (!textInput) return;

  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
}

/**
 * Get text input value
 */
export function getTextInputValue() {
  return elements.textInput?.value.trim() || '';
}

/**
 * Clear text input
 */
export function clearTextInput() {
  if (elements.textInput) {
    elements.textInput.value = '';
    autoResizeTextInput();
  }
  updateSendButton();
}

/**
 * Show restart progress bar
 */
export function showRestartProgress() {
  const { restartProgress } = elements;
  if (restartProgress) {
    restartProgress.classList.add('active');
  }
}

/**
 * Hide restart progress bar (without completion animation)
 */
export function hideRestartProgress() {
  const { restartProgress } = elements;
  if (restartProgress) {
    restartProgress.classList.remove('active');
  }
}

/**
 * Complete restart progress bar
 */
export function completeRestartProgress() {
  const { restartProgress } = elements;
  if (restartProgress) {
    restartProgress.classList.remove('active');
    restartProgress.classList.add('done');
    setTimeout(() => restartProgress.classList.remove('done'), 600);
  }
}

/**
 * Show success flash on progress bar
 */
export function showSuccessFlash() {
  const { restartProgress } = elements;
  if (!restartProgress) return;

  restartProgress.style.transition = 'width 400ms ease-out';
  restartProgress.style.width = '100%';

  setTimeout(() => {
    restartProgress.style.transition = 'opacity 400ms ease-out';
    restartProgress.style.opacity = '0';

    setTimeout(() => {
      restartProgress.style.width = '0';
      restartProgress.style.opacity = '1';
      restartProgress.style.transition = '';
    }, 400);
  }, 500);
}

/**
 * Show error state on button
 */
export function showError(message) {
  setState('idle', { label: 'Tap to talk', sub: message });
}
