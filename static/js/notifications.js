/**
 * Notifications Module - Toast UI + system notifications + alarm tone
 *
 * When the tab is hidden and permission is granted, fires an OS-level
 * Notification via the Web Notifications API. Otherwise falls back to
 * an in-page toast (top-center, persists until dismissed).
 * Alarm tone uses Web Audio API (no external files, doesn't trigger barge-in).
 */

/**
 * Request permission for OS-level notifications.
 * Safe to call multiple times — the browser no-ops after first grant/deny.
 */
export function requestPermission() {
    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission();
    }
}

/**
 * Show a notification — system notification when tab is hidden, toast otherwise.
 */
export function showNotification({ title, message }) {
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
        new Notification(title, { body: message });
        return;
    }

    const container = document.getElementById("notifications");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<button class="toast-close" aria-label="Close">&times;</button><strong>${_escapeHtml(title)}</strong><span>${_escapeHtml(message)}</span>`;
    container.appendChild(toast);

    const dismiss = () => {
        toast.classList.add("fade-out");
        toast.addEventListener("animationend", () => toast.remove());
    };
    toast.querySelector(".toast-close").addEventListener("click", dismiss);
}

/**
 * Play a two-tone chime alarm using Web Audio API
 */
export function playAlarmTone() {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;

    [660, 880].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.25, now + i * 0.2);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.2 + 0.3);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now + i * 0.2);
        osc.stop(now + i * 0.2 + 0.3);
    });
}

/**
 * Escape HTML to prevent XSS in toast content
 */
function _escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
