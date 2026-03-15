/**
 * Events Module - SSE client for server→client push events
 *
 * Generic event system: connects to SSE endpoint, dispatches to registered handlers.
 * Any tool/backend component can publish events; frontend registers handlers per type.
 */

import { API_EVENTS } from "./config.js";

let source = null;
const handlers = {};  // { "timer_expired": [fn, fn], ... }

/**
 * Register a handler for a specific event type
 */
export function onEvent(type, fn) {
    (handlers[type] ??= []).push(fn);
    // If already connected, add listener for this new type
    if (source) {
        source.addEventListener(type, _makeSSEHandler(type));
    }
}

/**
 * Unregister a handler for a specific event type
 */
export function offEvent(type, fn) {
    const list = handlers[type];
    if (list) handlers[type] = list.filter(f => f !== fn);
}

/**
 * Connect to the SSE event stream
 */
export function connectEventStream() {
    if (source) source.close();
    source = new EventSource(API_EVENTS);

    // Named events (SSE `event:` field) — register for all known types
    for (const type of Object.keys(handlers)) {
        source.addEventListener(type, _makeSSEHandler(type));
    }

    source.onerror = () => {
        // EventSource auto-reconnects — just log
        console.debug("[events] SSE reconnecting...");
    };
}

/**
 * Disconnect from the SSE event stream
 */
export function disconnectEventStream() {
    if (source) { source.close(); source = null; }
}

/**
 * Create an SSE event handler that dispatches to registered handlers
 */
function _makeSSEHandler(type) {
    return (e) => {
        try {
            const event = JSON.parse(e.data);
            (handlers[type] || []).forEach(fn => fn(event));
        } catch (err) {
            console.warn(`[events] Failed to parse ${type} event:`, err);
        }
    };
}
