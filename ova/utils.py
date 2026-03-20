import logging
import os
import sys
import warnings

# NOTE: Do NOT import torch at module level - it initializes CUDA
# which breaks vLLM's multiprocessing (forces spawn instead of fork)

# Suppress loky semaphore warning from NeMo/joblib at shutdown
warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked semaphore.*")

# Debug mode toggle from environment
DEBUG = os.getenv("OVA_DEBUG", "").lower() == "true"

# ANSI color codes for component-specific logging
COMPONENT_COLORS = {
    "asr": "\033[36m",   # Cyan
    "llm": "\033[32m",   # Green
    "tts": "\033[33m",   # Yellow
    "api": "\033[35m",   # Magenta
    "sys": "\033[37m",   # White/default
}
RESET_COLOR = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that colors the [COMPONENT] tag based on component name."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Extract component from logger name (e.g., "ova.asr" -> "asr")
        component = "sys"
        if hasattr(record, "name") and "." in record.name:
            parts = record.name.split(".")
            if len(parts) >= 2:
                component = parts[1].lower()

        # Get color for component
        color = COMPONENT_COLORS.get(component, COMPONENT_COLORS["sys"])

        # Format the base message
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname

        # Create colored component tag
        component_upper = component.upper()
        colored_tag = f"{color}[{component_upper}]{RESET_COLOR}"

        # Build final message
        message = record.getMessage()
        formatted = f"{timestamp} {level} {colored_tag} {message}"

        # Add exception info if present
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            formatted = formatted + "\n" + record.exc_text

        return formatted


def _setup_logging():
    """Configure the logging system with colored output."""
    # Create root handler with colored formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure external loggers to reduce noise
    # OpenAI SDK logs verbose HTTP request/response details + tracebacks at DEBUG;
    # our own [LLM] logger already surfaces clean error messages via _handle_api_error
    for name in ["openai", "httpcore", "httpx", "urllib3", "numba"]:
        logging.getLogger(name).setLevel(logging.WARNING)


# Initialize logging on module import
_setup_logging()


def get_logger(component: str) -> logging.Logger:
    """
    Get a component-specific logger with colored output.

    Args:
        component: One of "asr", "llm", "tts", "api", "sys"

    Returns:
        Logger instance for the specified component
    """
    component = component.lower()
    if component not in COMPONENT_COLORS:
        component = "sys"
    return logging.getLogger(f"ova.{component}")


# Default logger for backward compatibility (uses "sys" component)
logger = get_logger("sys")


def get_device():
    """Get compute device. Lazy imports torch to avoid early CUDA init."""
    import torch
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"CUDA available. Using {device}")
    else:
        device = "cpu"
        logger.warning("CUDA not available. Falling back to CPU (this will be slower)")
    return device