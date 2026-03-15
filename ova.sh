#!/usr/bin/env bash
set -euo pipefail

# Prevent loky/joblib from creating semaphores by using threading backend
export JOBLIB_MULTIPROCESSING=0
export SKLEARN_NO_OPENMP=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OVA_DIR="$ROOT_DIR/.ova"
BACKEND_PID="$OVA_DIR/backend.pid"
BACKEND_GROUP="$OVA_DIR/backend.group"
BACKEND_LOG="$OVA_DIR/backend.log"
FRONTEND_PID="$OVA_DIR/frontend.pid"
FRONTEND_GROUP="$OVA_DIR/frontend.group"
FRONTEND_LOG="$OVA_DIR/frontend.log"

# Load .env file if it exists (before setting defaults)
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

# Server configuration with defaults
BACKEND_HOST="${OVA_BACKEND_HOST:-localhost}"
BACKEND_PORT="${OVA_BACKEND_PORT:-5173}"
FRONTEND_HOST="${OVA_FRONTEND_HOST:-localhost}"
FRONTEND_PORT="${OVA_FRONTEND_PORT:-8080}"

OVA_PROFILE="${OVA_PROFILE:-default}"

CHAT_MODEL="ministral-3:3b-instruct-2512-q4_K_M"
HF_MODELS=("hexgrad/Kokoro-82M" "nvidia/parakeet-tdt-0.6b-v3" "Qwen/Qwen3-TTS-12Hz-1.7B-Base" "Qwen/Qwen3-ASR-0.6B")

usage() {
  cat <<'EOF'
Usage: ova [OPTIONS] <command>

Options:
  OVA_PROFILE=<profile>  Set the profile to use (default: default)

Commands:
  install   Sync uv environment and fetch models
  start     Start backend + frontend server (non-blocking)
  stop      Stop running services
  restart   Restart services (keeps LLM loaded for faster init)
  help      Show this message

Example:
  OVA_PROFILE=dua ova start
EOF
}

die() {
  echo "ova: $*" >&2
  exit 1
}

ensure_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing '$1' in PATH"
}

ensure_uv_lock() {
  [[ -f "$ROOT_DIR/uv.lock" ]] || die "uv.lock not found in project root"
}

is_running() {
  local pidfile=$1
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

port_open() {
  local port=$1
  python3 - <<PY
import socket
import sys

port = int("${port}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.2)
try:
    sock.connect(("127.0.0.1", port))
    sock.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

wait_for_port() {
  local name=$1
  local port=$2
  local pidfile=$3
  local logfile=$4
  local timeout=${5:-10}
  local success_msg=${6:-}

  local start
  start="$(date +%s)"
  while true; do
    if ! is_running "$pidfile"; then
      echo "$name failed to start (process exited). See log: $logfile"
      return 1
    fi

    if port_open "$port"; then
      if [[ -n "$success_msg" ]]; then
        echo "$success_msg"
      else
        echo "$name started successfully (port $port)"
      fi
      return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "$name failed to start within ${timeout}s. See log: $logfile"
      return 1
    fi

    sleep 0.25
  done
}

wait_for_log_message() {
  local name=$1
  local logfile=$2
  local pidfile=$3
  local message=$4
  local attempts=${5:-120}
  local interval=${6:-1}
  local success_msg=${7:-}

  for ((i=1; i<=attempts; i++)); do
    if ! is_running "$pidfile"; then
      echo "$name failed to start (process exited). See log: $logfile"
      return 1
    fi

    if [[ -f "$logfile" ]] && grep -Fq "$message" "$logfile"; then
      if [[ -n "$success_msg" ]]; then
        echo "$success_msg"
      else
        echo "$name started successfully"
      fi
      return 0
    fi

    sleep "$interval"
  done

  echo "$name is still starting after ${attempts}s (first run may take longer). See log: $logfile"
  return 1
}

start_detached() {
  local logfile=$1
  local pidfile=$2
  local groupfile=$3
  shift 3

  mkdir -p "$OVA_DIR"
  : > "$logfile"

  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$logfile" 2>&1 < /dev/null &
    echo "$!" > "$pidfile"
    echo "1" > "$groupfile"
  else
    nohup "$@" >"$logfile" 2>&1 < /dev/null &
    echo "$!" > "$pidfile"
    rm -f "$groupfile"
  fi
}

stop_service() {
  local name=$1
  local pidfile=$2
  local groupfile=$3

  if [[ ! -f "$pidfile" ]]; then
    echo "$name not running"
    return 0
  fi

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
    rm -f "$pidfile" "$groupfile"
    echo "$name not running"
    return 0
  fi

  if ! ps -p "$pid" >/dev/null 2>&1; then
    rm -f "$pidfile" "$groupfile"
    echo "$name not running"
    return 0
  fi

  if [[ -f "$groupfile" ]]; then
    kill -- -"$pid" >/dev/null 2>&1 || true
  else
    kill "$pid" >/dev/null 2>&1 || true
  fi

  for _ in {1..25}; do
    if ps -p "$pid" >/dev/null 2>&1; then
      sleep 0.2
    else
      break
    fi
  done

  if ps -p "$pid" >/dev/null 2>&1; then
    if [[ -f "$groupfile" ]]; then
      kill -9 -- -"$pid" >/dev/null 2>&1 || true
    else
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "$pidfile" "$groupfile"
  echo "$name stopped"
}

ensure_ollama_model() {
  local model=$1
  local models
  if models="$(ollama list 2>/dev/null || true)"; then
    if echo "$models" | awk 'NR>1{print $1}' | grep -qx "$model"; then
      echo "Ollama model present: $model"
      return 0
    fi
  fi

  echo "Pulling Ollama model: $model"
  ollama pull "$model"
}

ensure_hf_models() {
  ensure_cmd uvx
  local cache_list
  # Use 'hf cache ls' to list cached models (format: model/<org>/<name>)
  cache_list="$(uvx hf cache ls 2>/dev/null || true)"
  for repo_id in "${HF_MODELS[@]}"; do
    # Check if repo_id exists in cache (cache format is "model/org/name" or "dataset/org/name")
    if [[ -n "$cache_list" ]] && echo "$cache_list" | grep -Fq "$repo_id"; then
      echo "HF model already cached: $repo_id"
    else
      echo "Downloading HF model: $repo_id"
      uvx hf download "$repo_id"
    fi
  done
}

cmd="${1:-help}"

case "$cmd" in
  install)
    ensure_cmd uv
    ensure_uv_lock
    uv sync --frozen
    if [[ "${OVA_LLM_PROVIDER:-ollama}" != "openai" ]]; then
      ensure_cmd ollama
      ensure_ollama_model "$CHAT_MODEL"
    else
      echo "Skipping Ollama (OVA_LLM_PROVIDER=openai)"
    fi
    ensure_hf_models
    ;;
  start)
    ensure_cmd uv
    ensure_uv_lock
    DISABLE_FRONTEND="${OVA_DISABLE_FRONTEND_ACCESS:-false}"

    echo "Starting Outrageous Voice Assistant..."
    if [[ "$DISABLE_FRONTEND" == "true" ]]; then
      echo "Frontend disabled (OVA_DISABLE_FRONTEND_ACCESS=true)"
    else
      echo "Starting front-end..."
      if is_running "$FRONTEND_PID"; then
        if port_open "$FRONTEND_PORT"; then
          echo "Front-end already running (port $FRONTEND_PORT)"
        else
          echo "Front-end running but not responding on port $FRONTEND_PORT"
        fi
      else
        start_detached "$FRONTEND_LOG" "$FRONTEND_PID" "$FRONTEND_GROUP" \
          uv run python3 -m http.server "$FRONTEND_PORT" --bind "$FRONTEND_HOST"
        wait_for_port "Web server" "$FRONTEND_PORT" "$FRONTEND_PID" "$FRONTEND_LOG" 5 \
          "Front-end started successfully (port $FRONTEND_PORT)"
      fi
    fi

    echo "Starting back-end..."
    if is_running "$BACKEND_PID"; then
      if port_open "$BACKEND_PORT"; then
        echo "Back-end already running (port $BACKEND_PORT)"
      else
        echo "Back-end running but not responding on port $BACKEND_PORT"
      fi
    else
      UVICORN_RELOAD="${UVICORN_RELOAD:-}"
      if [[ -n "$UVICORN_RELOAD" ]]; then
        start_detached "$BACKEND_LOG" "$BACKEND_PID" "$BACKEND_GROUP" \
          bash -c "LD_LIBRARY_PATH='$ROOT_DIR/.venv/lib/cuda-compat:$LD_LIBRARY_PATH' PYTHONWARNINGS='ignore::SyntaxWarning' OVA_PROFILE='$OVA_PROFILE' uv run uvicorn ova.api:app --reload --host \"$BACKEND_HOST\" --port \"$BACKEND_PORT\""
      else
        start_detached "$BACKEND_LOG" "$BACKEND_PID" "$BACKEND_GROUP" \
          bash -c "LD_LIBRARY_PATH='$ROOT_DIR/.venv/lib/cuda-compat:$LD_LIBRARY_PATH' PYTHONWARNINGS='ignore::SyntaxWarning' OVA_PROFILE='$OVA_PROFILE' uv run uvicorn ova.api:app --host \"$BACKEND_HOST\" --port \"$BACKEND_PORT\""
      fi
      wait_for_log_message "Backend" "$BACKEND_LOG" "$BACKEND_PID" \
        "Application startup complete." 120 1 \
        "Back-end started successfully (port $BACKEND_PORT)"
    fi
    echo "All services are up and running."
    if [[ "$DISABLE_FRONTEND" == "true" ]]; then
      DISPLAY_HOST="$BACKEND_HOST"
      [[ "$DISPLAY_HOST" == "0.0.0.0" ]] && DISPLAY_HOST="localhost"
      echo "API-only mode: backend available at http://$DISPLAY_HOST:$BACKEND_PORT"
    else
      # Show localhost in URL if bound to 0.0.0.0 (more user-friendly)
      DISPLAY_HOST="$FRONTEND_HOST"
      [[ "$DISPLAY_HOST" == "0.0.0.0" ]] && DISPLAY_HOST="localhost"
      echo "Now just fire up your browser and go to http://$DISPLAY_HOST:$FRONTEND_PORT and enjoy!!!"
    fi
    ;;
  stop)
    stop_service "Backend" "$BACKEND_PID" "$BACKEND_GROUP"
    stop_service "Web server" "$FRONTEND_PID" "$FRONTEND_GROUP"
    # Clean up stale marker (safety for failed restarts)
    rm -f "$OVA_DIR/keep_llm"
    ;;
  restart)
    # Create marker to keep LLM loaded during restart
    mkdir -p "$OVA_DIR"
    touch "$OVA_DIR/keep_llm"
    # Stop services directly (don't call stop which would delete marker)
    stop_service "Backend" "$BACKEND_PID" "$BACKEND_GROUP"
    stop_service "Web server" "$FRONTEND_PID" "$FRONTEND_GROUP"
    # Start services
    "$0" start
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    die "unknown command: $cmd"
    ;;
esac
