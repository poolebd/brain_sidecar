#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/runtime/dev"
LOG_DIR="$RUNTIME_DIR/logs"
PID_DIR="$RUNTIME_DIR/pids"
STATE_DIR="$RUNTIME_DIR/state"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
UI_DIR="$ROOT_DIR/ui"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR"

BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

trim_spaces() {
  awk '{$1=$1; print}'
}

load_dotenv_without_overrides() {
  local env_file="$ROOT_DIR/.env"
  [[ -f "$env_file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    if [[ -z "${!key+x}" ]]; then
      value="$(ENV_FILE="$env_file" ENV_KEY="$key" bash -c 'set -a; source "$ENV_FILE"; printf "%s" "${!ENV_KEY-}"')"
      export "$key=$value"
    fi
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=.*/\2/p' "$env_file")
}

resolve_system_python() {
  if [[ -n "${BRAIN_SIDECAR_SYSTEM_PYTHON:-}" ]]; then
    printf '%s\n' "$BRAIN_SIDECAR_SYSTEM_PYTHON"
    return 0
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

python_dependencies_ready() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import os

required = [
    "fastapi",
    "uvicorn",
    "faster_whisper",
    "torch",
    "torchaudio",
    "speechbrain",
    "faiss",
    "pypdf",
    "docx",
    "pytest",
]

if os.environ.get("BRAIN_SIDECAR_ASR_BACKEND", "nemotron_streaming") == "nemotron_streaming":
    required.extend(["Cython", "nemo", "nemo.collections.asr"])

def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False

missing = [name for name in required if not has_module(name)]
raise SystemExit(1 if missing else 0)
PY
}

run_or_show_log_tail() {
  local description="$1"
  local log_file="$2"
  shift 2

  if ! "$@" >"$log_file" 2>&1; then
    echo "$description failed. See $log_file" >&2
    tail -n 30 "$log_file" >&2 2>/dev/null || true
    exit 1
  fi
}

bootstrap_python_environment() {
  local system_python pyproject_hash python_stamp current_hash venv_created install_needed
  system_python="$(resolve_system_python || true)"
  if [[ -z "$system_python" ]]; then
    echo "[env] python3.11 or python3 is required to bootstrap .venv." >&2
    exit 1
  fi

  venv_created=0
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[env] creating .venv with $system_python"
    "$system_python" -m venv "$VENV_DIR"
    venv_created=1
  fi

  if [[ ! -x "$PYTHON_BIN" || ! -x "$PIP_BIN" ]]; then
    echo "[env] expected virtualenv binaries were not created in $VENV_DIR." >&2
    exit 1
  fi

  pyproject_hash="$(hash_file "$ROOT_DIR/pyproject.toml")"
  python_stamp="$STATE_DIR/python-pyproject.sha256"
  current_hash="$(cat "$python_stamp" 2>/dev/null || true)"
  install_needed=0

  if [[ "$venv_created" == "1" ]]; then
    install_needed=1
  elif [[ -n "$current_hash" && "$pyproject_hash" != "$current_hash" ]]; then
    install_needed=1
  elif ! python_dependencies_ready; then
    install_needed=1
  fi

  if [[ "$install_needed" == "1" ]]; then
    local extras
    extras="gpu,speaker,recall,dev"
    if [[ "${BRAIN_SIDECAR_ASR_BACKEND:-nemotron_streaming}" == "nemotron_streaming" ]]; then
      extras="$extras,nemotron"
    fi
    echo "[env] installing Python dependencies (logs in $LOG_DIR)"
    run_or_show_log_tail "[env] pip upgrade" "$LOG_DIR/pip-upgrade.log" "$PYTHON_BIN" -m pip install --upgrade pip
    run_or_show_log_tail "[env] pip install" "$LOG_DIR/pip-install.log" "$PIP_BIN" install -e ".[$extras]"
    if [[ "${BRAIN_SIDECAR_ASR_BACKEND:-nemotron_streaming}" == "nemotron_streaming" ]]; then
      run_or_show_log_tail "[env] pip install NeMo ASR" "$LOG_DIR/pip-install-nemotron.log" \
        "$PIP_BIN" install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'
    fi
    echo "$pyproject_hash" >"$python_stamp"
  elif [[ -z "$current_hash" ]]; then
    echo "$pyproject_hash" >"$python_stamp"
  fi

  echo "[env] Python environment ready"
}

bootstrap_ui_environment() {
  local lock_file lock_hash ui_stamp current_hash install_needed
  if ! command -v npm >/dev/null 2>&1; then
    echo "[env] npm is required to run the Vite UI." >&2
    exit 1
  fi

  lock_file="$UI_DIR/package-lock.json"
  if [[ ! -f "$lock_file" ]]; then
    echo "[env] $lock_file is required for deterministic npm installs." >&2
    exit 1
  fi

  lock_hash="$(hash_file "$lock_file")"
  ui_stamp="$STATE_DIR/ui-package-lock.sha256"
  current_hash="$(cat "$ui_stamp" 2>/dev/null || true)"
  install_needed=0

  if [[ ! -d "$UI_DIR/node_modules" ]]; then
    install_needed=1
  elif [[ -n "$current_hash" && "$lock_hash" != "$current_hash" ]]; then
    install_needed=1
  fi

  if [[ "$install_needed" == "1" ]]; then
    echo "[env] installing UI dependencies with npm ci (log in $LOG_DIR/npm-ci.log)"
    run_or_show_log_tail "[env] npm ci" "$LOG_DIR/npm-ci.log" npm --prefix "$UI_DIR" ci
    echo "$lock_hash" >"$ui_stamp"
  elif [[ -z "$current_hash" ]]; then
    echo "$lock_hash" >"$ui_stamp"
  fi

  echo "[env] UI environment ready"
}

pid_alive() {
  local pid="$1"
  local stat
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  stat="$(ps -o stat= -p "$pid" 2>/dev/null | trim_spaces)"
  [[ -n "$stat" && "$stat" != Z* ]]
}

pid_owner_uid() {
  local pid="$1"
  ps -o uid= -p "$pid" 2>/dev/null | trim_spaces
}

pid_owner_name() {
  local pid="$1"
  ps -o user= -p "$pid" 2>/dev/null | trim_spaces
}

pid_cmd() {
  local pid="$1"
  ps -o cmd= -p "$pid" 2>/dev/null | trim_spaces
}

same_repo_pid() {
  local pid="$1"
  local cwd cmd
  cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
  cmd="$(pid_cmd "$pid")"

  [[ "$cwd" == "$ROOT_DIR" || "$cwd" == "$ROOT_DIR/"* ]] && return 0
  [[ "$cmd" == *"$ROOT_DIR"* ]] && return 0
  return 1
}

append_unique_pid() {
  local pid="$1"
  shift
  local existing
  for existing in "$@"; do
    [[ "$existing" == "$pid" ]] && return 1
  done
  printf '%s\n' "$pid"
}

port_listener_pids() {
  local port="$1"
  ss -ltnp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true
}

matching_backend_pids() {
  local pids=()
  local pid

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    pids+=("$pid")
  done < <(pgrep -f "brain_sidecar.server.app:create_app" || true)

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    pids+=("$pid")
  done < <(port_listener_pids "$BRAIN_SIDECAR_PORT")

  local seen=()
  for pid in "${pids[@]}"; do
    [[ "$pid" == "$$" ]] && continue
    pid_alive "$pid" || continue
    same_repo_pid "$pid" || continue
    append_unique_pid "$pid" "${seen[@]}" >/dev/null || continue
    seen+=("$pid")
    printf '%s\n' "$pid"
  done
}

matching_frontend_pids() {
  local pids=()
  local pid

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    pids+=("$pid")
  done < <(pgrep -f "vite.*--port[ =]?$BRAIN_SIDECAR_UI_PORT" || true)

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    pids+=("$pid")
  done < <(port_listener_pids "$BRAIN_SIDECAR_UI_PORT")

  local seen=()
  for pid in "${pids[@]}"; do
    [[ "$pid" == "$$" ]] && continue
    pid_alive "$pid" || continue
    same_repo_pid "$pid" || continue
    append_unique_pid "$pid" "${seen[@]}" >/dev/null || continue
    seen+=("$pid")
    printf '%s\n' "$pid"
  done
}

describe_pid() {
  local pid="$1"
  local owner cmd
  owner="$(pid_owner_name "$pid")"
  cmd="$(pid_cmd "$pid")"
  printf '  %s pid=%s %s\n' "${owner:-unknown}" "$pid" "$cmd"
}

ensure_pid_file_not_running() {
  local name="$1"
  local pid_file="$2"
  local pid owner_uid

  [[ -f "$pid_file" ]] || return 0
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if ! pid_alive "$pid"; then
    rm -f "$pid_file"
    return 0
  fi

  owner_uid="$(pid_owner_uid "$pid")"
  echo "$name is already running from $pid_file:"
  describe_pid "$pid"
  if [[ "$(id -u)" != "0" && "$owner_uid" != "$(id -u)" ]]; then
    echo "Run sudo ./stop.sh, then start again."
  else
    echo "Run ./stop.sh, then start again."
  fi
  exit 1
}

ensure_no_matching_processes() {
  local name="$1"
  local current_uid pid owner_uid
  shift
  current_uid="$(id -u)"

  local matches=("$@")
  [[ "${#matches[@]}" -gt 0 ]] || return 0

  echo "$name appears to already be running for this repo and port:"
  for pid in "${matches[@]}"; do
    describe_pid "$pid"
  done

  for pid in "${matches[@]}"; do
    owner_uid="$(pid_owner_uid "$pid")"
    if [[ "$current_uid" != "0" && "$owner_uid" != "$current_uid" ]]; then
      echo "Run sudo ./stop.sh, then start again."
      exit 1
    fi
  done

  echo "Run ./stop.sh, then start again."
  exit 1
}

curl_host() {
  local host="$1"
  if [[ "$host" == "0.0.0.0" || "$host" == "::" || "$host" == "[::]" ]]; then
    printf '127.0.0.1\n'
  else
    printf '%s\n' "$host"
  fi
}

launch_detached() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  shift 3

  {
    printf '\n[%s] starting %s\n' "$(timestamp)" "$name"
    printf '[%s] command:' "$(timestamp)"
    printf ' %q' "$@"
    printf '\n'
  } >>"$log_file"

  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >>"$log_file" 2>&1 </dev/null &
  else
    nohup "$@" >>"$log_file" 2>&1 </dev/null &
  fi

  local pid="$!"
  echo "$pid" >"$pid_file"
  echo "[start] $name pid $pid"
}

wait_for_http_or_exit() {
  local name="$1"
  local url="$2"
  local pid_file="$3"
  local log_file="$4"
  local retries="${5:-60}"
  local pid i
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  for ((i = 1; i <= retries; i += 1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[ready] $name: $url"
      return 0
    fi
    if ! pid_alive "$pid"; then
      echo "$name exited before becoming ready. See $log_file" >&2
      tail -n 30 "$log_file" >&2 2>/dev/null || true
      exit 1
    fi
    sleep 1
  done

  echo "$name did not become ready at $url. See $log_file" >&2
  tail -n 30 "$log_file" >&2 2>/dev/null || true
  exit 1
}

load_dotenv_without_overrides
export BRAIN_SIDECAR_DATA_DIR="${BRAIN_SIDECAR_DATA_DIR:-$ROOT_DIR/runtime}"
export BRAIN_SIDECAR_TEST_MODE_ENABLED="${BRAIN_SIDECAR_TEST_MODE_ENABLED:-0}"
export BRAIN_SIDECAR_HOST="${BRAIN_SIDECAR_HOST:-127.0.0.1}"
export BRAIN_SIDECAR_PORT="${BRAIN_SIDECAR_PORT:-8765}"
export BRAIN_SIDECAR_UI_HOST="${BRAIN_SIDECAR_UI_HOST:-127.0.0.1}"
export BRAIN_SIDECAR_UI_PORT="${BRAIN_SIDECAR_UI_PORT:-8766}"
export BRAIN_SIDECAR_ASR_BACKEND="${BRAIN_SIDECAR_ASR_BACKEND:-nemotron_streaming}"
export BRAIN_SIDECAR_NEMOTRON_CHUNK_MS="${BRAIN_SIDECAR_NEMOTRON_CHUNK_MS:-160}"
export BRAIN_SIDECAR_NEMOTRON_DTYPE="${BRAIN_SIDECAR_NEMOTRON_DTYPE:-float32}"

ensure_pid_file_not_running "Backend" "$BACKEND_PID_FILE"
ensure_pid_file_not_running "Frontend" "$FRONTEND_PID_FILE"

mapfile -t BACKEND_MATCHES < <(matching_backend_pids)
ensure_no_matching_processes "Backend" "${BACKEND_MATCHES[@]}"
mapfile -t FRONTEND_MATCHES < <(matching_frontend_pids)
ensure_no_matching_processes "Frontend" "${FRONTEND_MATCHES[@]}"

bootstrap_python_environment
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/gpu-env.sh"
bootstrap_ui_environment

launch_detached \
  "backend" \
  "$BACKEND_PID_FILE" \
  "$BACKEND_LOG" \
  "$PYTHON_BIN" -m uvicorn brain_sidecar.server.app:create_app \
    --factory \
    --host "$BRAIN_SIDECAR_HOST" \
    --port "$BRAIN_SIDECAR_PORT"

launch_detached \
  "frontend" \
  "$FRONTEND_PID_FILE" \
  "$FRONTEND_LOG" \
  npm --prefix "$UI_DIR" run dev -- \
    --host "$BRAIN_SIDECAR_UI_HOST" \
    --port "$BRAIN_SIDECAR_UI_PORT"

BACKEND_READY_HOST="$(curl_host "$BRAIN_SIDECAR_HOST")"
FRONTEND_READY_HOST="$(curl_host "$BRAIN_SIDECAR_UI_HOST")"

wait_for_http_or_exit "backend" "http://$BACKEND_READY_HOST:$BRAIN_SIDECAR_PORT/api/health/gpu" "$BACKEND_PID_FILE" "$BACKEND_LOG" 60
wait_for_http_or_exit "frontend" "http://$FRONTEND_READY_HOST:$BRAIN_SIDECAR_UI_PORT/" "$FRONTEND_PID_FILE" "$FRONTEND_LOG" 60

echo
echo "Brain Sidecar is running."
echo "Backend:  http://$BACKEND_READY_HOST:$BRAIN_SIDECAR_PORT"
echo "UI:       http://$FRONTEND_READY_HOST:$BRAIN_SIDECAR_UI_PORT"
echo "Hosted:   https://notes.shoalstone.net/ via the existing Caddy route"
echo
echo "Logs:"
echo "  $BACKEND_LOG"
echo "  $FRONTEND_LOG"
echo "PIDs:"
echo "  $BACKEND_PID_FILE"
echo "  $FRONTEND_PID_FILE"
echo
echo "Stop with: ./stop.sh"
