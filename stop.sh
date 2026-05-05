#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/runtime/dev"
PID_DIR="$RUNTIME_DIR/pids"
UI_DIR="$ROOT_DIR/ui"

BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

CURRENT_UID="$(id -u)"
CURRENT_USER="$(id -un)"
FOREIGN_PROCESS_BLOCKERS=""

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

can_manage_pid() {
  local pid="$1"
  [[ "$CURRENT_UID" == "0" ]] && return 0
  [[ "$(pid_owner_uid "$pid")" == "$CURRENT_UID" ]]
}

record_foreign_process() {
  local label="$1"
  local pid="$2"
  local owner cmd
  owner="$(pid_owner_name "$pid")"
  cmd="$(pid_cmd "$pid")"
  FOREIGN_PROCESS_BLOCKERS+=$'\n'"$label: ${owner:-unknown} pid=$pid $cmd"
}

process_group_signal_target() {
  local pid="$1"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | trim_spaces)"
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    printf -- '-%s\n' "$pid"
  else
    printf '%s\n' "$pid"
  fi
}

signal_pid() {
  local signal="$1"
  local pid="$2"
  local target
  target="$(process_group_signal_target "$pid")"
  kill "-$signal" "$target" >/dev/null 2>&1 || true
}

stop_pid() {
  local label="$1"
  local pid="$2"
  local i

  if ! pid_alive "$pid"; then
    return 0
  fi

  if ! can_manage_pid "$pid"; then
    record_foreign_process "$label" "$pid"
    return 0
  fi

  echo "[stop] stopping $label (pid $pid)..."
  signal_pid TERM "$pid"
  for ((i = 1; i <= 15; i += 1)); do
    if ! pid_alive "$pid"; then
      return 0
    fi
    sleep 1
  done

  echo "[stop] $label did not exit in time; forcing kill."
  signal_pid KILL "$pid"
  for ((i = 1; i <= 5; i += 1)); do
    if ! pid_alive "$pid"; then
      return 0
    fi
    sleep 1
  done
}

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  local pid

  [[ -f "$pid_file" ]] || return 0
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if ! pid_alive "$pid"; then
    rm -f "$pid_file"
    return 0
  fi

  stop_pid "$label" "$pid"
  if ! pid_alive "$pid"; then
    rm -f "$pid_file"
  fi
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

stop_matching_processes() {
  local label="$1"
  shift
  local pid

  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    stop_pid "$label" "$pid"
  done
}

load_dotenv_without_overrides
export BRAIN_SIDECAR_PORT="${BRAIN_SIDECAR_PORT:-8765}"
export BRAIN_SIDECAR_UI_PORT="${BRAIN_SIDECAR_UI_PORT:-8766}"

stop_pid_file "frontend" "$FRONTEND_PID_FILE"
stop_pid_file "backend" "$BACKEND_PID_FILE"

mapfile -t FRONTEND_MATCHES < <(matching_frontend_pids)
stop_matching_processes "frontend" "${FRONTEND_MATCHES[@]}"

mapfile -t BACKEND_MATCHES < <(matching_backend_pids)
stop_matching_processes "backend" "${BACKEND_MATCHES[@]}"

if [[ -n "$FOREIGN_PROCESS_BLOCKERS" ]]; then
  echo "Some Brain Sidecar processes were not stopped because they are owned by another user:"
  printf '%s\n' "$FOREIGN_PROCESS_BLOCKERS" | sed '/^$/d'
  if [[ "$CURRENT_UID" == "0" ]]; then
    echo "Confirm those processes are still active, then rerun ./stop.sh."
  else
    echo "Run sudo ./stop.sh once, then use ./start.sh and ./stop.sh as $CURRENT_USER."
  fi
  exit 1
fi

echo "Brain Sidecar processes stopped."
