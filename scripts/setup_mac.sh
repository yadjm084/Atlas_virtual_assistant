#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_FORMULA="${PYTHON_FORMULA:-python@3.10}"
INSTALL_FFMPEG="${INSTALL_FFMPEG:-1}"

log() {
  printf "\n[setup_mac] %s\n" "$1"
}

fail() {
  printf "\n[setup_mac] ERROR: %s\n" "$1" >&2
  exit 1
}

ensure_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "This script is for macOS only."
  fi
}

ensure_project_files() {
  [[ -f "${PROJECT_ROOT}/requirements.txt" ]] || fail "requirements.txt not found in ${PROJECT_ROOT}"
  [[ -f "${PROJECT_ROOT}/app.py" ]] || fail "app.py not found in ${PROJECT_ROOT}"
}

ensure_xcode_cli() {
  if xcode-select -p >/dev/null 2>&1; then
    return
  fi

  log "Xcode Command Line Tools are required. Launching the macOS installer prompt."
  xcode-select --install >/dev/null 2>&1 || true
  fail "Finish installing Xcode Command Line Tools, then rerun this script."
}

bootstrap_brew_path() {
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    return
  fi

  if [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

ensure_homebrew() {
  bootstrap_brew_path
  if command -v brew >/dev/null 2>&1; then
    return
  fi

  log "Homebrew not found. Installing Homebrew."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  bootstrap_brew_path
  command -v brew >/dev/null 2>&1 || fail "Homebrew installation finished, but brew is still not on PATH."
}

ensure_brew_package() {
  local formula="$1"

  if brew list "${formula}" >/dev/null 2>&1; then
    return
  fi

  log "Installing ${formula} with Homebrew."
  brew install "${formula}"
}

resolve_python_bin() {
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
    return
  fi

  ensure_brew_package "${PYTHON_FORMULA}"
  PYTHON_BIN="$(brew --prefix "${PYTHON_FORMULA}")/bin/python3.10"
  [[ -x "${PYTHON_BIN}" ]] || fail "Python 3.10 was installed, but ${PYTHON_BIN} was not found."
}

verify_python_version() {
  "${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10, got {sys.version.split()[0]}")
print(sys.version.split()[0])
PY
}

create_virtualenv() {
  log "Creating virtual environment at ${VENV_DIR}."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

install_python_dependencies() {
  log "Upgrading pip tooling."
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

  log "Installing Python dependencies from requirements.txt."
  "${VENV_DIR}/bin/pip" install -r "${PROJECT_ROOT}/requirements.txt"
}

verify_python_imports() {
  log "Running dependency import check."
  "${VENV_DIR}/bin/python" - <<'PY'
import audioread
import edge_tts
import gradio
import huggingface_hub
import imageio_ffmpeg
import librosa
import numpy
import pydub
import requests
import seqeval
import sklearn
import soundfile
import tensorflow
import torch
import transformers
import whisper

print("Import check passed.")
PY
}

main() {
  ensure_macos
  ensure_project_files
  ensure_xcode_cli
  ensure_homebrew

  if [[ "${INSTALL_FFMPEG}" == "1" ]]; then
    ensure_brew_package "ffmpeg"
  fi

  resolve_python_bin
  log "Using Python: ${PYTHON_BIN} ($(verify_python_version))"

  create_virtualenv
  install_python_dependencies
  verify_python_imports

  mkdir -p "${PROJECT_ROOT}/generated_audio"

  cat <<EOF

[setup_mac] Setup complete.

Activate the environment:
  source "${VENV_DIR}/bin/activate"

Optional TMDB credentials:
  export TMDB_API_KEY="your_api_key"
  export TMDB_BEARER_TOKEN="your_bearer_token"

Run Atlas:
  cd "${PROJECT_ROOT}"
  python app.py

Optional smoke test:
  python demo_smoke_test.py --skip-tts

Notes:
  - The first app run can still download the Whisper tiny model.
  - Atlas will prefer the local atlas-voice-data-wav dataset in this repo.
EOF
}

main "$@"
