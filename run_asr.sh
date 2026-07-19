#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  "")
    ENGINE="whisper"
    ;;
  "nemo")
    ENGINE="nemo"
    ;;
  "whisper")
    ENGINE="whisper"
    ;;
  *)
    echo "Usage: ./run_asr.sh [nemo|whisper]" >&2
    exit 1
    ;;
esac

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "${PYTHON} is not installed or not available in PATH." >&2
  exit 1
fi

exec "${PYTHON}" -m asr_service.cli "${ENGINE}"
