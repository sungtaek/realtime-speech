#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
APP="${APP:-asr_service.app:app}"

MODE="${1:-}"
case "${MODE}" in
  "")
    DEFAULT_BACKEND="faster_whisper"
    ;;
  "nemo")
    DEFAULT_BACKEND="nemo"
    ;;
  "whisper" | "faster_whisper")
    DEFAULT_BACKEND="faster_whisper"
    ;;
  *)
    echo "Usage: ./run_asr.sh [nemo|whisper]" >&2
    exit 1
    ;;
esac

export ASR_BACKEND="${ASR_BACKEND:-${DEFAULT_BACKEND}}"
export ASR_MODEL_NAME="${ASR_MODEL_NAME:-eesungkim/stt_kr_conformer_transducer_large}"
export ASR_TARGET_LANG="${ASR_TARGET_LANG:-}"
export ASR_STRIP_LANG_TAGS="${ASR_STRIP_LANG_TAGS:-true}"
export ASR_WHISPER_MODEL="${ASR_WHISPER_MODEL:-large-v3}"
export ASR_WHISPER_PARTIAL_MODEL="${ASR_WHISPER_PARTIAL_MODEL:-small}"
export ASR_WHISPER_COMPUTE_TYPE="${ASR_WHISPER_COMPUTE_TYPE:-float16}"
export ASR_WHISPER_PARTIAL_COMPUTE_TYPE="${ASR_WHISPER_PARTIAL_COMPUTE_TYPE:-${ASR_WHISPER_COMPUTE_TYPE}}"
export ASR_WHISPER_PARTIAL_BEAM_SIZE="${ASR_WHISPER_PARTIAL_BEAM_SIZE:-1}"
export ASR_WHISPER_NO_SPEECH_THRESHOLD="${ASR_WHISPER_NO_SPEECH_THRESHOLD:-0.6}"
export ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT="${ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT:-false}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn is not installed. Run: pip install -e \".[test]\"" >&2
  exit 1
fi

echo "Starting ASR service on http://${HOST}:${PORT}"
echo "Web test page: http://localhost:${PORT}/"
echo "ASR backend: ${ASR_BACKEND}"
if [[ "${ASR_BACKEND}" == "faster_whisper" ]]; then
  echo "ASR final model: ${ASR_WHISPER_MODEL}"
  echo "ASR partial model: ${ASR_WHISPER_PARTIAL_MODEL}"
  echo "ASR final compute type: ${ASR_WHISPER_COMPUTE_TYPE}"
  echo "ASR partial compute type: ${ASR_WHISPER_PARTIAL_COMPUTE_TYPE}"
  echo "ASR no speech threshold: ${ASR_WHISPER_NO_SPEECH_THRESHOLD}"
else
  echo "ASR model: ${ASR_MODEL_NAME}"
fi
if [[ -n "${ASR_TARGET_LANG}" ]]; then
  echo "ASR target language: ${ASR_TARGET_LANG}"
fi
exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
