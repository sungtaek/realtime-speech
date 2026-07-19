#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
APP="${APP:-asr_service.app:app}"

MODE="${1:-}"
case "${MODE}" in
  "")
    DEFAULT_BACKEND="whisper"
    ;;
  "nemo")
    DEFAULT_BACKEND="nemo"
    ;;
  "whisper")
    DEFAULT_BACKEND="whisper"
    ;;
  *)
    echo "Usage: ./run_asr.sh [nemo|whisper]" >&2
    exit 1
    ;;
esac

export ASR_ENGINE="${ASR_ENGINE:-${DEFAULT_BACKEND}}"
export ASR_NEMO_MODEL_NAME="${ASR_NEMO_MODEL_NAME:-eesungkim/stt_kr_conformer_transducer_large}"
export ASR_NEMO_MODEL_PATH="${ASR_NEMO_MODEL_PATH:-}"
export ASR_NEMO_STRIP_LANG_TAGS="${ASR_NEMO_STRIP_LANG_TAGS:-true}"
export ASR_TARGET_LANG="${ASR_TARGET_LANG:-}"
export ASR_WHISPER_MODEL="${ASR_WHISPER_MODEL:-large-v3}"
export ASR_WHISPER_PARTIAL_MODEL="${ASR_WHISPER_PARTIAL_MODEL:-small}"
export ASR_WHISPER_COMPUTE_TYPE="${ASR_WHISPER_COMPUTE_TYPE:-float16}"
export ASR_WHISPER_PARTIAL_COMPUTE_TYPE="${ASR_WHISPER_PARTIAL_COMPUTE_TYPE:-${ASR_WHISPER_COMPUTE_TYPE}}"
export ASR_WHISPER_BEAM_SIZE="${ASR_WHISPER_BEAM_SIZE:-5}"
export ASR_WHISPER_PARTIAL_BEAM_SIZE="${ASR_WHISPER_PARTIAL_BEAM_SIZE:-1}"
export ASR_WHISPER_NO_SPEECH_THRESHOLD="${ASR_WHISPER_NO_SPEECH_THRESHOLD:-0.6}"
export ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT="${ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT:-false}"
export ASR_VAD_AGGRESSIVENESS="${ASR_VAD_AGGRESSIVENESS:-2}"
export ASR_VAD_FRAME_MS="${ASR_VAD_FRAME_MS:-20}"
export ASR_VAD_START_TRIGGER_MS="${ASR_VAD_START_TRIGGER_MS:-160}"
export ASR_VAD_END_SILENCE_MS="${ASR_VAD_END_SILENCE_MS:-700}"
export ASR_VAD_PREROLL_MS="${ASR_VAD_PREROLL_MS:-300}"
export ASR_VAD_MAX_UTTERANCE_MS="${ASR_VAD_MAX_UTTERANCE_MS:-30000}"
export ASR_STREAM_PARTIAL_INTERVAL_MS="${ASR_STREAM_PARTIAL_INTERVAL_MS:-800}"
export ASR_STREAM_MIN_PARTIAL_MS="${ASR_STREAM_MIN_PARTIAL_MS:-600}"

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
echo "ASR engine: ${ASR_ENGINE}"
if [[ "${ASR_ENGINE}" == "whisper" ]]; then
  echo "ASR final model: ${ASR_WHISPER_MODEL}"
  echo "ASR partial model: ${ASR_WHISPER_PARTIAL_MODEL}"
  echo "ASR final compute type: ${ASR_WHISPER_COMPUTE_TYPE}"
  echo "ASR partial compute type: ${ASR_WHISPER_PARTIAL_COMPUTE_TYPE}"
  echo "ASR no speech threshold: ${ASR_WHISPER_NO_SPEECH_THRESHOLD}"
else
  echo "ASR model: ${ASR_NEMO_MODEL_NAME}"
fi
if [[ -n "${ASR_TARGET_LANG}" ]]; then
  echo "ASR target language: ${ASR_TARGET_LANG}"
fi
exec uvicorn "${APP}" --host "${HOST}" --port "${PORT}"
