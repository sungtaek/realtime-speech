# Real-time Speech Service

Speech service project for real-time speech features. The current implementation provides streaming ASR over FastAPI WebSocket, and the project is structured to add TTS next. ASR accepts 16 kHz, 16-bit, mono PCM audio chunks, uses WebRTC VAD for utterance start/end detection, and supports NeMo RNNT/TDT plus faster-whisper backends for partial and final transcription.

Default ASR models:

```text
backend: faster-whisper
partial: small
final: large-v3
```

## Install

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[test]"
```

Use a CUDA/PyTorch environment compatible with your GPU when running on an NVIDIA GPU. If your platform needs a specific CUDA wheel, install `torch` from the official PyTorch index before installing this project.

### macOS

```bash
brew install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[test]"
```

Run on CPU:

```bash
ASR_DEVICE=cpu ./run_asr.sh
```

Python 3.10 or 3.11 may be more stable than 3.12 if NeMo dependency resolution fails on macOS.

The project uses `webrtcvad-wheels`, which provides the `webrtcvad` Python module with prebuilt wheels on common Python versions.

## Current Scope

- ASR: implemented. Streaming PCM WebSocket, WebRTC VAD, NeMo RNNT/TDT, and faster-whisper backends.
- TTS: planned. Future TTS scripts and endpoints should use a `tts` prefix, for example `run_tts.sh` and `/v1/tts/...`.

## Run ASR

```bash
./run_asr.sh
```

Then open the browser test page:

```text
http://localhost:8000/
```

The test page supports both microphone input and audio file upload. Select an audio file, click `Send File`, and the browser will decode it and stream 16 kHz PCM chunks to the WebSocket endpoint.

Useful environment variables:

```bash
ASR_BACKEND=faster_whisper
ASR_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
ASR_MODEL_PATH=
ASR_DEVICE=cuda
ASR_TARGET_LANG=
ASR_STRIP_LANG_TAGS=true
ASR_WHISPER_MODEL=large-v3
ASR_WHISPER_PARTIAL_MODEL=small
ASR_WHISPER_COMPUTE_TYPE=float16
ASR_WHISPER_PARTIAL_COMPUTE_TYPE=float16
ASR_WHISPER_BEAM_SIZE=5
ASR_WHISPER_PARTIAL_BEAM_SIZE=1
ASR_WHISPER_NO_SPEECH_THRESHOLD=0.6
ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
ASR_VAD_AGGRESSIVENESS=2
ASR_VAD_FRAME_MS=20
ASR_START_TRIGGER_MS=160
ASR_END_SILENCE_MS=700
ASR_PREROLL_MS=300
ASR_PARTIAL_INTERVAL_MS=800
ASR_MIN_PARTIAL_MS=600
ASR_MAX_UTTERANCE_MS=30000
```

Set `ASR_MODEL_PATH=/path/to/model.nemo` to load a local checkpoint instead of `ASR_MODEL_NAME`.

By default, `run_asr.sh` uses faster-whisper:

```bash
./run_asr.sh
```

Equivalent explicit command:

```bash
./run_asr.sh whisper
```

To tune faster-whisper models:

```bash
ASR_WHISPER_MODEL=large-v3 \
ASR_WHISPER_PARTIAL_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

In this mode, partial results use `ASR_WHISPER_PARTIAL_MODEL`, and final results after `speech_end` use `ASR_WHISPER_MODEL`.

If CUDA runtime libraries are not installed, run Whisper on CPU:

```bash
ASR_DEVICE=cpu \
ASR_WHISPER_COMPUTE_TYPE=int8 \
ASR_WHISPER_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

For a faster but less accurate Whisper test:

```bash
ASR_WHISPER_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

You can also try NVIDIA's newer multilingual streaming RNNT model with a Korean language prompt:

```bash
ASR_MODEL_NAME=nvidia/nemotron-3.5-asr-streaming-0.6b \
ASR_TARGET_LANG=ko-KR \
./run_asr.sh nemo
```

The project installs NeMo from GitHub main so models requiring newer NeMo classes can load.

For automatic language detection with the multilingual model:

```bash
ASR_MODEL_NAME=nvidia/nemotron-3.5-asr-streaming-0.6b \
ASR_TARGET_LANG=auto \
./run_asr.sh nemo
```

You can also override the bind address:

```bash
HOST=127.0.0.1 PORT=9000 ./run_asr.sh
```

## WebSocket Protocol

Connect to:

```text
ws://localhost:8000/v1/asr/stream
```

Send binary messages containing raw little-endian signed 16-bit PCM at 16 kHz mono. Text messages are accepted only for control:

```json
{"type":"end"}
```

Server events:

```json
{"type":"speech_start","utterance_id":"...","time_ms":420}
{"type":"partial","utterance_id":"...","text":"hello wor","audio_ms":1200}
{"type":"speech_end","utterance_id":"...","audio_ms":1460,"time_ms":2140}
{"type":"final","utterance_id":"...","text":"hello world","audio_ms":1460}
```

## Test

```bash
pytest
```
