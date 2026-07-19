# Real-time Speech Service

Speech service project for real-time speech features. The current implementation provides streaming ASR over FastAPI WebSocket, and the project is structured to add TTS next. ASR accepts 16 kHz, 16-bit, mono PCM audio chunks, uses WebRTC VAD for utterance start/end detection, and supports NeMo RNNT/TDT plus Whisper engines for partial and final transcription.

Default ASR models:

```text
engine: whisper
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

- ASR: implemented. Streaming PCM WebSocket, WebRTC VAD, NeMo RNNT/TDT, and Whisper engines.
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

Common environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_ENGINE` | `whisper` via `run_asr.sh`; `nemo` when running the app directly | ASR engine. Use `whisper` or `nemo`. |
| `ASR_DEVICE` | `cuda` | Inference device. Use `cuda` or `cpu`. The code falls back to CPU when CUDA is unavailable. |
| `ASR_TARGET_LANG` | empty | Language hint. Whisper maps `ko`, `ko-KR`, and `korean` to `ko`; NeMo multilingual models can use values such as `ko-KR` or `auto`. |

VAD environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_VAD_AGGRESSIVENESS` | `2` | WebRTC VAD aggressiveness from `0` to `3`; higher is stricter. |
| `ASR_VAD_FRAME_MS` | `20` | VAD frame size in milliseconds. Must be `10`, `20`, or `30`. |
| `ASR_VAD_START_TRIGGER_MS` | `160` | Required speech duration before emitting `speech_start`. |
| `ASR_VAD_END_SILENCE_MS` | `700` | Required trailing silence duration before emitting `speech_end`. |
| `ASR_VAD_PREROLL_MS` | `300` | Audio kept before `speech_start` to avoid cutting the first syllable. |
| `ASR_VAD_MAX_UTTERANCE_MS` | `30000` | Maximum utterance duration before forced `speech_end`. |

Streaming policy environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_STREAM_PARTIAL_INTERVAL_MS` | `800` | Minimum interval between partial transcriptions. |
| `ASR_STREAM_MIN_PARTIAL_MS` | `600` | Minimum active speech duration before partial transcription starts. |

### Run Whisper Engine

`run_asr.sh` uses the Whisper engine by default:

```bash
./run_asr.sh
```

Equivalent explicit command:

```bash
./run_asr.sh whisper
```

Korean Whisper test with separate partial and final models:

```bash
ASR_WHISPER_MODEL=large-v3 \
ASR_WHISPER_PARTIAL_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

Partial results use `ASR_WHISPER_PARTIAL_MODEL`, and final results after `speech_end` use `ASR_WHISPER_MODEL`.

Whisper environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_WHISPER_MODEL` | `large-v3` | Final transcription model used after `speech_end`. |
| `ASR_WHISPER_PARTIAL_MODEL` | `small` | Partial transcription model used while speech is active. |
| `ASR_WHISPER_COMPUTE_TYPE` | `float16` | Final model compute type, for example `float16` or `int8`. |
| `ASR_WHISPER_PARTIAL_COMPUTE_TYPE` | same as `ASR_WHISPER_COMPUTE_TYPE` via `run_asr.sh`; unset when running the app directly | Partial model compute type. When unset, the code uses `ASR_WHISPER_COMPUTE_TYPE`. |
| `ASR_WHISPER_BEAM_SIZE` | `5` | Beam size for final transcription. |
| `ASR_WHISPER_PARTIAL_BEAM_SIZE` | `1` | Beam size for partial transcription. |
| `ASR_WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | Whisper no-speech threshold. Higher values can reduce silence/tail hallucination but may suppress quiet speech. |
| `ASR_WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | Whether Whisper conditions on previous decoded text. The default is `false` to reduce tail hallucination. |

GPU execution:

```bash
ASR_DEVICE=cuda \
ASR_WHISPER_COMPUTE_TYPE=float16 \
ASR_WHISPER_PARTIAL_COMPUTE_TYPE=float16 \
ASR_WHISPER_MODEL=small \
ASR_WHISPER_PARTIAL_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

CPU execution:

```bash
ASR_DEVICE=cpu \
ASR_WHISPER_COMPUTE_TYPE=int8 \
ASR_WHISPER_PARTIAL_COMPUTE_TYPE=int8 \
ASR_WHISPER_MODEL=small \
ASR_WHISPER_PARTIAL_MODEL=small \
ASR_TARGET_LANG=ko \
./run_asr.sh whisper
```

### Run NeMo Engine

Run with the default Korean NeMo model:

```bash
./run_asr.sh nemo
```

Set a NeMo pretrained model:

```bash
ASR_NEMO_MODEL_NAME=eesungkim/stt_kr_conformer_transducer_large \
./run_asr.sh nemo
```

Load a local `.nemo` checkpoint:

```bash
ASR_NEMO_MODEL_PATH=/path/to/model.nemo \
./run_asr.sh nemo
```

NeMo environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_NEMO_MODEL_NAME` | `eesungkim/stt_kr_conformer_transducer_large` via `run_asr.sh`; `nvidia/parakeet-tdt-0.6b-v2` when running the app directly | NeMo pretrained model name used when `ASR_NEMO_MODEL_PATH` is empty. |
| `ASR_NEMO_MODEL_PATH` | empty | Local `.nemo` checkpoint path. When set, this is loaded instead of `ASR_NEMO_MODEL_NAME`. |
| `ASR_NEMO_STRIP_LANG_TAGS` | `true` | Passes `strip_lang_tags` to NeMo multilingual transcribe calls when `ASR_TARGET_LANG` is set. |

Try NVIDIA's newer multilingual streaming RNNT model with a Korean language prompt:

```bash
ASR_NEMO_MODEL_NAME=nvidia/nemotron-3.5-asr-streaming-0.6b \
ASR_TARGET_LANG=ko-KR \
./run_asr.sh nemo
```

The project installs NeMo from GitHub main so models requiring newer NeMo classes can load.

Automatic language detection with the multilingual model:

```bash
ASR_NEMO_MODEL_NAME=nvidia/nemotron-3.5-asr-streaming-0.6b \
ASR_TARGET_LANG=auto \
./run_asr.sh nemo
```

### Bind Address

Override the bind address:

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
