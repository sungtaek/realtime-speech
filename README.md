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
python -m asr_service.cli whisper --device cpu
```

Python 3.10 or 3.11 may be more stable than 3.12 if NeMo dependency resolution fails on macOS.

The project uses `webrtcvad-wheels`, which provides the `webrtcvad` Python module with prebuilt wheels on common Python versions.

## Current Scope

- ASR: implemented. Streaming PCM WebSocket, WebRTC VAD, NeMo RNNT/TDT, and Whisper engines.
- TTS: planned. Future TTS scripts and endpoints should use a `tts` prefix, for example `run_tts.sh` and `/v1/tts/...`.

## Run ASR

Quick start with the default Whisper engine:

```bash
./run_asr.sh
```

Explicit engine selection:

```bash
./run_asr.sh whisper
./run_asr.sh nemo
```

`run_asr.sh` only accepts the engine name. Use the Python launcher when you need to change host, port, model, VAD, or streaming parameters:

```bash
python -m asr_service.cli whisper --host 0.0.0.0 --port 8000
```

Then open the browser test page:

```text
http://localhost:8000/
```

The test page supports both microphone input and audio file upload. Select an audio file, click `Send File`, and the browser will decode it and stream 16 kHz PCM chunks to the WebSocket endpoint.

Common parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `engine` | `whisper` | Positional ASR engine. Use `whisper` or `nemo`. |
| `--host` | `0.0.0.0` | Bind host. |
| `--port` | `8000` | Bind port. |
| `--device` | `cuda` | Inference device. Use `cuda` or `cpu`. The code falls back to CPU when CUDA is unavailable. |
| `--target-lang` | empty | Language hint. Whisper maps `ko`, `ko-KR`, and `korean` to `ko`; NeMo multilingual models can use values such as `ko-KR` or `auto`. |

VAD parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--vad-aggressiveness` | `2` | WebRTC VAD aggressiveness from `0` to `3`; higher is stricter. |
| `--vad-frame-ms` | `20` | VAD frame size in milliseconds. Must be `10`, `20`, or `30`. |
| `--vad-start-trigger-ms` | `160` | Required speech duration before emitting `speech_start`. |
| `--vad-end-silence-ms` | `700` | Required trailing silence duration before emitting `speech_end`. |
| `--vad-preroll-ms` | `300` | Audio kept before `speech_start` to avoid cutting the first syllable. |
| `--vad-max-utterance-ms` | `30000` | Maximum utterance duration before forced `speech_end`. |

Streaming policy parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--stream-partial-interval-ms` | `800` | Minimum interval between partial transcriptions. |
| `--stream-min-partial-ms` | `600` | Minimum active speech duration before partial transcription starts. |

### Run Whisper Engine

`run_asr.sh` uses the Whisper engine by default:

```bash
./run_asr.sh
```

Korean Whisper test with separate partial and final models:

```bash
python -m asr_service.cli whisper \
  --whisper-model large-v3 \
  --whisper-partial-model small \
  --target-lang ko
```

Partial results use `--whisper-partial-model`, and final results after `speech_end` use `--whisper-model`. Whisper compute type is selected automatically from the actual device: `float16` on CUDA and `int8` on CPU.

Whisper parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--whisper-model` | `large-v3` | Final transcription model used after `speech_end`. |
| `--whisper-partial-model` | `small` | Partial transcription model used while speech is active. |
| `--whisper-beam-size` | `5` | Beam size for final transcription. |
| `--whisper-partial-beam-size` | `1` | Beam size for partial transcription. |
| `--whisper-no-speech-threshold` | `0.6` | Whisper no-speech threshold. Higher values can reduce silence/tail hallucination but may suppress quiet speech. |
| `--whisper-condition-on-previous-text` | `false` | Whether Whisper conditions on previous decoded text. The default is `false` to reduce tail hallucination. |

GPU execution:

```bash
python -m asr_service.cli whisper \
  --device cuda \
  --whisper-model small \
  --whisper-partial-model small \
  --target-lang ko
```

CPU execution:

```bash
python -m asr_service.cli whisper \
  --device cpu \
  --whisper-model small \
  --whisper-partial-model small \
  --target-lang ko
```

### Run NeMo Engine

Run with the default Korean NeMo model:

```bash
./run_asr.sh nemo
```

Set a NeMo pretrained model:

```bash
python -m asr_service.cli nemo \
  --nemo-model-name eesungkim/stt_kr_conformer_transducer_large
```

Load a local `.nemo` checkpoint:

```bash
python -m asr_service.cli nemo \
  --nemo-model-path /path/to/model.nemo
```

NeMo parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--nemo-model-name` | `eesungkim/stt_kr_conformer_transducer_large` | NeMo pretrained model name used when `--nemo-model-path` is empty. |
| `--nemo-model-path` | empty | Local `.nemo` checkpoint path. When set, this is loaded instead of `--nemo-model-name`. |
| `--nemo-strip-lang-tags` | `true` | Passes `strip_lang_tags` to NeMo multilingual transcribe calls when `--target-lang` is set. |

Try NVIDIA's newer multilingual streaming RNNT model with a Korean language prompt:

```bash
python -m asr_service.cli nemo \
  --nemo-model-name nvidia/nemotron-3.5-asr-streaming-0.6b \
  --target-lang ko-KR
```

The project installs NeMo from GitHub main so models requiring newer NeMo classes can load.

Automatic language detection with the multilingual model:

```bash
python -m asr_service.cli nemo \
  --nemo-model-name nvidia/nemotron-3.5-asr-streaming-0.6b \
  --target-lang auto
```

### Bind Address

Override the bind address:

```bash
python -m asr_service.cli whisper --host 127.0.0.1 --port 9000
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
