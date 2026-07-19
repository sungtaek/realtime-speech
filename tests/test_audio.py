import numpy as np

from asr_service.audio import audio_ms_from_bytes, pcm16le_to_float32


def test_pcm16le_to_float32() -> None:
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()

    audio = pcm16le_to_float32(pcm)

    assert audio.dtype == np.float32
    assert np.allclose(audio, [-1.0, 0.0, 32767 / 32768])


def test_audio_ms_from_bytes() -> None:
    assert audio_ms_from_bytes(32000) == 1000

