import numpy as np


def pcm16le_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if not pcm:
        return np.empty(0, dtype=np.float32)
    audio = np.frombuffer(pcm, dtype="<i2")
    return audio.astype(np.float32) / 32768.0


def audio_ms_from_bytes(byte_count: int, sample_rate: int = 16000) -> int:
    return int(byte_count / 2 / sample_rate * 1000)

