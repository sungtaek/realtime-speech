import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from asr_service.audio import pcm16le_to_float32
from asr_service.asr import AsrEngine
from asr_service.vad import UtteranceVad, VadEvent


@dataclass
class StreamingAsrSession:
    asr: AsrEngine
    vad: UtteranceVad
    partial_interval_ms: int
    min_partial_ms: int
    _last_partial_at: float = field(default=0.0, init=False)
    _last_partial_text: str = field(default="", init=False)

    async def accept_audio(self, chunk: bytes) -> list[dict]:
        return [event async for event in self.iter_audio_events(chunk)]

    async def iter_audio_events(self, chunk: bytes) -> AsyncIterator[dict]:
        for event in self.vad.accept(chunk):
            async for message in self._iter_vad_event(event):
                yield message
        partial = await self._maybe_partial()
        if partial:
            yield partial

    async def finish(self) -> list[dict]:
        return [event async for event in self.iter_finish_events()]

    async def iter_finish_events(self) -> AsyncIterator[dict]:
        for event in self.vad.flush():
            async for message in self._iter_vad_event(event):
                yield message

    async def _iter_vad_event(self, event: VadEvent) -> AsyncIterator[dict]:
        if event.type == "speech_start":
            self._last_partial_at = 0.0
            self._last_partial_text = ""
            yield {
                "type": "speech_start",
                "utterance_id": event.utterance_id,
                "time_ms": event.time_ms,
            }
            return

        if event.type == "speech_end":
            yield {
                "type": "speech_end",
                "utterance_id": event.utterance_id,
                "audio_ms": event.audio_ms,
                "time_ms": event.time_ms,
            }
            text = await self.asr.transcribe_final_async(pcm16le_to_float32(event.audio))
            yield {
                "type": "final",
                "utterance_id": event.utterance_id,
                "text": text,
                "audio_ms": event.audio_ms,
            }
            return

    async def _maybe_partial(self) -> dict | None:
        if self.vad.trailing_silence_ms:
            return None

        audio_ms = self.vad.current_audio_ms
        if audio_ms < self.min_partial_ms:
            return None
        now = time.monotonic()
        if self._last_partial_at and (now - self._last_partial_at) * 1000 < self.partial_interval_ms:
            return None

        audio = self.vad.current_audio
        if not audio:
            return None

        text = await self.asr.transcribe_partial_async(pcm16le_to_float32(audio))
        self._last_partial_at = now
        if text == self._last_partial_text:
            return None
        self._last_partial_text = text
        return {
            "type": "partial",
            "utterance_id": self.vad.current_utterance_id,
            "text": text,
            "audio_ms": audio_ms,
        }
