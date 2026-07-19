from asr_service.session import StreamingAsrSession
from asr_service.vad import UtteranceVad, VadConfig
import pytest


class ScriptedVad:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = decisions
        self.index = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        decision = self.decisions[self.index] if self.index < len(self.decisions) else False
        self.index += 1
        return decision


class FakeAsr:
    async def transcribe_async(self, audio) -> str:
        return f"text-{audio.size}"

    async def transcribe_partial_async(self, audio) -> str:
        return f"partial-{audio.size}"

    async def transcribe_final_async(self, audio) -> str:
        return f"final-{audio.size}"


def frame(config: VadConfig) -> bytes:
    return b"\x01\x00" * (config.frame_bytes // 2)


@pytest.mark.asyncio
async def test_session_emits_partial_and_final() -> None:
    config = VadConfig(
        frame_ms=20,
        start_trigger_ms=40,
        end_silence_ms=40,
        preroll_ms=20,
    )
    session = StreamingAsrSession(
        asr=FakeAsr(),
        vad=UtteranceVad(config, vad=ScriptedVad([True, True, True, False, False])),
        partial_interval_ms=1,
        min_partial_ms=40,
    )

    messages = []
    for _ in range(5):
        messages.extend(await session.accept_audio(frame(config)))

    assert [message["type"] for message in messages] == [
        "speech_start",
        "partial",
        "speech_end",
        "final",
    ]
    assert messages[0]["utterance_id"] == "utt-1"
    assert messages[2]["utterance_id"] == "utt-1"
