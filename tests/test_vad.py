from asr_service.vad import UtteranceVad, VadConfig


class ScriptedVad:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = decisions
        self.index = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        decision = self.decisions[self.index] if self.index < len(self.decisions) else False
        self.index += 1
        return decision


def frame(config: VadConfig) -> bytes:
    return b"\x00" * config.frame_bytes


def test_vad_detects_start_and_end() -> None:
    config = VadConfig(
        frame_ms=20,
        start_trigger_ms=80,
        end_silence_ms=60,
        preroll_ms=40,
    )
    vad = UtteranceVad(
        config,
        vad=ScriptedVad([False, False, True, True, True, True, False, False, False]),
    )

    events = []
    for _ in range(9):
        events.extend(vad.accept(frame(config)))

    assert [event.type for event in events] == ["speech_start", "speech_end"]
    assert events[0].utterance_id == "utt-1"
    assert events[1].utterance_id == "utt-1"
    assert events[1].audio_ms >= 100


def test_vad_flush_ends_active_utterance() -> None:
    config = VadConfig(frame_ms=20, start_trigger_ms=40, end_silence_ms=200)
    vad = UtteranceVad(config, vad=ScriptedVad([True, True, True]))

    for _ in range(3):
        vad.accept(frame(config))

    events = vad.flush()

    assert len(events) == 1
    assert events[0].type == "speech_end"
    assert events[0].audio
