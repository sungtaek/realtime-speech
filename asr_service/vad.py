from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VadEvent:
    type: str
    utterance_id: str | None = None
    audio: bytes = b""
    audio_ms: int = 0
    time_ms: int = 0


@dataclass
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2
    start_trigger_ms: int = 160
    end_silence_ms: int = 700
    preroll_ms: int = 300
    max_utterance_ms: int = 30000

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    @property
    def start_frames(self) -> int:
        return max(1, self.start_trigger_ms // self.frame_ms)

    @property
    def end_silence_frames(self) -> int:
        return max(1, self.end_silence_ms // self.frame_ms)

    @property
    def preroll_frames(self) -> int:
        return max(0, self.preroll_ms // self.frame_ms)

    @property
    def max_utterance_frames(self) -> int:
        return max(1, self.max_utterance_ms // self.frame_ms)


class WebRtcVad:
    def __init__(self, aggressiveness: int) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "webrtcvad is required. Install project dependencies again: "
                "pip install -e \".[test]\""
            ) from exc

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(frame, sample_rate)


@dataclass
class UtteranceVad:
    config: VadConfig
    vad: object | None = None
    _pending: bytearray = field(default_factory=bytearray, init=False)
    _preroll: deque[bytes] = field(default_factory=deque, init=False)
    _speech_votes: deque[bool] = field(default_factory=deque, init=False)
    _speech_frames: list[bytes] = field(default_factory=list, init=False)
    _in_speech: bool = field(default=False, init=False)
    _silence_frames: int = field(default=0, init=False)
    _frames_seen: int = field(default=0, init=False)
    _utterance_seq: int = field(default=0, init=False)
    _current_utterance_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.vad is None:
            self.vad = WebRtcVad(self.config.aggressiveness)
        self._preroll = deque(maxlen=self.config.preroll_frames)
        self._speech_votes = deque(maxlen=self.config.start_frames)

    @property
    def current_audio(self) -> bytes:
        return b"".join(self._trimmed_current_frames())

    @property
    def current_audio_ms(self) -> int:
        return len(self._trimmed_current_frames()) * self.config.frame_ms

    @property
    def trailing_silence_ms(self) -> int:
        return self._silence_frames * self.config.frame_ms

    @property
    def current_utterance_id(self) -> str | None:
        return self._current_utterance_id

    def _trimmed_current_frames(self) -> list[bytes]:
        if self._silence_frames <= 0:
            return self._speech_frames
        keep_count = max(0, len(self._speech_frames) - self._silence_frames)
        return self._speech_frames[:keep_count]

    def accept(self, chunk: bytes) -> list[VadEvent]:
        self._pending.extend(chunk)
        events: list[VadEvent] = []
        while len(self._pending) >= self.config.frame_bytes:
            frame = bytes(self._pending[: self.config.frame_bytes])
            del self._pending[: self.config.frame_bytes]
            events.extend(self._accept_frame(frame))
        return events

    def flush(self) -> list[VadEvent]:
        if self._in_speech and self._speech_frames:
            return [self._end_event(trim_trailing_silence=True)]
        self._pending.clear()
        return []

    def _accept_frame(self, frame: bytes) -> list[VadEvent]:
        self._frames_seen += 1
        is_speech = bool(self.vad.is_speech(frame, self.config.sample_rate))
        if self._in_speech:
            return self._accept_speech_frame(frame, is_speech)
        return self._accept_idle_frame(frame, is_speech)

    def _accept_idle_frame(self, frame: bytes, is_speech: bool) -> list[VadEvent]:
        self._preroll.append(frame)
        self._speech_votes.append(is_speech)
        if len(self._speech_votes) < self.config.start_frames:
            return []
        speech_count = sum(1 for vote in self._speech_votes if vote)
        if speech_count < max(1, int(self.config.start_frames * 0.75)):
            return []

        self._utterance_seq += 1
        self._current_utterance_id = f"utt-{self._utterance_seq}"
        self._in_speech = True
        self._silence_frames = 0
        self._speech_frames = list(self._preroll)
        event = VadEvent(
            type="speech_start",
            utterance_id=self._current_utterance_id,
            time_ms=self._frames_seen * self.config.frame_ms,
        )
        return [event]

    def _accept_speech_frame(self, frame: bytes, is_speech: bool) -> list[VadEvent]:
        self._speech_frames.append(frame)
        self._silence_frames = 0 if is_speech else self._silence_frames + 1
        if self._silence_frames >= self.config.end_silence_frames:
            return [self._end_event(trim_trailing_silence=True)]
        if len(self._speech_frames) >= self.config.max_utterance_frames:
            return [self._end_event()]
        return []

    def _end_event(self, trim_trailing_silence: bool = False) -> VadEvent:
        utterance_id = self._current_utterance_id
        frames = self._speech_frames
        if trim_trailing_silence and self._silence_frames > 0:
            keep_count = max(0, len(frames) - self._silence_frames)
            frames = frames[:keep_count]
        audio = b"".join(frames)
        audio_ms = len(frames) * self.config.frame_ms
        self._pending.clear()
        self._preroll.clear()
        self._speech_votes.clear()
        self._speech_frames = []
        self._in_speech = False
        self._silence_frames = 0
        self._current_utterance_id = None
        return VadEvent(
            type="speech_end",
            utterance_id=utterance_id,
            audio=audio,
            audio_ms=audio_ms,
            time_ms=self._frames_seen * self.config.frame_ms,
        )
