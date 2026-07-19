import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from asr_service.config import Settings


class AsrEngine(Protocol):
    async def transcribe_async(self, audio: np.ndarray) -> str:
        ...

    async def transcribe_partial_async(self, audio: np.ndarray) -> str:
        ...

    async def transcribe_final_async(self, audio: np.ndarray) -> str:
        ...


def _extract_text(result: Any) -> str:
    item = result[0] if isinstance(result, list) else result
    if hasattr(item, "text"):
        return str(item.text)
    return str(item)


@dataclass
class NemoAsrEngine:
    model_name: str
    model_path: str | None
    device: str
    target_lang: str | None = None
    strip_lang_tags: bool = True

    def __post_init__(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        if self.model_path:
            self.model = nemo_asr.models.ASRModel.restore_from(self.model_path)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "freeze"):
            self.model.freeze()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        kwargs: dict[str, Any] = {
            "audio": [audio],
            "batch_size": 1,
            "return_hypotheses": True,
            "verbose": False,
        }
        if self.target_lang:
            kwargs["target_lang"] = self.target_lang
            kwargs["strip_lang_tags"] = self.strip_lang_tags

        try:
            result = self.model.transcribe(**kwargs)
        except TypeError:
            kwargs.pop("target_lang", None)
            kwargs.pop("strip_lang_tags", None)
            result = self.model.transcribe(**kwargs)
        return _extract_text(result).strip()

    async def transcribe_async(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, audio)

    async def transcribe_partial_async(self, audio: np.ndarray) -> str:
        return await self.transcribe_async(audio)

    async def transcribe_final_async(self, audio: np.ndarray) -> str:
        return await self.transcribe_async(audio)


@dataclass
class FasterWhisperAsrEngine:
    model_name: str
    partial_model_name: str
    device: str
    language: str | None = None
    compute_type: str = "float16"
    partial_compute_type: str | None = None
    beam_size: int = 5
    partial_beam_size: int = 1
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False

    def __post_init__(self) -> None:
        self._final_device = self.device
        self._partial_device = self.device
        self._final_compute_type = self.compute_type
        self._partial_compute_type = self.partial_compute_type or self.compute_type
        self.final_model = self._load_model(self.model_name, final=True)
        self.partial_model = (
            self.final_model
            if self.partial_model_name == self.model_name
            else self._load_model(self.partial_model_name, final=False)
        )

    def _load_model(self, model_name: str, final: bool):
        from faster_whisper import WhisperModel

        device = self._final_device if final else self._partial_device
        compute_type = self._final_compute_type if final else self._partial_compute_type

        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"

        if device == "cpu" and compute_type in {"float16", "bfloat16"}:
            compute_type = "int8"

        if final:
            self._final_device = device
            self._final_compute_type = compute_type
        else:
            self._partial_device = device
            self._partial_compute_type = compute_type

        return WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_final(audio)

    def transcribe_partial(self, audio: np.ndarray) -> str:
        return self._transcribe_with_model(audio, final=False)

    def transcribe_final(self, audio: np.ndarray) -> str:
        return self._transcribe_with_model(audio, final=True)

    def _transcribe_with_model(self, audio: np.ndarray, final: bool) -> str:
        if audio.size == 0:
            return ""

        model = self.final_model if final else self.partial_model
        beam_size = self.beam_size if final else self.partial_beam_size
        try:
            segments, _info = model.transcribe(
                audio,
                language=_whisper_language(self.language),
                beam_size=beam_size,
                vad_filter=False,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
            )
        except RuntimeError as exc:
            device = self._final_device if final else self._partial_device
            if device != "cuda" or not _is_cuda_library_error(exc):
                raise
            if final:
                self._final_device = "cpu"
                self._final_compute_type = "int8"
                self.final_model = self._load_model(self.model_name, final=True)
                model = self.final_model
            else:
                self._partial_device = "cpu"
                self._partial_compute_type = "int8"
                self.partial_model = self._load_model(self.partial_model_name, final=False)
                model = self.partial_model
            segments, _info = model.transcribe(
                audio,
                language=_whisper_language(self.language),
                beam_size=beam_size,
                vad_filter=False,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
            )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe_async(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, audio)

    async def transcribe_partial_async(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe_partial, audio)

    async def transcribe_final_async(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe_final, audio)


def _whisper_language(language: str | None) -> str | None:
    if not language or language == "auto":
        return None
    if language.lower() in {"ko", "kor", "korean"}:
        return "ko"
    if language.lower() == "ko-kr":
        return "ko"
    return language


def _is_cuda_library_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "libcublas" in message or "libcudnn" in message or ("cuda" in message and "not found" in message)


def create_asr_engine(settings: Settings) -> AsrEngine:
    if settings.engine == "nemo":
        return NemoAsrEngine(
            model_name=settings.nemo_model_name,
            model_path=settings.nemo_model_path,
            device=settings.device,
            target_lang=settings.target_lang,
            strip_lang_tags=settings.nemo_strip_lang_tags,
        )

    if settings.engine == "whisper":
        return FasterWhisperAsrEngine(
            model_name=settings.whisper_model,
            partial_model_name=settings.whisper_partial_model,
            device=settings.device,
            language=settings.target_lang,
            compute_type=settings.whisper_compute_type,
            partial_compute_type=settings.whisper_partial_compute_type,
            beam_size=settings.whisper_beam_size,
            partial_beam_size=settings.whisper_partial_beam_size,
            no_speech_threshold=settings.whisper_no_speech_threshold,
            condition_on_previous_text=settings.whisper_condition_on_previous_text,
        )

    raise ValueError(f"unsupported ASR engine: {settings.engine}")
