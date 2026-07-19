from asr_service.asr import _whisper_compute_type_for_device, _whisper_language


def test_whisper_language_mapping() -> None:
    assert _whisper_language(None) is None
    assert _whisper_language("auto") is None
    assert _whisper_language("ko-KR") == "ko"
    assert _whisper_language("korean") == "ko"
    assert _whisper_language("en") == "en"


def test_whisper_compute_type_for_device() -> None:
    assert _whisper_compute_type_for_device("cuda") == "float16"
    assert _whisper_compute_type_for_device("cpu") == "int8"
