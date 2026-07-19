import argparse

from asr_service.config import Settings, set_settings


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Run the ASR service.")
    parser.add_argument("engine", nargs="?", choices=["whisper", "nemo"], default=defaults.engine)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--device", default=defaults.device, choices=["cuda", "cpu"])
    parser.add_argument("--target-lang", default=defaults.target_lang)

    parser.add_argument("--nemo-model-name", default=defaults.nemo_model_name)
    parser.add_argument("--nemo-model-path", default=defaults.nemo_model_path)
    parser.add_argument("--nemo-strip-lang-tags", type=str_to_bool, default=defaults.nemo_strip_lang_tags)

    parser.add_argument("--whisper-model", default=defaults.whisper_model)
    parser.add_argument("--whisper-partial-model", default=defaults.whisper_partial_model)
    parser.add_argument("--whisper-beam-size", type=int, default=defaults.whisper_beam_size)
    parser.add_argument("--whisper-partial-beam-size", type=int, default=defaults.whisper_partial_beam_size)
    parser.add_argument("--whisper-no-speech-threshold", type=float, default=defaults.whisper_no_speech_threshold)
    parser.add_argument(
        "--whisper-condition-on-previous-text",
        type=str_to_bool,
        default=defaults.whisper_condition_on_previous_text,
    )

    parser.add_argument("--vad-aggressiveness", type=int, default=defaults.vad_aggressiveness)
    parser.add_argument("--vad-frame-ms", type=int, choices=[10, 20, 30], default=defaults.vad_frame_ms)
    parser.add_argument("--vad-start-trigger-ms", type=int, default=defaults.vad_start_trigger_ms)
    parser.add_argument("--vad-end-silence-ms", type=int, default=defaults.vad_end_silence_ms)
    parser.add_argument("--vad-preroll-ms", type=int, default=defaults.vad_preroll_ms)
    parser.add_argument("--vad-max-utterance-ms", type=int, default=defaults.vad_max_utterance_ms)

    parser.add_argument("--stream-partial-interval-ms", type=int, default=defaults.stream_partial_interval_ms)
    parser.add_argument("--stream-min-partial-ms", type=int, default=defaults.stream_min_partial_ms)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        engine=args.engine,
        device=args.device,
        target_lang=args.target_lang,
        nemo_model_name=args.nemo_model_name,
        nemo_model_path=args.nemo_model_path,
        nemo_strip_lang_tags=args.nemo_strip_lang_tags,
        whisper_model=args.whisper_model,
        whisper_partial_model=args.whisper_partial_model,
        whisper_beam_size=args.whisper_beam_size,
        whisper_partial_beam_size=args.whisper_partial_beam_size,
        whisper_no_speech_threshold=args.whisper_no_speech_threshold,
        whisper_condition_on_previous_text=args.whisper_condition_on_previous_text,
        vad_aggressiveness=args.vad_aggressiveness,
        vad_frame_ms=args.vad_frame_ms,
        vad_start_trigger_ms=args.vad_start_trigger_ms,
        vad_end_silence_ms=args.vad_end_silence_ms,
        vad_preroll_ms=args.vad_preroll_ms,
        vad_max_utterance_ms=args.vad_max_utterance_ms,
        stream_partial_interval_ms=args.stream_partial_interval_ms,
        stream_min_partial_ms=args.stream_min_partial_ms,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = settings_from_args(args)
    set_settings(settings)

    print(f"Starting ASR service on http://{args.host}:{args.port}")
    print(f"Web test page: http://localhost:{args.port}/")
    print(f"ASR engine: {settings.engine}")
    if settings.engine == "whisper":
        print(f"ASR final model: {settings.whisper_model}")
        print(f"ASR partial model: {settings.whisper_partial_model}")
    else:
        print(f"ASR model: {settings.nemo_model_path or settings.nemo_model_name}")
    if settings.target_lang:
        print(f"ASR target language: {settings.target_lang}")

    import uvicorn
    from asr_service.app import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
