import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from asr_service.asr import AsrEngine, create_asr_engine
from asr_service.config import Settings, get_settings
from asr_service.session import StreamingAsrSession
from asr_service.vad import UtteranceVad, VadConfig


def create_session(asr: AsrEngine, settings: Settings) -> StreamingAsrSession:
    vad_config = VadConfig(
        sample_rate=settings.sample_rate,
        frame_ms=settings.vad_frame_ms,
        aggressiveness=settings.vad_aggressiveness,
        start_trigger_ms=settings.start_trigger_ms,
        end_silence_ms=settings.end_silence_ms,
        preroll_ms=settings.preroll_ms,
        max_utterance_ms=settings.max_utterance_ms,
    )
    return StreamingAsrSession(
        asr=asr,
        vad=UtteranceVad(vad_config),
        partial_interval_ms=settings.partial_interval_ms,
        min_partial_ms=settings.min_partial_ms,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.asr = create_asr_engine(settings)
    yield


app = FastAPI(title="Speech Service - ASR", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "backend": settings.backend}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/v1/asr/stream")
async def asr_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = create_session(websocket.app.state.asr, websocket.app.state.settings)

    async def send_event(event: dict) -> bool:
        if websocket.client_state != WebSocketState.CONNECTED:
            return False
        try:
            await websocket.send_json(event)
            return True
        except (RuntimeError, WebSocketDisconnect):
            return False

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            if "bytes" in message and message["bytes"] is not None:
                try:
                    async for event in session.iter_audio_events(message["bytes"]):
                        if not await send_event(event):
                            return
                except Exception as exc:
                    if not await send_event({"type": "error", "message": str(exc)}):
                        return
                    continue
                continue

            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    if not await send_event({"type": "error", "message": "invalid JSON control message"}):
                        return
                    continue

                if control.get("type") == "end":
                    try:
                        async for event in session.iter_finish_events():
                            if not await send_event(event):
                                return
                    except Exception as exc:
                        if not await send_event({"type": "error", "message": str(exc)}):
                            return
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close()
                    return

                if not await send_event({"type": "error", "message": "unsupported control message"}):
                    return
    except WebSocketDisconnect:
        await session.finish()
