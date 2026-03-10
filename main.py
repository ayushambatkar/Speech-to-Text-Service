import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.speech_to_text_service import SpeechToTextService
from src.vosk_service import VoskTranscribeService

app = FastAPI(title="Flick S2T Streaming API", version="1.0.0")


# Which backend to use for transcription
class TranscribingService(str, Enum):
    whisper = "whisper"
    vosk = "vosk"


# Create backend instances at startup. Vosk requires a local model path;
# set via VOSK_MODEL_PATH env var or defaults to ./model
whisper_service = SpeechToTextService(model_size="base", device="cpu", compute_type="int8")
vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./model")
VOSK_MODEL_PATH="./models/vosk-model-small-en-in-0.4"
vosk_service = VoskTranscribeService(VOSK_MODEL_PATH, sample_rate=16000)

SERVICES = {"whisper": whisper_service, "vosk": vosk_service}

# Simple logger for websocket/audio events
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("s2t")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {"S2T Streaming API": "Live"}


# ---------------------------------------------------------------------------
# POST /transcribe  –  full response (non-streaming)
# ---------------------------------------------------------------------------


@app.post("/transcribe", summary="Transcribe audio file (full response)")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, webm, ogg, …)"),
    language: Optional[str] = Query(None, description="BCP-47 language code, e.g. 'en'. Auto-detect if omitted."),
    word_timestamps: bool = Query(False, description="Include per-word timestamps"),
    service: TranscribingService = Query(TranscribingService.vosk, description="Which transcription backend to use"),
):
    """
    Upload an audio file and receive the complete transcription as JSON once
    all segments have been processed.
    """
    audio_bytes = await file.read()
    loop = asyncio.get_running_loop()

    audio = await loop.run_in_executor(None, SpeechToTextService.decode_audio, audio_bytes)
    service_obj = SERVICES[service.value]

    segments_iter, info = await loop.run_in_executor(
        None, lambda: service_obj.transcribe(audio, language=language, word_timestamps=word_timestamps)
    )

    segments = []
    for seg in segments_iter:
        if isinstance(seg, dict):
            segments.append(seg)
        else:
            segments.append(SpeechToTextService.segment_to_dict(seg, word_timestamps))

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# POST /transcribe/stream  –  Server-Sent Events streaming
# ---------------------------------------------------------------------------


@app.post("/transcribe/stream", summary="Transcribe audio file (SSE streaming)")
async def transcribe_stream(
    file: UploadFile = File(..., description="Audio file (wav, mp3, webm, ogg, …)"),
    language: Optional[str] = Query(None, description="BCP-47 language code. Auto-detect if omitted."),
    word_timestamps: bool = Query(False, description="Include per-word timestamps"),
    service: TranscribingService = Query(TranscribingService.vosk, description="Which transcription backend to use"),
):
    """
    Upload an audio file and receive transcription segments as a
    **Server-Sent Events** stream.  Each event is a JSON object with a
    ``type`` field:

    * ``info``    - language detection metadata  
    * ``segment`` - one transcription segment  
    * ``done``    - signals end of stream  
    * ``error``   - an error occurred  
    """
    audio_bytes = await file.read()
    loop = asyncio.get_running_loop()

    audio = await loop.run_in_executor(
        None, SpeechToTextService.decode_audio, audio_bytes
    )
    queue: asyncio.Queue = asyncio.Queue()

    # Kick off transcription in background (pushes to queue)
    service_obj = SERVICES[service.value]
    asyncio.create_task(service_obj.transcribe_to_queue(audio, queue, language=language, word_timestamps=word_timestamps))

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# WS /ws/transcribe  -  WebSocket (send audio bytes then "DONE")
# ---------------------------------------------------------------------------


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, service: TranscribingService = Query(TranscribingService.vosk)):
    """
    Real-time transcription over WebSocket with **partial (incremental) results**.

    **Protocol**
    1. Client sends audio as **binary** frames (any format: wav, webm, ogg…).
       Frames are concatenated into a growing buffer.
    2. Each binary chunk immediately triggers a partial transcription pass
       (if no pass is already running). New segments are streamed back as
       soon as faster-whisper produces them — without waiting for DONE.
    3. Client sends the UTF-8 text ``"DONE"`` to finalise the utterance.
       The server waits for any in-flight transcription, emits any remaining
       segments, sends ``{"type": "done"}``, then resets for the next utterance.
    4. Client may send ``"RESET"`` to discard the buffer immediately.
    5. Connection stays open — the cycle repeats for each utterance.

    **Segment deduplication**: every transcription pass runs on the full
    accumulated audio, but only segments whose start time is ≥ the last
    emitted segment's end time are forwarded, so the client never receives
    the same text twice.
    """
    await websocket.accept()
    client = websocket.client or (None, None)
    logger.info("WebSocket connection accepted: %s", client)

    raw_buffer: bytearray = bytearray()  # latest audio chunk from client
    transcribe_lock = asyncio.Lock()  # prevents overlapping runs
    loop = asyncio.get_running_loop()
    service_obj = SERVICES[service.value]

    async def _run_partial(final: bool = False) -> None:
        """Decode the current chunk, transcribe, emit segments."""
        if not raw_buffer:
            return

        raw_snapshot = bytes(raw_buffer)  # snapshot current chunk/window

        try:
            # WS path receives:
            #   - first frame: 44-byte WAV header + PCM16
            #   - subsequent frames: raw PCM16 (mono, 16 kHz)
            pcm_bytes = raw_snapshot
            if pcm_bytes.startswith(b"RIFF") and pcm_bytes[8:12] == b"WAVE":
                # Strip standard 44-byte WAV header if present
                if len(pcm_bytes) <= 44:
                    return
                pcm_bytes = pcm_bytes[44:]

            # Decode PCM16 little-endian mono @16kHz directly into float32
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            # For partial passes, just log and wait for the next chunk.
            if not final:
                logger.info("Partial audio decode failed (waiting for more data): %s", exc)
                return
            # For final passes the client expects a result, so report the error.
            try:
                await websocket.send_json({"type": "error", "message": f"Audio decode failed: {exc}"})
            except Exception:
                logger.exception("Failed to send audio decode error to client")
            return

        # # For very small buffers there simply isn't enough audio for a
        # # meaningful partial transcription. Skip until we have at least ~1.5s.
        # try:
        #     duration_sec = len(audio) / 16_000.0
        # except Exception:
        #     duration_sec = 0.0
        # if not final and duration_sec < 1.5:
        #     return

        def _run_sync():
            if final:
                return service_obj.transcribe(audio, vad_filter=True, beam_size=5)
            return service_obj.transcribe_partial(audio, vad_filter=False, beam_size=2)

        try:
            segments, info = await loop.run_in_executor(None, _run_sync)
        except Exception as exc:
            logger.exception("Transcription failed")
            try:
                await websocket.send_json({"type": "error", "message": f"Transcription failed: {exc}"})
            except Exception:
                logger.exception("Failed to send transcription error to client")
            return

        # Normalize segments (Segment objects -> dicts) and emit them.
        for seg in segments:
            if not isinstance(seg, dict):
                seg = SpeechToTextService.segment_to_dict(seg, word_timestamps=False)

            try:
                await websocket.send_json({"type": "segment", **seg})
                now = datetime.now(timezone.utc)
                logger.info(
                    "Sent segment @%s start=%.3f end=%.3f text=%s",
                    now,
                    seg.get("start", 0.0),
                    seg.get("end", 0.0),
                    seg.get("text", "")[:160],
                )
            except WebSocketDisconnect:
                logger.info("Client disconnected while sending segment")
                raise
            except Exception:
                logger.exception("Failed to send segment to client")

        if final:
            try:
                await websocket.send_json({"type": "done"})
                logger.info("Utterance finalised and done sent")
            except Exception:
                logger.exception("Failed to send done message")
            raw_buffer.clear()

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WebSocketDisconnect received from client %s", client)
                break
            except RuntimeError as exc:
                # receive() may raise RuntimeError if a disconnect message
                # was already received; treat as closed.
                logger.info("RuntimeError receiving from websocket (treating as disconnect): %s", exc)
                break
            except Exception:
                logger.exception("Unexpected error while receiving websocket message; closing")
                break

            # ---- binary audio chunk ----
            if message.get("bytes"):
                chunk = message["bytes"]
                # Keep only the latest chunk from the client; they already
                # handle temporal chunking (e.g. 2–3s windows) on the frontend.
                raw_buffer.clear()
                raw_buffer.extend(chunk)
                logger.info("Received audio chunk (%d bytes)", len(chunk))

                # Trigger a partial transcription pass only if none is running.
                # If the lock is held, this chunk is silently accumulated and
                # will be included in the next available pass.
                if not transcribe_lock.locked():
                    try:
                        async with transcribe_lock:
                            await _run_partial(final=False)
                    except WebSocketDisconnect:
                        logger.info("Client disconnected during partial transcription")
                        break

            # ---- control commands ----
            elif message.get("text"):
                command = message["text"].strip().upper()

                if command == "DONE":
                    if not raw_buffer:
                        try:
                            await websocket.send_json({"type": "error", "message": "Buffer is empty."})
                        except Exception:
                            logger.exception("Failed to send empty-buffer error")
                        continue

                    # Wait for any in-flight partial pass, then do the final pass.
                    try:
                        async with transcribe_lock:
                            await _run_partial(final=True)
                    except WebSocketDisconnect:
                        logger.info("Client disconnected during final transcription")
                        break

                elif command == "RESET":
                    try:
                        async with transcribe_lock:
                            raw_buffer.clear()
                            last_emitted_end = 0.0
                        await websocket.send_json({"type": "reset"})
                        logger.info("Buffer reset by client")
                    except Exception:
                        logger.exception("Failed to reset buffer or notify client")

    except Exception:
        logger.exception("Unhandled exception in websocket handler")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket handler closed for %s", client)



