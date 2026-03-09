import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.speech_to_text_service import SpeechToTextService

app = FastAPI(title="Flick S2T Streaming API", version="1.0.0")

# Single model instance – loaded once at startup.
# Swap model_size to "small", "medium", "large-v3", etc. for better accuracy.
stt = SpeechToTextService(model_size="base", device="cpu", compute_type="int8")

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
):
    """
    Upload an audio file and receive the complete transcription as JSON once
    all segments have been processed.
    """
    audio_bytes = await file.read()
    loop = asyncio.get_running_loop()

    audio = await loop.run_in_executor(
        None, SpeechToTextService.decode_audio, audio_bytes
    )
    segments_iter, info = await loop.run_in_executor(
        None,
        lambda: stt.transcribe(audio, language=language, word_timestamps=word_timestamps),
    )

    segments = []
    for seg in segments_iter:
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
    asyncio.create_task(
        stt.transcribe_to_queue(audio, queue, language=language, word_timestamps=word_timestamps)
    )

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
async def websocket_transcribe(websocket: WebSocket):
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

    raw_buffer: bytearray = bytearray()   # growing raw audio bytes
    last_emitted_end: float = 0.0         # dedup cursor
    transcribe_lock = asyncio.Lock()      # prevents overlapping runs
    loop = asyncio.get_running_loop()

    async def _run_partial(final: bool = False) -> None:
        """Decode the current buffer, transcribe, emit only new segments."""
        nonlocal last_emitted_end

        if not raw_buffer:
            return

        raw_snapshot = bytes(raw_buffer)  # snapshot so buffer can keep growing

        try:
            audio = await loop.run_in_executor(
                None, SpeechToTextService.decode_audio, raw_snapshot
            )
        except Exception as exc:
            await websocket.send_json(
                {"type": "error", "message": f"Audio decode failed: {exc}"}
            )
            return

        try:
            segments, info = await loop.run_in_executor(
                None,
                lambda: stt.transcribe_sync(audio),
            )
        except Exception as exc:
            logger.exception("Transcription failed")
            try:
                await websocket.send_json({"type": "error", "message": f"Transcription failed: {exc}"})
            except Exception:
                logger.exception("Failed to send transcription error to client")
            return

        # Emit only segments beyond the dedup cursor
        for seg in segments:
            if seg["start"] >= last_emitted_end - 0.05:  # 50 ms tolerance
                try:
                    await websocket.send_json({"type": "segment", **seg})
                    now = datetime.now(datetime.timezone.utc)
                    logger.info("Sent segment @%s start=%.3f end=%.3f text=%s", now, seg["start"], seg["end"], seg["text"][:160])
                except WebSocketDisconnect:
                    logger.info("Client disconnected while sending segment")
                    raise
                except Exception:
                    logger.exception("Failed to send segment to client")
                last_emitted_end = seg["end"]

        if final:
            try:
                await websocket.send_json({"type": "done"})
                logger.info("Utterance finalised and done sent")
            except Exception:
                logger.exception("Failed to send done message")
            last_emitted_end = 0.0
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
                raw_buffer.extend(chunk) # append to buffer
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



