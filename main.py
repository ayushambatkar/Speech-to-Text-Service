import asyncio
import json
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.speech_to_text_service import SpeechToTextService

app = FastAPI(title="Flick S2T Streaming API", version="1.0.0")

# Single model instance – loaded once at startup.
# Swap model_size to "small", "medium", "large-v3", etc. for better accuracy.
stt = SpeechToTextService(model_size="base", device="cpu", compute_type="int8")


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
    Real-time transcription over WebSocket.

    **Protocol**  
    1. Client sends audio as **binary** frames (any format).  
       Multiple frames are concatenated, so you can stream chunks.  
    2. Client sends the UTF-8 text ``"DONE"`` to trigger transcription.  
    3. Server streams JSON messages (same shape as SSE: info / segment / done).  
    4. Client may send ``"RESET"`` to clear the buffer without transcribing.  
    5. Connection stays open - the cycle can repeat for multiple utterances.
    """
    await websocket.accept()
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            # ---- binary audio data ----
            if message.get("bytes"):
                audio_buffer.extend(message["bytes"])

            # ---- control commands ----
            elif message.get("text"):
                command = message["text"].strip().upper()

                if command == "DONE":
                    if not audio_buffer:
                        await websocket.send_json(
                            {"type": "error", "message": "Buffer is empty."}
                        )
                        continue

                    queue: asyncio.Queue = asyncio.Queue()
                    raw = bytes(audio_buffer)
                    audio_buffer.clear()

                    loop = asyncio.get_running_loop()
                    try:
                        audio = await loop.run_in_executor(
                            None, SpeechToTextService.decode_audio, raw
                        )
                    except Exception as exc:
                        await websocket.send_json(
                            {"type": "error", "message": f"Audio decode failed: {exc}"}
                        )
                        continue

                    asyncio.create_task(stt.transcribe_to_queue(audio, queue))

                    while True:
                        item = await queue.get()
                        if item is None:
                            await websocket.send_json({"type": "done"})
                            break
                        await websocket.send_json(item)

                elif command == "RESET":
                    audio_buffer.clear()
                    await websocket.send_json({"type": "reset"})

    except WebSocketDisconnect:
        pass



